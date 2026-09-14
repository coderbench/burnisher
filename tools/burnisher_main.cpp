// burnisher -- the native runtime. One binary, a handful of subcommands.
//
// Every subcommand that measures anything prints exactly ONE line of the form
//
//     BURNISH_JSON: {...}
//
// and the harness parses that line and nothing else. Parsing a number out of prose is how a
// harness comes to silently accept a changed output format, so a missing or duplicated line is
// an error on the harness side rather than a fallback to a regex.
//
// The JSON always carries an `effective` block saying what the run ACTUALLY did -- which
// implementation of each op, which dtype, which shape. eval/runner.py compares that against what
// it asked for and refuses the run if they differ. An arm that silently fell back produces a
// perfectly good number for a configuration nobody requested, and that is invisible from outside
// the process.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "burnisher/device.h"
#include "burnisher/models.h"
#include "burnisher/ops.h"
#include "burnisher/pipeline.h"
#include "burnisher/scheduler.h"

using namespace burnisher;

#ifdef BURNISHER_CUDA
namespace burnisher {
int probe_device_main();
}
#endif

namespace {

const char* kUsage = R"(burnisher -- a native C++/CUDA image generation runtime for Blackwell

usage: burnisher <command> [options]

  info                      build configuration and every registered op implementation
  selftest                  run the whole graph on synthetic weights at a small config
  generate                  one image, fixed seed, end to end
  bench                     time one stage, for the harness
  probe                     measure this device's sustained bandwidth and FLOPS
  weights-manifest          every tensor this runtime requires of a checkpoint, as JSON
  check-weights             load a real checkpoint and verify every required tensor
  noise                     the pinned initial latent for a seed, as .npy
  decode                    run the VAE decoder on a latent from a .npy file
  dit-step                  one denoiser forward pass, from .npy inputs
  encode                    run the text encoder on token ids from a file
  schedule                  run the sampler on a fixed synthetic trajectory, as .npy

common options:
  --impl NAME               registered implementation to use (default: stock)
  --device cpu|cuda         where tensors live (default: cpu). A CUDA kernel over host
                            tensors is a fault, not a slow path, so the two must agree.
  --dtype bf16|fp32         compute dtype (default: bf16)
  --resolution N            image side in pixels (default: 1024)
  --steps N                 denoise steps (default: 20)
  --seed N                  fixed seed (default: 20260911)
  --token-ids FILE          T5 token ids, one prompt per line (negative first under CFG)
  --noise FILE              the pinned initial latent (from `burnisher noise`). REQUIRED for a
                            gate run: regenerating it at the compute dtype starts the two sides
                            from different points and the comparison then measures the noise.
  --dump-latents FILE       write the denoised LATENT as .npy -- what the gate compares
  --dump-pixels FILE        write the decoded image as .npy
  --weights DIR             checkpoint directory (transformer/ vae/ text_encoder/);
                            omit to use synthetic weights, which exercise the graph but say
                            nothing about the loader or the tensor-name mapping
  --help                    this message

Scoring a change to this runtime is `burnish`, the harness in tools/. This binary only runs and
reports; it does not decide whether a number is good.
)";

struct Args {
    std::string command;
    std::map<std::string, std::string> opt;
    bool has(const std::string& k) const { return opt.count(k) != 0; }
    std::string get(const std::string& k, const std::string& d = "") const {
        auto it = opt.find(k);
        return it == opt.end() ? d : it->second;
    }
    long num(const std::string& k, long d) const {
        auto it = opt.find(k);
        return it == opt.end() ? d : std::stol(it->second);
    }
    double real(const std::string& k, double d) const {
        auto it = opt.find(k);
        return it == opt.end() ? d : std::stod(it->second);
    }
};

Args parse(int argc, char** argv) {
    Args a;
    int i = 1;
    if (i < argc && argv[i][0] != '-') a.command = argv[i++];
    for (; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) continue;
        std::string key = s.substr(2);
        const size_t eq = key.find('=');
        if (eq != std::string::npos) {
            a.opt[key.substr(0, eq)] = key.substr(eq + 1);
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
            a.opt[key] = argv[++i];
        } else {
            a.opt[key] = "1";
        }
    }
    return a;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

std::string json_map(const std::map<std::string, std::string>& m) {
    std::ostringstream os;
    os << "{";
    bool first = true;
    for (const auto& kv : m) {
        os << (first ? "" : ",") << "\"" << json_escape(kv.first) << "\":\""
           << json_escape(kv.second) << "\"";
        first = false;
    }
    os << "}";
    return os.str();
}

std::string stats_json(const OutputStats& s) {
    std::ostringstream os;
    os.precision(10);
    os << "{\"latent_mean\":" << s.latent_mean << ",\"latent_std\":" << s.latent_std
       << ",\"latent_absmax\":" << s.latent_absmax << "}";
    return os.str();
}

// Minimal .npy v1.0. Enough for a contiguous little-endian fp32 tensor, which is all that
// crosses this boundary; numpy on the other side does the rest.
void write_npy(const std::string& path, const Tensor& t) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < t.rank(); ++i) hdr << t.dim(i) << ", ";
    hdr << "), }";
    std::string h = hdr.str();
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    out.write(reinterpret_cast<const char*>(magic), 8);
    const uint16_t len = static_cast<uint16_t>(h.size());
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (int64_t i = 0; i < t.numel(); ++i) {
        const float v = t.get(i);
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
}

// Minimal .npy v1.0 reader. Contiguous little-endian fp32 only, which is what crosses this
// boundary; anything else is refused rather than reinterpreted.
Tensor read_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error(path + " is not a .npy file");
    }
    uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 2);
    std::string header(len, '\0');
    in.read(&header[0], len);
    if (header.find("'<f4'") == std::string::npos &&
        header.find("\"<f4\"") == std::string::npos) {
        throw std::runtime_error(path + ": only little-endian fp32 is read. Converting silently "
                                        "would change the thing being compared.");
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error(path + ": Fortran order is not read");
    }
    const size_t open_paren = header.find('(');
    const size_t close_paren = header.find(')', open_paren);
    std::vector<int64_t> shape;
    {
        std::string dims = header.substr(open_paren + 1, close_paren - open_paren - 1);
        std::string tok;
        std::istringstream is(dims);
        while (std::getline(is, tok, ',')) {
            const size_t a = tok.find_first_not_of(" \t");
            if (a == std::string::npos) continue;
            shape.push_back(std::stoll(tok.substr(a)));
        }
    }
    Tensor t(shape, DType::F32);
    for (int64_t i = 0; i < t.numel(); ++i) {
        float v;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!in) throw std::runtime_error(path + ": truncated");
        t.set(i, v);
    }
    return t;
}

size_t peak_rss_bytes() {
    // Host peak RSS: the memory objective for a run on the CPU. A run on the device reports the
    // device allocator's high-water mark instead -- see `peak_memory_bytes`.
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return static_cast<size_t>(std::stoul(line.substr(6))) * 1024;
        }
    }
    return 0;
}

// The memory objective the frontier scores, for the resource this run actually used. It was
// host RSS on every run, including runs on the device, so the "peak VRAM" of a CUDA run was how
// much host memory the process touched -- the wrong resource, scored as if it were the right one.
size_t peak_memory_bytes(Device d) {
    return d == Device::CUDA ? device::peak_allocated_bytes() : peak_rss_bytes();
}

void device_sync_if(Device d) {
    if (d == Device::CUDA) device::synchronize();
}

Device device_arg(const Args& a) {
    const std::string d = a.get("device", "cpu");
    if (d == "cpu") return Device::CPU;
    if (d == "cuda") return Device::CUDA;
    throw std::runtime_error("--device must be cpu or cuda, not '" + d + "'");
}

// Weights on the device the model will run on. The host source stays alive because the device
// copy is made from it lazily, per tensor, on first use.
std::shared_ptr<WeightSource> place(std::shared_ptr<WeightSource> host, Device dev, DType dt,
                                    std::shared_ptr<WeightSource>* keep_alive) {
    if (dev != Device::CUDA) return host;
    *keep_alive = host;
    return std::make_shared<DeviceWeights>(*host, dt);
}

DType dtype_arg(const Args& a) {
    DType t;
    const std::string name = a.get("dtype", "bf16");
    if (!dtype_from_name(name, &t)) {
        throw std::runtime_error("unknown dtype '" + name + "'");
    }
    return t;
}

// A small-but-structurally-identical config, for running the whole graph without 22 GB of
// weights. Same op sequence, same control flow, same shapes-modulo-size -- so `selftest` really
// does exercise the code a real run takes, which is the only reason it is worth having.
void small_configs(T5Config* t5, DiTConfig* dit, VaeConfig* vae) {
    t5->num_layers = 2; t5->d_model = 64; t5->d_ff = 128; t5->d_kv = 8; t5->num_heads = 8;
    t5->vocab_size = 512;
    dit->num_layers = 2; dit->num_heads = 4; dit->head_dim = 16; dit->caption_channels = 64;
    dit->sample_size = 8;
    vae->block_out_channels = {8, 16, 16, 16};
    vae->layers_per_block = 1;
    // The real checkpoint uses 32 groups over >=128 channels. At eight channels that is not a
    // group norm, it is an error, so the small config scales the group count down with the
    // channel count rather than keeping a constant that cannot divide it.
    vae->norm_num_groups = 4;
}

int cmd_info(const Args&) {
    // Registration is an explicit call, so `info` has to make it too -- otherwise it reports an
    // empty build and the first thing anyone learns about this binary is wrong.
    register_all_ops();
    std::cout << "burnisher 0.1.0\n";
#ifdef BURNISHER_CUDA
    std::cout << "cuda: enabled\n";
#else
    std::cout << "cuda: DISABLED (built without BURNISHER_BUILD_CUDA)\n";
#endif
    std::cout << "\nregistered implementations:\n";
    for (const auto& op : list_all_impls()) {
        std::cout << "  " << op.op << "\n";
        for (const auto& impl : op.impls) {
            std::cout << "    " << impl.name << "  -- " << impl.doc << "\n";
        }
    }
    std::cout << "\nAn implementation registers a NAME beside the existing one rather than\n"
                 "replacing a file, so base and candidate run in one process, one model load,\n"
                 "one thermal state. `burnisher bench --impl NAME` fails if NAME is absent.\n";
    return 0;
}

int cmd_selftest(const Args& a) {
    register_all_ops();
    T5Config t5; DiTConfig dit; VaeConfig vae;
    small_configs(&t5, &dit, &vae);
    SchedulerConfig sched;

    auto weights = std::make_shared<SyntheticWeights>(DType::F32);
    declare_pixart_shapes(*weights, t5, dit, vae);

    PipelineConfig cfg;
    cfg.resolution = static_cast<int>(vae.scale_factor()) * dit.sample_size;
    cfg.steps = static_cast<int>(a.num("steps", 3));
    cfg.caption_len = 6;
    cfg.compute = DType::F32;
    cfg.impl = a.get("impl", "stock");

    Pipeline p(cfg, weights, weights, weights, t5, dit, vae, sched);
    Tensor ids({2, cfg.caption_len}, DType::F32);
    for (int64_t i = 0; i < ids.numel(); ++i) ids.set(i, static_cast<float>((i * 7) % 500));

    StageTimings t{};
    Tensor pixels = p.generate(ids, &t);
    const OutputStats s = OutputStats::of(pixels);

    std::cout << "selftest: " << pixels.describe() << " at " << cfg.resolution << "px, "
              << cfg.steps << " steps\n"
              << "  text " << t.text_encode_s * 1e3 << " ms, denoise " << t.denoise_s * 1e3
              << " ms, vae " << t.vae_decode_s * 1e3 << " ms\n"
              << "  mean " << s.latent_mean << " std " << s.latent_std
              << " absmax " << s.latent_absmax << "\n";
    if (!(s.latent_std > 1e-6) || !std::isfinite(s.latent_absmax)) {
        std::cerr << "!! the pipeline produced a constant or non-finite output; it ran and "
                     "generated nothing\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}

int cmd_bench(const Args& a) {
    register_all_ops();
    const std::string stage = a.get("stage", "dit-step");
    const std::string impl = a.get("impl", "stock");
    const DType dt = dtype_arg(a);
    const Device dev = device_arg(a);
    const int64_t resolution = a.num("resolution", 1024);
    const int64_t caption_len = a.num("caption-len", 300);
    const int64_t batch = a.num("batch", 2);
    const int warmup = static_cast<int>(a.num("warmup", 3));
    const int iters = static_cast<int>(a.num("iters", 10));

    // Fails here rather than falling back, and the message lists what IS registered.
    const ImplSelection impls = ImplSelection::from_request(impl, dev);

    T5Config t5; DiTConfig dit; VaeConfig vae;
    if (a.has("small")) small_configs(&t5, &dit, &vae);

    // Synthetic weights exercise the graph; real ones exercise the loader, the tensor-name
    // mapping and the dtypes. Both are useful and they answer different questions, so which one
    // was used is reported in `effective` rather than inferred.
    std::shared_ptr<WeightSource> host_weights;
    const std::string weights_dir = a.get("weights");
    if (!weights_dir.empty()) {
        if (a.has("small")) {
            throw std::runtime_error("--small and --weights are contradictory: a real checkpoint "
                                     "has the real geometry");
        }
        const std::string component = (stage == "t5-encode") ? "text_encoder"
                                    : (stage == "vae-decode") ? "vae" : "transformer";
        host_weights = std::make_shared<CheckpointWeights>(
            CheckpointWeights::component(weights_dir, component));
    } else {
        auto synth = std::make_shared<SyntheticWeights>(dt);
        declare_pixart_shapes(*synth, t5, dit, vae);
        host_weights = synth;
    }
    // Placed on the device the stage will run on. Every other command does this; `bench` did
    // not, so a CUDA run here met a host weight table and faulted -- which is the guard working,
    // but only after the calibration had already started.
    std::shared_ptr<WeightSource> weights_keep;
    std::shared_ptr<WeightSource> weights =
        place(host_weights, dev, dt, &weights_keep);

    const int64_t f = vae.scale_factor();
    const int64_t latent = resolution / f;
    using clock = std::chrono::steady_clock;

    std::vector<double> times;
    OutputStats stats{};
    // Inputs are built on the HOST -- they are filled with scalar stores -- and then placed.
    // A tensor filled by a host loop and handed straight to a CUDA kernel is the fault this
    // runtime's device boundary exists to make loud.
    const auto place_in = [&](Tensor t) {
        return (dev == Device::CUDA) ? t.to_device() : t;
    };
    const auto run_stage = [&]() -> Tensor {
        if (stage == "t5-encode") {
            T5Encoder e(t5, *weights, dt);
            Tensor ids({batch, caption_len}, dt);
            for (int64_t i = 0; i < ids.numel(); ++i)
                ids.set(i, static_cast<float>((i * 7) % (t5.vocab_size - 1)));
            Tensor ids_f({batch, caption_len}, DType::F32);
            for (int64_t i = 0; i < ids.numel(); ++i) ids_f.set(i, ids.get(i));
            Tensor d_ids = place_in(ids_f);
            return e.forward(d_ids, impls);
        }
        if (stage == "dit-step") {
            PixArtDiT d(dit, *weights, dt);
            Tensor z({batch, dit.in_channels, latent, latent}, dt);
            for (int64_t i = 0; i < z.numel(); ++i)
                z.set(i, static_cast<float>(std::sin(static_cast<double>(i) * 0.37)));
            Tensor cap({batch, caption_len, dit.caption_channels}, dt);
            for (int64_t i = 0; i < cap.numel(); ++i)
                cap.set(i, static_cast<float>(std::cos(static_cast<double>(i) * 0.11)));
            // Three quarters padding, which is what a short caption in a 300-token window looks
            // like. A bench that masked nothing would time a cheaper attention than the pipeline
            // runs and would not notice.
            Tensor mask({batch, caption_len}, dt);
            for (int64_t b = 0; b < batch; ++b)
                for (int64_t i = 0; i < caption_len; ++i)
                    mask.set(b * caption_len + i,
                             i < std::max<int64_t>(1, caption_len / 4) ? 1.0f : 0.0f);
            Tensor dz = place_in(z), dcap = place_in(cap), dmask = place_in(mask);
            return d.forward(dz, 500.0, dcap, dmask, impls);
        }
        if (stage == "vae-decode") {
            VaeDecoder v(vae, *weights, dt);
            Tensor z({1, vae.latent_channels, latent, latent}, dt);
            for (int64_t i = 0; i < z.numel(); ++i)
                z.set(i, static_cast<float>(std::sin(static_cast<double>(i) * 0.21)));
            Tensor dz = place_in(z);
            return v.forward(dz, impls);
        }
        throw std::runtime_error("unknown stage '" + stage + "'");
    };

    for (int i = 0; i < warmup; ++i) run_stage();
    device_sync_if(dev);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = clock::now();
        Tensor out = run_stage();
        // Timed across a synchronise, or a CUDA measurement times the launch queue rather than
        // the work -- which reads as a spectacular and entirely fictional speedup.
        device_sync_if(dev);
        const auto t1 = clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
        if (i == 0) stats = OutputStats::of((dev == Device::CUDA) ? out.to_host() : out);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];

    std::map<std::string, std::string> effective{
        {"stage", stage}, {"dtype", dtype_name(dt)}, {"impl", impl},
        {"resolution", std::to_string(resolution)}, {"batch", std::to_string(batch)},
        {"caption_len", std::to_string(caption_len)},
        {"weights", weights_dir.empty() ? "synthetic" : "checkpoint"},
        {"device", a.get("device", "cpu")},
    };
    for (const auto& kv : impls.as_map()) effective["impl." + kv.first] = kv.second;

    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << median
       << ",\"peak_vram_bytes\":" << peak_memory_bytes(device_arg(a))
       << ",\"iters\":" << iters << "},\"effective\":" << json_map(effective)
       << ",\"output_stats\":" << stats_json(stats) << "}";
    std::cout << os.str() << "\n";
    return 0;
}

// What this runtime requires of a checkpoint: every tensor name and shape, from the same
// declaration the models are built against.
//
// It exists so the claim "these tensor names are right" can stop being an assertion. The names
// here were written from the reference implementation's module structure, which is usually right
// and is not evidence; `scripts/verify_checkpoint_layout.py` fetches the real checkpoint's
// safetensors header by HTTP range request -- 68 kB rather than 22 GB -- and compares.
int cmd_weights_manifest(const Args& a) {
    T5Config t5; DiTConfig dit; VaeConfig vae;
    if (a.has("small")) small_configs(&t5, &dit, &vae);
    SyntheticWeights w(DType::F32);
    declare_pixart_shapes(w, t5, dit, vae);

    // Which component each tensor belongs to, so the checker knows which shard to look in.
    const auto component = [](const std::string& n) -> const char* {
        if (n.rfind("encoder.", 0) == 0 || n == "shared.weight") return "text_encoder";
        if (n.rfind("decoder.", 0) == 0 || n.rfind("post_quant_conv", 0) == 0) return "vae";
        return "transformer";
    };

    std::ostringstream os;
    os << "{\n \"_what\": \"Every tensor this runtime requires of a checkpoint, from the same "
          "declaration the models are built against. Generated by `burnisher weights-manifest`; "
          "checked against a real checkpoint by scripts/verify_checkpoint_layout.py.\",\n"
       << " \"tensors\": {\n";
    bool first = true;
    for (const auto& kv : w.declared()) {
        os << (first ? "" : ",\n") << "  \"" << kv.first << "\": {\"component\": \""
           << component(kv.first) << "\", \"shape\": [";
        for (size_t i = 0; i < kv.second.size(); ++i) os << (i ? "," : "") << kv.second[i];
        os << "]}";
        first = false;
    }
    os << "\n }\n}";
    std::cout << os.str() << "\n";
    return 0;
}

// Load a real checkpoint through the real loader and check every tensor the models will ask for.
//
// `scripts/verify_checkpoint_layout.py` checks the same thing against the checkpoint's HEADER,
// over the network, without downloading it. This checks it through `SafeTensors` -- the mmap, the
// offset arithmetic, the dtype mapping, the shard search -- against bytes on disk. The two
// overlap on purpose: the cheap one runs in CI on every push, and this one is what you run once
// after downloading 22 GB, before discovering at step 19 of 20 that a tensor was missing.
int cmd_check_weights(const Args& a) {
    const std::string dir = a.get("weights");
    if (dir.empty()) {
        std::cerr << "!! --weights DIR is required\n";
        return 2;
    }
    T5Config t5; DiTConfig dit; VaeConfig vae;
    SyntheticWeights declared(DType::F32);
    declare_pixart_shapes(declared, t5, dit, vae);

    struct Component { const char* name; };
    const Component components[] = {{"transformer"}, {"vae"}, {"text_encoder"}};
    std::vector<std::string> only;
    if (a.has("component")) only.push_back(a.get("component"));

    int missing = 0, wrong = 0, checked = 0;
    size_t bytes = 0;
    for (const auto& c : components) {
        if (!only.empty() && only[0] != c.name) continue;
        std::unique_ptr<CheckpointWeights> w;
        try {
            w = std::make_unique<CheckpointWeights>(
                CheckpointWeights::component(dir, c.name));
        } catch (const std::exception& e) {
            std::cout << "  " << c.name << ": SKIPPED -- " << e.what() << "\n";
            continue;
        }
        bytes += w->total_bytes();
        int comp_missing = 0, comp_wrong = 0, comp_checked = 0;
        for (const auto& kv : declared.declared()) {
            const std::string& name = kv.first;
            const bool is_t5 = (name.rfind("encoder.", 0) == 0 || name == "shared.weight");
            const bool is_vae = (name.rfind("decoder.", 0) == 0 ||
                                 name.rfind("post_quant_conv", 0) == 0);
            const char* belongs = is_t5 ? "text_encoder" : (is_vae ? "vae" : "transformer");
            if (std::string(belongs) != c.name) continue;
            ++comp_checked;
            if (!w->has(name)) {
                if (comp_missing < 10) std::cout << "  MISSING  " << name << "\n";
                ++comp_missing;
                continue;
            }
            const Tensor t = w->get(name);
            if (t.numel() != numel_of(kv.second)) {
                if (comp_wrong < 10) {
                    std::cout << "  SHAPE    " << name << ": checkpoint " << t.describe()
                              << ", runtime wants " << numel_of(kv.second) << " elements\n";
                }
                ++comp_wrong;
            }
        }
        std::cout << "  " << c.name << ": " << comp_checked << " required, "
                  << comp_missing << " missing, " << comp_wrong << " wrong shape, "
                  << w->total_bytes() / 1000000 << " MB mapped\n";
        missing += comp_missing;
        wrong += comp_wrong;
        checked += comp_checked;
    }
    std::cout << "\n  " << checked << " tensors checked through the real loader, "
              << bytes / 1000000 << " MB mapped\n";
    if (missing || wrong) {
        std::cout << "\nFAIL: " << missing << " missing, " << wrong << " wrong shape. The "
                     "runtime cannot load this checkpoint.\n";
        return 1;
    }
    std::cout << "\nok: every tensor the runtime requires is present with the right geometry\n";
    return 0;
}

// The initial latent for a seed, written as .npy.
//
// It exists so the starting noise is an INPUT to both this runtime and the reference rather than
// something each produces for itself. Two RNG implementations agreeing bit for bit is not a
// thing worth depending on, and if they disagree the latents diverge from step zero and the
// correctness gate measures the random number generator.
int cmd_noise(const Args& a) {
    PipelineConfig cfg;
    cfg.resolution = static_cast<int>(a.num("resolution", 1024));
    cfg.seed = static_cast<uint64_t>(a.num("seed", 20260911));
    cfg.compute = DType::F32;
    VaeConfig vae;
    T5Config t5; DiTConfig dit;
    SchedulerConfig sched;
    auto none = std::make_shared<SyntheticWeights>(DType::F32);
    Pipeline p(cfg, none, none, none, t5, dit, vae, sched);
    Tensor z = p.initial_latent();

    const std::string out = a.get("out");
    if (out.empty()) {
        std::cerr << "!! --out FILE is required\n";
        return 2;
    }
    write_npy(out, z);
    const OutputStats st = OutputStats::of(z);
    std::cout << "BURNISH_JSON: {\"effective\":{\"seed\":" << cfg.seed
              << ",\"resolution\":" << cfg.resolution << "},\"output_stats\":"
              << stats_json(st) << "}\n";
    return 0;
}

// One stage, one input file, one output file.
//
// It exists so a single stage can be compared against the reference implementation without
// running a whole generation: `scripts/differential_test.py` feeds both the same latent and
// compares the results. That is the correctness gate's question asked at a scale a CPU can
// answer, and it is how the two defects in docs/STATUS.md would have been caught earlier.
int cmd_decode(const Args& a) {
    register_all_ops();
    const std::string dir = a.get("weights");
    const std::string in = a.get("latent");
    const std::string out = a.get("out");
    if (dir.empty() || in.empty() || out.empty()) {
        std::cerr << "!! decode needs --weights DIR --latent IN.npy --out OUT.npy\n";
        return 2;
    }
    VaeConfig vae;
    const DType dt = dtype_arg(a);
    const Device dev = device_arg(a);
    Tensor latent = read_npy(in);
    if (latent.rank() != 4 || latent.dim(1) != vae.latent_channels) {
        throw std::runtime_error("decode: expected [1, " +
                                 std::to_string(vae.latent_channels) + ", h, w], got " +
                                 latent.describe());
    }
    std::shared_ptr<WeightSource> host =
        std::make_shared<CheckpointWeights>(CheckpointWeights::component(dir, "vae"));
    std::shared_ptr<WeightSource> keep;
    auto w = place(host, dev, dt, &keep);
    VaeDecoder dec(vae, *w, dt);
    const ImplSelection impls = ImplSelection::from_request(a.get("impl", "stock"), dev);

    // The scaling divide belongs to the PIPELINE, not the decoder, and the reference's
    // `vae.decode()` does not do it either. Applying it here would make the comparison off by
    // 1/0.13025 and look like a catastrophic disagreement.
    Tensor x = latent.to(dt);
    if (dev == Device::CUDA) x = x.to_device();
    device_sync_if(dev);
    const auto t0 = std::chrono::steady_clock::now();
    Tensor pixels = dec.forward(x, impls);
    // Timed AFTER a synchronise, or a CUDA measurement times the launch queue rather than the
    // work -- which reads as a spectacular and entirely fictional speedup.
    device_sync_if(dev);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    Tensor host_pixels = (dev == Device::CUDA) ? pixels.to_host() : pixels;
    write_npy(out, host_pixels);
    const OutputStats st = OutputStats::of(host_pixels);
    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << secs
       << ",\"peak_vram_bytes\":" << peak_memory_bytes(dev) << "},\"effective\":"
       << json_map({{"stage", "vae-decode"}, {"dtype", dtype_name(dt)},
                    {"impl", a.get("impl", "stock")}, {"weights", "checkpoint"},
                    {"device", a.get("device", "cpu")}})
       << ",\"output_stats\":" << stats_json(st) << "}";
    std::cout << os.str() << "\n";
    return 0;
}

// One DiT forward pass, every input read from a file.
//
// Same purpose as `decode`: it lets `scripts/differential_test.py` feed this runtime and the
// reference implementation identical tensors and compare the results. Nothing is generated
// twice, so a disagreement is the model and not the inputs.
int cmd_dit_step(const Args& a) {
    register_all_ops();
    const std::string dir = a.get("weights");
    if (dir.empty() || a.get("latent").empty() || a.get("caption").empty() ||
        a.get("mask").empty() || a.get("out").empty()) {
        std::cerr << "!! dit-step needs --weights DIR --latent L.npy --caption C.npy "
                     "--mask M.npy --out O.npy [--timestep T]\n";
        return 2;
    }
    DiTConfig dit;
    // --layers truncates the block stack, for bisecting a disagreement by depth. A discrepancy
    // that grows linearly with depth is accumulation; one that appears at a particular block is
    // a bug in it.
    if (a.has("layers")) dit.num_layers = static_cast<int>(a.num("layers", dit.num_layers));
    const DType dt = dtype_arg(a);
    const Device dev = device_arg(a);
    Tensor latent = read_npy(a.get("latent")).to(dt);
    Tensor caption = read_npy(a.get("caption")).to(dt);
    Tensor mask = read_npy(a.get("mask")).to(dt);
    if (dev == Device::CUDA) {
        latent = latent.to_device();
        caption = caption.to_device();
        mask = mask.to_device();
    }
    std::shared_ptr<WeightSource> host = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "transformer"));
    std::shared_ptr<WeightSource> keep;
    auto w = place(host, dev, dt, &keep);
    PixArtDiT model(dit, *w, dt);
    const ImplSelection impls = ImplSelection::from_request(a.get("impl", "stock"), dev);
    // One untimed warm-up: the first call uploads every weight and instantiates every kernel,
    // and timing that measures the loader.
    if (a.num("warmup", 1) > 0) model.forward(latent, a.real("timestep", 500.0), caption, mask,
                                              impls);
    device_sync_if(dev);
    const auto t0 = std::chrono::steady_clock::now();
    Tensor out = model.forward(latent, a.real("timestep", 500.0), caption, mask, impls);
    device_sync_if(dev);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    Tensor host_out = (dev == Device::CUDA) ? out.to_host() : out;
    write_npy(a.get("out"), host_out);
    const OutputStats st = OutputStats::of(host_out);
    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << secs
       << ",\"peak_vram_bytes\":" << peak_memory_bytes(dev) << "},\"effective\":"
       << json_map({{"stage", "dit-step"}, {"dtype", dtype_name(dt)},
                    {"impl", a.get("impl", "stock")}, {"weights", "checkpoint"},
                    {"device", a.get("device", "cpu")}})
       << ",\"output_stats\":" << stats_json(st) << "}";
    std::cout << os.str() << "\n";
    return 0;
}

// The text encoder, from a token-ids file (one prompt per line, whitespace-separated ids).
int cmd_encode(const Args& a) {
    register_all_ops();
    const std::string dir = a.get("weights");
    const std::string ids_path = a.get("token-ids");
    if (dir.empty() || ids_path.empty() || a.get("out").empty()) {
        std::cerr << "!! encode needs --weights DIR --token-ids FILE --out O.npy\n";
        return 2;
    }
    std::vector<std::vector<int64_t>> rows;
    {
        std::ifstream f(ids_path);
        if (!f) throw std::runtime_error("cannot open " + ids_path);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::vector<int64_t> row;
            long long v;
            while (is >> v) row.push_back(static_cast<int64_t>(v));
            if (!row.empty()) rows.push_back(std::move(row));
        }
    }
    if (rows.empty()) throw std::runtime_error(ids_path + ": no token ids");
    size_t width = 0;
    for (const auto& r : rows) width = std::max(width, r.size());
    Tensor ids({static_cast<int64_t>(rows.size()), static_cast<int64_t>(width)}, DType::F32);
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < width; ++c) {
            ids.set(static_cast<int64_t>(r * width + c),
                    c < rows[r].size() ? static_cast<float>(rows[r][c]) : 0.0f);
        }
    }
    T5Config t5;
    // --layers truncates the encoder stack, for bisecting a disagreement by depth. A defect is
    // present at one layer; amplification starts small and climbs.
    if (a.has("layers")) t5.num_layers = static_cast<int>(a.num("layers", t5.num_layers));
    const DType dt = dtype_arg(a);
    const Device dev = device_arg(a);
    if (dev == Device::CUDA) ids = ids.to_device();
    std::shared_ptr<WeightSource> host = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "text_encoder"));
    std::shared_ptr<WeightSource> keep;
    auto w = place(host, dev, dt, &keep);
    T5Encoder enc(t5, *w, dt);
    const ImplSelection impls = ImplSelection::from_request(a.get("impl", "stock"), dev);
    device_sync_if(dev);
    const auto t0 = std::chrono::steady_clock::now();
    Tensor h = enc.forward(ids, impls);
    device_sync_if(dev);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    Tensor host_h = (dev == Device::CUDA) ? h.to_host() : h;
    write_npy(a.get("out"), host_h);
    const OutputStats st = OutputStats::of(host_h);
    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << secs
       << ",\"peak_vram_bytes\":" << peak_memory_bytes(dev) << "},\"effective\":"
       << json_map({{"stage", "t5-encode"}, {"dtype", dtype_name(dt)},
                    {"impl", a.get("impl", "stock")}, {"weights", "checkpoint"},
                    {"device", a.get("device", "cpu")}})
       << ",\"output_stats\":" << stats_json(st) << "}";
    std::cout << os.str() << "\n";
    return 0;
}

// The sampler alone, on a synthetic trajectory, so it can be compared against the reference
// scheduler without weights or a model.
//
// The sampler is part of the ORACLE and it is the one oracle-critical component with no weights:
// a scheduler that differs by one index convention produces a plausible image from the same seed
// and fails the latent comparison with no clue as to why. The "model output" here is a fixed
// function of the step index, so both sides see identical inputs and any difference is the
// solver.
int cmd_schedule(const Args& a) {
    const int steps = static_cast<int>(a.num("steps", 20));
    const int64_t n = a.num("size", 16);
    const DType dt = dtype_arg(a);
    SchedulerConfig cfg;
    DPMSolverMultistep sched(cfg);
    sched.set_timesteps(steps);

    Tensor sample({n}, dt);
    for (int64_t i = 0; i < n; ++i) sample.set(i, static_cast<float>(std::sin(i * 0.7)));
    Tensor eps({n}, dt);
    // The trajectory, one row per step plus the final state. Always fp32 so the DUMP does not
    // add a rounding the comparison would then attribute to the sampler.
    Tensor traj({steps + 1, n}, DType::F32);
    for (int64_t i = 0; i < n; ++i) traj.set(i, sample.get(i));
    for (int i = 0; i < steps; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            eps.set(j, static_cast<float>(std::cos(j * 0.3 + i * 0.11)));
        }
        sched.step(eps, i, sample);
        for (int64_t j = 0; j < n; ++j) traj.set((i + 1) * n + j, sample.get(j));
    }
    write_npy(a.get("out"), traj);

    std::ostringstream ts;
    for (int i = 0; i < steps; ++i) ts << (i ? "," : "") << sched.timestep(i);
    std::cout << "BURNISH_JSON: {\"effective\":{\"steps\":" << steps
              << ",\"size\":" << n << ",\"dtype\":\"" << dtype_name(dt)
              << "\"},\"timesteps\":[" << ts.str()
              << "],\"output_stats\":" << stats_json(OutputStats::of(traj)) << "}\n";
    return 0;
}

int cmd_probe(const Args&) {
#ifndef BURNISHER_CUDA
    std::cerr << "!! burnisher probe needs a CUDA build. This binary was built without it, so\n"
                 "   there is no device to measure. Every roofline in eval/ then stands on a\n"
                 "   VENDOR peak, and the achieved fractions computed against it are LOWER\n"
                 "   bounds on how done each cell really is.\n";
    return 3;
#else
    return probe_device_main();
#endif
}

int cmd_generate(const Args& a) {
    register_all_ops();
    const std::string dir = a.get("weights");
    const std::string ids_path = a.get("token-ids");
    if (dir.empty() || ids_path.empty()) {
        std::cerr <<
            "!! `generate` needs --weights DIR and --token-ids FILE.\n"
            "\n"
            "   --weights is a checkpoint in the reference layout (transformer/, vae/,\n"
            "   text_encoder/). Running a real generation on synthetic weights would produce a\n"
            "   plausible image from numbers that mean nothing; `selftest` exercises the graph\n"
            "   without a checkpoint.\n"
            "\n"
            "   --token-ids is a text file of integer T5 token ids, one PROMPT per line and ids\n"
            "   separated by spaces. Under classifier-free guidance the NEGATIVE prompt comes\n"
            "   first. Ids rather than text on purpose: the T5 tokenizer is a SentencePiece\n"
            "   model, and vendoring one would put a second oracle in this repository.\n"
            "   docs/CORRECTNESS.md has the procedure for producing them and pinning a digest.\n";
        return 2;
    }

    T5Config t5; DiTConfig dit; VaeConfig vae;
    SchedulerConfig sched;
    PipelineConfig cfg;
    cfg.resolution = static_cast<int>(a.num("resolution", 1024));
    cfg.steps = static_cast<int>(a.num("steps", 20));
    cfg.seed = static_cast<uint64_t>(a.num("seed", 20260911));
    cfg.guidance_scale = a.real("guidance-scale", 0.0);
    if (cfg.classifier_free_guidance && cfg.guidance_scale <= 0.0) {
        throw std::runtime_error(
            "--guidance-scale is required under classifier-free guidance. It changes the "
            "latents, so it is part of the oracle; the harness passes the generation's pinned "
            "value and there is deliberately no default here for it to drift from.");
    }
    cfg.compute = dtype_arg(a);
    cfg.impl = a.get("impl", "stock");
    cfg.device = device_arg(a);
    cfg.classifier_free_guidance = !a.has("no-cfg");

    // Token ids: one prompt per line, ids separated by whitespace. Rows are padded to the
    // longest line with the T5 pad id (0), which is what the reference does -- a ragged batch
    // would silently give the two CFG branches different caption lengths.
    std::vector<std::vector<int64_t>> rows;
    {
        std::ifstream f(ids_path);
        if (!f) throw std::runtime_error("cannot open " + ids_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            std::istringstream is(line);
            std::vector<int64_t> row;
            long long v;
            while (is >> v) row.push_back(static_cast<int64_t>(v));
            if (!row.empty()) rows.push_back(std::move(row));
        }
    }
    const int64_t want_rows = cfg.classifier_free_guidance ? 2 : 1;
    if (static_cast<int64_t>(rows.size()) != want_rows) {
        throw std::runtime_error(
            "token ids: " + std::to_string(rows.size()) + " prompt(s), but guidance asks for " +
            std::to_string(want_rows) + ". Under classifier-free guidance the file holds two "
            "lines and the NEGATIVE prompt is the first.");
    }
    size_t width = 0;
    for (const auto& r : rows) width = std::max(width, r.size());
    cfg.caption_len = static_cast<int>(width);
    Tensor ids({want_rows, static_cast<int64_t>(width)}, DType::F32);
    for (int64_t r = 0; r < want_rows; ++r) {
        for (size_t c = 0; c < width; ++c) {
            ids.set(r * static_cast<int64_t>(width) + static_cast<int64_t>(c),
                    c < rows[r].size() ? static_cast<float>(rows[r][c]) : 0.0f);
        }
    }

    std::shared_ptr<CheckpointWeights> text_host = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "text_encoder"));
    std::shared_ptr<CheckpointWeights> den_host = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "transformer"));
    std::shared_ptr<CheckpointWeights> dec_host = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "vae"));
    std::cerr << ">> mapped " << (text_host->total_bytes() + den_host->total_bytes() +
                                  dec_host->total_bytes()) / 1000000
              << " MB of checkpoint\n";
    // Placed on the device the pipeline will run on. The host sources are kept alive because the
    // device copies are made from them lazily, per tensor, on first use.
    std::shared_ptr<WeightSource> k1, k2, k3;
    auto text = place(text_host, cfg.device, cfg.compute, &k1);
    auto den = place(den_host, cfg.device, cfg.compute, &k2);
    auto dec = place(dec_host, cfg.device, cfg.compute, &k3);

    Pipeline p(cfg, text, den, dec, t5, dit, vae, sched);
    StageTimings t{};
    Tensor final_latent;
    // The starting noise is an INPUT when one is given.
    //
    // Without this the runtime regenerates it from the seed AT THE COMPUTE DTYPE, so a bf16 run
    // starts from bf16-rounded noise while an fp32 reference starts from fp32 noise. The two
    // trajectories then differ from step zero and the correctness gate measures the rounding of
    // a random number rather than the kernels. That is not a small effect: guided diffusion
    // amplifies a 0.4% difference in the initial latent into a completely different image.
    Tensor noise;
    if (!a.get("noise").empty()) {
        noise = read_npy(a.get("noise"));
        const int64_t f = vae.scale_factor();
        const int64_t h = cfg.resolution / f;
        if (noise.rank() != 4 || noise.dim(2) != h || noise.dim(3) != h) {
            throw std::runtime_error("--noise is " + noise.describe() + ", expected [1, " +
                                     std::to_string(vae.latent_channels) + ", " +
                                     std::to_string(h) + ", " + std::to_string(h) + "]");
        }
    }
    Tensor pixels = p.generate(ids, &t, &final_latent, noise.defined() ? &noise : nullptr);
    // The gate's statistics are about the LATENT, because the latent is what the gate compares.
    const OutputStats st = OutputStats::of(final_latent);

    // --dump-latents writes the denoised LATENT, before the VAE. The VAE decode is itself under
    // optimization, so comparing pixels would fold two questions into one and let a decoder
    // change hide a denoiser change. --dump-pixels is separate and is for looking at.
    write_npy(a.get("dump-latents"), final_latent);
    write_npy(a.get("dump-pixels"), pixels);

    std::map<std::string, std::string> effective{
        {"impl", cfg.impl}, {"dtype", dtype_name(cfg.compute)},
        {"resolution", std::to_string(cfg.resolution)},
        {"steps", std::to_string(cfg.steps)}, {"seed", std::to_string(cfg.seed)},
        {"caption_len", std::to_string(cfg.caption_len)},
        {"cfg", cfg.classifier_free_guidance ? "1" : "0"},
        {"device", a.get("device", "cpu")},
        {"noise", a.get("noise").empty() ? "seeded" : "pinned"},
    };
    for (const auto& kv : p.impls().as_map()) {
        effective["impl." + kv.first] = kv.second;
    }
    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << t.total_s
       << ",\"text_encode_s\":" << t.text_encode_s
       << ",\"denoise_s\":" << t.denoise_s
       << ",\"vae_decode_s\":" << t.vae_decode_s
       << ",\"peak_vram_bytes\":" << peak_memory_bytes(device_arg(a))
       << "},\"effective\":" << json_map(effective)
       << ",\"output_stats\":" << stats_json(st) << "}";
    std::cout << os.str() << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    if (a.command.empty() || a.has("help") || a.command == "help") {
        std::cout << kUsage;
        return a.command.empty() && !a.has("help") ? 1 : 0;
    }
    try {
        if (a.command == "info") return cmd_info(a);
        if (a.command == "selftest") return cmd_selftest(a);
        if (a.command == "bench") return cmd_bench(a);
        if (a.command == "probe") return cmd_probe(a);
        if (a.command == "weights-manifest") return cmd_weights_manifest(a);
        if (a.command == "check-weights") return cmd_check_weights(a);
        if (a.command == "noise") return cmd_noise(a);
        if (a.command == "decode") return cmd_decode(a);
        if (a.command == "dit-step") return cmd_dit_step(a);
        if (a.command == "encode") return cmd_encode(a);
        if (a.command == "schedule") return cmd_schedule(a);
        if (a.command == "generate") return cmd_generate(a);
    } catch (const std::exception& e) {
        std::cerr << "!! " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command '" << a.command << "'\n\n" << kUsage;
    return 2;
}
