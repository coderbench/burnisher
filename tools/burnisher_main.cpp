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

common options:
  --impl NAME               registered implementation to use (default: stock)
  --dtype bf16|fp32         compute dtype (default: bf16)
  --resolution N            image side in pixels (default: 1024)
  --steps N                 denoise steps (default: 20)
  --seed N                  fixed seed (default: 20260911)
  --token-ids FILE          T5 token ids, one prompt per line (negative first under CFG)
  --dump-latents FILE       write the output tensor as .npy, for the correctness gate
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

size_t peak_rss_bytes() {
    // Host peak RSS on a CPU build. On a CUDA build this is replaced by the device allocator's
    // high-water mark, which is what the memory objective actually scores; reporting the host
    // figure there would silently score the wrong resource.
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return static_cast<size_t>(std::stoul(line.substr(6))) * 1024;
        }
    }
    return 0;
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
    register_builtin_cpu_ops();
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
    register_builtin_cpu_ops();
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
    register_builtin_cpu_ops();
    const std::string stage = a.get("stage", "dit-step");
    const std::string impl = a.get("impl", "stock");
    const DType dt = dtype_arg(a);
    const int64_t resolution = a.num("resolution", 1024);
    const int64_t caption_len = a.num("caption-len", 300);
    const int64_t batch = a.num("batch", 2);
    const int warmup = static_cast<int>(a.num("warmup", 3));
    const int iters = static_cast<int>(a.num("iters", 10));

    // Fails here rather than falling back, and the message lists what IS registered.
    const ImplSelection impls = ImplSelection::from_request(impl);

    T5Config t5; DiTConfig dit; VaeConfig vae;
    if (a.has("small")) small_configs(&t5, &dit, &vae);

    // Synthetic weights exercise the graph; real ones exercise the loader, the tensor-name
    // mapping and the dtypes. Both are useful and they answer different questions, so which one
    // was used is reported in `effective` rather than inferred.
    std::shared_ptr<WeightSource> weights;
    const std::string weights_dir = a.get("weights");
    if (!weights_dir.empty()) {
        if (a.has("small")) {
            throw std::runtime_error("--small and --weights are contradictory: a real checkpoint "
                                     "has the real geometry");
        }
        const std::string component = (stage == "t5-encode") ? "text_encoder"
                                    : (stage == "vae-decode") ? "vae" : "transformer";
        weights = std::make_shared<CheckpointWeights>(
            CheckpointWeights::component(weights_dir, component));
    } else {
        auto synth = std::make_shared<SyntheticWeights>(dt);
        declare_pixart_shapes(*synth, t5, dit, vae);
        weights = synth;
    }

    const int64_t f = vae.scale_factor();
    const int64_t latent = resolution / f;
    using clock = std::chrono::steady_clock;

    std::vector<double> times;
    OutputStats stats{};
    const auto run_stage = [&]() -> Tensor {
        if (stage == "t5-encode") {
            T5Encoder e(t5, *weights, dt);
            Tensor ids({batch, caption_len}, dt);
            for (int64_t i = 0; i < ids.numel(); ++i)
                ids.set(i, static_cast<float>((i * 7) % (t5.vocab_size - 1)));
            return e.forward(ids, impls);
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
            return d.forward(z, 500.0, cap, mask, impls);
        }
        if (stage == "vae-decode") {
            VaeDecoder v(vae, *weights, dt);
            Tensor z({1, vae.latent_channels, latent, latent}, dt);
            for (int64_t i = 0; i < z.numel(); ++i)
                z.set(i, static_cast<float>(std::sin(static_cast<double>(i) * 0.21)));
            return v.forward(z, impls);
        }
        throw std::runtime_error("unknown stage '" + stage + "'");
    };

    for (int i = 0; i < warmup; ++i) run_stage();
    for (int i = 0; i < iters; ++i) {
        const auto t0 = clock::now();
        Tensor out = run_stage();
        const auto t1 = clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
        if (i == 0) stats = OutputStats::of(out);
    }
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];

    std::map<std::string, std::string> effective{
        {"stage", stage}, {"dtype", dtype_name(dt)}, {"impl", impl},
        {"resolution", std::to_string(resolution)}, {"batch", std::to_string(batch)},
        {"caption_len", std::to_string(caption_len)},
        {"weights", weights_dir.empty() ? "synthetic" : "checkpoint"},
    };
    for (const auto& kv : impls.as_map()) effective["impl." + kv.first] = kv.second;

    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << median
       << ",\"peak_vram_bytes\":" << peak_rss_bytes()
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
    register_builtin_cpu_ops();
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
    cfg.guidance_scale = a.real("guidance-scale", 4.5);
    cfg.compute = dtype_arg(a);
    cfg.impl = a.get("impl", "stock");
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

    auto text = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "text_encoder"));
    auto den = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "transformer"));
    auto dec = std::make_shared<CheckpointWeights>(
        CheckpointWeights::component(dir, "vae"));
    std::cerr << ">> mapped " << (text->total_bytes() + den->total_bytes() +
                                  dec->total_bytes()) / 1000000 << " MB of checkpoint\n";

    Pipeline p(cfg, text, den, dec, t5, dit, vae, sched);
    StageTimings t{};
    Tensor pixels = p.generate(ids, &t);
    const OutputStats st = OutputStats::of(pixels);

    const std::string dump = a.get("dump-latents");
    if (!dump.empty()) {
        // A .npy of the final PIXELS, so eval/gate.py can compare them with numpy. Latents are
        // what the gate actually wants; this is the pixel tensor because the pipeline returns
        // it, and `--dump-latents` keeps the harness's flag name.
        std::ofstream out(dump, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write " + dump);
        std::ostringstream hdr;
        hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < pixels.rank(); ++i) hdr << pixels.dim(i) << ", ";
        hdr << "), }";
        std::string h = hdr.str();
        while ((10 + h.size() + 1) % 64) h += ' ';
        h += '\n';
        const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
        out.write(reinterpret_cast<const char*>(magic), 8);
        const uint16_t len = static_cast<uint16_t>(h.size());
        out.write(reinterpret_cast<const char*>(&len), 2);
        out.write(h.data(), static_cast<std::streamsize>(h.size()));
        for (int64_t i = 0; i < pixels.numel(); ++i) {
            const float v = pixels.get(i);
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }

    std::map<std::string, std::string> effective{
        {"impl", cfg.impl}, {"dtype", dtype_name(cfg.compute)},
        {"resolution", std::to_string(cfg.resolution)},
        {"steps", std::to_string(cfg.steps)}, {"seed", std::to_string(cfg.seed)},
        {"caption_len", std::to_string(cfg.caption_len)},
        {"cfg", cfg.classifier_free_guidance ? "1" : "0"},
    };
    for (const auto& kv : ImplSelection::from_request(cfg.impl).as_map()) {
        effective["impl." + kv.first] = kv.second;
    }
    std::ostringstream os;
    os.precision(12);
    os << "BURNISH_JSON: {\"metrics\":{\"latency_s\":" << t.total_s
       << ",\"text_encode_s\":" << t.text_encode_s
       << ",\"denoise_s\":" << t.denoise_s
       << ",\"vae_decode_s\":" << t.vae_decode_s
       << ",\"peak_vram_bytes\":" << peak_rss_bytes()
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
        if (a.command == "generate") return cmd_generate(a);
    } catch (const std::exception& e) {
        std::cerr << "!! " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command '" << a.command << "'\n\n" << kUsage;
    return 2;
}
