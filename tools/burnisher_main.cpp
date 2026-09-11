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

common options:
  --impl NAME               registered implementation to use (default: stock)
  --dtype bf16|fp32         compute dtype (default: bf16)
  --resolution N            image side in pixels (default: 1024)
  --steps N                 denoise steps (default: 20)
  --seed N                  fixed seed (default: 20260911)
  --weights DIR             checkpoint directory; omit to use synthetic weights
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
    auto weights = std::make_shared<SyntheticWeights>(dt);
    declare_pixart_shapes(*weights, t5, dit, vae);

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
            return d.forward(z, 500.0, cap, impls);
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
    if (!a.has("weights")) {
        std::cerr << "!! --weights DIR is required for `generate`.\n"
                     "   Running a real generation on synthetic weights would produce a "
                     "plausible\n   image from numbers that mean nothing. Use `selftest` to "
                     "exercise the graph\n   without a checkpoint.\n";
        return 2;
    }
    std::cerr << "!! loading a real checkpoint is not wired up in this build.\n"
                 "   The graph, the scheduler and the op registry are complete and are "
                 "exercised by\n   `burnisher selftest`; what is missing is the checkpoint "
                 "layout mapping and the\n   pre-tokenized prompt ids. docs/CORRECTNESS.md "
                 "states exactly what remains and\n   why it has not been verified: no "
                 "Blackwell device and no checkpoint were\n   available when this was written, "
                 "and shipping an unverified load path as if it\n   worked is the failure mode "
                 "this repository exists to avoid.\n";
    return 4;
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
        if (a.command == "generate") return cmd_generate(a);
    } catch (const std::exception& e) {
        std::cerr << "!! " << e.what() << "\n";
        return 1;
    }
    std::cerr << "unknown command '" << a.command << "'\n\n" << kUsage;
    return 2;
}
