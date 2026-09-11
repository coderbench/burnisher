// The three stages of a PixArt-Sigma generation, as explicit graphs over the op registry.
//
// Written as straight-line code over named ops rather than as a generic module system. The thing
// under optimization is the kernel, and every layer of indirection between a contributor and the
// kernel is a layer they have to read before they can start. It also means the op sequence in
// `pixart_dit.cpp` and the op list in `eval/burnscore/geometry.py` can be compared line by line,
// which is what keeps the published roofline honest about the thing that actually runs.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "burnisher/ops.h"
#include "burnisher/safetensors.h"
#include "burnisher/tensor.h"

namespace burnisher {

// Where weights come from. Two backings: a mapped checkpoint, and a deterministic synthetic
// source used by the tests. The synthetic one is what lets the whole graph be exercised on a
// machine with no checkpoint, at a small config, which is the only way these files get tested
// at all before they meet 22 GB of weights.
class WeightSource {
  public:
    virtual ~WeightSource() = default;
    virtual bool has(const std::string& name) const = 0;
    virtual Tensor get(const std::string& name) const = 0;
    // Throws with the name, rather than returning an undefined tensor that faults three frames
    // later in a kernel.
    Tensor require(const std::string& name) const;
};

class CheckpointWeights : public WeightSource {
  public:
    explicit CheckpointWeights(std::vector<std::string> paths);
    bool has(const std::string& name) const override;
    Tensor get(const std::string& name) const override;
    size_t total_bytes() const;

  private:
    std::vector<SafeTensors> shards_;
};

// Deterministic pseudo-random weights, seeded by tensor name. Same name, same numbers, every
// run and every process -- so a test that compares two graph evaluations is comparing the graph
// and not the RNG.
class SyntheticWeights : public WeightSource {
  public:
    explicit SyntheticWeights(DType dtype, uint64_t seed = 20260911);
    bool has(const std::string&) const override { return true; }
    Tensor get(const std::string& name) const override;
    void declare(const std::string& name, std::vector<int64_t> shape);

  private:
    DType dtype_;
    uint64_t seed_;
    std::map<std::string, std::vector<int64_t>> shapes_;
};

struct T5Config {
    int num_layers = 24;
    int d_model = 4096;
    int d_ff = 10240;
    int d_kv = 64;
    int num_heads = 64;
    int vocab_size = 32128;
    bool gated = true;
    double eps = 1e-6;
};

struct DiTConfig {
    int num_layers = 28;
    int num_heads = 16;
    int head_dim = 72;
    int patch_size = 2;
    int in_channels = 4;
    int out_channels = 8;
    int caption_channels = 4096;
    double mlp_ratio = 4.0;
    double eps = 1e-6;
    int sample_size = 128;
    int d() const { return num_heads * head_dim; }
    int d_ff() const { return static_cast<int>(d() * mlp_ratio); }
};

struct VaeConfig {
    std::vector<int> block_out_channels{128, 256, 512, 512};
    int layers_per_block = 2;
    int latent_channels = 4;
    int norm_num_groups = 32;
    double scaling_factor = 0.13025;
    double eps = 1e-6;
    int scale_factor() const {
        return 1 << (static_cast<int>(block_out_channels.size()) - 1);
    }
};

class T5Encoder {
  public:
    T5Encoder(T5Config cfg, const WeightSource& w, DType compute);
    // token_ids: [batch, seq] as int32 stored in an F32 tensor (the ids are small and exact).
    Tensor forward(const Tensor& token_ids, const ImplSelection& impls) const;
    const T5Config& config() const { return cfg_; }

  private:
    T5Config cfg_;
    const WeightSource& w_;
    DType dtype_;
};

class PixArtDiT {
  public:
    PixArtDiT(DiTConfig cfg, const WeightSource& w, DType compute);
    // latent: [batch, in_channels, h, w]; caption: [batch, caption_len, caption_channels]
    Tensor forward(const Tensor& latent, double timestep, const Tensor& caption,
                   const ImplSelection& impls) const;
    const DiTConfig& config() const { return cfg_; }

  private:
    DiTConfig cfg_;
    const WeightSource& w_;
    DType dtype_;
};

class VaeDecoder {
  public:
    VaeDecoder(VaeConfig cfg, const WeightSource& w, DType compute);
    // latent: [1, latent_channels, h, w] -> [1, 3, h*f, w*f]
    Tensor forward(const Tensor& latent, const ImplSelection& impls) const;
    const VaeConfig& config() const { return cfg_; }

  private:
    VaeConfig cfg_;
    const WeightSource& w_;
    DType dtype_;
};

// Declare every weight the three models will ask for, at the given configs.
//
// This doubles as a check that the fixture and the models agree about the graph: SyntheticWeights
// throws on an undeclared name, so a weight the model requires and this function forgot shows up
// as a loud failure in `burnisher selftest` rather than as a missing tensor at load time against
// a 22 GB checkpoint.
void declare_pixart_shapes(SyntheticWeights& w, const T5Config& t5, const DiTConfig& dit,
                           const VaeConfig& vae);

// Shape helpers shared by the models and by the tests.
Tensor sinusoidal_timestep_embedding(double t, int dim, DType dtype);
Tensor patchify(const Tensor& x, int patch, DType dtype);
Tensor unpatchify(const Tensor& x, int batch, int channels, int grid, int patch, DType dtype);

}  // namespace burnisher
