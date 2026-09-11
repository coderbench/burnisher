"""Model geometry -> the ops a stage executes, with flops and bytes for each.

This is the arithmetic the whole repository rests on. Everything downstream -- every ceiling,
every achieved fraction, every "how much room is left in this cell" -- is a ratio computed from
numbers this file produces. So it is written to be checked rather than trusted: `selfcheck()`
recomputes each model's parameter count from the enumerated ops and compares it against the
published file size, and `eval/tests/test_geometry.py` fails if they disagree by more than a
percent. A geometry that reproduces a 610,000,000-parameter checkpoint to within 0.3% is one
that read the config correctly; one that does not is a table of confident fiction.

Two distinctions this file makes that a naive flop counter does not, and both change answers:

**Weight bytes READ is not parameter bytes STORED.** A 32128 x 4096 embedding table is 263 MB
resident and 2.4 MB read when 300 tokens are gathered from it. Pricing the read at the resident
size would put the T5 encoder's roofline in the wrong regime entirely. `weight_bytes` is what
the invocation reads; `param_bytes` is what it occupies. The latency roofline uses the first and
the memory objective uses the second.

**Attention traffic depends on the implementation, and the ceiling must not.** A materializing
attention writes and re-reads an Sq x Sk score matrix -- 1.07 GB for the VAE mid-block at
1024px -- and a flash-style one never materializes it. That is a real difference between two
implementations of the same op and therefore exactly the thing a contributor is scored for
fixing. So the ceiling is computed against `attn_impl="flash"` always, and the as-implemented
figure is reported beside it. A ceiling that moved when you improved the implementation would
not be a ceiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .dtypes import width


class GeometryError(ValueError):
    """A config does not describe a model this file knows how to enumerate."""


@dataclass(frozen=True)
class Op:
    """One executed operation, priced.

    `count` is how many times this identical op appears in the stage (once per layer, usually).
    Keeping it as a multiplier rather than emitting 28 copies keeps the op list readable, which
    matters because the op list is the thing a contributor reads to find out where the time goes.
    """
    name: str
    kind: str                  # gemm | attention | conv | norm | elementwise | gather
    flops: float
    weight_bytes: float = 0.0  # parameter bytes READ by one invocation
    read_bytes: float = 0.0    # activation bytes read
    write_bytes: float = 0.0   # activation bytes written
    param_bytes: float = 0.0   # parameter bytes RESIDENT (>= weight_bytes)
    count: int = 1
    note: str = ""

    @property
    def total_flops(self) -> float:
        return self.flops * self.count

    @property
    def total_bytes(self) -> float:
        return (self.weight_bytes + self.read_bytes + self.write_bytes) * self.count

    @property
    def total_param_bytes(self) -> float:
        return (self.param_bytes or self.weight_bytes) * self.count

    @property
    def arithmetic_intensity(self) -> float:
        """flop per byte of this op alone. Compared against the device ridge point, this is what
        says whether a cell is the compute kind or the bandwidth kind -- and therefore which of
        the two roofline terms a contributor has to attack."""
        b = self.weight_bytes + self.read_bytes + self.write_bytes
        return (self.flops / b) if b else float("inf")


@dataclass
class StageProfile:
    """Everything a roofline needs about one stage at one shape and dtype."""
    stage: str
    shape: dict
    wdtype: str
    adtype: str
    ops: list = field(default_factory=list)
    input_bytes: float = 0.0
    output_bytes: float = 0.0
    invocations: int = 1
    notes: list = field(default_factory=list)

    @property
    def flops(self) -> float:
        return sum(o.total_flops for o in self.ops) * self.invocations

    @property
    def weight_bytes(self) -> float:
        return sum(o.weight_bytes * o.count for o in self.ops) * self.invocations

    @property
    def param_bytes(self) -> float:
        """Resident parameter bytes. NOT multiplied by invocations -- weights are loaded once
        and re-read, which is the entire reason step caching and weight-format work pay."""
        return sum(o.total_param_bytes for o in self.ops)

    @property
    def activation_bytes(self) -> float:
        return sum((o.read_bytes + o.write_bytes) * o.count for o in self.ops) * self.invocations

    @property
    def traffic_bytes(self) -> float:
        """What the stage moves as currently decomposed: weights plus every activation hop."""
        return sum(o.total_bytes for o in self.ops) * self.invocations

    @property
    def unavoidable_bytes(self) -> float:
        """The traffic no implementation can remove: read every weight once per invocation,
        read the stage input, write the stage output.

        Everything else is an intermediate, and an intermediate is removable in principle by
        fusion. This is deliberately generous to the contributor -- it is the term that makes
        the ceiling stay put while they fuse.
        """
        return self.weight_bytes + (self.input_bytes + self.output_bytes) * self.invocations

    def by_kind(self) -> dict:
        out = {}
        for o in self.ops:
            e = out.setdefault(o.kind, {"flops": 0.0, "bytes": 0.0})
            e["flops"] += o.total_flops * self.invocations
            e["bytes"] += o.total_bytes * self.invocations
        return out

    def hot_ops(self, n: int = 8) -> list:
        return sorted(self.ops, key=lambda o: -o.total_flops)[:n]


# ---------------------------------------------------------------------------
# op constructors
# ---------------------------------------------------------------------------

def gemm(name, m, n, k, *, wb, ab, count=1, note="", bias=False, gathered_rows=None):
    """[m,k] x [k,n] -> [m,n]. `wb`/`ab` are weight and activation element widths.

    `gathered_rows` models an embedding lookup: the table is `k x n` resident but only
    `gathered_rows` of it are read.
    """
    params = k * n + (n if bias else 0)
    read_params = params if gathered_rows is None else gathered_rows * n
    return Op(name=name, kind="gemm",
              flops=2.0 * m * n * k,
              weight_bytes=read_params * wb,
              param_bytes=params * wb,
              read_bytes=m * k * ab,
              write_bytes=m * n * ab,
              count=count, note=note)


def attention(name, *, batch, heads, q_len, kv_len, head_dim, ab, count=1,
              impl="flash", note=""):
    """Scaled dot-product attention, priced for one of two implementations.

    flops are 2 x (QK^T) + 2 x (AV) = 4 * B * h * Sq * Sk * dh, and they do not depend on the
    implementation. The traffic does, and by a lot: materializing writes and re-reads
    B * h * Sq * Sk elements, which at the VAE mid-block's 16384 tokens is 1.07 GB of round trip
    for 550 GFLOP of math.
    """
    if impl not in ("flash", "materialized"):
        raise GeometryError(f"attention impl must be 'flash' or 'materialized', not {impl!r}")
    qkv_elems = batch * heads * head_dim * (2 * q_len + 2 * kv_len)   # q,out + k,v
    score_elems = batch * heads * q_len * kv_len
    read = qkv_elems * ab * 0.5
    write = qkv_elems * ab * 0.5
    if impl == "materialized":
        # written once by the QK^T kernel, read by softmax, written by softmax, read by AV.
        read += 2.0 * score_elems * ab
        write += 2.0 * score_elems * ab
    return Op(name=name, kind="attention",
              flops=4.0 * batch * heads * q_len * kv_len * head_dim,
              read_bytes=read, write_bytes=write, count=count,
              note=note or f"{impl}, {q_len}x{kv_len} over {heads} heads")


def conv2d(name, *, h_out, w_out, c_in, c_out, k=3, h_in=None, w_in=None,
           wb, ab, count=1, note="", bias=True):
    h_in = h_in if h_in is not None else h_out
    w_in = w_in if w_in is not None else w_out
    params = c_in * c_out * k * k + (c_out if bias else 0)
    return Op(name=name, kind="conv",
              flops=2.0 * h_out * w_out * c_in * c_out * k * k,
              weight_bytes=params * wb, param_bytes=params * wb,
              read_bytes=h_in * w_in * c_in * ab,
              write_bytes=h_out * w_out * c_out * ab,
              count=count, note=note or f"{k}x{k} {c_in}->{c_out} @ {h_out}x{w_out}")


def norm(name, *, numel, ab, channels=0, wb=0.0, count=1, affine=True, flops_per_elem=5.0,
         note=""):
    params = (2 * channels) if (affine and channels) else 0
    return Op(name=name, kind="norm", flops=flops_per_elem * numel,
              weight_bytes=params * wb, param_bytes=params * wb,
              read_bytes=numel * ab, write_bytes=numel * ab, count=count, note=note)


def elementwise(name, *, numel, ab, flops_per_elem=1.0, inputs=1, count=1, note=""):
    return Op(name=name, kind="elementwise", flops=flops_per_elem * numel,
              read_bytes=inputs * numel * ab, write_bytes=numel * ab, count=count, note=note)


# ---------------------------------------------------------------------------
# T5 encoder
# ---------------------------------------------------------------------------

def t5_encoder(cfg, *, seq, batch=1, wdtype="bf16", adtype="bf16", attn_impl="flash"):
    """T5 v1.1 encoder stack (gated-gelu, RMSNorm, no bias, relative position bias).

    PixArt-Sigma runs this once per prompt at seq=300 and then never again for the whole
    denoise loop. That single fact is what the dominance screen turns on: the encoder holds 89%
    of the checkpoint's parameters and, at twenty steps, about two percent of the wall clock.
    """
    wb, ab = width(wdtype), width(adtype)
    L = int(cfg["num_layers"]); d = int(cfg["d_model"]); dff = int(cfg["d_ff"])
    dkv = int(cfg["d_kv"]); h = int(cfg["num_heads"]); vocab = int(cfg["vocab_size"])
    inner = h * dkv
    B, S = batch, seq
    M = B * S

    p = StageProfile(stage="t5-encode", wdtype=wdtype, adtype=adtype,
                     shape={"seq": S, "batch": B, "layers": L, "d_model": d},
                     input_bytes=B * S * 4,          # token ids, int32
                     output_bytes=M * d * ab)
    ops = p.ops
    # The embedding table is 263 MB resident and `M` rows read. Pricing the read at the
    # resident size is the single easiest way to get this stage's regime wrong.
    ops.append(gemm("embed_tokens", 0, d, vocab, wb=wb, ab=ab, gathered_rows=M,
                    note="gather: whole table resident, M rows read"))
    ops[-1] = replace(ops[-1], flops=0.0, read_bytes=M * 4, write_bytes=M * d * ab)

    ops.append(norm("layer_norm.self_attn", numel=M * d, ab=ab, channels=d, wb=wb, count=L,
                    affine=True, flops_per_elem=4.0, note="T5 RMSNorm, scale only"))
    ops.append(gemm("attn.q", M, inner, d, wb=wb, ab=ab, count=L))
    ops.append(gemm("attn.k", M, inner, d, wb=wb, ab=ab, count=L))
    ops.append(gemm("attn.v", M, inner, d, wb=wb, ab=ab, count=L))
    ops.append(attention("attn.sdpa", batch=B, heads=h, q_len=S, kv_len=S, head_dim=dkv,
                         ab=ab, count=L, impl=attn_impl))
    ops.append(gemm("attn.o", M, d, inner, wb=wb, ab=ab, count=L))
    ops.append(norm("layer_norm.ffn", numel=M * d, ab=ab, channels=d, wb=wb, count=L,
                    affine=True, flops_per_elem=4.0))
    gated = str(cfg.get("feed_forward_proj", "gated-gelu")).startswith("gated")
    ops.append(gemm("ffn.wi_0", M, dff, d, wb=wb, ab=ab, count=L))
    if gated:
        ops.append(gemm("ffn.wi_1", M, dff, d, wb=wb, ab=ab, count=L,
                        note="gated-gelu: second input projection"))
        ops.append(elementwise("ffn.gelu_mul", numel=M * dff, ab=ab, flops_per_elem=9.0,
                               inputs=2, count=L))
    else:
        ops.append(elementwise("ffn.relu", numel=M * dff, ab=ab, flops_per_elem=1.0, count=L))
    ops.append(gemm("ffn.wo", M, d, dff, wb=wb, ab=ab, count=L))
    ops.append(norm("final_layer_norm", numel=M * d, ab=ab, channels=d, wb=wb,
                    affine=True, flops_per_elem=4.0))
    # Relative position bias: one h x 32-bucket table, added to scores of every layer. Tiny in
    # parameters, and it is a real read of B*h*S*S bytes per layer if not folded into the
    # attention kernel -- which is the kind of thing the flash path removes.
    ops.append(Op(name="attn.rel_pos_bias", kind="elementwise",
                  flops=float(B * h * S * S), read_bytes=0.0, write_bytes=0.0,
                  param_bytes=32 * h * wb, weight_bytes=32 * h * wb, count=1,
                  note="computed once, broadcast into every layer's scores"))
    p.notes.append("T5 v1.1: no bias anywhere, RMSNorm, gated-gelu FFN.")
    return p


# ---------------------------------------------------------------------------
# PixArt-Sigma DiT
# ---------------------------------------------------------------------------

def pixart_dit(cfg, *, resolution, caption_len, batch=1, wdtype="bf16", adtype="bf16",
               attn_impl="flash", vae_scale=8):
    """One forward pass of the PixArt-Sigma Transformer2DModel (ada_norm_single).

    `batch` is 2 under classifier-free guidance and that is not a detail: CFG doubles this
    stage and nothing else in the pipeline, so it doubles the denoise loop's share of the wall
    clock and halves the text encoder's.
    """
    wb, ab = width(wdtype), width(adtype)
    L = int(cfg["num_layers"]); h = int(cfg["num_attention_heads"])
    dh = int(cfg["attention_head_dim"]); d = h * dh
    patch = int(cfg["patch_size"]); cin = int(cfg["in_channels"]); cout = int(cfg["out_channels"])
    cap_ch = int(cfg["caption_channels"]); xattn_d = int(cfg.get("cross_attention_dim", d))
    dff = int(round(d * float(cfg.get("mlp_ratio", 4.0))))

    latent = resolution // vae_scale
    if latent % patch:
        raise GeometryError(f"latent {latent} is not divisible by patch size {patch}")
    grid = latent // patch
    N = grid * grid                         # image tokens
    Mtok = batch * N
    Mcap = batch * caption_len

    p = StageProfile(stage="dit-step", wdtype=wdtype, adtype=adtype,
                     shape={"resolution": resolution, "latent": latent, "tokens": N,
                            "batch": batch, "layers": L, "d": d, "caption_len": caption_len},
                     input_bytes=batch * latent * latent * cin * ab,
                     output_bytes=batch * latent * latent * cout * ab)
    ops = p.ops

    # --- once per forward ---
    ops.append(conv2d("pos_embed.proj", h_out=grid, w_out=grid, c_in=cin * batch,
                      c_out=d, k=patch, wb=wb, ab=ab,
                      note=f"patchify {patch}x{patch}: {cin}->{d}"))
    ops.append(elementwise("pos_embed.add_sincos", numel=Mtok * d, ab=ab, inputs=2))
    # AdaLayerNormSingle: timestep -> 256 sinusoid -> MLP -> SiLU -> Linear(d, 6d).
    ops.append(gemm("adaln_single.emb.0", batch, d, 256, wb=wb, ab=ab))
    ops.append(gemm("adaln_single.emb.2", batch, d, d, wb=wb, ab=ab))
    ops.append(gemm("adaln_single.linear", batch, 6 * d, d, wb=wb, ab=ab,
                    note="the ONE big modulation projection: once per step, not per layer"))
    # caption_projection: T5 hidden 4096 -> d, gelu, d -> d.
    ops.append(gemm("caption_projection.linear_1", Mcap, d, cap_ch, wb=wb, ab=ab))
    ops.append(elementwise("caption_projection.act", numel=Mcap * d, ab=ab, flops_per_elem=9.0))
    ops.append(gemm("caption_projection.linear_2", Mcap, d, d, wb=wb, ab=ab))

    # --- per layer ---
    ops.append(norm("norm1", numel=Mtok * d, ab=ab, channels=0, count=L, affine=False,
                    note="LayerNorm, elementwise_affine=False: the scale/shift come from AdaLN"))
    ops.append(Op(name="norm1.modulate", kind="elementwise",
                  flops=2.0 * Mtok * d, read_bytes=Mtok * d * ab, write_bytes=Mtok * d * ab,
                  param_bytes=6 * d * wb, weight_bytes=6 * d * wb, count=L,
                  note="scale_shift_table + shared modulation: x*(1+scale)+shift. THE fusion "
                       "target -- it is a full activation round trip for two flops per element"))
    ops.append(gemm("attn1.to_q", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(gemm("attn1.to_k", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(gemm("attn1.to_v", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(attention("attn1.sdpa", batch=batch, heads=h, q_len=N, kv_len=N, head_dim=dh,
                         ab=ab, count=L, impl=attn_impl,
                         note=f"self-attention over {N} image tokens"))
    ops.append(gemm("attn1.to_out", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(Op(name="attn1.gate_residual", kind="elementwise", flops=2.0 * Mtok * d,
                  read_bytes=2 * Mtok * d * ab, write_bytes=Mtok * d * ab, count=L))

    ops.append(gemm("attn2.to_q", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(gemm("attn2.to_k", Mcap, d, xattn_d, wb=wb, ab=ab, count=L, bias=True,
                    note="cross-attention K over the caption, not the image"))
    ops.append(gemm("attn2.to_v", Mcap, d, xattn_d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(attention("attn2.sdpa", batch=batch, heads=h, q_len=N, kv_len=caption_len,
                         head_dim=dh, ab=ab, count=L, impl=attn_impl,
                         note=f"cross-attention {N}x{caption_len}"))
    ops.append(gemm("attn2.to_out", Mtok, d, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(Op(name="attn2.residual", kind="elementwise", flops=1.0 * Mtok * d,
                  read_bytes=2 * Mtok * d * ab, write_bytes=Mtok * d * ab, count=L))

    ops.append(norm("norm2", numel=Mtok * d, ab=ab, channels=0, count=L, affine=False))
    ops.append(Op(name="norm2.modulate", kind="elementwise", flops=2.0 * Mtok * d,
                  read_bytes=Mtok * d * ab, write_bytes=Mtok * d * ab, count=L))
    ops.append(gemm("ff.net.0.proj", Mtok, dff, d, wb=wb, ab=ab, count=L, bias=True))
    ops.append(elementwise("ff.gelu", numel=Mtok * dff, ab=ab, flops_per_elem=9.0, count=L,
                           note="gelu-approximate (tanh)"))
    ops.append(gemm("ff.net.2", Mtok, d, dff, wb=wb, ab=ab, count=L, bias=True))
    ops.append(Op(name="ff.gate_residual", kind="elementwise", flops=2.0 * Mtok * d,
                  read_bytes=2 * Mtok * d * ab, write_bytes=Mtok * d * ab, count=L))

    # --- output ---
    ops.append(norm("norm_out", numel=Mtok * d, ab=ab, channels=0, affine=False))
    ops.append(Op(name="norm_out.modulate", kind="elementwise", flops=2.0 * Mtok * d,
                  read_bytes=Mtok * d * ab, write_bytes=Mtok * d * ab,
                  param_bytes=2 * d * wb, weight_bytes=2 * d * wb))
    ops.append(gemm("proj_out", Mtok, patch * patch * cout, d, wb=wb, ab=ab, bias=True))
    p.notes.append(f"{N} image tokens at {resolution}px; caption {caption_len}; batch {batch} "
                   f"({'CFG' if batch == 2 else 'no CFG'}).")
    return p


# ---------------------------------------------------------------------------
# AutoencoderKL decoder
# ---------------------------------------------------------------------------

def vae_decoder(cfg, *, resolution, batch=1, wdtype="bf16", adtype="bf16",
                attn_impl="materialized", tile=None):
    """The AutoencoderKL decoder: latent -> pixels.

    Modelled at the resolution each block actually runs at, because that is the whole story.
    The last two up-blocks run at 512x512 and 1024x1024 with 256 and 128 channels, and they hold
    most of the stage's flops and nearly all of its activation traffic: one 1024x1024x128
    activation is 268 MB in bf16, and a ResNet block touches several. That is why the stage is
    tiling-sensitive and why `tile` is a parameter here rather than an implementation secret.

    `attn_impl` defaults to "materialized" because that is what a reference decoder does at the
    mid-block, and the 16384x16384 score matrix it writes is 1.07 GB. The ceiling uses "flash".
    """
    wb, ab = width(wdtype), width(adtype)
    chans = list(cfg["block_out_channels"])
    lpb = int(cfg.get("layers_per_block", 2))
    zc = int(cfg.get("latent_channels", 4))
    groups = int(cfg.get("norm_num_groups", 32))
    scale = 2 ** (len(chans) - 1)
    latent = resolution // scale
    if tile:
        raise GeometryError("tiled decode changes the op list, not a multiplier on it; build "
                            "the profile at the tile resolution and count the tiles")

    rev = list(reversed(chans))
    p = StageProfile(stage="vae-decode", wdtype=wdtype, adtype=adtype,
                     shape={"resolution": resolution, "latent": latent, "batch": batch,
                            "latent_channels": zc, "blocks": len(chans)},
                     input_bytes=batch * latent * latent * zc * ab,
                     output_bytes=batch * resolution * resolution * 3 * ab)
    ops = p.ops
    B = batch

    def resnet(prefix, res, cin, cout, count=1):
        n_in = B * res * res * cin
        n_out = B * res * res * cout
        ops.append(norm(f"{prefix}.norm1", numel=n_in, ab=ab, channels=cin, wb=wb, count=count,
                        note=f"GroupNorm({groups}) @ {res}x{res}x{cin}"))
        ops.append(elementwise(f"{prefix}.silu1", numel=n_in, ab=ab, flops_per_elem=4.0,
                               count=count))
        ops.append(conv2d(f"{prefix}.conv1", h_out=res, w_out=res, c_in=cin * B, c_out=cout,
                          k=3, wb=wb, ab=ab, count=count))
        ops.append(norm(f"{prefix}.norm2", numel=n_out, ab=ab, channels=cout, wb=wb, count=count))
        ops.append(elementwise(f"{prefix}.silu2", numel=n_out, ab=ab, flops_per_elem=4.0,
                               count=count))
        ops.append(conv2d(f"{prefix}.conv2", h_out=res, w_out=res, c_in=cout * B, c_out=cout,
                          k=3, wb=wb, ab=ab, count=count))
        if cin != cout:
            ops.append(conv2d(f"{prefix}.conv_shortcut", h_out=res, w_out=res, c_in=cin * B,
                              c_out=cout, k=1, wb=wb, ab=ab, count=count))
        ops.append(Op(name=f"{prefix}.residual_add", kind="elementwise", flops=float(n_out),
                      read_bytes=2 * n_out * ab, write_bytes=n_out * ab, count=count))

    ops.append(conv2d("post_quant_conv", h_out=latent, w_out=latent, c_in=zc * B, c_out=zc,
                      k=1, wb=wb, ab=ab))
    ops.append(conv2d("conv_in", h_out=latent, w_out=latent, c_in=zc * B, c_out=rev[0], k=3,
                      wb=wb, ab=ab))

    resnet("mid.resnets.0", latent, rev[0], rev[0])
    mid_tokens = latent * latent
    ops.append(norm("mid.attn.group_norm", numel=B * mid_tokens * rev[0], ab=ab,
                    channels=rev[0], wb=wb))
    for proj in ("to_q", "to_k", "to_v", "to_out"):
        ops.append(gemm(f"mid.attn.{proj}", B * mid_tokens, rev[0], rev[0], wb=wb, ab=ab,
                        bias=True))
    ops.append(attention("mid.attn.sdpa", batch=B, heads=1, q_len=mid_tokens,
                         kv_len=mid_tokens, head_dim=rev[0], ab=ab, impl=attn_impl,
                         note=f"spatial self-attention over {mid_tokens} positions -- the "
                              f"score matrix is {mid_tokens}^2 elements"))
    resnet("mid.resnets.1", latent, rev[0], rev[0])

    res = latent
    prev = rev[0]
    for i, cout in enumerate(rev):
        is_final = (i == len(rev) - 1)
        resnet(f"up.{i}.resnets.0", res, prev, cout)
        if lpb >= 1:
            resnet(f"up.{i}.resnets.rest", res, cout, cout, count=lpb)
        prev = cout
        if not is_final:
            new = res * 2
            ops.append(Op(name=f"up.{i}.upsample.interpolate", kind="elementwise",
                          flops=float(B * new * new * cout),
                          read_bytes=B * res * res * cout * ab,
                          write_bytes=B * new * new * cout * ab,
                          note="nearest 2x"))
            ops.append(conv2d(f"up.{i}.upsample.conv", h_out=new, w_out=new, c_in=cout * B,
                              c_out=cout, k=3, wb=wb, ab=ab,
                              note=f"runs at the NEW resolution {new}x{new}"))
            res = new

    ops.append(norm("conv_norm_out", numel=B * res * res * prev, ab=ab, channels=prev, wb=wb))
    ops.append(elementwise("conv_act", numel=B * res * res * prev, ab=ab, flops_per_elem=4.0))
    ops.append(conv2d("conv_out", h_out=res, w_out=res, c_in=prev * B, c_out=3, k=3,
                      wb=wb, ab=ab))
    if res != resolution:
        raise GeometryError(f"decoder reached {res}px, expected {resolution}px -- the up-block "
                            f"count and the scale factor disagree")
    p.notes.append(f"{lpb + 1} ResNets per up-block ({lpb} + 1), reversed channels {rev}.")
    return p


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def selfcheck(candidates: dict) -> list:
    """Recompute each model's parameter count from the enumerated ops and compare it against
    the published file size. This is the only thing standing between this module and a table of
    confident fiction, so it is a function rather than a comment.
    """
    out = []
    c = candidates["candidates"]["pixart-sigma-xl2-1024"]
    for label, profile, declared in (
        ("t5-encode",
         t5_encoder(c["text_encoder"], seq=300, wdtype="fp32", adtype="fp32"),
         c["text_encoder"]["file_bytes_fp32"]),
        ("dit-step",
         pixart_dit(c["denoiser"], resolution=1024, caption_len=300, batch=1,
                    wdtype="fp32", adtype="fp32"),
         c["denoiser"]["file_bytes_fp32"]),
    ):
        got = profile.param_bytes
        out.append({"stage": label, "computed_param_bytes": got,
                    "published_file_bytes": declared,
                    "ratio": got / declared if declared else None})
    # The VAE file holds encoder AND decoder; only the decoder is enumerated here, so the check
    # is against the decoder's share rather than the file.
    vd = vae_decoder(c["vae"], resolution=1024, wdtype="fp32", adtype="fp32")
    out.append({"stage": "vae-decode", "computed_param_bytes": vd.param_bytes,
                "published_file_bytes": c["vae"]["decoder_params"] * 4,
                "ratio": vd.param_bytes / (c["vae"]["decoder_params"] * 4)})
    return out
