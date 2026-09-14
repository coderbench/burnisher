# Contributing

Make Burnisher faster and get a measured number for it. Every submission is built, checked for
correctness and timed by the evaluator. Nobody grades it, and no maintainer decides the score.

- [At a glance](#at-a-glance)
- [What you need](#what-you-need)
- [1. Pick something](#1-pick-something)
- [2. Write your kernel](#2-write-your-kernel)
- [3. Check it](#3-check-it)
- [4. Measure it yourself (optional)](#4-measure-it-yourself-optional)
- [5. Open the pull request](#5-open-the-pull-request)
- [6. What comes back](#6-what-comes-back)
- [Common mistakes](#common-mistakes)
- [Other ways to contribute](#other-ways-to-contribute)

## At a glance

| step | what you do | needs a GPU? |
|:--|:--|:--|
| [1. Pick](#1-pick-something) | find the cell with room, and the kernel that costs the time | no |
| [2. Write](#2-write-your-kernel) | copy that kernel, make the copy faster, register it under a new name | no |
| [3. Check](#3-check-it) | build it and compare it with the reference | only for the GPU tests |
| [4. Measure](#4-measure-it-yourself-optional) | run the evaluator's own scoring command | yes, optional |
| [5. Submit](#5-open-the-pull-request) | open a pull request and fill in the template | no |
| [6. Read the label](#6-what-comes-back) | the evaluator labels the pull request with your score | no |

## What you need

| to | you need |
|:--|:--|
| write, build, check and submit | a C++17 compiler, CMake 3.24 or newer, Python 3 with NumPy |
| compile a CUDA kernel | the CUDA toolkit (12.8 or 13) and cuDNN 9 (`libcudnn9-dev-cuda-<major>` from NVIDIA's apt repository). Compiling needs no GPU. |
| compare a stage with the reference | the checkpoint ([how to get it](README.md#2-get-the-checkpoint)), and `torch`, `diffusers` and `transformers` |
| run your kernel, or measure it | an RTX 5090 |

You can submit without a GPU. The evaluator measures every submission on the pinned card.

## 1. Pick something

**See where the room is.** `tools/burnish roofline` prints one row per cell:

| column | meaning |
|:--|:--|
| `achieved` | how close the cell is to its ceiling today, as a share |
| `left` | the most any implementation could still gain there |
| `floor` | the cell's noise floor: a gain smaller than this is invisible |
| `res` | `no` means the room left is under 20 floors, so a gain must be large to show |
| `vs pt` | how many times slower than the same stage in PyTorch on the same card |

The same table, with every ceiling explained, is [`docs/ROOFLINE.md`](docs/ROOFLINE.md).

**Find open work.** [`issues/README.md`](issues/README.md) lists it, each item with its arithmetic.
Good places to start:

| issue | what it is about |
|:--|:--|
| [`dit-attention`](issues/dit-attention.md) | DiT self-attention at 4k to 16k tokens |
| [`vae-decode`](issues/vae-decode.md) | VAE decode: fusion and the mid-block attention |
| [`fused-adaln`](issues/fused-adaln.md) | fusing AdaLN modulation into its neighbours |
| [`text-encoder`](issues/text-encoder.md) | T5-XXL: most of the checkpoint, little of the clock |
| [`weight-formats`](issues/weight-formats.md) | NVFP4 and MXFP4 on Blackwell |

## 2. Write your kernel

### Where the kernels are

| file | what is in it |
|:--|:--|
| [`src/cuda/ops_cuda.cu`](src/cuda/ops_cuda.cu) | the `cuda` kernels, and `register_cuda_ops()` at the end, which registers them |
| [`include/burnisher/ops.h`](include/burnisher/ops.h) | each op's argument struct: the inputs, outputs and shapes a kernel receives |
| [`src/cpu/ops_cpu.cpp`](src/cpu/ops_cpu.cpp) | the `stock` CPU kernels: slow, simple, and the definition of correct |

The kernels that cost the most time:

| op | arguments | `cuda` kernel | what it does today |
|:--|:--|:--|:--|
| `attention` | `AttentionArgs` | `attention_cuda` | cuDNN fused SDPA for bf16 without bias; cuBLAS scores and cuDNN softmax otherwise |
| `gemm` | `GemmArgs` | `gemm_cuda` | cuBLAS matmul, with the bias and activation as a separate pass |
| `conv2d` | `Conv2dArgs` | `conv2d_cuda` | cuDNN, deterministic algorithm, in float |
| `norm` | `NormArgs` | `norm_cuda` | GroupNorm in fixed chunks; LayerNorm and RMSNorm one block per row |
| `modulate` | `ModulateArgs` | `modulate_cuda` | unfused AdaLN modulation, the main fusion target |

Ten smaller ops complete the graph. `build-cuda/burnisher info` lists every op and every name
registered for it.

### Copy, change, register

Never change an existing kernel. Put your version beside it:

```cpp
// src/cuda/ops_cuda.cu, beside attention_cuda
void attention_flash_sm120(const AttentionArgs& a) {
    // start from a copy of attention_cuda, then make it faster
}

// at the end of register_cuda_ops()
register_impl<AttentionArgs>("attention", "flash-sm120", attention_flash_sm120,
                             "what changed, and why it is faster");
```

Your kernel can also live in a new file under `src/cuda/`. Add that file to the
`target_sources(burnisher_core ...)` line in [`CMakeLists.txt`](CMakeLists.txt), or the build will
not see it.

### The rules that decide whether you are measured

- **The evaluator runs your new name against `cuda`, in the same binary.** Every op you did not
  register runs `cuda` in both arms, so registering one kernel is normal.
- **A kernel changed in place is not evaluated** (`burnish:no-candidate`). It would be measured
  against itself.
- **A copy must really change.** Registering a kernel already on main under a new name, or a copy
  that is 95% or more the same code, is `burnish:reregistered`. Changing one constant is not enough.
- **Register one new name.** If you register several, write the one to measure in the pull
  request's `Implementation name` line.
- **Stay correct and deterministic.** The same input must give byte-identical output on every run.
  Avoid per-process autotuning, atomic reductions and TF32 ([`docs/CORRECTNESS.md`](docs/CORRECTNESS.md#rules)).
- **Be fast at every size.** Each cell is also timed at a resolution drawn after your code is frozen.
  A kernel tuned to one shape only is `burnish:shape-overfit`.

## 3. Check it

| command | what it checks | needs |
|:--|:--|:--|
| `scripts/check.sh` | the CPU build, the harness tests and the docs | nothing |
| `scripts/build_cuda.sh`, then `build-cuda/burnisher info` | your kernel compiles, and its name is registered | the CUDA toolkit and cuDNN |
| `build-cuda/test_cuda_ops <your-impl>` | an attention, convolution or GroupNorm kernel against the CPU oracle, in fp32 and bf16 | a GPU |
| `scripts/differential_test.py --weights DIR --stage <stage> --impl <your-impl> --device cuda` | one whole stage against the reference implementation | a GPU, the checkpoint, `torch`, `diffusers`, `transformers` |

`<stage>` is `t5-encode`, `dit-step` or `vae-decode`. A large difference with the same mean and
standard deviation as the reference usually means values in the wrong order.

What the evaluator's correctness gate checks: [`docs/CORRECTNESS.md`](docs/CORRECTNESS.md).

## 4. Measure it yourself (optional)

The evaluator measures every submission, so this step is only for knowing your number before a
round does. It needs an RTX 5090 and the checkpoint.

**Make the pinned starting noise.** The CPU build can make it:

```bash
build/burnisher noise --out noise1024.npy                    # BG-1
build/burnisher noise --resolution 512 --out noise512.npy    # BG-2
sha256sum noise1024.npy noise512.npy
```

Each hash must equal `noise_sha256` in that generation's manifest
([BG-1](eval/cells/BG-1/reference-latents/manifest.json),
[BG-2](eval/cells/BG-2/reference-latents/manifest.json)).

**Score it, with the command the evaluator runs:**

```bash
scripts/build_cuda.sh
eval/score_submission.sh --base <commit you branched from> --worktree . \
    --impl-base cuda --impl-candidate <your-impl> --pr <n> \
    --ledger <directory outside this repo> --weights <checkpoint dir> --noise noise1024.npy
tools/burnish receipt show <receipt it wrote>
```

It scores BG-1. For the 512px generation, add `--generation BG-2` and pass `noise512.npy`. How to
read the receipt: [`docs/SCORING.md`](docs/SCORING.md#a-receipt).

## 5. Open the pull request

1. **Branch from `main`** and keep one kernel per pull request.
2. **Touch no instrument path:** `eval/`, `configs/`, `schemas/`, `tools/burnish`, `.github/`,
   `.gittensor/`, or `scripts/` other than `scripts/build*`. A kernel pull request that does is not
   scored (`burnish:skipped-instrument`).
3. **Write one-line commit messages** of 72 characters or fewer, with no body and no trailers:
   `<type>: <short imperative description>`, where `type` is `feat`, `fix`, `perf`, `docs`, `test`,
   `build`, `refactor` or `chore`.

   ```
   perf: fuse AdaLN into the modulation kernel
   ```

4. **Fill in the [template](.github/PULL_REQUEST_TEMPLATE.md).** Tick **Kernel**, and write your
   name on the implementation line exactly in this form, because the evaluator reads it:

   ```
   **Implementation name:** `flash-sm120`
   ```

## 6. What comes back

- **Rounds run every two hours,** oldest pull request first, up to twelve per round.
- **Each is built, gated, then timed** against `cuda`, and labelled.
- **One merge per round:** the biggest verified gain.
- **Every new commit is measured again,** whatever the current label.

| label | what it means | what to do |
|:--|:--|:--|
| `burnish:gap+N.NNNN` | **Paid.** The share of the remaining gap you closed ([how](docs/SCORING.md#the-number)). | nothing |
| `burnish:merge-first` | the best verified gain of its round | wait for the merge |
| `burnish:needs-rebase` | a verified gain, but a bigger one was merged this round | rebase on `main` and push |
| `burnish:unresolved` | the effect was inside the noise; it says nothing about the idea | make the gain larger, or aim at a cell with room |
| `burnish:no-gain` | measurably not faster | rethink the change |
| `burnish:moved-along-frontier` | faster, but it cost more memory or fidelity than the speed was worth | cut the memory or precision cost |
| `burnish:shape-overfit` | fast at the scored size, not at the held-out one | make it fast at every resolution |
| `burnish:correctness-fail`, `burnish:determinism-fail` | wrong, or not reproducible | fix it and push |
| `burnish:build-fail` | did not build | fix it and push |
| `burnish:no-candidate` | no single new name to measure | register a new name, or fill in `Implementation name` |
| `burnish:reregistered` | an existing kernel under a new name | change the kernel, not only its name |
| `burnish:skipped-instrument` | touches an instrument path | move those files to their own pull request |
| `burnish:eval-error` | the evaluator's fault | nothing; it is retried |

Every label, including the copy labels: [`docs/EVAL.md`](docs/EVAL.md#labels).

**Check any verdict yourself,** with no GPU, from the published measurements:

```bash
tools/burnish audit pr-000042-raw.json pr-000042.json
```

A worked example of a real run: [`examples/README.md`](examples/README.md).

## Common mistakes

| mistake | result |
|:--|:--|
| editing `attention_cuda` itself | `burnish:no-candidate` |
| renaming a copy without really changing it | `burnish:reregistered` |
| committing a script or config change with the kernel | `burnish:skipped-instrument` |
| registering two names and leaving `Implementation name` empty | `burnish:no-candidate` |
| a new source file missing from `CMakeLists.txt` | `burnish:build-fail` |
| a kernel tuned for 1024px only | `burnish:shape-overfit` |
| copying another author's open pull request | `burnish:copycat`: closed, account blocked |

## Other ways to contribute

- **A new cell is paid** (`burnish:cell-opened`): a new resolution, dtype or model, as a new
  generation. How: [`docs/CARTOGRAPHY.md`](docs/CARTOGRAPHY.md).
- **Existing generations under `eval/cells/` are frozen.** A change means a new generation.
- **Instrument changes** are reviewed by a maintainer and not scored. Send them as their own pull
  request, and before removing a guard, read the incident its comment names
  ([`docs/EVAL.md`](docs/EVAL.md#the-instrument-guard)).
- **Docs and refactors** are welcome and score zero.
- **Building on others is fine; copying is not.** Starting from a kernel on `main`, or iterating on
  your own pull requests, is normal. Copying another author's open pull request is
  `burnish:copycat` ([`docs/EVAL.md`](docs/EVAL.md#copies)).
