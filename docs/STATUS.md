# What is real in this repository, and what is not

Overselling the surface is the single failure mode that kills a subnet. A contributor who burns a
week of GPU time for a result inside the noise floor does not come back. So this page is the
first one to read and it is deliberately blunt.

## The one-sentence version

**The instrument is complete and tested. The runtime runs end to end on the CPU and reproduces
itself exactly. Nothing has been measured on a GPU, because no Blackwell device and no CUDA
toolkit were available when this was built.**

## What has been built and checked

| thing | state | how you can check it yourself |
|:--|:--|:--|
| The six-question screen | complete, runs | `burnish screen` |
| Per-cell arithmetic rooflines | complete, generated from configs | `burnish roofline` |
| Gap-closed scoring | complete, 79 tests | `python3 -m unittest discover -s eval -t eval` |
| Measured noise floor, paired bootstrap | complete, tested | same |
| Frontier over latency / VRAM / fidelity | complete, tested | same |
| Receipts and the append-only ledger | complete, tested | same |
| Trusted-instrument overlay | complete | `eval/run_from_base.sh` |
| Op registry with named implementations | complete, 4303 assertions | `ctest --test-dir build` |
| CPU reference ops | complete | `burnisher info` |
| T5 encoder / PixArt DiT / VAE decoder graphs | complete | `burnisher selftest` |
| DPM-Solver++ scheduler | complete, pinned against the reference construction | `ctest` |
| End-to-end pipeline, byte-identical replays | complete | `burnisher selftest` |
| Correctness gate (determinism + reference) | complete, exercised against a fake device | `python3 -m unittest discover -s eval -t eval` |
| Paired bench and calibration runners | complete, exercised against a fake device | `python3 -m unittest discover -s eval -t eval` |
| CUDA device probe | **written, never compiled** | CI job `cuda-compile` |
| CUDA op backend | **does not exist** | `issues/cuda-op-backend.md` |
| Checkpoint load path | **not wired up** | `issues/checkpoint-load.md` |
| Reference latents for the gate | **do not exist** | `issues/checkpoint-load.md` |
| Every cell's achieved fraction | **null** | `burnish roofline` |
| Every cell's noise floor | **null** | `burnish roofline` |
| Device peaks behind every ceiling | **vendor, not probed** | `configs/devices.json` |

### On "exercised against a fake device"

`eval/tests/fakes/` holds a stub `nvidia-smi` and a stub runtime that speaks the `BURNISH_JSON`
protocol. `eval/tests/test_device_runners.py` drives `bench.py` and `calibrate.py` through them
end to end, including the closed loop: calibrate, bench, score, receipt.

**The fakes replace the device, not the guards.** The idle check still shells out, still parses,
and still refuses when the stub reports a busy device. The fallback check still compares the
runtime's report against the request. What is removed is the silicon, and that is the only way
code that runs exclusively beside a GPU gets tested at all.

Writing those tests found two real defects: `bench.py` read the whole of `/dev/urandom` (a stream
that never ends) when choosing a held-out shape, and it never produced the
`latent_l2_vs_reference` objective the generation declares — so the frontier would have come out
as exactly zero for both arms and every result would have read `MOVED_ALONG_FRONTIER`.

## The four things that are not known, stated precisely

**1. How full any cell is.** Every cell publishes an arithmetic ceiling and `achieved: null`. A
ceiling says how big the box is. It says nothing about how full it is, and a cell at 95% of its
ceiling looks identical in the published table to one at 8%. Nothing in this repository should be
read as a claim that there is room in a cell — only that there could be, and at most how much.

**2. What any cell's noise floor is.** The scoring model credits a gain only when it clears that
cell's own measured run-to-run spread. Those spreads are null. Until `burnish calibrate` runs,
the scorer refuses to produce a receipt at all — it raises rather than substituting a constant,
because a guessed floor is precisely the mistake this scoring model exists to replace.

**3. Whether the peaks the ceilings stand on are reachable.** `configs/devices.json` carries
vendor figures with a `confidence` field. No kernel reaches a vendor peak. The direction of the
error matters: **every achieved fraction computed against a vendor peak is a LOWER bound on how
done a cell really is, so the real remaining room is smaller than the table implies, not larger.**
`burnish probe` measures sustained bandwidth and an achievable bf16 GEMM rate on the part and
rewrites the basis to `measured`.

**4. Whether the CUDA code compiles.** It has never been near a compiler. `src/cuda/device.cu` is
the probe and it is the only CUDA in the tree; there is no CUDA implementation of any op, so the
runtime cannot currently run on a device at all. The CI job `cuda-compile` exists to make that
stop being true and is `continue-on-error` today.

## What a first session on the pinned hardware would produce

In this order, because each step unblocks the next:

1. `scripts/build_cuda.sh` — does the probe compile for sm_120? This is the first time anyone
   will know.
2. `burnish probe` — sustained bandwidth and achievable bf16 GEMM rate. Copy into
   `configs/devices.json` with `source: measured`, regenerate the generation and the roofline
   table. **Every ceiling in the repository changes at this point**, and every one of them
   changes in the direction of less room.
3. The CUDA op backend (`issues/cuda-op-backend.md`), then the checkpoint load path
   (`issues/checkpoint-load.md`). Neither is a measurement; both are prerequisites for one.
4. `burnish gate --determinism` — ten replays, byte-identical. If this fails, stop: nothing
   downstream means anything until it passes.
5. `burnish gate` against the pinned reference latents.
6. `burnish calibrate --repeats 9 --write` — the achieved fractions and the noise floors. The
   roofline table becomes real here, and some cells may turn out to be unresolvable, which is a
   result and should be published as one rather than quietly dropped.
7. Only now can `burnish bench` produce a receipt.

## Things that would be easy to get wrong later

- **`peak_vram_bytes` on a CPU build is host RSS.** On a CUDA build it must be the device
  allocator's high-water mark. Scoring the wrong resource would make the entire memory axis
  meaningless and would look completely reasonable.
- **The checkpoint tensor names in `declare_pixart_shapes()` were written from the reference
  implementation's module structure, not checked against the real file.** They are usually right
  and that is not evidence.
- **The 2D position embedding is sin-then-cos and the timestep embedding is cos-then-sin.** Two
  conventions in one model is not a design, it is history. Both are marked `ORACLE` in the source.
- **The tolerance in BG-1 is a stated, falsifiable threshold and is expected to move once**, the
  first time `burnish gate --calibrate-tolerance` measures the real bf16-vs-fp32 drift.
