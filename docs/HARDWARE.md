# The measurement box

## What is pinned

BG-1 is scored on **one RTX 5090** (GB202, sm_120, 32 GiB GDDR7), built from source, CUDA 12.8.
`configs/devices.json` also carries RTX PRO 6000 Blackwell (sm_120, 96 GiB) and DGX Spark
(GB10, sm_121, 128 GB unified) because `burnish roofline --device` recomputes the ceilings for
them, and the comparison is informative — but only the 5090's ceilings are the scored ones.

## Clocks cannot be pinned, and everything follows from that

Graphics clocks cannot be pinned in a container. Absolute throughput drifts with temperature over
minutes, so:

- **Only paired interleaved same-box deltas mean anything.** `base, candidate, base, candidate…`,
  adjacent, never blocked. Running one arm to completion and then the other puts the thermal ramp
  between the arms and attributes it to whichever went second.
- **Absolute times from two different sessions are not comparable.** A receipt records the device
  fingerprint so it says which regime it was taken in, and the drift guard in `eval/burnscore/
  compute.py` refuses a run whose base arm has moved more than three floors from its calibration.
- **A noise floor is a property of a session's procedure**, which is why `burnish calibrate`
  measures the base against itself with every guard a scored comparison uses, rather than in a
  quieter loop.

## Never run two benchmarks at once

Two processes on one GPU do not merely go slower. They race for VRAM, one fails to allocate, the
harness retries into the same contention, and what comes out is a number for a run that never
happened — reported as a slowdown, not as an error.

Three layers guard against it and all three are cheap:

1. `burnish` checks `nvidia-smi --query-compute-apps` and refuses to start if anything is on the
   device.
2. `eval/runner.py` takes an advisory `flock` on `/tmp/burnisher-eval.lock`, so a second `burnish`
   waits rather than racing.
3. Arms that fail to load settle and retry once — the driver does not always have a large model's
   memory back by the time the next arm asks for it, and that failure has nothing to do with what
   is being measured.

## Kill by PID

`pkill -f <pattern>` over ssh kills your own session when the remote command line contains the
pattern. This has happened. Find the PID with `nvidia-smi --query-compute-apps=pid` and kill that.

## Getting a build

```bash
scripts/build.sh        # CPU: the correctness oracle and the whole harness. No GPU needed.
scripts/build_cuda.sh   # CUDA: the only build that can measure anything. Needs nvcc.
scripts/check.sh        # everything that can be checked without a GPU
```

`CMAKE_CUDA_ARCHITECTURES` defaults to 120. Use 121 for DGX Spark.

## The order to run things in on a fresh box

Each step unblocks the next, and steps 4 and 5 will refuse to run if the one before them did not
pass. See `docs/STATUS.md` for what each will produce for the first time.

```bash
scripts/build_cuda.sh
burnish probe                                   # measured peaks -> configs/devices.json
burnish generation write && burnish roofline    # ceilings on measured peaks
burnish gate --determinism --repeats 10         # stop here if this fails
burnish gate --impl stock --output gate-base.json     # the base arm, against the reference
burnish calibrate --repeats 9 --write           # achieved fractions and noise floors

# then, per submission:
burnish gate --impl <name> --output gate.json   # correctness first, always
burnish bench --impl-candidate <name> \
    --gate-result gate.json --gate-base-result gate-base.json --output raw.json
burnish score raw.json --ledger <outside the worktree>
```

## Why the probe matters more than it looks

Until `burnish probe` runs, every ceiling in the repository stands on a **vendor** peak, and no
kernel reaches a vendor peak. The error is directional: every achieved fraction computed against
one is a *lower* bound on how done a cell really is, so the real remaining room is **smaller**
than the published table implies.

Telling somebody there is 55% left when there is 8% is how a subnet loses a contributor. The probe
is twenty lines of measurement that turns the whole table from an ordering into a budget.
