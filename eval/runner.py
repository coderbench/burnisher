"""Shared machinery for every command that touches the device.

Each guard here is an incident. None of them is defensive programming, and none should be
removed without first finding out which failure it encodes -- they are all named, because a
guard whose reason is lost is a guard somebody deletes.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# INCIDENT: two eval processes on one GPU. The second fails to allocate, the harness retries
# into the same contention, and what comes out is a number for a run that never happened. It
# does not look like an error -- it looks like a slowdown. An advisory lock turns a corrupt
# measurement into a wait.
GPU_LOCK_PATH = os.environ.get("BURNISH_EVAL_LOCK", "/tmp/burnisher-eval.lock")

# INCIDENT: a large model's memory is not back from the driver by the time the next arm asks
# for it, so a back-to-back run fails to load for a reason that has nothing to do with what is
# being measured.
SETTLE_SECONDS = 0.0 if os.environ.get("BURNISH_EVAL_FAST") else 3.0

LOAD_FAILURE_MARKERS = ("out of memory", "cudaErrorMemoryAllocation", "[FAIL] load",
                        "CUDA error")

# Every environment name the runtime reads. The BASE arm must have all of them scrubbed from the
# inherited environment, not merely left unset by the caller.
#
# INCIDENT (inherited, and it cost three releases elsewhere): an operator with the tuning
# variable exported in their shell runs a "control" that is not the control, and the harness
# reports ~0% for a comparison of the candidate against itself. Nothing about a contaminated
# control looks wrong in the output, which is why this is a function the tests can assert rather
# than an inline dict comprehension.
RUNTIME_ENV_PREFIXES = ("BURNISHER_", "BURNISH_RT_")


class RunnerError(RuntimeError):
    """The device, the binary, or the environment cannot produce a trustworthy measurement."""


class GpuLock:
    """Exclusive use of the device for the duration of a measurement."""

    def __init__(self, path=GPU_LOCK_PATH, verbose=True):
        self.path, self.verbose, self.fh = path, verbose, None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self.verbose:
                print(f">> another burnish holds {self.path}; waiting rather than racing it "
                      f"for VRAM", flush=True)
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False


def scrubbed_environment(base=None) -> dict:
    """A copy of `base` with every runtime-controlled name removed.

    A function, not an inline expression, so `eval/tests/test_runner.py` can assert it. This is
    the only way a guard like this stays correct.
    """
    env = dict(os.environ if base is None else base)
    for key in [k for k in env if k.startswith(RUNTIME_ENV_PREFIXES)]:
        del env[key]
    # Autotuning that picks a different algorithm per process makes a build non-deterministic
    # in a way that looks like a policy effect. Pin it for every measured run, both arms.
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def require_idle_device(verbose=True) -> None:
    """Verify nothing else is on the device before a load. See GPU_LOCK_PATH for why."""
    if shutil.which("nvidia-smi") is None:
        raise RunnerError("nvidia-smi is not on PATH; this command needs a real device")
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                          "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    busy = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if busy:
        raise RunnerError(
            "the device is not idle:\n  " + "\n  ".join(busy) +
            "\nTwo benchmarks on one GPU race for VRAM and the loser becomes a plausible-"
            "looking\nnumber. Wait, or stop the other process BY PID -- never with `pkill -f`, "
            "which over\nssh matches and kills your own session when the remote command line "
            "contains the\npattern.")
    if verbose:
        print(">> device is idle")


def device_fingerprint() -> dict:
    """What the receipt needs to say which box produced it."""
    fields = ("name", "driver_version", "memory.total", "pci.bus_id", "uuid")
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    fp = dict(zip(fields, parts))
    # INCIDENT (inherited): graphics clocks cannot be pinned in a container, so absolute numbers
    # drift with temperature over minutes. Recorded so a receipt says what regime it was taken
    # in, and so nobody is tempted to compare two receipts' absolute times.
    fp["_clocks_note"] = ("Clocks are not pinned and cannot be in a container. Only paired "
                          "interleaved same-box deltas mean anything; absolute times from two "
                          "different sessions are not comparable.")
    return fp


def run_once(cmd, env_extra=None, *, timeout=1800, scrub=True, retry_on_load_failure=True):
    """One invocation of the runtime, with the settle-and-retry the driver makes necessary."""
    env = scrubbed_environment() if scrub else dict(os.environ)
    env.update(env_extra or {})
    time.sleep(SETTLE_SECONDS)
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if p.returncode != 0 and retry_on_load_failure and any(m in out for m in
                                                           LOAD_FAILURE_MARKERS):
        print(">> load failure; settling and retrying once", flush=True)
        time.sleep(SETTLE_SECONDS * 4)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    return p.returncode, out, time.time() - t0


_JSON_LINE = re.compile(r"^BURNISH_JSON:\s*(\{.*\})\s*$", re.M)


def parse_result(text, label):
    """Pull the runtime's structured result out of its output.

    The runtime prints exactly one `BURNISH_JSON: {...}` line per invocation. Parsing a number
    out of prose is how a harness comes to silently accept a changed output format, so a missing
    or duplicated line is an error rather than a fallback to regex.
    """
    matches = _JSON_LINE.findall(text)
    if not matches:
        raise RunnerError(f"{label}: no BURNISH_JSON line in the runtime's output. The last "
                          f"4 kB was:\n{text[-4000:]}")
    if len(matches) > 1:
        raise RunnerError(f"{label}: {len(matches)} BURNISH_JSON lines. One invocation reports "
                          f"one result; two means the runtime ran twice and the pairing is gone.")
    return json.loads(matches[0])


def require_ran_what_it_claimed(result, expected, label):
    """The runtime echoes the configuration it actually used; compare it against what was asked.

    INCIDENT CLASS: an arm that silently fell back. A requested implementation that is not
    registered, a tile size clamped to fit, a dtype demoted because the kernel was missing --
    each produces a perfectly good number for a configuration nobody asked for, and each is
    invisible from outside the run. The runtime is required to report what it did; this compares.
    """
    got = result.get("effective", {})
    wrong = {k: (v, got.get(k)) for k, v in expected.items()
             if k in got and got[k] != v}
    if wrong:
        detail = "; ".join(f"{k}: asked {a!r}, ran {b!r}" for k, (a, b) in wrong.items())
        raise RunnerError(
            f"{label}: the runtime did not run what was asked -- {detail}. An arm that silently "
            f"fell back produces a perfectly good number for a configuration nobody requested.")
    missing = [k for k in expected if k not in got]
    if missing:
        raise RunnerError(
            f"{label}: the runtime did not report {', '.join(missing)} in its `effective` block, "
            f"so there is no way to tell whether it ran what was asked. A runtime that stopped "
            f"echoing its configuration must not be scored.")


def require_not_degenerate(result, label):
    """A generation that produced a black image or a NaN latent runs fast and means nothing."""
    stats = result.get("output_stats") or {}
    for key in ("latent_mean", "latent_std", "latent_absmax"):
        if key not in stats:
            raise RunnerError(f"{label}: the runtime reported no {key}; a degenerate output is "
                              f"indistinguishable from a fast one without it")
    if not all(isinstance(stats[k], (int, float)) for k in stats
               if k.startswith("latent")):
        raise RunnerError(f"{label}: output statistics are not numbers")
    if stats["latent_std"] <= 1e-6:
        raise RunnerError(
            f"{label}: the output latent has std {stats['latent_std']:.3e} -- it is constant. "
            f"A pipeline that produced a flat tensor ran fast and generated nothing.")
    import math
    if not math.isfinite(stats["latent_absmax"]) or stats["latent_absmax"] > 1e4:
        raise RunnerError(f"{label}: latent absmax {stats['latent_absmax']} -- the run diverged")


def interleave(pairs, repeats):
    """The order arms are executed in: base, candidate, base, candidate, ...

    Adjacent, not blocked. Running all of one arm and then all of the other puts a thermal ramp
    between the arms and attributes it to whichever ran second.
    """
    for k in range(repeats):
        for name in pairs:
            yield k, name
