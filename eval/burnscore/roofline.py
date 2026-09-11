"""Stage geometry + device peaks -> the arithmetic ceiling for a cell, and the room left in it.

The contract this module exists to keep is the one in the README: a contributor must be able to
see how much room is left in a cell BEFORE spending a week in it. That means two numbers and
one honest caveat.

**The ceiling.** `max(flops / peak_flops, unavoidable_bytes / bandwidth)` -- the time the stage
would take if arithmetic units and the memory system were the only things that existed. It is a
lower bound on time, it is not reachable, and it is stated as such everywhere it is printed.
`unavoidable_bytes` is weights-read-once plus stage input plus stage output; every intermediate
is excluded, because an intermediate is removable by fusion and a ceiling that moved when a
contributor fused would not be a ceiling.

**The fraction of it currently achieved.** `ceiling_seconds / measured_seconds`. This is the
number that decides what a PR is worth, and it is the one that CANNOT be computed from a config
file. Until the cell has been measured on the pinned hardware it is `None` and every consumer
of this module reports it as `null` with a reason rather than substituting a plausible figure.

**The caveat.** A ceiling computed against a vendor peak nobody reaches understates every
achieved fraction by the same factor. That is harmless for ranking cells against each other and
actively harmful for the question above -- telling somebody there is 55% left when there is 8%
is exactly how a subnet loses a contributor. So every result carries `peak_basis`, and
`burnish probe` exists to replace `vendor` with `measured`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from .dtypes import peak_key


class RooflineError(ValueError):
    """A device spec or a profile cannot produce a bound."""


def _spec(device: dict, key: str):
    """Read a device field, returning (value, source). Fields are {"value":..,"source":..}."""
    if key not in device:
        raise RooflineError(f"device {device.get('name', '?')} has no field {key!r}; a ceiling "
                            f"against a peak nobody named would be a ceiling against zero")
    field = device[key]
    if isinstance(field, dict):
        return float(field["value"]), str(field.get("source", "unstated"))
    return float(field), "unstated"


@dataclass(frozen=True)
class Bound:
    """One cell's arithmetic ceiling, with everything needed to reproduce or dispute it."""
    cell: str
    stage: str
    device: str
    wdtype: str
    adtype: str
    flops: float
    unavoidable_bytes: float
    traffic_bytes: float
    peak_flops: float
    peak_bandwidth_bytes: float
    peak_basis: str
    compute_seconds: float
    memory_seconds: float
    ceiling_seconds: float
    decomposed_seconds: float
    bound_by: str
    arithmetic_intensity: float
    ridge_point: float
    param_bytes: float
    basis: str = "model"          # NEVER "measured": this is arithmetic, not a run.

    def to_json(self) -> dict:
        d = asdict(self)
        d["_basis_note"] = ("Arithmetic. Computed from a config file and a device peak; no run "
                            "produced it. A ceiling is not evidence about a speedup.")
        return d

    @property
    def headroom_note(self) -> str:
        return (f"bound by {self.bound_by}; arithmetic intensity {self.arithmetic_intensity:.1f} "
                f"flop/byte against a ridge point of {self.ridge_point:.1f}")


def bound_for(profile, device: dict, *, cell: str, device_name: str = None) -> Bound:
    """The ceiling for one (stage, shape, dtype) cell on one device."""
    pk = peak_key(profile.wdtype)
    peak_flops_t, src_f = _spec(device, pk)
    bw_gbs, src_b = _spec(device, "memory_bandwidth_gbs")
    peak_flops = peak_flops_t * 1e12
    bw = bw_gbs * 1e9
    if peak_flops <= 0 or bw <= 0:
        raise RooflineError(f"{cell}: non-positive peak ({peak_flops} flop/s, {bw} B/s)")

    flops = profile.flops
    unavoidable = profile.unavoidable_bytes
    t_compute = flops / peak_flops
    t_memory = unavoidable / bw
    ceiling = max(t_compute, t_memory)

    # The bound at the CURRENT op decomposition: every op pays its own max(). This is what the
    # present implementation must beat, and the distance between it and `ceiling` is exactly
    # what fusion is worth. Reporting only one of the two hides one of the two mechanisms.
    decomposed = 0.0
    for op in profile.ops:
        ob = op.total_bytes
        of = op.total_flops
        decomposed += max(of / peak_flops, ob / bw)
    decomposed *= profile.invocations

    basis = "measured" if src_f == "measured" and src_b == "measured" else (
        "vendor" if "vendor" in (src_f, src_b) else src_f)
    return Bound(
        cell=cell, stage=profile.stage, device=device_name or device.get("name", "?"),
        wdtype=profile.wdtype, adtype=profile.adtype,
        flops=flops, unavoidable_bytes=unavoidable, traffic_bytes=profile.traffic_bytes,
        peak_flops=peak_flops, peak_bandwidth_bytes=bw, peak_basis=basis,
        compute_seconds=t_compute, memory_seconds=t_memory, ceiling_seconds=ceiling,
        decomposed_seconds=decomposed,
        bound_by="compute" if t_compute >= t_memory else "memory",
        arithmetic_intensity=(flops / unavoidable) if unavoidable else float("inf"),
        ridge_point=peak_flops / bw, param_bytes=profile.param_bytes)


def achieved_fraction(bound: Bound, measured_seconds):
    """How much of the ceiling this cell currently reaches, in (0, 1].

    `None` in, `None` out, deliberately and loudly: an unmeasured cell has no achieved fraction
    and substituting one would make every gap-closed score downstream a fiction.
    """
    if measured_seconds is None:
        return None
    m = float(measured_seconds)
    if not math.isfinite(m) or m <= 0:
        raise RooflineError(f"{bound.cell}: measured time {measured_seconds!r} is not a duration")
    frac = bound.ceiling_seconds / m
    if frac > 1.0 + 1e-9:
        raise RooflineError(
            f"{bound.cell}: measured {m * 1e3:.3f} ms is FASTER than the arithmetic ceiling "
            f"{bound.ceiling_seconds * 1e3:.3f} ms ({frac:.3f}x of it). One of three things is "
            f"true and all of them are bugs: the geometry undercounts the work, the device peak "
            f"is overstated, or the run did not do the work it claimed. Do not publish this "
            f"cell until it is resolved -- a ceiling that is beaten is not a ceiling.")
    return frac


def gap_closed(bound: Bound, base_seconds, cand_seconds):
    """The score: the fraction of the REMAINING roofline gap that a change closes.

        a  = ceiling / t                       achieved fraction
        g  = (a_cand - a_base) / (1 - a_base)

    Scale-free, comparable across cells, and correctly harder near the ceiling: taking a cell
    from 40% to 55% closes 0.25 of what was left, and so does taking it from 90% to 92.5%.

    It also self-terminates, which is the property that makes the axis-supply problem an
    incentive rather than an admin chore. As a cell approaches its ceiling the denominator
    shrinks toward zero, so the same absolute speedup is worth progressively less there and
    progressively more somewhere else. Nobody has to police a contributor grinding an exhausted
    cell; the score stops paying for it.

    Returns a dict rather than a float because the components are what make a receipt auditable,
    and because `g` alone cannot be checked by anyone.
    """
    if base_seconds is None or cand_seconds is None:
        return None
    a_base = achieved_fraction(bound, base_seconds)
    a_cand = achieved_fraction(bound, cand_seconds)
    remaining = 1.0 - a_base
    if remaining <= 1e-12:
        # The base is already at the ceiling. Any further "gain" is a measurement artifact or a
        # broken ceiling, and dividing by it would manufacture an enormous score.
        return {"gap_closed": 0.0, "achieved_base": a_base, "achieved_candidate": a_cand,
                "remaining_before": remaining, "speedup": base_seconds / cand_seconds,
                "saturated": True,
                "note": "the base already sits at the arithmetic ceiling for this cell; there "
                        "is no remaining gap to close and nothing here is scored"}
    return {
        "gap_closed": (a_cand - a_base) / remaining,
        "achieved_base": a_base,
        "achieved_candidate": a_cand,
        "remaining_before": remaining,
        "remaining_after": 1.0 - a_cand,
        "speedup": base_seconds / cand_seconds,
        "saturated": False,
        "ceiling_seconds": bound.ceiling_seconds,
        "base_seconds": base_seconds,
        "candidate_seconds": cand_seconds,
    }


def room_left(bound: Bound, measured_seconds):
    """What a contributor is told before they start.

    Deliberately phrased as the speedup still available rather than as a percentage of anything,
    because "there is 60% of the roofline left" and "the cell can get 2.5x faster" are the same
    fact and only one of them is the question being asked.
    """
    a = achieved_fraction(bound, measured_seconds)
    if a is None:
        return {"achieved": None, "max_further_speedup": None,
                "why": "this cell has not been measured on the pinned hardware; a ceiling "
                       "without an achieved fraction says how big the box is and nothing about "
                       "how full it is"}
    return {"achieved": a, "max_further_speedup": 1.0 / a if a > 0 else None,
            "remaining_fraction": 1.0 - a,
            "seconds_recoverable": measured_seconds - bound.ceiling_seconds}
