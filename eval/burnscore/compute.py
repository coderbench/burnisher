"""Raw paired measurements -> a scored receipt.

    raw records (per cell, per variant, per config, per repeat)
        -> pairing and coverage checks               the ones that catch a corrupt comparison
        -> per cell: gap closed, from the anchored achieved fraction and the paired ratio
        -> per cell: paired bootstrap, and the cell's own measured noise floor
        -> weighted aggregate                        = gap closed for the submission
        -> frontier hypervolume over latency, memory and fidelity
        -> status, derived                           = the receipt

Every refusal below is a guard, and every guard is an incident. Removing one because it is
inconvenient is how an evaluator comes to print a confident number for a run that never
happened. If one is in your way, find out which incident it encodes first -- they are named.
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from collections import defaultdict

from .bootstrap import paired_bootstrap, sign_test
from .cells import aggregate as aggregate_cells, cell_objectives, GenerationError
from .floor import floor_as_gap_closed, resolution_gate
from .frontier import FAILURE_STATUSES, frontier_delta
from .roofline import gap_closed as gap_closed_for


class ComputeError(ValueError):
    """The raw results cannot be turned into a score."""


def _require(cond, msg):
    if not cond:
        raise ComputeError(msg)


def _geomean(xs):
    return math.exp(statistics.fmean(math.log(float(x)) for x in xs))


def group(records):
    """(cell, variant) -> {repeat: [records]}, with the shape checks a comparison depends on."""
    out = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k in ("cell", "variant", "repeat", "status"):
            _require(k in r, f"raw record is missing {k!r}: {r}")
        _require(r["variant"] in ("base", "candidate"),
                 f"variant must be 'base' or 'candidate', not {r['variant']!r}")
        out[(r["cell"], r["variant"])][int(r["repeat"])].append(r)
    return out


# How far the base arm may sit from the anchor's measured time before the anchor stops describing
# it. Measured, not guessed: on a second RTX 5090 with a different driver and host, BG-1's base
# times came in +2.2%, +9.4% and -1.2% from the anchor's (eval/cells/BG-1/second-card-check.json),
# and the small, host-heavy t5 cell moved most. A 10% band would refuse that card's t5 on a noisy
# afternoon; 25% leaves room for it and still catches a large change to the base code.
BASE_ANCHOR_BAND_PCT = 25.0
# How noisy a run may be, in multiples of the cell's frozen noise floor, before it is refused.
BASE_SPREAD_FLOORS = 3.0


def compute(generation, records, *, held_out_records=None, allow_partial=False,
            reference_drift_guard=True, device=None):
    """Score a submission. `generation` is frozen; `records` are what the runner measured.

    `device` is the box the records were measured on, from the raw file's provenance. The score
    does not depend on it: no box has to be calibrated before it can score, which is what lets a
    rented card be swapped for another without half an hour of setup. It is accepted so callers
    can pass provenance through unchanged.
    """
    uncalibrated = [c.id for c in generation.cells.values()
                    if c.implemented and not c.calibrated]
    _require(not uncalibrated,
             f"{generation.name} is UNCALIBRATED for {', '.join(uncalibrated)}: no measured "
             f"achieved fraction and/or no measured noise floor. A gap-closed score needs a "
             f"denominator somebody measured, and a receipt computed without one would carry a "
             f"checksum, verify cleanly, and mean nothing. Run `burnish calibrate` on the "
             f"pinned hardware first.")

    grouped = group(records)
    scored_cells = sorted({c for (c, _v) in grouped})
    coverage = generation.coverage(scored_cells)
    _require(coverage["complete"] or allow_partial,
             f"incomplete matrix: {', '.join(coverage['missing_cells'])} was not run. Dropping "
             f"the cell a change hurts is the cheapest way to raise a score. Pass "
             f"allow_partial=True to compute it anyway -- the receipt will say PARTIAL and "
             f"credit nothing.")

    per_cell = {}
    run_cells = {}
    cell_gaps = {}
    any_unresolved = False
    for cell_id in scored_cells:
        cell = generation.cell(cell_id)
        if not cell.implemented:
            raise ComputeError(
                f"{cell_id} is declared but NOT IMPLEMENTED in {generation.name}; it has a "
                f"published ceiling and no reference. Measuring it means landing that "
                f"reference, which is a cartography contribution against a new generation.")
        base = grouped.get((cell_id, "base"), {})
        cand = grouped.get((cell_id, "candidate"), {})
        _require(base and cand, f"{cell_id}: needs both a base and a candidate arm")
        shared = sorted(set(base) & set(cand))
        _require(len(shared) == len(base) == len(cand),
                 f"{cell_id}: unpaired repeats -- base has {sorted(base)}, candidate has "
                 f"{sorted(cand)}. Repeat k of each arm is ONE interleaved measurement; an arm "
                 f"with extra repeats is not better sampled, it is unpaired, and clocks cannot "
                 f"be pinned in a container so an unpaired delta means nothing.")
        # The generation DECLARES its sampling plan, and until this check existed nothing
        # enforced it: `repeats` sat in the frozen definition as decoration while the runner's
        # own flag decided the real number. A declared parameter nobody checks is worse than an
        # undeclared one, because it reads as a guarantee.
        #
        # Enforced as a minimum rather than an exact count: more repeats is a better-sampled run
        # of the same experiment and there is no reason to refuse it. Fewer is a different
        # experiment wearing this generation's name.
        want = max(2, int(generation.repeats or 2))
        _require(len(shared) >= want,
                 f"{cell_id}: {len(shared)} paired repeat(s), but {generation.name} declares "
                 f"{want}. That number is part of the frozen definition because it decides what "
                 f"resolves: the paired bootstrap resamples repeat INDICES, and two of them "
                 f"admit three distinct resamples, so an interval over them is the larger and "
                 f"smaller of two numbers wearing a confidence level. Run more repeats, or "
                 f"score against a generation that declares fewer.")

        b_times, c_times = [], []
        for k in shared:
            for arm, sink in ((base[k], b_times), (cand[k], c_times)):
                _require(len(arm) == 1,
                         f"{cell_id} repeat {k}: {len(arm)} records for one arm. Two records "
                         f"for one repeat means the runner ran it twice or two benchmarks "
                         f"raced -- either way the pairing is gone.")
                rec = arm[0]
                if rec["status"] in FAILURE_STATUSES:
                    raise ComputeError(
                        f"{cell_id} repeat {k} ({rec['variant']}): status {rec['status']}. A "
                        f"failed run is the ABSENCE of a measurement, not a slow one. It must "
                        f"not be averaged into a ratio -- an arm that lost its work did not run "
                        f"slower, it ran less.")
                t = rec.get("metrics", {}).get("latency_s")
                _require(isinstance(t, (int, float)) and not isinstance(t, bool)
                         and math.isfinite(float(t)) and float(t) > 0,
                         f"{cell_id} repeat {k}: latency_s={t!r} is not a duration")
                sink.append(float(t))

        # The ceiling in THIS run's seconds. The achieved fraction belongs to the base code and was
        # measured once, when the generation was anchored; the base arm's time here belongs to
        # this card. Their product is the ceiling expressed on this card, so all a run contributes
        # to the score is the paired base/candidate ratio -- and a card that is uniformly slower,
        # or slower at a resource the code is not limited by, scores the same.
        #
        # This replaced per-box calibration, which divided this card's time into this card's
        # probed ceiling. That is invariant for a uniformly slower card, but it moves the score by
        # the card's peak difference whenever the code is bound by something else -- launch
        # overhead, today -- and it cost every rented box half an hour before it could score.
        base_geo = _geomean(b_times)
        run_ceiling = cell.achieved * base_geo
        run_cell = dataclasses.replace(cell, ceiling_seconds=run_ceiling)
        run_cells[cell_id] = run_cell

        # Two guards on the base arm, each sized for what it catches. Neither needs this box to
        # have been calibrated.
        #
        # 1. Was the box quiet? The base arm's own spread across repeats, against the cell's
        #    frozen floor. A floor moves between sessions, so the frozen one is the worst ever
        #    measured; a run noisier than several of those is a bad afternoon on this box, not a
        #    measurement of the submission.
        spread_pct = (max(b_times) / min(b_times) - 1.0) * 100.0
        if reference_drift_guard and spread_pct > max(cell.floor_pct, 1e-9) * BASE_SPREAD_FLOORS:
            raise ComputeError(
                f"{cell_id}: the BASE arm's own repeats spread {spread_pct:.3f}% in this run, more "
                f"than {BASE_SPREAD_FLOORS:.0f}x this cell's frozen noise floor of "
                f"{cell.floor_pct:.3f}%. The box was too noisy to judge anything against; that is "
                f"not the submission's fault. Re-run when the device is quiet.")
        # 2. Is the base code the code the anchor measured? Its time here against the anchor's,
        #    with a band wide enough for different cards and hosts of the pinned class and
        #    narrower than a large change to the base. Outside it, the anchored achieved fraction
        #    no longer describes this base, and every score in the cell would inherit the error.
        base_vs_anchor_pct = None
        if cell.measured_seconds:
            base_vs_anchor_pct = (base_geo / cell.measured_seconds - 1.0) * 100.0
            if reference_drift_guard and abs(base_vs_anchor_pct) > BASE_ANCHOR_BAND_PCT:
                raise ComputeError(
                    f"{cell_id}: the BASE arm took {base_geo:.4f} s where the anchor measured "
                    f"{cell.measured_seconds:.4f} s ({base_vs_anchor_pct:+.1f}%), outside the "
                    f"+/-{BASE_ANCHOR_BAND_PCT:.0f}% band cards of the pinned class fall in. "
                    f"Either the base code is not the code this generation was anchored on -- "
                    f"re-anchor it once, on any card, with `burnish calibrate --write` -- or this "
                    f"is not the pinned hardware.")

        g = gap_closed_for(_BoundView(run_cell), base_geo, _geomean(c_times))
        ci = paired_bootstrap(run_ceiling, b_times, c_times,
                              level=generation.confidence_level,
                              resamples=generation.bootstrap_resamples,
                              seed=generation.bootstrap_seed)
        floor_gap = floor_as_gap_closed(cell.floor_pct, cell.achieved)
        # BOTH gates. The interval says the effect is not zero; the floor says the effect is
        # bigger than this cell's own noise. An axis whose effect sits inside its own noise is
        # open, not solved, and an interval alone does not say that -- enough repeats make a
        # tiny, real, thermally-driven bias significant.
        resolved = ci.excludes(floor_gap) or ci.upper < -floor_gap
        # An UNRESOLVED cell contributes exactly zero, and does not block the submission.
        #
        # This is the difference between "we cannot tell" and "nothing happened", and getting it
        # wrong in either direction is bad. Blocking would mean a submission that improves the
        # DiT and leaves the VAE untouched scores nothing -- an untouched cell can never resolve,
        # because there is nothing there to resolve. Crediting the raw figure would mean paying
        # for whichever way the noise happened to point. Zero is the honest contribution of a
        # cell whose measurement did not clear its own floor, and the raw figure is still
        # reported beside it so nobody has to take this on trust.
        credited_cell = g["gap_closed"] if resolved else 0.0
        if not resolved:
            any_unresolved = True
        per_cell[cell_id] = {
            "gap_closed": g["gap_closed"],
            "credited_gap_closed": credited_cell,
            "achieved_base": g["achieved_base"],
            "achieved_candidate": g["achieved_candidate"],
            "remaining_before": g["remaining_before"],
            "speedup": g["speedup"],
            "ceiling_seconds": cell.ceiling_seconds,
            "ceiling_basis": "model",
            "run_ceiling_seconds": run_ceiling,
            "anchor_achieved": cell.achieved,
            "anchor_measured_seconds": cell.measured_seconds,
            "base_vs_anchor_pct": base_vs_anchor_pct,
            "base_spread_pct": spread_pct,
            "floor_pct": cell.floor_pct,
            "floor_as_gap_closed": floor_gap,
            "resolved": resolved,
            "resolution_note": ("credited only when the interval clears this cell's own "
                                "measured floor, not a constant threshold"),
            "confidence_interval": ci.to_json(),
            "sign_test": sign_test(b_times, c_times),
            "base_seconds": b_times, "candidate_seconds": c_times,
            "repeats": len(shared),
            "cell_room_after": resolution_gate(cell.floor_pct, g["achieved_candidate"]),
        }
        cell_gaps[cell_id] = credited_cell

    agg = aggregate_cells(cell_gaps, generation)
    agg["method"] = ("weighted arithmetic mean of per-cell gap-closed, with unresolved cells "
                     "contributing zero")

    # A cell that got measurably WORSE is not noise and is not zeroed. It enters the aggregate
    # at its full negative value and is also named, because a submission that buys a large win
    # in one cell with a smaller real loss in another has done something a maintainer needs to
    # see rather than something an average should absorb silently.
    regressed = sorted(c for c, v in per_cell.items()
                       if v["resolved"] and v["gap_closed"] < 0)

    # The submission-level interval is the weighted combination of the per-cell intervals'
    # endpoints. It is deliberately conservative: a submission qualifies only if the aggregate
    # lower bound clears the weighted floor, and no cell may be unresolved and still credited.
    w = {c: generation.cell(c).weight for c in per_cell}
    tw = sum(w.values()) or 1.0
    lower = sum(per_cell[c]["confidence_interval"]["lower"] * w[c] for c in per_cell) / tw
    upper = sum(per_cell[c]["confidence_interval"]["upper"] * w[c] for c in per_cell) / tw
    weighted_floor = sum(per_cell[c]["floor_as_gap_closed"] * w[c] for c in per_cell) / tw
    interval = {
        "lower": lower, "upper": upper, "level": generation.confidence_level,
        "weighted_floor_as_gap_closed": weighted_floor,
        "resolved": lower > weighted_floor,
        "cells_unresolved": sorted(c for c, v in per_cell.items() if not v["resolved"]),
        "cells_regressed": regressed,
        "rule": ("the weighted lower bound of the paired bootstrap must clear the weighted "
                 "per-cell noise floor. TWO quantities, not one: an interval alone does not say "
                 "an effect is bigger than the noise -- with enough repeats a tiny thermal bias "
                 "becomes significant -- and a floor alone does not say the effect is real. "
                 "Cells that did not individually resolve contribute zero and are named."),
    }

    # The frontier is computed PER CELL, over that cell's own configurations, against that
    # cell's own bounds. Pooling every cell's records into one hypervolume looks tidier and is
    # wrong: in a pooled space the fastest CELL dominates the slowest one no matter which arm
    # either came from, so a real improvement in the slow cell is invisible and the delta comes
    # out as exactly zero. That is not a hypothetical -- it is what this function did until the
    # test for a clean win caught it.
    fr_cells = {}
    for cell_id in per_cell:
        objs = cell_objectives(generation, run_cells[cell_id])
        fr_cells[cell_id] = frontier_delta(
            [r for r in records if r["cell"] == cell_id and r["variant"] == "base"],
            [r for r in records if r["cell"] == cell_id and r["variant"] == "candidate"],
            objs, generation.reference_point)
    # A cell where BOTH arms produced no operating point is a harness fault, not a tie. It
    # happens when the runner stopped reporting an objective the generation declares, and the
    # symptom is a frontier delta of exactly zero -- which reads as "quality-neutral" rather than
    # as "nobody measured this dimension". Caught here because it is invisible downstream.
    for cell_id, d in fr_cells.items():
        if d["points_base"] == 0 and d["points_candidate"] == 0:
            raise ComputeError(
                f"{cell_id}: neither arm produced an operating point. Every record was dropped, "
                f"which means the runner did not report an objective this generation declares "
                f"({', '.join(o.key for o in generation.objectives)}). A frontier computed from "
                f"no points is exactly zero and looks like a neutral result; it is a missing "
                f"measurement.")
    rel = 0.0
    for cell_id, d in fr_cells.items():
        r = d["relative_delta"]
        # A cell whose base frontier has zero volume cannot express a relative change. It is
        # counted as no movement rather than as an infinite one.
        rel += (r if r is not None else 0.0) * w_of(generation, cell_id)
    total_cw = sum(w_of(generation, c) for c in fr_cells) or 1.0
    fr = {
        "delta": rel / total_cw,
        "per_cell": fr_cells,
        "method": ("weighted mean of per-cell relative hypervolume change, each cell's frontier "
                   "computed over its own configurations against its own frozen bounds"),
        "objectives": [o.key for o in generation.objectives],
        "expanded": (rel / total_cw) > 0,
        "_why_per_cell": ("Latency is not comparable between cells -- a 64 ms DiT step and a "
                          "278 ms VAE decode are different quantities. Normalizing both against "
                          "one generation-wide bracket makes the latency axis nearly constant "
                          "and hands the whole frontier to memory and fidelity."),
    }

    held = None
    if held_out_records:
        held = _held_out_verdict(generation, held_out_records, agg["gap_closed"])
    return {"per_cell": per_cell, "aggregate": agg, "interval": interval, "frontier": fr,
            "coverage": coverage, "held_out": held, "regressed_cells": regressed}


def w_of(generation, cell_id):
    return generation.cell(cell_id).weight


class _BoundView:
    """Adapter so `roofline.gap_closed` can score a calibrated Cell without rebuilding geometry.

    Exists because the ceiling in a frozen generation is the authority during scoring, not a
    figure recomputed at evaluation time from a config the submission might have touched.
    """
    def __init__(self, cell):
        self.cell = cell.id
        self.ceiling_seconds = cell.ceiling_seconds


def _held_out_verdict(generation, held_out_records, published_gap):
    """Did the gain survive a shape the candidate was not tuned on?

    A kernel fast only on the benchmarked shape scores nothing. The held-out shape is chosen by
    the evaluator at run time, from the base commit, after the candidate is frozen -- which is
    what makes this a guard rather than a second benchmark to tune against.
    """
    grouped = group(held_out_records)
    cells = sorted({c for (c, _v) in grouped})
    results = {}
    for cell_id in cells:
        base = grouped.get((cell_id, "base"), {})
        cand = grouped.get((cell_id, "candidate"), {})
        if not (base and cand):
            continue
        shared = sorted(set(base) & set(cand))
        bt = [float(base[k][0]["metrics"]["latency_s"]) for k in shared]
        ct = [float(cand[k][0]["metrics"]["latency_s"]) for k in shared]
        speedup = _geomean(bt) / _geomean(ct)
        results[cell_id] = {"speedup": speedup, "repeats": len(shared),
                            "sign_test": sign_test(bt, ct)}
    if not results:
        return {"survived": None, "per_shape": {},
                "why": "no held-out records were supplied; the guard did not run"}
    worst = min(r["speedup"] for r in results.values())
    # The published gain must not evaporate off-shape. A candidate that is 15% faster on the
    # scored shape and 1% faster on a neighbouring one has tuned a shape, not written a kernel.
    survived = (published_gap <= 0) or (worst >= 1.0)
    return {"survived": survived, "worst_speedup": worst, "per_shape": results,
            "why": ("the gain is present on a shape the candidate was not tuned on"
                    if survived else
                    f"the candidate is SLOWER on a held-out shape (worst {worst:.4f}x). A "
                    f"kernel fast only on the benchmarked shape is a tuned constant, not a "
                    f"contribution.")}
