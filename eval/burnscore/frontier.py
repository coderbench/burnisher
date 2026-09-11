"""The second half of the score: latency is not the only axis, so a scalar is not the answer.

A change that is 10% faster and needs 4 GB more VRAM has not made the runtime better; it has
picked a different point on a trade-off that was already available. A change that stays inside
the correctness tolerance while measurably degrading output has done the same thing, more
quietly. Both are legitimate engineering and neither is an expansion, and a scoring model that
reported only the speedup would pay for both as though they were.

So each arm's operating configurations become points in a normalized higher-is-better space,
the non-dominated subset is its frontier, and the volume that frontier dominates is compared.
Three axes:

    speed      1/latency, against frozen per-cell bounds
    memory     peak device bytes, minimized -- the bound is the card, because a configuration
               that does not fit is not a slow point, it is the absence of a point
    quality    distance from the pinned reference latents, minimized, with the correctness
               tolerance as the zero -- so "inside the gate but worse" is visible as a smaller
               number rather than invisible as a pass

Two rules carried over intact from the engagement that learned them:

**Bounds are frozen per generation.** Normalizing against today's best makes every historical
receipt mean something different the moment a new PR lands, and a ledger whose past entries
silently change is not a ledger.

**A failure is not a bad number.** An OOM, a timeout, a blown tolerance or a crash is the
ABSENCE of an operating point, not a slow one. It must never normalize into a small positive
score that still contributes volume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from itertools import combinations


class FrontierError(ValueError):
    """The objectives or the measurements cannot produce a frontier."""


# A record carrying one of these produced no operating point. Named explicitly, because a
# status this set does not know about falls through to the missing-objective branch and is
# counted as a harness fault -- which is the right default, and only works if the list is
# honest about what a failure looks like here.
FAILURE_STATUSES = frozenset({
    "OOM", "TIMEOUT", "CRASH", "NOT_RUN", "EVAL_ERROR",
    # The correctness gate rejected it. It is not traded off against speed, ever.
    "CORRECTNESS_FAIL", "TOLERANCE_FAIL", "NONDETERMINISTIC",
    # A generation that produced a black or NaN image runs fast and means nothing. The gate
    # catches it, and naming the status keeps a raw file written by something else from being
    # scored as if the arm had merely been quick.
    "DEGENERATE_OUTPUT",
    # A gain that appeared on the published shape and vanished on the held-out one.
    "SHAPE_OVERFIT",
})


@dataclass(frozen=True)
class Objective:
    key: str
    direction: str          # "max" or "min"
    lo: float               # raw value normalizing to 0.0
    hi: float               # raw value normalizing to 1.0
    unit: str = ""

    def __post_init__(self):
        if self.direction not in ("max", "min"):
            raise FrontierError(f"{self.key}: direction must be 'max' or 'min'")
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise FrontierError(f"{self.key}: bounds must be finite")
        if self.lo == self.hi:
            raise FrontierError(f"{self.key}: lo == hi, so the objective carries no information")
        if self.direction == "max" and self.hi <= self.lo:
            raise FrontierError(f"{self.key}: a maximized objective needs hi > lo")
        if self.direction == "min" and self.hi >= self.lo:
            raise FrontierError(
                f"{self.key}: a minimized objective needs hi < lo, so hi is the GOOD end. "
                f"State the bounds in the direction the objective actually improves -- the "
                f"alternative is a sign error that normalizes cleanly and inverts the score.")

    def normalize(self, raw) -> float:
        if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise FrontierError(f"{self.key}: {raw!r} is not a measurement")
        v = float(raw)
        if not math.isfinite(v):
            raise FrontierError(f"{self.key}: measurement is {v}; a failed run carries a "
                                f"failure status, not a non-finite number")
        s = (v - self.lo) / (self.hi - self.lo)
        return 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Objective":
        missing = [k for k in ("key", "direction", "lo", "hi") if k not in d]
        if missing:
            raise FrontierError(f"objective missing {', '.join(missing)}")
        return Objective(key=str(d["key"]), direction=str(d["direction"]),
                         lo=float(d["lo"]), hi=float(d["hi"]), unit=str(d.get("unit", "")))


def normalize_point(metrics: dict, objectives, status="OK"):
    """One configuration -> a normalized point, or None when it produced no point.

    A missing objective returns None rather than zero. Treating it as zero would quietly reward
    a runner that stopped reporting the dimension its candidate is worst in.
    """
    if status in FAILURE_STATUSES:
        return None
    pt = []
    for o in objectives:
        if o.key not in metrics or metrics[o.key] is None:
            return None
        pt.append(o.normalize(metrics[o.key]))
    return tuple(pt)


def dominates(a, b) -> bool:
    if len(a) != len(b):
        raise FrontierError(f"points of different dimension: {len(a)} vs {len(b)}")
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_frontier(points):
    """The non-dominated subset, deduplicated, in a deterministic order.

    Deterministic order matters more than it looks: the frontier feeds a hypervolume that has to
    be reproducible bit for bit from the raw results, or the receipt cannot be audited. Sorting
    also makes the answer independent of the order the runner happened to execute in.
    """
    uniq = sorted({tuple(float(v) for v in p) for p in points})
    return [c for c in uniq if not any(dominates(o, c) for o in uniq if o != c)]


def hypervolume(points, reference) -> float:
    """Volume dominated by `points` and bounded below by `reference` (the worst corner).

    Monotone by construction: adding a non-dominated point can only increase it and losing one
    can only decrease it. That is what makes it a fair single number for a portfolio of
    operating points, and why no separate regression penalty is needed -- a lost region reduces
    the volume by exactly its own size.

    2D is closed form; higher dimensions are exact inclusion-exclusion, which is exponential in
    the number of FRONTIER points and therefore fine for the handful a cell carries, and exact
    rather than approximate, which a receipt requires.
    """
    if not points:
        return 0.0
    dim = len(reference)
    for p in points:
        if len(p) != dim:
            raise FrontierError(f"point of dimension {len(p)} against a {dim}-d reference")
    clamped = [tuple(max(float(v), float(r)) for v, r in zip(p, reference)) for p in points]
    front = [p for p in pareto_frontier(clamped)
             if all(v > r for v, r in zip(p, reference))]
    if not front:
        return 0.0
    if dim == 1:
        return max(p[0] for p in front) - float(reference[0])
    if dim == 2:
        return _hv2(front, reference)
    return _hv_incl_excl(front, reference)


def _hv2(front, reference) -> float:
    rx, ry = float(reference[0]), float(reference[1])
    total, prev_y = 0.0, ry
    for x, y in sorted(front, key=lambda p: (-p[0], -p[1])):
        if y <= prev_y:
            continue
        total += (x - rx) * (y - prev_y)
        prev_y = y
    return total


def _hv_incl_excl(front, reference) -> float:
    ref = [float(r) for r in reference]
    total = 0.0
    for size in range(1, len(front) + 1):
        sign = 1.0 if size % 2 else -1.0
        for subset in combinations(front, size):
            vol = 1.0
            for axis in range(len(ref)):
                edge = min(p[axis] for p in subset) - ref[axis]
                if edge <= 0.0:
                    vol = 0.0
                    break
                vol *= edge
            total += sign * vol
    return total if total > 0.0 else 0.0


def frontier_delta(base_records, cand_records, objectives, reference):
    """dHV between two arms' portfolios, plus the diagnosis of what moved.

    The verdict is the part that matters. A positive latency change with a non-positive dHV is
    reported as MOVED_ALONG_FRONTIER, and that is not a euphemism for a loss -- it is the
    accurate description of a trade the runtime could already make.
    """
    b_pts = [p for p in (normalize_point(r.get("metrics", {}), objectives,
                                         r.get("status", "OK")) for r in base_records) if p]
    c_pts = [p for p in (normalize_point(r.get("metrics", {}), objectives,
                                         r.get("status", "OK")) for r in cand_records) if p]
    hv_b = hypervolume(b_pts, reference)
    hv_c = hypervolume(c_pts, reference)
    lost = [p for p in pareto_frontier(b_pts)
            if not any(dominates(q, p) or q == p for q in c_pts)]
    gained = [p for p in pareto_frontier(c_pts)
              if not any(dominates(q, p) or q == p for q in b_pts)]
    return {
        "hypervolume_base": hv_b, "hypervolume_candidate": hv_c,
        "delta": hv_c - hv_b,
        "relative_delta": ((hv_c - hv_b) / hv_b) if hv_b > 0 else None,
        "points_base": len(b_pts), "points_candidate": len(c_pts),
        "dropped_base_points": len(base_records) - len(b_pts),
        "dropped_candidate_points": len(cand_records) - len(c_pts),
        "frontier_points_lost": [list(p) for p in lost],
        "frontier_points_gained": [list(p) for p in gained],
        "expanded": (hv_c - hv_b) > 0,
        "objectives": [o.key for o in objectives],
        "_dropped_note": ("A dropped point is a configuration that produced no operating point "
                          "-- it OOMed, timed out, blew the tolerance, or failed to report an "
                          "objective. It is not counted as a slow point, and a candidate that "
                          "drops one of the base's frontier points loses that volume."),
    }


def verdict(gap_closed, dhv, *, resolved) -> str:
    """One word for what happened, derived rather than chosen."""
    if not resolved:
        return "UNRESOLVED"
    if gap_closed > 0 and dhv > 0:
        return "FRONTIER_EXPANDED"
    if gap_closed > 0 and dhv <= 0:
        return "MOVED_ALONG_FRONTIER"
    if gap_closed <= 0 and dhv > 0:
        return "EXPANDED_OFF_LATENCY"
    return "NO_GAIN"
