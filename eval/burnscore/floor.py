"""Each cell's own measured noise floor, and the bar a submission has to clear in that cell.

The thing this replaces is a constant. SparkInfer discards anything under 2%; that number is a
guess at the noise, and it is wrong in both directions at once. On a quiet cell a 0.5% gain is
real and gets thrown away. On a noisy one a 3% gain is nothing and gets paid. The quantity being
guessed at is measurable, so measure it.

**How it is measured.** Two arms, both of them the unmodified base, run interleaved exactly the
way a real comparison runs. Every guard that applies to a scored comparison applies here, which
is the point: the floor has to be the noise of THIS measurement procedure, not of some quieter
one. The paired ratios of a control against itself should centre on 1.0, and how far they wander
is the floor.

**Two floors, and the larger wins.**

* the *spread* -- how far the paired control-vs-control ratios actually moved, peak to peak
* the *resolution* -- the smallest difference the instrument can represent at all

The second exists because a bench that prints three significant figures cannot resolve a
0.01% difference no matter how quiet the box is, and a cell whose repeats happened to print the
same number three times would otherwise publish a floor of zero and accept anything.

**Stated in the currency of the score.** A floor in percent of runtime is not directly
comparable to a gap-closed score, so `floor_as_gap_closed` converts it: how much gap a change
worth exactly one noise floor would appear to close in this cell. Near the ceiling that number
is large -- which is correct and is the whole reason this conversion is published. A cell at 95%
of roofline with a 1% floor cannot resolve anything smaller than a fifth of its remaining gap,
and a contributor deserves to know that before starting rather than after.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict


class FloorError(ValueError):
    """A floor cannot be computed from what was supplied."""


@dataclass(frozen=True)
class Floor:
    cell: str
    repeats: int
    median_ratio: float
    spread_pct: float
    stdev_pct: float
    resolution_pct: float
    floor_pct: float
    decided_by: str
    ratios: tuple
    basis: str = "measured"

    def to_json(self) -> dict:
        d = asdict(self)
        d["ratios"] = list(self.ratios)
        return d

    def clears(self, effect_pct: float) -> bool:
        """Is an observed effect bigger than this cell's own noise, whichever way it points?"""
        return abs(effect_pct) > self.floor_pct


def paired_ratios(arm_a, arm_b):
    """Repeat k of one arm against repeat k of the other. Pairing is not optional.

    Graphics clocks cannot be pinned in a container, so absolute numbers drift with temperature
    over minutes. The only trustworthy quantity is a same-box delta between two runs that
    happened next to each other, and an estimator that pooled the arms before dividing would
    discard exactly that.
    """
    a = [float(x) for x in arm_a]
    b = [float(x) for x in arm_b]
    if len(a) != len(b):
        raise FloorError(
            f"unpaired repeats: {len(a)} and {len(b)}. An interleaved pair is one measurement; "
            f"an arm with extra repeats is not better sampled, it is unpaired, and on a box "
            f"whose clocks drift that is the whole ballgame.")
    if not a:
        raise FloorError("no repeats")
    for x in a + b:
        if not math.isfinite(x) or x <= 0:
            raise FloorError(f"{x!r} is not a duration; a failed run carries a failure status, "
                             f"not a number")
    return [x / y for x, y in zip(a, b)]


def measure_floor(cell: str, control_a, control_b, *, timer_resolution_s=None,
                  reported_digits=None):
    """The floor of `cell`, from two interleaved arms that are both the unmodified base.

    `timer_resolution_s` or `reported_digits` fixes the instrument term. Supply one: a floor
    computed with neither can come out as zero, and a zero floor accepts noise as a result.
    """
    ratios = paired_ratios(control_a, control_b)
    n = len(ratios)
    med = statistics.median(ratios)
    spread = (max(ratios) - min(ratios)) / med * 100.0 if med else float("inf")
    stdev = (statistics.stdev(ratios) / med * 100.0) if n > 1 and med else 0.0

    if timer_resolution_s is not None:
        typical = statistics.median([float(x) for x in control_a])
        resolution = (timer_resolution_s / typical) * 100.0
    elif reported_digits is not None:
        # Half of the last printed digit, relative. A bench printing 4 significant figures
        # cannot represent a difference below 0.005%.
        resolution = 0.5 * 10.0 ** (-(int(reported_digits) - 1)) * 100.0
    else:
        raise FloorError(
            f"{cell}: measure_floor needs timer_resolution_s or reported_digits. Without one, "
            f"a cell whose repeats happened to agree publishes a floor of zero and then accepts "
            f"anything -- which is how a broken evaluator prints a confident number.")

    if n < 2:
        raise FloorError(
            f"{cell}: {n} repeat pair(s). One pair carries no information about spread; a floor "
            f"derived from it would be a floor of zero wearing a number.")

    floor = max(spread, resolution)
    return Floor(cell=cell, repeats=n, median_ratio=med, spread_pct=spread, stdev_pct=stdev,
                 resolution_pct=resolution, floor_pct=floor,
                 decided_by="spread" if spread >= resolution else "instrument_resolution",
                 ratios=tuple(ratios))


def floor_as_gap_closed(floor_pct: float, achieved_base: float) -> float:
    """This cell's floor, expressed as the gap-closed score it would masquerade as.

    A change worth exactly one floor makes the cell `floor_pct` faster, so
    `a' = a / (1 - floor)`, and the gap that appears closed is `(a' - a) / (1 - a)`.

    Published beside every cell, because the same 1% floor means completely different things at
    40% and at 95% of roofline -- 0.011 of the gap in one and 0.19 in the other. A contributor
    reading only the percent would take the wrong cell.
    """
    if not (0.0 < achieved_base < 1.0):
        raise FloorError(f"achieved_base must be in (0,1), got {achieved_base!r}")
    f = floor_pct / 100.0
    if f >= 1.0:
        raise FloorError(f"a floor of {floor_pct}% is not a floor, it is the whole measurement")
    a2 = min(achieved_base / (1.0 - f), 1.0)
    return (a2 - achieved_base) / (1.0 - achieved_base)


def resolution_gate(floor_pct: float, achieved_base: float, *, ratio=2.0) -> dict:
    """Screen question 2, asked of a cell that has now been measured.

    An axis whose room sits inside its own noise is OPEN, not solved -- and a cell whose room is
    only a few floors wide is one where a real improvement cannot be told from a quiet afternoon.
    """
    room = 1.0 - achieved_base
    floor_gap = floor_as_gap_closed(floor_pct, achieved_base)
    resolvable = (floor_gap > 0) and (1.0 / floor_gap) >= ratio
    return {
        "achieved": achieved_base, "room_fraction": room, "floor_pct": floor_pct,
        "floor_as_gap_closed": floor_gap,
        "floors_of_room": (1.0 / floor_gap) if floor_gap > 0 else float("inf"),
        "ratio_required": ratio, "resolvable": resolvable,
        "verdict": ("measurable" if resolvable else
                    "UNRESOLVABLE AT THIS FLOOR: the whole remaining gap is worth fewer than "
                    f"{ratio:.0f} noise floors, so a contributor cannot be shown to have moved "
                    "it. Quieten the cell, lengthen the run, or declare the cell closed -- do "
                    "not publish it as open."),
    }
