"""Paired bootstrap over interleaved repeats. A qualification gate, never a multiplier.

Confidence answers "how sure are we"; the score answers "how much was created". Multiplying one
by the other produces a number that answers neither and cannot be argued with either. So the
interval qualifies a result and the score stays the score:

    lower bound of gap-closed above this cell's floor-equivalent   -> credited
    otherwise                                                      -> resolved: false

Resampling is PAIRED over repeat index, because the runs are paired: repeat k of base and repeat
k of candidate ran adjacently on one box under one thermal state. An estimator that resampled
the arms independently would throw away the only thing that makes a same-box delta mean anything
on hardware whose clocks cannot be pinned.

The statistic resampled is `gap_closed` itself rather than a speedup, because gap_closed is what
gets published and a confidence interval on a different quantity is not a confidence interval on
the published one. It is nonlinear in the timings, which is precisely why it is recomputed from
resampled timings on every draw rather than propagated analytically.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, asdict

MIN_RESAMPLES = 1000


class BootstrapError(ValueError):
    """The repeats cannot support an interval."""


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    level: float
    resamples: int
    seed: int
    repeats: int
    method: str = "paired_percentile_bootstrap"

    def to_json(self) -> dict:
        return asdict(self)

    def excludes(self, threshold: float) -> bool:
        """Is the whole interval above `threshold`? The gate, stated once."""
        return self.lower > threshold


def _gap_closed(ceiling_s, base_times, cand_times):
    """gap-closed from a set of paired repeats, combined the way the score is combined.

    Times are combined by GEOMETRIC mean before the achieved fractions are formed. The score is
    built from ratios, and the mean of ratios is not the ratio of arithmetic means -- combining
    the wrong way changes go/no-go on drifting clocks, which is a mistake this lineage has
    already made once and paid for.
    """
    gb = math.exp(statistics.fmean(math.log(t) for t in base_times))
    gc = math.exp(statistics.fmean(math.log(t) for t in cand_times))
    a_b = ceiling_s / gb
    a_c = ceiling_s / gc
    rem = 1.0 - a_b
    if rem <= 1e-12:
        return 0.0
    return (a_c - a_b) / rem


def paired_bootstrap(ceiling_s, base_times, cand_times, *, level=0.99, resamples=20000,
                     seed=20260911):
    """Percentile bootstrap of gap-closed over paired repeats.

    Everything that makes the answer reproducible is an input and is recorded: the seed, the
    resample count and the level. A receipt that could not be recomputed from the raw timings is
    not a receipt.
    """
    b = [float(x) for x in base_times]
    c = [float(x) for x in cand_times]
    if len(b) != len(c):
        raise BootstrapError(
            f"unpaired repeats: {len(b)} base, {len(c)} candidate. Interleaved pairing is the "
            f"entire basis of a same-box delta.")
    n = len(b)
    if n == 0:
        raise BootstrapError("no repeats")
    for x in b + c:
        if not math.isfinite(x) or x <= 0:
            raise BootstrapError(f"{x!r} is not a duration")
    if not (0.5 < level < 1.0):
        raise BootstrapError("confidence level must be in (0.5, 1.0)")
    if resamples < MIN_RESAMPLES:
        raise BootstrapError(f"{resamples} resamples cannot resolve a {level:.0%} interval; "
                             f"minimum {MIN_RESAMPLES}")
    if ceiling_s <= 0:
        raise BootstrapError("a non-positive ceiling cannot produce a gap")

    point = _gap_closed(ceiling_s, b, c)
    if n == 1:
        # One pair carries no information about spread. Saying so is the honest answer; a
        # zero-width interval would let a single run qualify, which is how a harness comes to
        # pay for a quiet afternoon.
        return Interval(point=point, lower=float("-inf"), upper=float("inf"), level=level,
                        resamples=0, seed=seed, repeats=1, method="insufficient_repeats")

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(_gap_closed(ceiling_s, [b[i] for i in idx], [c[i] for i in idx]))
    draws.sort()
    tail = (1.0 - level) / 2.0
    return Interval(point=point, lower=draws[_rank(len(draws), tail)],
                    upper=draws[_rank(len(draws), 1.0 - tail)], level=level,
                    resamples=resamples, seed=seed, repeats=n)


def _rank(count: int, q: float) -> int:
    """Nearest-rank index, clamped. Deterministic, and free of interpolation choices that would
    make two correct implementations disagree in the last digit of a published interval."""
    i = int(math.floor(q * count))
    return 0 if i < 0 else (count - 1 if i >= count else i)


def sign_test(base_times, cand_times) -> dict:
    """How many pairs went the candidate's way, and the exact two-sided binomial p.

    Reported beside the bootstrap because they fail differently. A bimodal cell -- one where a
    single stall either lands in the window or does not -- produces a wide bootstrap interval
    and a clean sign test, and reporting only the first would throw away a real result. The
    reverse case, a tight interval driven by one enormous pair, shows up as a weak sign test.
    Two rules that disagree are information, and both are published.
    """
    pairs = list(zip(base_times, cand_times))
    wins = sum(1 for b, c in pairs if c < b)
    n = sum(1 for b, c in pairs if c != b)
    if n == 0:
        return {"pairs": len(pairs), "candidate_faster": 0, "comparable": 0, "p_value": 1.0,
                "note": "every pair was identical"}
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return {"pairs": len(pairs), "candidate_faster": wins, "comparable": n,
            "p_value": min(1.0, 2.0 * tail),
            "note": "exact two-sided binomial over paired repeats"}
