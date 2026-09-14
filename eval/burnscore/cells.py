"""A generation: the frozen definition of what is measured and what the numbers mean.

Everything in a generation is FROZEN for its lifetime. A receipt stays attached to the
generation that produced it, and if the meaning of the evaluation changes materially the answer
is BG-2, never an edit to BG-1. The generation's SHA-256 goes into every receipt and
`burnish receipt verify` refuses one whose generation has moved underneath it.

A cell is `(stage, shape, dtype)`. It carries, and must carry before it can be scored:

    ceiling_seconds      the arithmetic bound              -- computable from a config file
    achieved             the fraction currently reached    -- needs a measurement
    floor_pct            the run-to-run spread             -- needs repeated control runs

A cell holding the first and not the other two is `calibrated: false` and cannot produce a
score. It is still PUBLISHED, with its ceiling and an explicit null, because a contributor
deciding where to spend a week is better served by "the box is this big, nobody has measured how
full it is" than by silence -- and far better than by a plausible number nobody measured.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .frontier import Objective


class GenerationError(ValueError):
    """A generation definition is unusable."""


@dataclass
class Cell:
    id: str
    stage: str
    shape: dict
    wdtype: str
    adtype: str
    implemented: bool = True
    weight: float = 1.0
    ceiling_seconds: float = None
    achieved: float = None
    floor_pct: float = None
    floor_repeats: int = None
    measured_seconds: float = None
    notes: str = ""

    @property
    def calibrated(self) -> bool:
        return (self.achieved is not None and self.floor_pct is not None
                and self.ceiling_seconds is not None)

    def require_calibrated(self):
        if not self.calibrated:
            missing = [n for n, v in (("ceiling_seconds", self.ceiling_seconds),
                                      ("achieved", self.achieved),
                                      ("floor_pct", self.floor_pct)) if v is None]
            raise GenerationError(
                f"cell {self.id} is not calibrated: {', '.join(missing)} is null. A cell scored "
                f"without a measured achieved fraction produces a gap-closed number whose "
                f"denominator nobody measured, and it would verify cleanly and mean nothing. "
                f"Run `burnish calibrate --cell {self.id}` on the pinned hardware first.")

    def to_json(self) -> dict:
        return {"id": self.id, "stage": self.stage, "shape": self.shape,
                "wdtype": self.wdtype, "adtype": self.adtype,
                "implemented": self.implemented, "weight": self.weight,
                "ceiling_seconds": self.ceiling_seconds, "achieved": self.achieved,
                "floor_pct": self.floor_pct, "floor_repeats": self.floor_repeats,
                "measured_seconds": self.measured_seconds,
                "calibrated": self.calibrated, "notes": self.notes}


@dataclass
class Generation:
    name: str
    description: str
    model: dict
    device: str
    objectives: list
    reference_point: list
    cells: dict = field(default_factory=dict)
    aggregation: str = "weighted_mean"
    confidence_level: float = 0.99
    bootstrap_resamples: int = 20000
    bootstrap_seed: int = 20260911
    repeats: int = 3
    tolerance: dict = field(default_factory=dict)
    held_out: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)
    # The instrument settings the NOISE FLOOR was measured with, read from reference.json.
    #
    # These are not tuning knobs. A cell's floor describes the run-to-run spread of a particular
    # measurement procedure -- a median over `iters` timed invocations after `warmup` untimed
    # ones -- and averaging over more invocations produces a quieter measurement than the floor
    # describes. Comparing an effect measured one way against a floor measured another way is
    # comparing two different instruments, so the bench reads these and refuses to run with
    # anything else.
    #
    # They disagreed for a while: the floor was calibrated at warmup 2 / iters 5 and the bench
    # had 3 / 10 hardcoded with no flag to change it. That made every scored measurement twice
    # as expensive as it needed to be AND quieter than the floor it was judged against.
    calibrated_warmup: int = None
    calibrated_iters: int = None
    # Which physical device the anchor calibration was measured on. Provenance only: a run on any
    # other card of the pinned class is scored against the same anchor, because the score takes
    # nothing absolute from the run -- see `compute`.
    calibration_device: str = None
    calibration_device_name: str = None
    calibration_driver: str = None
    calibration_path: str = None

    @property
    def digest(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(canonical).hexdigest()

    def cell(self, cell_id: str) -> Cell:
        if cell_id not in self.cells:
            raise GenerationError(
                f"{self.name} has no cell {cell_id!r}. Cells are frozen; scoring against one "
                f"the generation does not declare is how a submission picks its own benchmark. "
                f"Adding a cell is a cartography contribution against a NEW generation.")
        return self.cells[cell_id]

    def scorable_cells(self) -> list:
        return [c for c in self.cells.values() if c.implemented and c.calibrated]

    def published_cells(self) -> list:
        return list(self.cells.values())

    def coverage(self, scored_ids) -> dict:
        want = {c.id for c in self.scorable_cells()}
        got = set(scored_ids)
        missing = sorted(want - got)
        return {"complete": not missing, "missing_cells": missing,
                "scored": sorted(got), "expected": sorted(want),
                "_note": ("A partial matrix credits nothing. Dropping the cell a change hurts "
                          "is the cheapest way to raise a score, and a rule that stops a drop "
                          "from PAYING removes the incentive rather than policing it.")}


def load(path, calibration=None) -> Generation:
    """Load a frozen generation and its anchor calibration.

      generation.json   WHAT IS MEASURED. Cells, shapes, dtypes, tolerances, objectives, the
                        sampling plan, the held-out list, the model pins. Frozen: receipts stay
                        attached to it, and editing one silently re-scores history.

      reference.json    THE ANCHOR. Each cell's achieved fraction and the base time behind it,
                        measured once for the generation on one card of the pinned class, and
                        the worst noise floor measured in any session on any card.

    The anchor is not a property of the box that scores. `compute` takes only the paired ratio
    from a run and expresses the ceiling in that run's seconds as `achieved x base time`, so a
    uniformly slower card, or one slower at a resource the code is not limited by, produces the
    same score with no calibration of its own. Anchoring again is needed only when the base code
    itself changes, and the scorer says so when it does.

    `calibration` overrides the committed reference.json, for scoring against a different anchor.
    """
    p = Path(path)
    doc = json.loads(p.read_text())
    objectives = [Objective.from_json(o) for o in doc["objectives"]]
    cells = {}
    ref_path = Path(calibration) if calibration else (p.parent / "reference.json")
    calib = json.loads(ref_path.read_text()) if ref_path.exists() else {"cells": {}}
    for c in doc["cells"]:
        cal = calib.get("cells", {}).get(c["id"], {})
        cells[c["id"]] = Cell(
            id=c["id"], stage=c["stage"], shape=c["shape"],
            wdtype=c["wdtype"], adtype=c.get("adtype", c["wdtype"]),
            implemented=bool(c.get("implemented", True)),
            weight=float(c.get("weight", 1.0)),
            # The anchor's ceiling when it records one, the generation's otherwise. Display only:
            # scoring expresses the ceiling in each run's own seconds -- see `compute`.
            ceiling_seconds=cal.get("ceiling_seconds") or c.get("ceiling_seconds"),
            achieved=cal.get("achieved"), floor_pct=cal.get("floor_pct"),
            floor_repeats=cal.get("floor_repeats"),
            measured_seconds=cal.get("measured_seconds"), notes=c.get("notes", ""))
    # Which card the anchor was measured on. Provenance, not a requirement.
    probe = calib.get("device_probe") or {}
    gen_kwargs_extra = {
        "calibration_device": probe.get("uuid"),
        "calibration_device_name": probe.get("name"),
        "calibration_driver": probe.get("driver_version"),
        "calibration_path": str(ref_path) if ref_path.exists() else None,
    }
    gen = Generation(
        name=doc["name"], description=doc["description"], model=doc["model"],
        device=doc["device"], objectives=objectives,
        reference_point=list(doc["reference_point"]), cells=cells,
        aggregation=doc.get("aggregation", "weighted_mean"),
        confidence_level=float(doc.get("confidence_level", 0.99)),
        bootstrap_resamples=int(doc.get("bootstrap_resamples", 20000)),
        bootstrap_seed=int(doc.get("bootstrap_seed", 20260911)),
        repeats=int(doc.get("repeats", 3)),
        tolerance=doc.get("tolerance", {}), held_out=doc.get("held_out", {}), raw=doc,
        calibrated_warmup=(calib.get("calibrated_with") or {}).get("warmup"),
        calibrated_iters=(calib.get("calibrated_with") or {}).get("iters"),
        **gen_kwargs_extra)
    if gen.raw.get("_status"):
        # Kept loadable so the ceiling table still prints; refused by the scorer.
        pass
    return gen


def cell_objectives(generation: Generation, cell: Cell) -> list:
    """The generation's objectives with this cell's LATENCY bounds substituted in.

    The generation-wide latency bounds are a documented fallback and they are nearly useless as
    a scoring range: against a 60-second timeout, a 64 ms cell and a 278 ms cell both normalize
    to about 0.998, so the latency axis carries almost no information and the frontier is
    decided entirely by memory and fidelity. Cells here differ in absolute time by a factor of
    four and across the declared resolutions by far more.

    So each cell is normalized against its own bracket:

        hi (score 1.0) = the cell's ceiling -- the best time that can exist
        lo (score 0.0) = 1.5x the base time

    `compute` passes the cell with its ceiling expressed in the run's own seconds, so both ends
    sit on the card that ran and the bracket is the same fraction of the ceiling on every card.
    Anchoring the good end at the ceiling also makes this axis and the gap-closed score agree
    about what progress is, rather than having the frontier reward something subtly different.
    """
    cell.require_calibrated()
    base_s = cell.ceiling_seconds / cell.achieved
    out = []
    for o in generation.objectives:
        if o.key == "latency_s":
            out.append(Objective(key=o.key, direction="min", lo=base_s * 1.5,
                                 hi=cell.ceiling_seconds, unit=o.unit))
        else:
            out.append(o)
    return out


def aggregate(per_cell: dict, generation: Generation) -> dict:
    """Combine per-cell gap-closed into one number for the receipt.

    A WEIGHTED ARITHMETIC mean, not a geometric one, and the choice is load-bearing. gap-closed
    is already scale-free and already lives on a common [0,1]-ish scale by construction, so the
    reason to use a geometric mean -- combining quantities with different units or wildly
    different magnitudes -- does not apply. A geometric mean would also be undefined the moment
    one cell goes negative, which is a thing that happens on every honest matrix.
    """
    rows = []
    total_w = 0.0
    acc = 0.0
    for cell_id, g in per_cell.items():
        cell = generation.cell(cell_id)
        rows.append({"cell": cell_id, "gap_closed": g, "weight": cell.weight})
        acc += g * cell.weight
        total_w += cell.weight
    if total_w <= 0:
        raise GenerationError("no weighted cells to aggregate")
    return {"gap_closed": acc / total_w, "per_cell": rows, "total_weight": total_w,
            "method": "weighted arithmetic mean of per-cell gap-closed"}
