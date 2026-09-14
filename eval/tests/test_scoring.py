"""The scorer's own behaviour, exercised without hardware.

An evaluator is where the bugs are: a broken one prints a confident number rather than an error.
These tests are the thing standing between this repository and that outcome, so they assert the
GUARDS as much as the arithmetic -- every refusal below corresponds to a way a comparison can be
silently corrupted.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_EX = HERE.parent.parent / "examples"
sys.path.insert(0, str(HERE.parent))

from burnscore import cells as C, compute as CP, floor as F, frontier as FR, geometry as G
from burnscore import bootstrap as B, ledger as L, receipt as R, roofline as RL
from burnscore.pipeline import pixart_stages
from tests import fixtures


def _cands():
    return json.loads((HERE.parent.parent / "configs" / "candidates.json").read_text())


def _device():
    return json.loads((HERE.parent.parent / "configs" / "devices.json").read_text())["rtx5090"]


class TestGeometry(unittest.TestCase):
    def test_parameter_counts_reproduce_published_checkpoint_sizes(self):
        """The only thing between this module and a table of confident fiction."""
        for row in G.selfcheck(_cands()):
            self.assertAlmostEqual(row["ratio"], 1.0, delta=0.01,
                                   msg=f"{row['stage']}: enumerated geometry gives "
                                       f"{row['computed_param_bytes']} bytes against a "
                                       f"published {row['published_file_bytes']}. The config "
                                       f"was misread; every ceiling downstream is wrong.")

    def test_attention_flops_do_not_depend_on_implementation(self):
        kw = dict(batch=2, heads=16, q_len=4096, kv_len=4096, head_dim=72, ab=2.0)
        self.assertEqual(G.attention("a", impl="flash", **kw).flops,
                         G.attention("a", impl="materialized", **kw).flops)

    def test_materialized_attention_costs_more_traffic(self):
        kw = dict(batch=1, heads=1, q_len=16384, kv_len=16384, head_dim=512, ab=2.0)
        flash = G.attention("a", impl="flash", **kw)
        mat = G.attention("a", impl="materialized", **kw)
        self.assertGreater(mat.total_bytes, flash.total_bytes + 1e9,
                           "the VAE mid-block score matrix is about a gigabyte; a model that "
                           "priced both implementations the same could not see the tiling work")

    def test_embedding_read_is_not_the_resident_table(self):
        cfg = _cands()["candidates"]["pixart-sigma-xl2-1024"]["text_encoder"]
        p = G.t5_encoder(cfg, seq=300, batch=1)
        embed = next(o for o in p.ops if o.name == "embed_tokens")
        self.assertLess(embed.weight_bytes * 50, embed.param_bytes,
                        "300 gathered rows must not be priced as a 32128-row table")

    def test_unavoidable_bytes_excludes_intermediates(self):
        cfg = _cands()["candidates"]["pixart-sigma-xl2-1024"]
        p = G.pixart_dit(cfg["denoiser"], resolution=1024, caption_len=300, batch=2)
        self.assertLess(p.unavoidable_bytes, p.traffic_bytes,
                        "a ceiling that counted every intermediate would move when a "
                        "contributor fused, and would not be a ceiling")


class TestRoofline(unittest.TestCase):
    def setUp(self):
        cfg = _cands()["candidates"]["pixart-sigma-xl2-1024"]
        p = G.pixart_dit(cfg["denoiser"], resolution=1024, caption_len=300, batch=2)
        self.bound = RL.bound_for(p, _device(), cell="dit-step/1024/bf16")

    def test_gap_closed_is_scale_free(self):
        """The defining property: the same fraction of remaining gap scores the same anywhere."""
        c = self.bound.ceiling_seconds
        near = RL.gap_closed(self.bound, c / 0.90, c / 0.925)["gap_closed"]
        far = RL.gap_closed(self.bound, c / 0.40, c / 0.55)["gap_closed"]
        self.assertAlmostEqual(near, far, places=9)
        self.assertAlmostEqual(near, 0.25, places=9)

    def test_reaching_the_ceiling_closes_the_whole_gap(self):
        c = self.bound.ceiling_seconds
        self.assertAlmostEqual(RL.gap_closed(self.bound, c / 0.3, c)["gap_closed"], 1.0,
                               places=9)

    def test_the_inverted_reward_is_fixed(self):
        """The headline defect this scoring model exists to correct.

        Under a raw-percent regime a 20% gain on a cell at 10% of roofline outscores a 2% gain
        on a cell at 95%, which is backwards: the first is ordinary and the second is
        extraordinary. Gap-closed reverses it, and by a wide margin.
        """
        c = self.bound.ceiling_seconds
        easy = RL.gap_closed(self.bound, c / 0.10, c / 0.10 / 1.20)["gap_closed"]
        hard = RL.gap_closed(self.bound, c / 0.95, c / 0.95 / 1.02)["gap_closed"]
        self.assertGreater(hard, easy * 10,
                           "a 2% gain at 95% of roofline must dominate a 20% gain at 10%")

    def test_a_cell_cannot_yield_more_than_its_whole_remaining_gap(self):
        """Self-termination, stated correctly.

        The score does not decay per PR -- the same FRACTION of remaining gap always pays the
        same, which is the point. What terminates is the cell: the total closable gap is 1.0,
        the ledger compounds toward it, and the physically available speedup shrinks to
        `1/achieved`. At 95% of roofline no submission can ever be worth more than a 1.053x
        speedup there, however clever it is.
        """
        c = self.bound.ceiling_seconds
        self.assertAlmostEqual(RL.room_left(self.bound, c / 0.95)["max_further_speedup"],
                               1.0 / 0.95, places=9)
        perfect = RL.gap_closed(self.bound, c / 0.95, c)["gap_closed"]
        self.assertAlmostEqual(perfect, 1.0, places=9)

    def test_the_noise_floor_is_what_actually_closes_a_cell(self):
        """Near the ceiling a fixed floor eats the remaining gap, and the cell stops resolving.

        This is the practical terminator and it is a property of the floor rather than of the
        score, which is why both are published per cell.
        """
        self.assertTrue(F.resolution_gate(0.5, 0.30)["resolvable"])
        self.assertFalse(F.resolution_gate(0.5, 0.99)["resolvable"])

    def test_beating_the_ceiling_is_refused(self):
        with self.assertRaises(RL.RooflineError):
            RL.achieved_fraction(self.bound, self.bound.ceiling_seconds * 0.5)

    def test_unmeasured_cell_has_no_achieved_fraction(self):
        self.assertIsNone(RL.achieved_fraction(self.bound, None))
        self.assertIsNone(RL.room_left(self.bound, None)["achieved"])

    def test_ceiling_never_claims_to_be_measured(self):
        self.assertEqual(self.bound.basis, "model")


class TestFloor(unittest.TestCase):
    def test_unpaired_control_arms_are_refused(self):
        with self.assertRaises(F.FloorError):
            F.paired_ratios([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_a_floor_needs_an_instrument_term(self):
        with self.assertRaises(F.FloorError):
            F.measure_floor("c", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    def test_identical_repeats_do_not_produce_a_zero_floor(self):
        f = F.measure_floor("c", [1.0] * 3, [1.0] * 3, reported_digits=4)
        self.assertGreater(f.floor_pct, 0.0)
        self.assertEqual(f.decided_by, "instrument_resolution")

    def test_one_pair_is_refused(self):
        with self.assertRaises(F.FloorError):
            F.measure_floor("c", [1.0], [1.0], reported_digits=4)

    def test_the_same_floor_means_more_near_the_ceiling(self):
        """The finding that makes a constant threshold indefensible."""
        low = F.floor_as_gap_closed(1.0, 0.40)
        high = F.floor_as_gap_closed(1.0, 0.95)
        self.assertLess(low, high / 10.0)

    def test_resolution_gate_closes_a_cell_with_no_room_left(self):
        self.assertFalse(F.resolution_gate(1.0, 0.95)["resolvable"])
        self.assertTrue(F.resolution_gate(1.0, 0.40)["resolvable"])


class TestBootstrap(unittest.TestCase):
    def test_pairing_is_required(self):
        with self.assertRaises(B.BootstrapError):
            B.paired_bootstrap(1.0, [1.0, 2.0], [1.0])

    def test_one_repeat_yields_an_infinite_interval(self):
        i = B.paired_bootstrap(1.0, [2.0], [1.9])
        self.assertEqual(i.method, "insufficient_repeats")
        self.assertFalse(i.excludes(0.0))

    def test_too_few_resamples_is_refused(self):
        with self.assertRaises(B.BootstrapError):
            B.paired_bootstrap(1.0, [2.0, 2.0], [1.9, 1.9], resamples=10)

    def test_a_real_effect_excludes_zero_and_noise_does_not(self):
        c = 1.0
        real = B.paired_bootstrap(c, [2.0, 2.01, 1.99, 2.02, 1.98],
                                  [1.8, 1.81, 1.79, 1.82, 1.78])
        self.assertTrue(real.excludes(0.0))
        noise = B.paired_bootstrap(c, [2.0, 2.05, 1.95, 2.02, 1.98],
                                   [2.01, 1.96, 2.04, 1.99, 2.03])
        self.assertFalse(noise.excludes(0.0))

    def test_bootstrap_is_reproducible_from_its_recorded_seed(self):
        a = B.paired_bootstrap(1.0, [2, 2.1, 1.9], [1.8, 1.9, 1.7], seed=7, resamples=2000)
        b = B.paired_bootstrap(1.0, [2, 2.1, 1.9], [1.8, 1.9, 1.7], seed=7, resamples=2000)
        self.assertEqual(a.to_json(), b.to_json())


class TestFrontier(unittest.TestCase):
    def setUp(self):
        self.objs = [FR.Objective("latency_s", "min", 60.0, 0.0),
                     FR.Objective("peak_vram_bytes", "min", 34.36e9, 0.0)]
        self.ref = [0.0, 0.0]
        self.base = [{"metrics": {"latency_s": 2.0, "peak_vram_bytes": 11e9}, "status": "OK"}]

    def test_a_minimized_objective_with_inverted_bounds_is_refused(self):
        with self.assertRaises(FR.FrontierError):
            FR.Objective("latency_s", "min", 0.0, 60.0)

    def test_faster_but_fatter_is_a_move_along_not_an_expansion(self):
        cand = [{"metrics": {"latency_s": 1.8, "peak_vram_bytes": 15e9}, "status": "OK"}]
        d = FR.frontier_delta(self.base, cand, self.objs, self.ref)
        self.assertFalse(d["expanded"])
        self.assertEqual(FR.verdict(0.12, d["delta"], resolved=True), "MOVED_ALONG_FRONTIER")

    def test_faster_at_the_same_cost_expands(self):
        cand = [{"metrics": {"latency_s": 1.8, "peak_vram_bytes": 11e9}, "status": "OK"}]
        d = FR.frontier_delta(self.base, cand, self.objs, self.ref)
        self.assertTrue(d["expanded"])

    def test_a_failure_is_an_absent_point_not_a_bad_one(self):
        for bad in ("OOM", "CORRECTNESS_FAIL", "DEGENERATE_OUTPUT", "SHAPE_OVERFIT"):
            cand = [{"metrics": {"latency_s": 0.01, "peak_vram_bytes": 1e9}, "status": bad}]
            d = FR.frontier_delta(self.base, cand, self.objs, self.ref)
            self.assertEqual(d["points_candidate"], 0, bad)
            self.assertLess(d["delta"], 0, f"{bad} must not gain volume by being fast")

    def test_a_missing_objective_is_absent_rather_than_zero(self):
        cand = [{"metrics": {"latency_s": 1.0}, "status": "OK"}]
        self.assertEqual(FR.frontier_delta(self.base, cand, self.objs, self.ref)
                         ["points_candidate"], 0)

    def test_hypervolume_is_monotone(self):
        a = [(0.5, 0.5)]
        b = [(0.5, 0.5), (0.9, 0.2)]
        self.assertGreater(FR.hypervolume(b, self.ref), FR.hypervolume(a, self.ref))

    def test_hypervolume_matches_closed_form_in_three_dimensions(self):
        self.assertAlmostEqual(FR.hypervolume([(0.5, 0.4, 0.2)], [0, 0, 0]), 0.04, places=12)
        two = FR.hypervolume([(0.5, 0.4, 0.2), (0.2, 0.8, 0.5)], [0, 0, 0])
        self.assertAlmostEqual(two, 0.04 + 0.08 - (0.2 * 0.4 * 0.2), places=12)


class TestComputeAndReceipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gen = C.load(fixtures.calibrated_generation(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _score(self, **kw):
        recs = fixtures.records(self.gen, **kw)
        return CP.compute(self.gen, recs), recs

    def test_the_real_bg1_reference_is_calibrated_and_internally_consistent(self):
        """The honest state of this repository, asserted so it cannot rot into a guess.

        This test asserted the opposite -- that BG-1 shipped uncalibrated and refused to score
        -- for as long as that was true. It is no longer true, and an assertion that a
        measurement is absent is worthless once it has been taken. What survives is the reason
        the assertion existed: that no cell may carry a number nobody measured. That is checked
        here against the arithmetic, because a calibration is the one artefact in the tree whose
        parts can be made to check each other -- achieved is ceiling over measured, by
        definition, so a hand-edited achieved fraction stops agreeing with the seconds beside it.
        """
        real = C.load(HERE.parent / "cells" / "BG-1" / "generation.json")
        ref = json.loads((HERE.parent / "cells" / "BG-1" / "reference.json").read_text())
        self.assertEqual(sorted(c.id for c in real.scorable_cells()), sorted(ref["cells"]))
        # A cell declared but not yet measured is the intended state for new cartography: the
        # map may name a place before anyone has been there. What it may NOT do is offer that
        # place for scoring, which is the assertion above -- scorable_cells is exactly the
        # calibrated set, no more.
        uncal = [c.id for c in real.cells.values() if c.id not in ref["cells"]]
        for cid in uncal:
            self.assertIsNone(real.cells[cid].achieved,
                              f"{cid} has an achieved fraction but no calibration behind it")
        for cell in real.cells.values():
            cal = ref["cells"].get(cell.id)
            if cal is None:
                continue
            self.assertIsNotNone(cell.achieved, f"{cell.id} lost its calibration")
            self.assertIsNotNone(cell.floor_pct)
            self.assertEqual(cal["basis"], "measured")
            # achieved == ceiling / measured. Three numbers, one relation: edit any one of them
            # by hand and this stops holding.
            self.assertAlmostEqual(cal["achieved"],
                                   cal["ceiling_seconds"] / cal["measured_seconds"], places=10,
                                   msg=f"{cell.id}: achieved does not equal ceiling/measured. "
                                       f"One of the three was typed rather than measured.")
            # A run cannot beat its own arithmetic ceiling. If it appears to, the ceiling is
            # wrong, and every gap-closed score computed against it is wrong with it.
            self.assertLessEqual(cal["achieved"], 1.0, f"{cell.id} beats its own roofline")
            # The floor is the larger of the paired spread and the instrument's resolution, and
            # it must say which of the two decided it -- a floor nobody can attribute is a
            # threshold, and a threshold is the thing this scoring model exists to replace.
            self.assertIn(cal["floor_decided_by"], ("spread", "resolution", "worst-of-sessions"))
            own = max(cal["floor_spread_pct"], cal["floor_resolution_pct"])
            if cal.get("floor_decided_by") == "worst-of-sessions":
                # A pooled floor must be the worst of sessions that are themselves in the tree, so
                # it is re-derived from them rather than trusted.
                other = json.loads((HERE.parent / "cells" / "BG-1" /
                                    cal["floor_sessions_max_from"]).read_text())
                want = max(own, other["cells"][cell.id]["floor_pct"])
            else:
                want = own
            self.assertAlmostEqual(cal["floor_pct"], want, places=12,
                                   msg=f"{cell.id}: the floor is not the spread, the resolution, "
                                       f"or the worst of the committed sessions")
            self.assertGreaterEqual(cal["floor_repeats"], 2,
                                    f"{cell.id}: a spread needs repeated runs to be a spread")

    def test_an_uncalibrated_cell_still_refuses_to_score(self):
        """The refusal itself is the guard, and it outlives BG-1's own calibration.

        Every new cell enters the tree uncalibrated -- that is what paying for cartography
        means -- so this path is walked by every cell that is ever added, not just by the state
        BG-1 happened to ship in.
        """
        real = C.load(HERE.parent / "cells" / "BG-1" / "generation.json")
        for cell in real.cells.values():
            cell.achieved = None
            cell.floor_pct = None
        self.assertEqual(real.scorable_cells(), [])
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(real, [])
        self.assertIn("UNCALIBRATED", str(cm.exception))

    def test_a_receipt_says_whether_it_can_name_the_code_it_scored(self):
        """Three null fields do not read as "unknown" to anybody skimming. One flag does.

        The fields go null whenever the runner has no git metadata. That is honest, but it is
        also exactly the state a submission would want a receipt to be in, so the receipt has to
        say it out loud rather than leave it to be noticed.
        """
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        blind = dict(fixtures.provenance())
        for k in ("candidate_commit", "base_commit", "instrument_from"):
            blind.pop(k, None)
        rec = R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True, provenance=blind)
        self.assertFalse(rec["provenance"]["code_provenance_complete"])
        self.assertEqual(sorted(rec["provenance"]["code_provenance_missing"]),
                         ["base_commit", "candidate_commit", "instrument_from"])
        R.verify_receipt(rec, self.gen)

        full = dict(fixtures.provenance())
        full.update(candidate_commit="a" * 40, base_commit="a" * 40, instrument_from="deadbee")
        rec2 = R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True, provenance=full)
        self.assertTrue(rec2["provenance"]["code_provenance_complete"])
        self.assertEqual(rec2["provenance"]["code_provenance_missing"], [])
        self.assertNotIn("_code_provenance_note", rec2["provenance"])

    def test_the_bench_measures_with_the_instrument_that_measured_the_noise(self):
        """A floor and an effect from two different instruments do not compare.

        A cell's floor is the run-to-run spread of one procedure: the median of `iters` timed
        invocations after `warmup` untimed ones. Average over more and the measurement is
        quieter than its floor describes; average over fewer and it is noisier. Neither is a
        little bit wrong -- the comparison is between two different instruments.

        This was wrong in both directions at once. The floors were calibrated at warmup 2 /
        iters 5; `bench.measure()` had 3 / 10 hardcoded with no flag to change it. Every scored
        run did 13 invocations per record where 7 would have done -- roughly twice the GPU time
        per submission -- to produce a number quieter than the floor it was judged against.
        """
        import bench as B
        self.assertEqual(B.instrument_settings(self.gen), (2, 5))

        real = C.load(HERE.parent / "cells" / "BG-1" / "generation.json")
        ref = json.loads((HERE.parent / "cells" / "BG-1" / "reference.json").read_text())
        cal = ref["calibrated_with"]
        self.assertEqual(B.instrument_settings(real), (cal["warmup"], cal["iters"]),
                         "the bench would measure BG-1 with settings its floors were not "
                         "calibrated with")

        # And a generation that cannot say how its floors were measured is refused outright,
        # rather than quietly falling back to a default that means nothing.
        blind = C.load(HERE.parent / "cells" / "BG-1" / "generation.json")
        blind.calibrated_warmup = None
        with self.assertRaises(Exception) as cm:
            B.instrument_settings(blind)
        self.assertIn("calibrated", str(cm.exception))

    def test_fewer_repeats_than_the_generation_declares_are_refused(self):
        """`repeats` was in the frozen definition and enforced nowhere -- decoration.

        A declared parameter nobody checks is worse than an undeclared one, because a reader
        takes it for a guarantee. The runner's own `--repeats` flag decided the real number and
        the generation's declaration had no effect on anything.

        Enforced as a MINIMUM: more repeats is a better-sampled run of the same experiment and
        there is no reason to refuse it. Fewer is a different experiment wearing this
        generation's name -- and specifically, two repeats cannot support the paired bootstrap,
        which resamples repeat indices with replacement and gets three distinct resamples out of
        two indices.
        """
        self.assertEqual(self.gen.repeats, 3)
        recs = fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 1.15})

        # Exactly what the generation declares: fine.
        CP.compute(self.gen, [r for r in recs if r["repeat"] < 3])
        # More than it declares: also fine, it is the same experiment sampled better.
        CP.compute(self.gen, recs)
        # Fewer: refused, and the message says the number and where it came from.
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, [r for r in recs if r["repeat"] < 2])
        self.assertIn("declares 3", str(cm.exception))

    def test_interleaving_is_worth_more_than_a_noise_floor(self):
        """Why the arms alternate instead of main running to completion first.

        The obvious arrangement -- measure `main`, then measure the pull request, then compare --
        is wrong on rented hardware, and not by a little. The box drifts WITHIN a single bench
        run: on the committed example the base arm moved +0.480% across three repeats in
        `dit-step` and -0.368% in `vae-decode`, where that cell's entire measured noise floor is
        0.259%. Run one arm to completion and then the other, and all of that drift lands
        between the arms and is attributed to whichever went second.

        Scored both ways, the same measurements give:

            vae-decode   interleaved  -0.000003   correctly unresolved
                         blocked      +0.000021   payable
                         floor         0.000021

        A null result becomes a payment, at exactly the threshold, because the box cooled by a
        third of a percent between the two halves of the run. This test pins the magnitude so
        the ordering cannot be "simplified" later by somebody who reads the docstring and
        assumes the effect is negligible.
        """
        import math
        raw = json.loads((ROOT_EX / "BG-1-pr-000001-raw.json").read_text())
        # The anchor that run was scored against: its floors are what "a noise floor" meant then.
        cal = json.loads((ROOT_EX / "BG-1-anchor-v0.json").read_text())["cells"]

        def geo(xs):
            return math.exp(sum(map(math.log, xs)) / len(xs))

        worst = 0.0
        for cell_id, c in cal.items():
            b = [r["metrics"]["latency_s"] for r in raw["records"]
                 if r["cell"] == cell_id and r["variant"] == "base"]
            cd = [r["metrics"]["latency_s"] for r in raw["records"]
                  if r["cell"] == cell_id and r["variant"] == "candidate"]
            if len(b) < 2:
                continue
            ceil = c["ceiling_seconds"]

            def gap(bt, ct):
                ab, ac = ceil / bt, ceil / ct
                return (ac - ab) / (1 - ab)

            paired = gap(geo(b), geo(cd))
            blocked = gap(b[0], cd[-1])          # main in the cool phase, PR in the warm one
            floor_gap = CP.floor_as_gap_closed(c["floor_pct"], c["achieved"]) \
                if hasattr(CP, "floor_as_gap_closed") else None
            if floor_gap is None:
                from burnscore.floor import floor_as_gap_closed as fg
                floor_gap = fg(c["floor_pct"], c["achieved"])
            worst = max(worst, abs(blocked - paired) / floor_gap)

        self.assertGreater(
            worst, 0.5,
            f"blocked ordering now differs from interleaved by only {worst:.2f} noise floors. "
            f"If the box has genuinely stopped drifting within a run that is worth knowing, but "
            f"check the measurements before relaxing the ordering -- this was 1.1 floors, which "
            f"is enough to turn a null result into a payment.")

    def test_a_real_speedup_scores_and_resolves(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        cell = out["per_cell"]["dit-step/1024/bf16"]
        self.assertGreater(cell["gap_closed"], 0.0)
        self.assertTrue(cell["resolved"])
        self.assertAlmostEqual(cell["speedup"], 1.15, places=3)

    def test_a_gain_inside_the_floor_does_not_resolve(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.001}, jitter=0.0)
        self.assertFalse(out["per_cell"]["dit-step/1024/bf16"]["resolved"])
        self.assertFalse(out["interval"]["resolved"])

    def test_unpaired_repeats_are_refused(self):
        _, recs = self._score()
        recs = [r for r in recs if not (r["cell"] == "dit-step/1024/bf16"
                                        and r["variant"] == "candidate" and r["repeat"] == 4)]
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs)
        self.assertIn("unpaired", str(cm.exception))

    def test_a_partial_matrix_is_refused_by_default(self):
        _, recs = self._score()
        recs = [r for r in recs if r["cell"] != "vae-decode/1024/bf16"]
        with self.assertRaises(CP.ComputeError):
            CP.compute(self.gen, recs)
        out = CP.compute(self.gen, recs, allow_partial=True)
        self.assertFalse(out["coverage"]["complete"])

    def test_a_failed_run_is_not_averaged_into_a_ratio(self):
        _, recs = self._score()
        for r in recs:
            if r["cell"] == "dit-step/1024/bf16" and r["repeat"] == 2:
                r["status"] = "OOM"
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs)
        self.assertIn("ran less", str(cm.exception))

    def test_two_records_for_one_repeat_are_refused(self):
        _, recs = self._score()
        recs.append(dict(recs[0]))
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs)
        self.assertIn("pairing", str(cm.exception))

    def test_a_base_arm_far_from_its_anchor_is_caught(self):
        """The achieved fraction comes from the anchor; a base it no longer describes is refused."""
        _, recs = self._score()
        for r in recs:
            if r["cell"] == "dit-step/1024/bf16":
                r["metrics"]["latency_s"] *= 1.35
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs)
        self.assertIn("BASE arm took", str(cm.exception))

    def test_a_declared_but_unimplemented_cell_cannot_be_scored(self):
        _, recs = self._score()
        recs.append({"cell": "dit-step/1024/nvfp4", "variant": "base", "repeat": 0,
                     "status": "OK", "metrics": {"latency_s": 0.01}})
        recs.append({"cell": "dit-step/1024/nvfp4", "variant": "candidate", "repeat": 0,
                     "status": "OK", "metrics": {"latency_s": 0.005}})
        with self.assertRaises(CP.ComputeError) as cm:
            CP.compute(self.gen, recs, allow_partial=True)
        self.assertIn("NOT IMPLEMENTED", str(cm.exception))

    def test_held_out_regression_is_reported_as_overfit(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.20})
        held = fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 0.97})
        verdict = CP._held_out_verdict(self.gen, held, out["per_cell"])
        self.assertFalse(verdict["survived"])
        self.assertEqual(verdict["cells_failed"], ["dit-step/1024/bf16"])

    def test_a_gain_that_shrinks_into_the_noise_off_shape_is_overfit(self):
        """Still nominally faster is not enough: it has to clear the bar the published gain did."""
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.20})
        floor = out["per_cell"]["dit-step/1024/bf16"]["floor_pct"]
        held = fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 1 + floor / 400})
        self.assertFalse(CP._held_out_verdict(self.gen, held, out["per_cell"])["survived"])

    def test_an_untouched_cell_does_not_decide_the_held_out_verdict(self):
        """A neighbour reading 0.99x by noise must not make a real kernel look overfit."""
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.20})
        held = fixtures.records(self.gen, speedups={"dit-step/1024/bf16": 1.20,
                                                    "t5-encode/1024/bf16": 0.99})
        self.assertTrue(CP._held_out_verdict(self.gen, held, out["per_cell"])["survived"])

    def test_an_untouched_noisy_cell_does_not_block_a_resolved_gain(self):
        """Unresolved cells contribute zero AND do not decide whether the submission resolved."""
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        self.assertTrue(out["per_cell"]["dit-step/1024/bf16"]["resolved"])
        widened = CP.compute(self.gen, fixtures.records(
            self.gen, speedups={"dit-step/1024/bf16": 1.15}))
        self.assertTrue(widened["interval"]["resolved"])
        for c, v in widened["per_cell"].items():
            if c != "dit-step/1024/bf16":
                self.assertFalse(v["resolved"], c)

    def _receipt(self, out, **over):
        kw = dict(generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
                  interval=out["interval"], frontier=out["frontier"], correctness="PASS",
                  determinism=True, coverage=out["coverage"], held_out=True,
                  provenance=fixtures.provenance())
        kw.update(over)
        return R.build_receipt(**kw)

    def test_a_clean_win_credits_and_verifies(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        rec = self._receipt(out)
        self.assertEqual(rec["status"], "FRONTIER_EXPANDED")
        self.assertGreater(rec["score"]["credited_gap_closed"], 0)
        R.verify_receipt(rec, self.gen)

    def test_no_letter_grade_appears_anywhere_in_a_receipt(self):
        """No XS/S/M/L/XL anywhere in this system, now or later."""
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        blob = json.dumps(self._receipt(out))
        for band in ('"XS"', '"XL"', '": "S"', '": "M"', '": "L"',
                     "impact_band", '"tier"', "score_band", "grade"):
            self.assertNotIn(band, blob, f"{band} appeared in a receipt")

    def test_a_run_that_beats_the_arithmetic_ceiling_is_refused(self):
        """1.9x from 55% of roofline is not a result; it is a bug in one of three places."""
        with self.assertRaises(Exception) as cm:
            self._score(speedups={"dit-step/1024/bf16": 1.9})
        self.assertIn("FASTER than the arithmetic ceiling", str(cm.exception))

    def test_correctness_failure_credits_nothing_however_fast(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.5})
        rec = self._receipt(out, correctness="FAIL")
        self.assertEqual(rec["status"], "CORRECTNESS_FAIL")
        self.assertEqual(rec["score"]["credited_gap_closed"], 0.0)
        self.assertGreater(rec["score"]["gap_closed"], 0.0, "the measured figure is still kept")
        R.verify_receipt(rec, self.gen)

    def test_determinism_failure_outranks_a_correctness_pass(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.5})
        self.assertEqual(self._receipt(out, determinism=False)["status"], "DETERMINISM_FAIL")

    def test_a_partial_matrix_credits_nothing(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.5})
        out["coverage"]["complete"] = False
        rec = self._receipt(out)
        self.assertEqual(rec["status"], "PARTIAL")
        self.assertEqual(rec["score"]["credited_gap_closed"], 0.0)
        self.assertIsNotNone(rec["credit_withheld"])

    def test_a_modelled_score_basis_is_refused(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        rec = self._receipt(out)
        rec["score"]["basis"] = "model"
        rec["content_digest"] = R.content_digest(rec)
        with self.assertRaises(R.ReceiptError) as cm:
            R.verify_receipt(rec, self.gen)
        self.assertIn("Only a measured run is evidence", str(cm.exception))

    def test_a_tampered_receipt_fails_its_own_digest(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        rec = self._receipt(out)
        rec["score"]["gap_closed"] = 0.99
        with self.assertRaises(R.ReceiptError):
            R.verify_receipt(rec, self.gen)

    def test_a_receipt_is_refused_when_its_generation_moved(self):
        out, _ = self._score(speedups={"dit-step/1024/bf16": 1.15})
        rec = self._receipt(out)
        moved = copy.deepcopy(self.gen)
        moved.raw = dict(moved.raw, confidence_level=0.95)
        with self.assertRaises(R.ReceiptError) as cm:
            R.verify_receipt(rec, moved)
        self.assertIn("frozen", str(cm.exception))


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gen = C.load(fixtures.calibrated_generation(self.tmp.name))
        self.root = Path(self.tmp.name) / "ledger"

    def tearDown(self):
        self.tmp.cleanup()

    def _receipt(self, speedup, pr):
        out = CP.compute(self.gen, fixtures.records(
            self.gen, speedups={"dit-step/1024/bf16": speedup}))
        return R.build_receipt(
            generation=self.gen, per_cell=out["per_cell"], aggregate=out["aggregate"],
            interval=out["interval"], frontier=out["frontier"], correctness="PASS",
            determinism=True, coverage=out["coverage"], held_out=True, pr=pr,
            provenance=fixtures.provenance(candidate_commit=f"{pr:040d}"))

    def test_rewriting_a_finalized_receipt_is_refused(self):
        L.append_receipt(self.root, self._receipt(1.15, 1))
        again = self._receipt(1.30, 1)
        with self.assertRaises(L.LedgerError) as cm:
            L.append_receipt(self.root, again)
        self.assertIn("supersedes", str(cm.exception))

    def test_writing_the_same_receipt_twice_is_idempotent(self):
        rec = self._receipt(1.15, 1)
        p1 = L.append_receipt(self.root, rec)
        self.assertEqual(p1, L.append_receipt(self.root, rec))

    def test_gap_closed_compounds_toward_the_ceiling_rather_than_summing(self):
        for i, s in enumerate((1.15, 1.15, 1.15), start=1):
            L.append_receipt(self.root, self._receipt(s, i))
        cur = L.show(self.root, "BG-1")
        summed = sum(h["credited_gap_closed"] for h in cur["history"])
        self.assertLess(cur["gap_closed_cumulative"], summed)
        self.assertLess(cur["gap_closed_cumulative"], 1.0)

    def test_audit_reports_a_corrupted_receipt(self):
        p = L.append_receipt(self.root, self._receipt(1.15, 1))
        doc = json.loads(p.read_text())
        doc["score"]["gap_closed"] = 0.99
        p.write_text(json.dumps(doc))
        report = L.audit(self.root, "BG-1", self.gen)
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
