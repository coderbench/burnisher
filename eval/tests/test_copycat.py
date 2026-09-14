#!/usr/bin/env python3
"""The copycat detector, on synthetic submissions shaped like the failures it was built from."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from burnscore import copycat as CC

KERNEL = """template <typename T>
__global__ void k_fused(const T* x, const T* scale, const T* shift, T* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const float xs = (float)x[i];
        const float m = xs * (1.0f + (float)scale[i]) + (float)shift[i];
        out[i] = (T)(m > 0.0f ? m : 0.01f * m);
        if (i % 64 == 0) { out[i] = (T)((float)out[i] * 0.999f); }
    }
}
void fused_cuda(const FusedArgs& a) {
    const int64_t n = a.batch * a.tokens * a.channels;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.x, T, k_fused<T><<<grid, kBlock>>>((const T*)a.x->data(),
             (const T*)a.scale->data(), (const T*)a.shift->data(), (T*)a.out->data(), n));
    check_launch("fused");
}"""

RENAMED = """template<typename V>
__global__ void my_kernel(const V* inp, const V* s, const V* b, V* dst, int64_t count)
{
  for (int64_t j = 0; j < count; ++j)
  {
    const float v = (float)inp[j];   // pull the input
    const float y = v * (1.0f + (float)s[j]) + (float)b[j];
    dst[j] = (V)(y > 0.0f ? y : 0.01f * y);
    if (j % 64 == 0) { dst[j] = (V)((float)dst[j] * 0.999f); }
  }
}
void my_cuda(const FusedArgs& args) {
    const int64_t total = args.batch * args.tokens * args.channels;
    const int g = (int)std::min<int64_t>(65535, (total + kBlock - 1) / kBlock);
    DISPATCH(*args.x, V, my_kernel<V><<<g, kBlock>>>((const V*)args.x->data(),
             (const V*)args.scale->data(), (const V*)args.shift->data(), (V*)args.out->data(), total));
    check_launch("mine");
}"""

NEWCODE = """int pick_splits(int64_t seqlen, int64_t chunk) {
    int want = 16;
    while (want < 256 && seqlen > (int64_t)want * chunk * 3) {
        want <<= 1;
    }
    switch (want) {
        case 16: return 32;
        case 256: return 128;
        default: return want;
    }
}"""

OTHER = """void conv_cuda(const Conv2dArgs& a) {
    const int64_t oh = (a.h + 2 * a.pad - a.kh) / a.stride + 1;
    const int64_t ow = (a.w + 2 * a.pad - a.kw) / a.stride + 1;
    for (int64_t c = 0; c < a.cout; ++c)
        for (int64_t y = 0; y < oh; ++y)
            for (int64_t x = 0; x < ow; ++x) {
                double acc = a.bias ? a.bias[c] : 0.0;
                for (int64_t k = 0; k < a.cin * a.kh * a.kw; ++k) acc += a.w8[c * a.cin + k];
                a.out[(c * oh + y) * ow + x] = (float)acc;
            }
}"""


def diff(added: str, path="src/cuda/ops_cuda.cu", context=""):
    lines = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}", "@@ -1,1 +1,1 @@"]
    lines += [" " + l for l in context.splitlines()]
    lines += ["+" + l for l in added.splitlines()]
    return "\n".join(lines) + "\n"


def entry(pr, author, first_seen, text):
    fp = CC.fingerprint_diff(text)
    return {"pr": pr, "author": author, "first_seen": first_seen,
            "added": fp["added"], "base": fp["base"]}


class TestCopies(unittest.TestCase):
    def setUp(self):
        self.orig = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL))

    def test_a_verbatim_copy_is_a_copy_and_names_the_original(self):
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", diff(KERNEL)), [self.orig])
        self.assertEqual(v["outcome"], "COPY")
        self.assertEqual(v["original"]["pr"], 10)
        self.assertEqual(v["original"]["author"], "alice")

    def test_renaming_reformatting_and_comments_do_not_hide_a_copy(self):
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", diff(RENAMED)), [self.orig])
        self.assertEqual(v["outcome"], "COPY", v)

    def test_copying_a_merged_submission_is_caught_after_it_lands(self):
        """Re-registering a merged kernel under a new name: the base arm never runs it, so it
        would be paid again for a gain that already landed."""
        on_main = CC.fingerprint_sources({"src/cuda/ops_cuda.cu": KERNEL})
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", diff(RENAMED)), [self.orig],
                     on_main=on_main)
        self.assertEqual((v["outcome"], v["original"]["pr"]), ("COPY", 10))


class TestNotCopies(unittest.TestCase):
    def setUp(self):
        self.orig = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL))

    def test_independent_code_is_clear(self):
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", diff(OTHER)), [self.orig])
        self.assertEqual(v["outcome"], "CLEAR")

    def test_a_self_resubmission_is_iteration_not_copying(self):
        v = CC.judge(entry(11, "alice", "2026-09-02T00:00:00Z", diff(KERNEL)), [self.orig])
        self.assertEqual(v["outcome"], "CLEAR")

    def test_a_later_submission_is_never_the_original(self):
        """A low pull-request number is not priority; being observed first is."""
        early = entry(9, "bob", "2026-09-03T00:00:00Z", diff(KERNEL))
        v = CC.judge(self.orig, [early])
        self.assertEqual(v["outcome"], "CLEAR")

    def test_code_already_in_the_diffs_context_is_not_evidence(self):
        """A build fix whose lines were already there."""
        ref = entry(10, "alice", "2026-09-01T00:00:00Z", diff(OTHER, context=KERNEL))
        cand = entry(11, "bob", "2026-09-02T00:00:00Z", diff(KERNEL))
        v = CC.judge(cand, [ref])
        self.assertEqual(v["outcome"], "CLEAR")

    def test_code_that_is_already_on_main_is_not_evidence_against_another_submission(self):
        """A helper matched across pull requests because it was on main."""
        on_main = CC.fingerprint_sources({"src/cuda/helpers.cu": KERNEL})
        ref = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL + "\n" + OTHER))
        cand = entry(11, "bob", "2026-09-02T00:00:00Z", diff(KERNEL + "\n" + NEWCODE))
        self.assertEqual(CC.judge(cand, [ref], on_main=on_main)["outcome"], "CLEAR")

    def test_starting_from_the_baseline_kernel_is_not_copying(self):
        """CONTRIBUTING.md: copy the old kernel, register the new one beside it."""
        on_main = CC.fingerprint_sources({"src/cuda/ops_cuda.cu": KERNEL})
        variant = KERNEL.replace("0.01f * m", "0.02f * m").replace("k_fused", "k_fused_v2")
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", diff(variant)), [], on_main=on_main)
        self.assertEqual(v["outcome"], "CLEAR")

    def test_boilerplate_that_many_submissions_add_is_not_evidence(self):
        subs = [entry(10 + i, f"user{i}", f"2026-09-0{i + 1}T00:00:00Z", diff(KERNEL)) for i in range(5)]
        boiler = CC.boilerplate([s["added"] for s in subs])
        v = CC.judge(subs[-1], subs[:-1], boiler=boiler)
        self.assertEqual(v["outcome"], "CLEAR")

    def test_moving_code_is_not_duplicating_main(self):
        on_main = CC.fingerprint_sources({"src/cuda/ops_cuda.cu": KERNEL})
        moved = "\n".join([f"diff --git a/src/a.cu b/src/a.cu", "--- a/src/a.cu", "+++ b/src/a.cu",
                           "@@ -1,1 +1,1 @@"] + ["-" + l for l in KERNEL.splitlines()]
                          + [f"diff --git a/src/b.cu b/src/b.cu", "--- a/src/b.cu", "+++ b/src/b.cu",
                             "@@ -1,1 +1,1 @@"] + ["+" + l for l in KERNEL.splitlines()]) + "\n"
        fp = CC.fingerprint_diff(moved)
        cand = {"pr": 11, "author": "bob", "first_seen": "2026-09-02T00:00:00Z",
                "added": fp["added"], "base": fp["base"]}
        self.assertEqual(CC.judge(cand, [], on_main=on_main)["outcome"], "CLEAR")


class TestReview(unittest.TestCase):
    def test_a_tiny_identical_change_is_reviewed_never_called_a_copy(self):
        small = "if (layer < 0 || layer >= (int)weights_.size()) {\n    fprintf(stderr, \"bad layer %d\\n\", layer);\n    return;\n}"
        ref = entry(10, "alice", "2026-09-01T00:00:00Z", diff(small, path="src/core/engine.cpp"))
        cand = entry(11, "bob", "2026-09-02T00:00:00Z", diff(small, path="src/core/engine.cpp"))
        v = CC.judge(cand, [ref])
        self.assertEqual((v["outcome"], v["kind"]), ("REVIEW", "tiny-identical"))

    def test_earlier_work_embedded_in_a_larger_submission_is_reviewed(self):
        big = KERNEL + "\n" + OTHER + "\n" + OTHER.replace("conv", "deconv").replace("acc +=", "acc -=")
        ref = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL))
        cand = entry(11, "bob", "2026-09-02T00:00:00Z", diff(big))
        cand_small_ref = dict(ref)
        v = CC.judge(cand, [cand_small_ref])
        self.assertEqual((v["outcome"], v["kind"]), ("REVIEW", "contains-earlier"))
        self.assertEqual(v["original"]["pr"], 10)

    def test_the_original_is_whoever_was_observed_first(self):
        first = entry(12, "carol", "2026-08-30T00:00:00Z", diff(KERNEL))
        second = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL))
        v = CC.judge(entry(13, "bob", "2026-09-02T00:00:00Z", diff(KERNEL)), [second, first])
        self.assertEqual(v["original"]["author"], "carol")

    def test_evidence_quotes_the_copied_lines(self):
        orig = entry(10, "alice", "2026-09-01T00:00:00Z", diff(KERNEL))
        text = diff(RENAMED)
        v = CC.judge(entry(11, "bob", "2026-09-02T00:00:00Z", text), [orig])
        lines = [e["line"] for e in CC.evidence(text, v["shared"])]
        self.assertTrue(any("0.01f" in l for l in lines), lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
