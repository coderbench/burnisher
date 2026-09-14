#!/usr/bin/env python3
"""Submitted code runs as an account that cannot reach the evaluator. Tested without root.

The account switch itself needs root, so it is not exercised here. What is: which environment
reaches the account, that the runtime and the build go through it, that leftovers are found by uid,
that output the evaluator reads back cannot be a link, and that the access check reports what the
account can reach -- run here as the current user, the same code the account runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import gate as G
import pr_bot as B
import runner as RN
import sandbox as SB

AS_ROOT = os.geteuid() == 0


class TestTheEnvironmentIsAnAllowlist(unittest.TestCase):
    JAIL = SB.Sandbox("burnish-sandbox", 1234, 1234, "/home/burnish-sandbox")

    def test_the_token_and_the_evaluators_paths_do_not_reach_the_account(self):
        env = self.JAIL.environment({
            "GH_TOKEN": "ghp_x", "GITHUB_TOKEN": "x", "SSH_AUTH_SOCK": "/run/ssh",
            "BURNISH_LEDGER": "/ledger", "BURNISH_SANDBOX_USER": "burnish-sandbox",
            "PATH": "/root/.local/bin:/usr/bin", "HOME": "/root", "PYTHONPATH": "/root/lib"})
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK", "BURNISH_LEDGER",
                     "BURNISH_SANDBOX_USER", "PYTHONPATH"):
            self.assertNotIn(name, env)
        self.assertEqual(env["HOME"], "/home/burnish-sandbox")
        self.assertNotIn("/root", env["PATH"])

    def test_what_the_runtime_and_the_build_read_is_kept(self):
        base = {"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "BURNISH_RT_TILE": "128", "CMAKE_CUDA_ARCHITECTURES": "121", "LANG": "C.UTF-8"}
        self.assertEqual({k: self.JAIL.environment(base)[k] for k in base}, base)

    def test_the_switch_drops_the_evaluators_groups_and_session(self):
        opts = self.JAIL.spawn_options({})
        self.assertEqual((opts["user"], opts["group"], opts["extra_groups"]), (1234, 1234, []))
        self.assertTrue(opts["start_new_session"])


class TestWhichAccountsAreRefused(unittest.TestCase):
    def test_root_isolates_nothing(self):
        with self.assertRaises(SB.SandboxError):
            SB.Sandbox.named("root")

    def test_an_account_that_does_not_exist_says_how_to_create_it(self):
        with self.assertRaises(SB.SandboxError) as cm:
            SB.Sandbox.named("burnish-no-such-account")
        self.assertIn("setup_sandbox.sh", str(cm.exception))

    @unittest.skipIf(AS_ROOT, "the refusal is for an evaluator that is not root")
    def test_switching_accounts_without_root_is_refused_not_skipped(self):
        with self.assertRaises(SB.SandboxError) as cm:
            SB.Sandbox("x", 65534, 65534, "/").run(["true"])
        self.assertIn("root", str(cm.exception))


class TestLeftoversAreKilledByUid(unittest.TestCase):
    def test_every_live_process_of_the_account_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as proc:
            for pid, uid, state in ((100, 1234, "S"), (200, 0, "S"), (300, 1234, "R"),
                                    (400, 1234, "Z")):
                Path(proc, str(pid)).mkdir()
                Path(proc, str(pid), "status").write_text(
                    f"Name:\tx\nState:\t{state} (x)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
            Path(proc, "self").mkdir()

            def kill(pid, _sig):
                Path(proc, str(pid), "status").unlink()

            killed = SB.Sandbox("x", 1234, 1234, "/").kill_leftovers(proc, kill)
        self.assertEqual(sorted(killed), [100, 300], "root's process is not the account's, and a "
                                                     "zombie can do nothing")


class TestWhatTheEvaluatorReadsBack(unittest.TestCase):
    def test_a_link_where_output_belongs_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            real, link = Path(tmp, "latent.npy"), Path(tmp, "link.npy")
            real.write_bytes(b"x")
            link.symlink_to("/etc/hostname")
            self.assertEqual(SB.require_plain_file(real), real)
            with self.assertRaises(SB.SandboxError):
                SB.require_plain_file(link)

    def test_the_gate_will_not_hash_a_latent_that_is_a_link(self):
        """A runtime that plants a link at --dump-latents would have the gate read its target."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp, "burnisher")
            fake.write_text(
                "#!/usr/bin/env python3\nimport os, sys\n"
                "out = sys.argv[sys.argv.index('--dump-latents') + 1]\n"
                "os.symlink('/etc/hostname', out)\n"
                "print('BURNISH_JSON: {\"output_stats\": {\"latent_mean\": 0.0, "
                "\"latent_std\": 1.0, \"latent_absmax\": 2.0}}')\n")
            fake.chmod(0o755)
            generation = SimpleNamespace(model={"steps": 1, "resolution": 64},
                                         raw={"model": {"guidance_scale": 4.5}})
            saved, RN.SETTLE_SECONDS = RN.SETTLE_SECONDS, 0.0
            env_user = os.environ.pop(SB.USER_ENV, None)
            try:
                with self.assertRaises(RN.RunnerError) as cm:
                    G.generate(fake, generation, Path(tmp, "ids.txt"), 1, "cuda", out_dir=tmp,
                               label="det-0", weights=tmp, device="cuda", dtype="fp32",
                               noise=Path(tmp, "noise.npy"))
            finally:
                RN.SETTLE_SECONDS = saved
                if env_user is not None:
                    os.environ[SB.USER_ENV] = env_user
            self.assertIn("not a regular file", str(cm.exception))

    def test_a_lock_in_tmp_can_be_created_first_by_anyone(self):
        self.assertTrue(SB.world_writable_parent("/tmp/burnisher-eval.lock"))
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o700)
            self.assertFalse(SB.world_writable_parent(Path(tmp, "eval.lock")))


class TestTheRuntimeGoesThroughTheSandbox(unittest.TestCase):
    def test_run_once_launches_as_the_named_account(self):
        calls = []

        class Jail:
            def run(self, cmd, **kw):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

        saved = SB.from_environment, RN.SETTLE_SECONDS
        SB.from_environment, RN.SETTLE_SECONDS = (lambda environ=None: Jail()), 0.0
        try:
            code, out, _ = RN.run_once(["/nonexistent/burnisher", "bench"])
        finally:
            SB.from_environment, RN.SETTLE_SECONDS = saved
        self.assertEqual((code, out, calls), (0, "ok\n", [["/nonexistent/burnisher", "bench"]]))

    def test_a_misnamed_account_is_a_runner_error_not_an_unsandboxed_run(self):
        saved = os.environ.get(SB.USER_ENV)
        os.environ[SB.USER_ENV] = "burnish-no-such-account"
        try:
            with self.assertRaises(RN.RunnerError):
                RN.run_once(["true"])
        finally:
            if saved is None:
                os.environ.pop(SB.USER_ENV, None)
            else:
                os.environ[SB.USER_ENV] = saved


class TestTheBotRefusesToRunUnsandboxed(unittest.TestCase):
    def test_no_account_and_no_opt_out_is_a_refusal(self):
        with self.assertRaises(RuntimeError) as cm:
            B.sandbox(SimpleNamespace(sandbox_user="", no_sandbox=False))
        self.assertIn("BURNISH_SANDBOX_USER", str(cm.exception))
        self.assertIsNone(B.sandbox(SimpleNamespace(sandbox_user="", no_sandbox=True)))

    def test_the_scripts_it_starts_launch_the_runtime_as_the_account_or_not_at_all(self):
        saved = os.environ.get(SB.USER_ENV)
        os.environ[SB.USER_ENV] = "stale"
        try:
            self.assertNotIn(SB.USER_ENV, B.child_env(None, BURNISHER_BIN="/b"))
            jail = SB.Sandbox("burnish-sandbox", 1234, 1234, "/home/burnish-sandbox")
            self.assertEqual(B.child_env(jail)[SB.USER_ENV], "burnish-sandbox")
        finally:
            if saved is None:
                os.environ.pop(SB.USER_ENV, None)
            else:
                os.environ[SB.USER_ENV] = saved

    def test_the_build_and_the_scoring_go_through_the_sandbox(self):
        src = (ROOT / "eval" / "pr_bot.py").read_text()
        body = src[src.index("def _evaluate("):src.index("def _evaluate_cartography(")]
        self.assertIn("build_submission(", body)
        self.assertNotIn("subprocess.run([\"./scripts/build_cuda.sh\"]", body)
        start = body.index('/ "score_submission.sh"')
        self.assertIn("env=child_env(jail, BURNISHER_BIN=str(binary))",
                      body[start:body.index("print(", start)])

    def test_the_bot_and_the_round_check_the_box_before_evaluating(self):
        for name in ("pr_bot.py", "round.py"):
            self.assertIn("box_is_safe(a, [GPU_LOCK_PATH", (ROOT / "eval" / name).read_text(), name)


class TestCredentialsInARemote(unittest.TestCase):
    def test_a_token_in_a_url_is_found_and_never_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True,
                                            capture_output=True)
            git("init", "-q")
            git("remote", "add", "origin", "https://someone:ghp_secret@github.com/o/r.git")
            git("remote", "add", "clean", "https://github.com/o/r.git")
            git("remote", "add", "ssh", "git@github.com:o/r.git")
            found = SB.credentials_in_remotes(tmp)
        self.assertEqual(found, ["origin"])
        self.assertNotIn("ghp_secret", json.dumps(found))


@unittest.skipIf(AS_ROOT, "root can read and write everything, so the check has nothing to find")
class TestTheAccessCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "open-secret").write_text("token")
        (t / "closed-secret").write_text("token")
        (t / "closed-secret").chmod(0o000)
        (t / "state").mkdir()
        (t / "locked").mkdir()
        (t / "locked" / "receipt.json").write_text("{}")
        (t / "locked" / "receipt.json").chmod(0o444)
        (t / "locked").chmod(0o555)
        (t / "weights").write_text("w")
        (t / "weights").chmod(0o000)
        self.t = t

    def tearDown(self):
        for p in ("closed-secret", "weights", "locked"):
            (self.t / p).chmod(0o755)
        self.tmp.cleanup()

    def spec(self):
        t = self.t
        return {"secrets": [str(t / "open-secret"), str(t / "closed-secret")],
                "protected": [str(t / "state"), str(t / "locked")],
                "readable": [str(t / "weights"), str(t / "missing")]}

    def test_it_reports_exactly_what_is_reachable_and_what_is_not(self):
        t = self.t
        self.assertEqual(SB._probe(self.spec()), [
            f"can read {t / 'open-secret'}", f"can write {t / 'state'}",
            f"cannot read {t / 'weights'}", f"cannot reach {t / 'missing'}"])

    def test_a_listening_service_other_than_ssh_is_reported_once(self):
        """A root notebook server on 8888, seen over IPv4 and IPv6. Docker's resolver and an
        established connection are not services the account can drive."""
        t = self.t
        header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt uid\n"
        (t / "tcp").write_text(header +
            "   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0\n"
            "   1: 00000000:22B8 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0\n"
            "   2: 0B00007F:B0B1 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0\n"
            "   3: 0100007F:1F90 0100007F:D431 01 00000000:00000000 00:00000000 00000000 0\n")
        (t / "tcp6").write_text(header +
            "   0: 00000000000000000000000000000000:22B8 00000000000000000000000000000000:0000 "
            "0A 00000000:00000000 00:00000000 00000000 0\n")
        spec = {"secrets": [], "protected": [], "readable": [],
                "net": [str(t / "tcp"), str(t / "tcp6")], "allowed_ports": [22]}
        self.assertEqual(SB._probe(spec), ["can connect to the service listening on port 8888"])

    def test_the_source_shipped_to_the_account_gives_the_same_answer(self):
        r = subprocess.run([sys.executable, "-c", SB.PROBE, json.dumps(self.spec())],
                           capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(r.stdout), SB._probe(self.spec()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
