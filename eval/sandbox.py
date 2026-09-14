"""Run submitted code as an account that cannot reach the evaluator.

The evaluator builds and runs every pull request's code. It did so as the account that holds the
GitHub token, writes the ledger, keeps the copycat record and the block list, and owns the checkout
the next round is scored with. A build script in a pull request could read the token, rewrite that
history, or leave a process behind for the next round. No exploit was needed:
`scripts/build_cuda.sh` is contributor surface, and it ran as written.

So submitted code -- the build, its tests, and every launch of the runtime -- runs as a separate
unprivileged account, named by BURNISH_SANDBOX_USER and created by `eval/setup_sandbox.sh`:

  - Its environment is an allowlist. A denylist of secret names is a list of the secrets somebody
    thought of.
  - It builds its own copy of the head commit, in its own home. The evaluator runs git in the
    worktree, and git run in a tree submitted code could write is git configured by submitted code.
  - Every process running as the account is killed when a step ends, by uid, so nothing outlives
    the step to tamper with what the evaluator reads next.
  - `preflight` checks, AS the account, that it cannot read the secrets or write the state. File
    modes are box configuration and configuration drifts; a check that runs every round does not.

Not a container. Rented boxes are containers already and cannot nest one, and the network is not
cut. That is acceptable only because the account can read nothing secret, so there is nothing to
send -- which is what `preflight` verifies.
"""
from __future__ import annotations

import grp
import inspect
import json
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

USER_ENV = "BURNISH_SANDBOX_USER"

# The toolkit and the system, and nothing from the evaluator's PATH: a directory on it that the
# account can write would be a way back in.
SAFE_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# What the build and the runtime legitimately read. Everything else is dropped, whatever it is called.
PASSED_PREFIXES = ("CUDA_", "NVIDIA_", "CUBLAS_", "CMAKE_", "BURNISHER_", "BURNISH_RT_", "LC_")
PASSED_NAMES = ("LANG", "TZ")

# Membership in any of these is as good as root on most boxes.
PRIVILEGED_GROUPS = frozenset({"root", "sudo", "wheel", "admin", "adm", "docker", "lxd", "disk",
                               "shadow"})

# Credentials written into a remote URL, as `https://user:token@host/...`. Never echoed back.
_CREDENTIAL_REMOTE = re.compile(r"^(\S+)\s+https?://[^/\s@]+@", re.M)


class SandboxError(RuntimeError):
    """The sandbox cannot be used as configured. Never a reason to run without it."""


@dataclass(frozen=True)
class Sandbox:
    user: str
    uid: int
    gid: int
    home: str

    @classmethod
    def named(cls, user: str) -> "Sandbox":
        try:
            pw = pwd.getpwnam(user)
        except KeyError:
            raise SandboxError(f"there is no account {user!r}; eval/setup_sandbox.sh creates it") \
                from None
        if pw.pw_uid == 0 or pw.pw_gid == 0:
            raise SandboxError(f"{user!r} is root or in root's group, which isolates nothing")
        groups = set()
        for gid in os.getgrouplist(user, pw.pw_gid):
            try:
                groups.add(grp.getgrgid(gid).gr_name)
            except KeyError:
                pass
        if groups & PRIVILEGED_GROUPS:
            raise SandboxError(f"{user!r} is in {', '.join(sorted(groups & PRIVILEGED_GROUPS))}, "
                               f"which is as good as root")
        return cls(user, pw.pw_uid, pw.pw_gid, pw.pw_dir)

    def environment(self, base=None) -> dict:
        """The allowlisted part of `base`, with the account's own PATH, HOME and TMPDIR."""
        base = os.environ if base is None else base
        env = {k: v for k, v in base.items()
               if k.startswith(PASSED_PREFIXES) or k in PASSED_NAMES}
        env.update(PATH=SAFE_PATH, HOME=self.home, USER=self.user, LOGNAME=self.user,
                   TMPDIR=self.home)
        return env

    def spawn_options(self, env=None) -> dict:
        # extra_groups=[] drops the evaluator's supplementary groups, which the switch would keep.
        return {"user": self.uid, "group": self.gid, "extra_groups": [],
                "env": self.environment(env), "start_new_session": True, "umask": 0o077,
                "cwd": self.home}

    def run(self, cmd, *, env=None, **kwargs) -> subprocess.CompletedProcess:
        """`subprocess.run` as the account, then kill whatever it left running."""
        if os.geteuid() != 0:
            raise SandboxError(f"running submitted code as {self.user!r} needs the evaluator to "
                               f"run as root; it is uid {os.geteuid()}")
        options = self.spawn_options(env)
        options.update(kwargs)
        try:
            return subprocess.run(cmd, **options)
        finally:
            self.kill_leftovers()

    def kill_leftovers(self, proc="/proc", kill=os.kill) -> list:
        """SIGKILL every live process running as the account. By uid, never by name."""
        killed = []
        for _ in range(50):
            found = [pid for pid, uids, state in _processes(proc)
                     if self.uid in uids and state != "Z"]
            if not found:
                return killed
            for pid in found:
                try:
                    kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if pid not in killed:
                    killed.append(pid)
            time.sleep(0.05)
        raise SandboxError(f"processes running as {self.user!r} keep appearing ({found}); not "
                           f"continuing while they could touch the next step")

    def writable_dir(self, prefix: str) -> Path:
        """A new directory in the account's home that the account owns."""
        path = tempfile.mkdtemp(prefix=prefix, dir=self.home)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fchown(fd, self.uid, self.gid)
        finally:
            os.close(fd)
        return Path(path)

    def readable_copy(self, files, prefix: str) -> Path:
        """Copies of `files` the account can read and not change, in one new directory.

        Files that do not exist are skipped: the runtime reports a missing input better than this.
        """
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.home))
        path.chmod(0o755)
        for f in map(Path, files):
            if f.is_file():
                shutil.copyfile(f, path / f.name)
                (path / f.name).chmod(0o644)
        return path


def _processes(proc):
    """(pid, uids, state) for every process visible under `proc`."""
    for name in os.listdir(proc):
        if not name.isdigit():
            continue
        try:
            text = Path(proc, name, "status").read_text()
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
        uids = {int(u) for u in fields.get("Uid", "").split()}
        yield int(name), uids, fields.get("State", "").strip()[:1]


def from_environment(environ=None):
    """The sandbox BURNISH_SANDBOX_USER names, or None when it is unset."""
    name = (os.environ if environ is None else environ).get(USER_ENV, "").strip()
    return Sandbox.named(name) if name else None


def require_plain_file(path) -> Path:
    """`path` is a regular file, not a link.

    For output the evaluator reads back from submitted code: a link left where a latent belongs
    would have the evaluator read whatever it points at.
    """
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise SandboxError(f"{path} is not a regular file; submitted code left something else "
                           f"where its output belongs")
    return Path(path)


def world_writable_parent(path) -> bool:
    """Is `path` in a directory anyone can write, like /tmp? The account could create it first."""
    try:
        return bool(os.stat(Path(path).parent).st_mode & stat.S_IWOTH)
    except FileNotFoundError:
        return False


def credentials_in_remotes(repo) -> list:
    """Names of git remotes whose URL carries credentials. The URLs themselves are not returned."""
    out = subprocess.run(["git", "-C", str(repo), "remote", "-v"], capture_output=True,
                         text=True).stdout
    return sorted(set(_CREDENTIAL_REMOTE.findall(out)))


def _probe(spec):
    """Run AS the account: what it can reach that it must not, and what it needs and cannot reach.

    Self-contained, because it is shipped to the account as source: the account may not be able to
    read the directory this module lives in.
    """
    import os

    def paths(root, errors):
        if os.path.isdir(root):
            for d, _dirs, files in os.walk(root, onerror=errors.append):
                yield d
                for f in files:
                    yield os.path.join(d, f)
        elif os.path.lexists(root):
            yield root

    found = []
    for root in spec["secrets"]:
        hit = next((p for p in paths(root, []) if os.path.isfile(p) and not os.path.islink(p)
                    and os.access(p, os.R_OK)), None)
        if hit:
            found.append(f"can read {hit}")
    for root in spec["protected"]:
        hit = next((p for p in paths(root, []) if not os.path.islink(p)
                    and os.access(p, os.W_OK)), None)
        if hit:
            found.append(f"can write {hit}")
    for root in spec["readable"]:
        if not os.path.exists(root):
            found.append(f"cannot reach {root}")
            continue
        errors = []
        hit = next((p for p in paths(root, errors) if not os.access(
            p, os.R_OK | (os.X_OK if os.path.isdir(p) else 0))), None)
        if hit or errors:
            found.append(f"cannot read {hit or errors[0].filename}")
    # A local service runs whatever its clients send, as its owner. A notebook server started as
    # root is the usual one on a rented image, and its token is in its world-readable command line.
    ports = []
    for table in spec.get("net", []):
        try:
            rows = open(table).read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) < 4 or cols[3] != "0A":                  # 0A is LISTEN
                continue
            addr, port = cols[1].rsplit(":", 1)
            port = int(port, 16)
            # 0B00007F is 127.0.0.11, Docker's embedded name resolver, which only answers lookups.
            if port not in spec.get("allowed_ports", []) and addr != "0B00007F" \
                    and port not in ports:
                ports.append(port)
    found += [f"can connect to the service listening on port {p}" for p in ports]
    return found


PROBE = ("import json, sys\n" + textwrap.dedent(inspect.getsource(_probe))
         + "\nprint(json.dumps(_probe(json.loads(sys.argv[1]))))\n")


def preflight(sandbox: Sandbox, *, secrets=(), protected=(), readable=(),
              allowed_ports=(22,)) -> list:
    """Every way this box lets the account reach the evaluator, checked as the account. Empty is go."""
    problems = []
    try:
        if os.stat(sandbox.home).st_uid != sandbox.uid:
            problems.append(f"{sandbox.home} does not belong to {sandbox.user}; its builds go there")
    except FileNotFoundError:
        problems.append(f"{sandbox.user}'s home {sandbox.home} does not exist; its builds go there")
        return problems
    spec = {key: [str(p) for p in paths if p]
            for key, paths in (("secrets", secrets), ("protected", protected),
                               ("readable", readable))}
    spec.update(net=["/proc/net/tcp", "/proc/net/tcp6"], allowed_ports=list(allowed_ports))
    try:
        r = sandbox.run([sys.executable, "-c", PROBE, json.dumps(spec)], capture_output=True,
                        text=True, timeout=1800, cwd="/")
        if r.returncode != 0:
            problems.append(f"the access check could not run as {sandbox.user}: "
                            f"{r.stderr.strip()[-300:]}")
        else:
            problems += json.loads(r.stdout)
        smi = shutil.which("nvidia-smi")
        if smi:
            d = sandbox.run([smi, "-L"], capture_output=True, text=True, timeout=60, cwd="/")
            if d.returncode != 0:
                problems.append(f"{sandbox.user} cannot see the GPU (nvidia-smi -L exited "
                                f"{d.returncode}), so every submission would fail")
    except SandboxError as exc:
        problems.append(str(exc))
    return problems
