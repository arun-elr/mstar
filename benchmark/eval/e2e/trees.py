"""The two trees ``compare`` serves: refs resolved in this checkout, exported once per commit."""

from __future__ import annotations

import fcntl
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = "mstar-project/mstar"


class TreeError(RuntimeError):
    pass


@dataclass
class Ref:
    spec: str      # what was asked for
    sha: str
    label: str     # main, PR #123, a branch name, a short commit
    rust_tree: str | None = None


def git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise TreeError(f"git {' '.join(args)}: {r.stderr.strip()[:400]}")
    return r.stdout.strip()


def upstream_remote(repo: Path) -> str:
    """The remote that points at mstar-project/mstar (PR heads live there), else ``origin``."""
    for line in git(repo, "remote", "-v").splitlines():
        name, url = line.split()[:2]
        if UPSTREAM in url.removesuffix(".git"):
            return name
    return "origin"


def resolve(repo: Path, spec: str, remote: str | None = None) -> Ref:
    """``main`` or another branch (fetched from the upstream remote, then origin, else local), a PR number
    (``548``, ``#548``), or a commit."""
    s = spec.strip()
    remote = remote or upstream_remote(repo)
    m = re.fullmatch(r"#?(\d{1,6})", s)
    if m:
        git(repo, "fetch", "-q", remote, f"pull/{m.group(1)}/head")
        return _ref(repo, s, git(repo, "rev-parse", "FETCH_HEAD"), f"PR #{m.group(1)}")
    if not re.fullmatch(r"[0-9a-f]{7,40}", s):
        for r in dict.fromkeys((remote, "origin")):
            fetched = subprocess.run(["git", "-C", str(repo), "fetch", "-q", r, s], capture_output=True, check=False)
            if fetched.returncode == 0:
                return _ref(repo, s, git(repo, "rev-parse", "FETCH_HEAD"), s)
    sha = git(repo, "rev-parse", "--verify", "-q", f"{s}^{{commit}}", check=False)
    if not sha:
        raise TreeError(f"cannot resolve {spec!r}: not a PR number, a branch on {remote} or origin, or a commit")
    return _ref(repo, s, sha, sha[:9] if sha.startswith(s) else s)


def _ref(repo: Path, spec: str, sha: str, label: str) -> Ref:
    tree = git(repo, "rev-parse", f"{sha}:rust", check=False)
    return Ref(spec, sha, label, tree[:16] or None)


def export(repo: Path, sha: str, dest: Path) -> Path:
    """``git archive`` of the commit into ``dest``, once: built beside it and renamed into place under a lock, so
    concurrent runs exporting the same commit never see or disturb a half-written tree."""
    if (dest / ".complete").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / f".{dest.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / ".complete").exists():
            return dest
        work = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
        try:
            tar = work / "src.tar"
            with open(tar, "wb") as f:
                if subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha], stdout=f,
                                  check=False).returncode:
                    raise TreeError(f"git archive {sha} failed")
            tree = work / "tree"
            with tarfile.open(tar) as t:
                t.extractall(tree, filter="data")
            (tree / ".complete").touch()
            shutil.rmtree(dest, ignore_errors=True)   # a partial tree from an older, unlocked attempt
            tree.rename(dest)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return dest
