#!/usr/bin/env python3
"""Check every archive blob against the official M* Git commit before GPU use."""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import tarfile

COMMIT = "5902054c0f9fa057a80ae5a7e15e67f19d4bcde2"


def main() -> None:
    repo = pathlib.Path(sys.argv[1])
    archive = pathlib.Path(sys.argv[2])
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", COMMIT]
    )
    expected = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        header, path = item.split(b"\t", 1)
        mode, kind, oid = header.split(b" ")
        if kind != b"blob":
            raise SystemExit(f"non-blob tree entry: {path!r}")
        expected[path.decode()] = (mode, oid.decode())
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isdir():
                continue
            path = member.name.removeprefix("./")
            if path in seen or path not in expected:
                raise SystemExit(f"duplicate/unexpected archive file: {path}")
            seen.add(path)
            if member.isfile():
                data = tar.extractfile(member).read()
                mode = b"100755" if member.mode & 0o111 else b"100644"
            elif member.issym():
                data = member.linkname.encode()
                mode = b"120000"
            else:
                raise SystemExit(f"unsupported archive member: {path}")
            oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if (mode, oid) != expected[path]:
                raise SystemExit(f"archive differs from commit: {path}")
    if seen != set(expected):
        raise SystemExit(f"archive missing {len(set(expected) - seen)} commit files")
    print(f"archive verified against {COMMIT}: {len(seen)} blobs")


if __name__ == "__main__":
    main()
