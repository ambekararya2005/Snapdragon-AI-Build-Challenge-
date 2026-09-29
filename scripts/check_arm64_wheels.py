"""Dev-time: which pinned requirements have a wheel for native ARM64 Windows + CPython 3.11?

Reads a requirements file (default requirements-snapdragon.txt), asks the PyPI JSON API for each pinned
`name==version`, and classifies it:
    wheel    a win_arm64 wheel for cp311 (or abi3) exists, or a pure-Python py3-none-any wheel
    sdist    only a source distribution: pip would compile it on the device (needs MSVC ARM64 build tools)
    missing  neither for this version; the newest version with a win_arm64 cp311 wheel is shown if any
Network: PyPI only (pypi.org JSON API). Dev-time tool; nothing here runs in Kavach.

    python scripts\\check_arm64_wheels.py [requirements-snapdragon.txt] [--markdown]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "cp311"


def pypi(name: str, version: str | None = None) -> dict:
    url = f"https://pypi.org/pypi/{name}/{version}/json" if version else f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def wheel_kind(filenames: list[str]) -> str | None:
    """'win_arm64' / 'pure' if an installable wheel for cp311 on win_arm64 is among filenames, else None."""
    for f in filenames:
        if not f.endswith(".whl"):
            continue
        tags = f[:-4].split("-")[-3:]                   # python tag, abi tag, platform tag
        pys, abi, plat = tags[0].split("."), tags[1], tags[2].split(".")
        if "win_arm64" in plat and (PY in pys or (abi == "abi3" and any(p.startswith("cp3") for p in pys))):
            return "win_arm64"
        if "any" in plat and any(p in ("py3", "py2.py3", PY) for p in pys):
            return "pure"
    return None


def newest_arm64(name: str) -> str | None:
    releases = pypi(name)["releases"]

    def key(v: str):
        return [int(x) if x.isdigit() else -1 for x in re.split(r"[.\-]", v)]

    for v in sorted(releases, key=key, reverse=True):
        if re.search(r"[a-z]", v.replace("post", "")):   # skip pre-releases (rc, b, a, dev)
            continue
        if wheel_kind([f["filename"] for f in releases[v]]) == "win_arm64":
            return v
    return None


def check(req: Path) -> list[dict]:
    rows = []
    for line in req.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", spec)
        if not m:
            continue
        name, version = m.groups()
        files = [f["filename"] for f in pypi(name, version)["urls"]]
        kind = wheel_kind(files)
        status = kind and ("wheel" if kind == "win_arm64" else "wheel (pure Python)")
        if not status:
            status = "sdist only" if any(f.endswith((".tar.gz", ".zip")) for f in files) else "missing"
        row = {"package": name, "version": version, "status": status}
        if not kind:
            row["newest_arm64"] = newest_arm64(name)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="win_arm64 / cp311 wheel availability of pinned requirements")
    p.add_argument("requirements", nargs="?", default=str(ROOT / "requirements-snapdragon.txt"))
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)
    rows = check(Path(args.requirements))
    if args.markdown:
        print("| package | pinned | win_arm64 cp311 | newest version with a win_arm64 wheel |")
        print("|---|---|---|---|")
    for r in rows:
        extra = r.get("newest_arm64", "") if "newest_arm64" in r else ""
        if args.markdown:
            print(f"| {r['package']} | {r['version']} | {r['status']} | {extra if 'newest_arm64' in r else '-'} |")
        else:
            print(f"{r['package']:<22} {r['version']:<12} {r['status']:<22} "
                  f"{('newest with arm64 wheel: ' + str(extra)) if 'newest_arm64' in r else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
