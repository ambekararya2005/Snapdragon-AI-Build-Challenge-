"""Verify Kavach's privacy claims on this PC: no network connections, no raw data on disk, derived-only log.

Runs `python main.py --plain --seconds N` (the real pipeline: screen capture + OCR, loopback audio + ASR,
process watch, fusion; headless) as a child process and, while it runs:
  1. network  - polls the child process tree's sockets (TCP + UDP, psutil) every 0.5 s; must stay 0
  2. disk     - after the run, lists files created/changed during it under the repo (minus .venv*, .git,
                caches, weights) and %TEMP%; the only expected one is logs/incidents.jsonl, and no image /
                audio / text dump anywhere. Files are not attributed per process (another program may write
                to %TEMP% meanwhile); Sysinternals Process Monitor gives a per-process proof.
  3. log      - every logs/incidents.jsonl line written during the run has only the allowed keys and
                reason ids matching the strict id pattern (fusion.incidents): ids and numbers, no text
Prints counts, paths and key names only.

    python scripts\\verify_privacy.py [--seconds 60] [--config config.yaml]
Play the demo while it runs (open demo/test_pages/mock_bank_transfer.html, then
python scripts\\play_scenario.py) to exercise OCR, ASR and the incident log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "weights"}
RAW_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".wav", ".mp3", ".flac", ".raw", ".pcm", ".txt", ".npy"}
LOG_KEYS = {"time", "ts", "kind", "incident", "score", "band", "reason_ids", "decision_ms", "time_to_alert_ms",
            "new_types", "override_until", "types", "button"}


def changed_files(roots: list[Path], since: float) -> list[Path]:
    out = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv")]
            for f in filenames:
                p = Path(dirpath) / f
                try:
                    st = p.stat()
                except OSError:
                    continue
                if max(st.st_mtime, st.st_ctime) >= since:
                    out.append(p)
    return out


def tree_sockets(proc: psutil.Process) -> list[tuple[int, str, str]]:
    socks = []
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return socks
    for p in procs:
        try:
            get = getattr(p, "net_connections", None) or p.connections
            for c in get(kind="inet"):
                raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "-"
                socks.append((p.pid, str(c.status), raddr))
        except psutil.Error:
            continue
    return socks


def check_log(path: Path, since: float) -> tuple[int, list[str]]:
    from fusion.incidents import REASON_ID, UI_KIND

    if not path.is_file():
        return 0, []
    problems, n = [], 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        rec = json.loads(line)
        if rec.get("ts", 0) < since:
            continue
        n += 1
        extra = set(rec) - LOG_KEYS
        if extra:
            problems.append(f"line {i}: unexpected keys {sorted(extra)}")
        bad = [r for r in rec.get("reason_ids", []) + rec.get("types", []) + rec.get("new_types", [])
               if not REASON_ID.match(r)]
        if bad:
            problems.append(f"line {i}: {len(bad)} reason id(s) not matching the id pattern")
        if not (rec.get("kind") in ("alert", "caution", "override", "overlay_shown") or UI_KIND.match(rec.get("kind", ""))):
            problems.append(f"line {i}: unknown kind")
    return n, problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify Kavach privacy claims (network, disk, incident log)")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--config", help="alternate config (passed to main.py)")
    args = p.parse_args(argv)
    from kavach_config import get_config

    log_path = ROOT / (get_config().fusion.get("incident_log") or "logs/incidents.jsonl")
    start = time.time() - 1.0
    cmd = [sys.executable, str(ROOT / "main.py"), "--plain", "--seconds", str(args.seconds)]
    if args.config:
        cmd += ["--config", args.config]
    print(f"running Kavach headless for {args.seconds:.0f} s and watching its sockets ...", flush=True)
    child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc = psutil.Process(child.pid)
    seen: set[tuple[int, str, str]] = set()
    polls = 0
    while child.poll() is None:
        seen.update(tree_sockets(proc))
        polls += 1
        time.sleep(0.5)
    rc = child.returncode

    files = changed_files([ROOT, Path(tempfile.gettempdir())], start)
    in_repo = sorted(str(f.relative_to(ROOT)) for f in files if f.is_relative_to(ROOT))
    raw = sorted(str(f) for f in files if f.suffix.lower() in RAW_SUFFIXES and f != log_path)
    n_log, problems = check_log(log_path, start)

    ok_net = not seen
    ok_disk = all(Path(f) == log_path.relative_to(ROOT) for f in in_repo) and not raw
    ok_log = not problems
    print(f"1. network: {len(seen)} socket(s) seen in {polls} polls of the Kavach process tree "
          f"-> {'OK (0)' if ok_net else 'FAIL'}")
    for pid, status, raddr in sorted(seen):
        print(f"     pid {pid} {status} {raddr}")
    print(f"2. disk: repo files changed during the run: {in_repo or 'none'}; image/audio/text files created "
          f"under repo or %TEMP%: {len(raw)} -> {'OK' if ok_disk else 'CHECK'}")
    for f in raw[:20]:
        print(f"     {f}")
    print(f"3. incident log: {n_log} line(s) written during the run, {len(problems)} problem(s) "
          f"-> {'OK' if ok_log else 'FAIL'}")
    for pr in problems[:20]:
        print(f"     {pr}")
    print(f"Kavach exit code {rc}")
    return 0 if (ok_net and ok_disk and ok_log and rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
