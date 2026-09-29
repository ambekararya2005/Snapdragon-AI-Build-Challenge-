"""Scenario runner: the TRD's 20 end-to-end scenarios (10 scam, 10 normal) on the real pipeline.

For each scenario in bench/scenarios.yaml: start the pipeline headless (screen + OCR, audio + ASR,
processes, fusion; no overlay, no incident log), start the (fake) remote tool, open the scenario's demo
page in its own browser window in front, speak the call script through the speakers with SAPI (WASAPI
loopback hears it), run for duration_s, then tear everything down and bring the neutral idle page to
the front. OCR / ASR models are loaded once and shared by every run; each run gets a fresh pipeline,
risk engine and rolling transcript.

Per scenario: max score, final band, whether alert / caution fired, and the time to alert:
  decision  AlertEvent.decision_ms: trigger (observation time of the signal that crossed the alert
            band: frame capture / audio chunk end / process poll) -> engine decision.
  e2e       stimulus -> alert decision. The stimulus is the latest thing the runner did before the
            trigger (page on screen, tool started, spoken line finished). Conservative: when the
            crossing words were in a line still being spoken, the previous line's end is used.
Summary: scam caught % (outcome >= expected band), false alarms among normal scenarios (alerts,
cautions), time to alert p50/p95 over the alerted scam scenarios.

Outputs in bench/results/: scenarios.md, scenarios.csv and scenario_runs.jsonl. The .jsonl keeps each
run's derived signal stream (screen label scores, tactic ids, tool names, timestamps; never text, audio
or images), so --replay / --compare re-score every scenario under another fusion config without the
hardware. Replay covers the fusion section only (weights, decay, caps, gate, bands): detector changes
(lexicons, screen_classifier, intent) need a live run.

    python -m bench.scenarios --dry-run                   # validate spec + assets, print the plan
    python -m bench.scenarios                             # all 20, live (~25 min)
    python -m bench.scenarios --only fake_kyc,work_call   # re-run some; other rows are kept
    python -m bench.scenarios --only news_audio_bank --repeat 3   # row = worst of 3, all 3 listed
    python -m bench.scenarios --report                    # rewrite scenarios.md/.csv from recorded runs
    python -m bench.scenarios --replay                    # re-score recorded runs with the current config
    python -m bench.scenarios --compare bench/tuning/x.yaml  # before/after for all scenarios -> tuning.md
                                                          # (x.yaml: only the keys to change, merged over config.yaml)

A live run takes over the desktop for its duration: keep the PC idle, system audio on and not muted,
and nothing else playing. Output is ids and numbers only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import logging
import math
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.risk import ALERT, CAUTION, QUIET, RemoteToolSignal, RiskEngine  # noqa: E402
from kavach.pipeline import Pipeline  # noqa: E402

log = logging.getLogger("bench.scenarios")

SPEC_PATH = ROOT / "bench" / "scenarios.yaml"
RESULTS_DIR = ROOT / "bench" / "results"
PAGES_DIR = ROOT / "demo" / "test_pages"
SCRIPTS_DIR = ROOT / "demo" / "scripts"
NEUTRAL_PAGE = "neutral.html"
RUNS_FILE = "scenario_runs.jsonl"
TYPES = ("scam", "normal")
REMOTE_MODES = ("none", "fake", "fake_tray", "anydesk", "anydesk_idle")
ANYDESK_NOTE = "AnyDesk not installed: waitfor.exe stands in (fake remote tool, user process)"
RANK = {QUIET: 0, CAUTION: 1, ALERT: 2}
TARGETS = {"scam_caught_pct": 90.0, "normal_alerts": 0, "normal_cautions": 1, "tta_p95_s": 5.0}

# Fake remote tool: a hidden waitfor.exe (ships with Windows; waits for a local signal name, no window,
# no network) registered as a known tool for the bench run only. The signal name is its only argument,
# so the fake_tray variant carries the tool's idle arg and gets role "tray" like AnyDesk's own tray process.
FAKE_TOOL = {"name": "FakeRemote", "exe_names": ["waitfor.exe"], "idle_args": ["kavachbenchidle"]}
FAKE_LIVE_ARG, FAKE_IDLE_ARG = "kavachbenchlive", "kavachbenchidle"
ANYDESK_PATHS = (r"C:\Program Files (x86)\AnyDesk\AnyDesk.exe", r"C:\Program Files\AnyDesk\AnyDesk.exe")

CSV_FIELDS = ("id", "source", "type", "expected", "outcome", "ok", "max_score", "final_score", "final_band", "alert",
              "caution", "trigger_s", "alert_s", "decision_ms", "e2e_ms", "stimulus", "reasons", "page",
              "remote", "call", "duration_s", "notes")


# ---------------------------------------------------------------- spec

class SpecError(ValueError):
    """bench/scenarios.yaml is malformed or points at missing assets."""


@dataclass(frozen=True)
class Scenario:
    id: str
    type: str                          # scam | normal
    page: str | None                   # demo/test_pages/<page>
    remote: str                        # REMOTE_MODES
    call: str | None                   # demo/scripts/<call>
    duration_s: float
    expected: str                      # quiet | caution | alert
    desc: str = ""
    call_delay_s: float = 3.0

    @property
    def page_path(self) -> Path | None:
        return PAGES_DIR / self.page if self.page else None

    @property
    def call_path(self) -> Path | None:
        return SCRIPTS_DIR / self.call if self.call else None


def load_spec(path: str | Path = SPEC_PATH) -> list[Scenario]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: list[Scenario] = []
    for i, s in enumerate(raw.get("scenarios") or [], 1):
        try:
            sc = Scenario(id=str(s["id"]), type=s["type"], page=s.get("page"), remote=s.get("remote") or "none",
                          call=s.get("call"), duration_s=float(s["duration_s"]), expected=s["expected"],
                          desc=str(s.get("desc", "")), call_delay_s=float(s.get("call_delay_s", 3.0)))
        except (KeyError, TypeError, ValueError) as e:
            raise SpecError(f"scenario #{i}: {type(e).__name__}: {e}") from None
        problems = []
        if any(o.id == sc.id for o in out):
            problems.append("duplicate id")
        if sc.type not in TYPES:
            problems.append(f"type must be one of {TYPES}")
        if sc.remote not in REMOTE_MODES:
            problems.append(f"remote must be one of {REMOTE_MODES}")
        if sc.expected not in RANK:
            problems.append(f"expected must be one of {tuple(RANK)}")
        if sc.duration_s <= sc.call_delay_s:
            problems.append("duration_s must be longer than call_delay_s")
        if sc.page_path and not sc.page_path.is_file():
            problems.append(f"page not found: {sc.page}")
        if sc.call_path and not sc.call_path.is_file():
            problems.append(f"call script not found: {sc.call}")
        if problems:
            raise SpecError(f"{sc.id}: {'; '.join(problems)}")
        out.append(sc)
    if not out:
        raise SpecError(f"no scenarios in {path}")
    return out


def select(scenarios: Sequence[Scenario], only: str | None) -> list[Scenario]:
    if not only:
        return list(scenarios)
    ids = [s.strip() for s in only.split(",") if s.strip()]
    known = {s.id for s in scenarios}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise SpecError(f"unknown scenario id(s): {', '.join(unknown)}")
    return [s for s in scenarios if s.id in ids]


def script_lines(path: Path) -> list[str]:
    """Demo call script: one spoken line per non-empty line; '#' lines are comments."""
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def page_title(path: Path) -> str:
    m = re.search(r"<title>(.*?)</title>", path.read_text(encoding="utf-8"), re.S | re.I)
    return html.unescape(m.group(1)).strip() if m else path.stem


# ---------------------------------------------------------------- derived signal records (no text)

def encode_signal(stage: str, payload: Any, ts: float, t_post: float, t0: float) -> dict[str, Any] | None:
    """Pipeline signal -> JSON-safe dict with times relative to t0. Scores, tactic ids and tool names only."""
    def rel(x: float) -> float:
        return round(float(x) - t0, 3)

    d: dict[str, Any] = {"t": rel(t_post), "ts": rel(ts), "stage": stage}
    if isinstance(payload, RemoteToolSignal):
        d.update(kind="remote", tool=str(payload.tool), live=bool(payload.live))
    elif hasattr(payload, "labels") and hasattr(payload, "screen_tactics"):
        d.update(kind="screen", labels={k: round(float(v), 3) for k, v in (payload.labels or {}).items() if k != "normal"},
                 tactics=sorted(payload.screen_tactics or ()))
    elif hasattr(payload, "tactics"):
        d.update(kind="call", tactics={
            t: rel(h.last_ts) if isinstance(getattr(h, "last_ts", None), (int, float)) else None
            for t, h in (payload.tactics or {}).items()},
            evidence=sorted(getattr(payload, "evidence", None) or []))   # lexicon ids + guard:/fuzzy: decisions
    else:
        return None
    return d


def decode_signal(d: Mapping[str, Any]) -> Any:
    """encode_signal() dict -> an object RiskEngine.update() accepts."""
    kind = d.get("kind")
    if kind == "remote":
        return RemoteToolSignal(d["tool"], bool(d["live"]))
    if kind == "screen":
        return SimpleNamespace(labels=dict(d["labels"]), screen_tactics=frozenset(d["tactics"]))
    if kind == "call":
        return SimpleNamespace(tactics={t: SimpleNamespace(last_ts=v) for t, v in d["tactics"].items()})
    raise ValueError(f"unknown signal kind {kind!r}")


# ---------------------------------------------------------------- outcome of one run

@dataclass
class Outcome:
    band: str = QUIET                  # highest band reached: alert if an alert fired, else caution / quiet
    max_score: int = 0
    reasons: list[str] = field(default_factory=list)    # reason ids at the max score
    final_score: int = 0
    final_band: str = QUIET
    alert: bool = False
    caution: bool = False
    trigger_s: float | None = None     # first alert's trigger, s from scenario start
    alert_s: float | None = None       # first alert's decision, s from scenario start
    decision_ms: float | None = None   # trigger -> decision
    e2e_ms: float | None = None        # stimulus -> decision
    stimulus: str = ""                 # speech:<tactic> | page | tool | line <n>


class Tracker:
    """Collects RiskStates / fusion events of one run (live subscriber or replay) into an Outcome."""

    def __init__(self, t0: float = 0.0):
        self.t0 = t0
        self.o = Outcome()
        self._max_band = QUIET

    def state(self, st: Any) -> None:
        if st.score > self.o.max_score:
            self.o.max_score, self.o.reasons = int(st.score), list(st.reason_ids)
        self.o.final_score, self.o.final_band = int(st.score), st.band
        if RANK[st.band] > RANK[self._max_band]:
            self._max_band = st.band

    def event(self, ev: Any) -> None:
        kind = getattr(ev, "kind", "")
        if kind == "alert" and not self.o.alert:
            self.o.alert = True
            self.o.alert_s = round(ev.ts - self.t0, 3)
            self.o.trigger_s = round(ev.t_trigger - self.t0, 3)
            self.o.decision_ms = float(ev.decision_ms)
        elif kind == "caution":
            self.o.caution = True

    def finish(self, stimuli: Sequence[Mapping[str, Any]], signals: Sequence[Mapping[str, Any]] = (),
               onsets: Mapping[str, float] | None = None) -> Outcome:
        o = self.o
        o.band = ALERT if o.alert else CAUTION if (o.caution or RANK[self._max_band] >= RANK[CAUTION]) else QUIET
        if o.alert and o.trigger_s is not None:
            t, label = alert_stimulus(o.trigger_s, stimuli, signals, onsets or {})
            if t is not None:
                o.e2e_ms = round((o.alert_s - t) * 1000.0, 1)
                o.stimulus = label
        return o


def stimulus_before(stimuli: Iterable[Mapping[str, Any]], t: float) -> Mapping[str, Any] | None:
    """Latest stimulus at or before t (page shown, tool started, spoken line finished)."""
    cands = [s for s in stimuli if s["t"] <= t + 1e-6 and s["kind"] != "line_start"]
    return max(cands, key=lambda s: s["t"]) if cands else None


def alert_stimulus(trigger_s: float, stimuli: Sequence[Mapping[str, Any]], signals: Sequence[Mapping[str, Any]],
                   onsets: Mapping[str, float]) -> tuple[float | None, str]:
    """(time, label) the risky combination appeared, for the first alert.

    Speech trigger (the alert's trigger is a call result): the tactics new in that result, dated by
    `onsets` (when the script's phrase completing the tactic was spoken, see speech_onsets); the
    earliest counts. Otherwise (screen / remote trigger, or a tactic the script does not contain): the
    latest runner action before the trigger - page on screen, tool started, spoken line finished."""
    calls = [s for s in signals if s["kind"] == "call" and abs(float(s["ts"]) - trigger_s) < 2e-3]
    if calls:
        sig = calls[0]
        prev = [s for s in signals if s["kind"] == "call" and s["t"] < sig["t"]]
        before = set(prev[-1]["tactics"]) if prev else set()
        new = [t for t in sig["tactics"] if t not in before] or list(sig["tactics"])
        dated = sorted((min(float(onsets[t]), trigger_s), t) for t in new if t in onsets)
        if dated:
            return dated[0][0], f"speech:{dated[0][1]}"
    stim = stimulus_before(stimuli, trigger_s)
    if stim is None:
        return None, ""
    return stim["t"], f"line {stim['i']}" if stim["kind"] == "line" else stim["kind"]


def speech_onsets(lines: Sequence[str], starts: Sequence[float], ends: Sequence[float],
                  intent_cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Ground truth for speech: when each call tactic became detectable in the spoken script.

    Per spoken line, the shortest word prefix whose cumulative script text reaches the tactic (same
    detector and config as the pipeline, but on the exact script instead of ASR output), placed in
    time by its share of the line's characters between the line's start and end (TTS speaks at an
    even pace, so this is an estimate within a fraction of a second)."""
    from detect.intent import detect

    onsets: dict[str, float] = {}
    spoken: list[str] = []
    for i, line in enumerate(lines[:min(len(starts), len(ends))]):
        new = [t for t in detect("\n".join([*spoken, line]), intent_cfg).tactics if t not in onsets]
        words = line.split()
        for k in range(1, len(words) + 1):
            if not new:
                break
            partial = " ".join(words[:k])
            got = detect("\n".join([*spoken, partial]), intent_cfg).tactics
            for t in [t for t in new if t in got]:
                onsets[t] = round(starts[i] + len(partial) / len(line) * (ends[i] - starts[i]), 3)
                new.remove(t)
        for t in new:                                          # needed the whole line's context
            onsets[t] = round(ends[i], 3)
        spoken.append(line)
    return onsets


def replay(record: Mapping[str, Any], config: Mapping[str, Any]) -> Outcome:
    """Re-score a recorded run under `config` (its fusion section), the way the pipeline does:
    update + score per signal at its arrival time, and a score every fusion.tick_s."""
    engine = RiskEngine(config)
    f = config.get("fusion", config) or {}
    tick = float(f.get("tick_s") or 0.5)
    duration = float(record["duration_s"])
    signals = list(record.get("signals") or [])
    timeline = [(float(s["t"]), 0, i) for i, s in enumerate(signals)]
    timeline += [(k * tick, 1, -1) for k in range(int(duration / tick) + 1)]
    tr = Tracker(0.0)
    for t, is_tick, i in sorted(timeline):
        if not is_tick:
            sig = signals[i]
            engine.update(decode_signal(sig), t, float(sig["ts"]))
        st = engine.score(t)
        tr.state(st)
        for ev in st.events:
            tr.event(ev)
    return tr.finish(record.get("stimuli") or [], signals, record.get("onsets") or {})


# ---------------------------------------------------------------- rows, metrics, reports

def is_ok(type_: str, expected: str, outcome: str) -> bool:
    """Scam: reached at least the expected band. Normal: stayed at or below it."""
    if type_ == "scam":
        return RANK[outcome] >= RANK[expected]
    return RANK[outcome] <= RANK[expected]


def make_row(record: Mapping[str, Any], o: Outcome, source: str = "live") -> dict[str, Any]:
    """source: "live" (this outcome was measured under the current config) or "replay"."""
    notes = [n for n in record.get("notes") or [] if not n.startswith("AnyDesk not installed")]
    if str(record.get("remote", "")).startswith("anydesk") and str(record.get("remote_used", "")).startswith("fake"):
        notes.insert(0, ANYDESK_NOTE)
    repeats = record.get("repeats") or []
    if source == "live" and len(repeats) > 1:
        notes.append(f"{len(repeats)} live runs (row = worst): " + ", ".join(f"{r['band']} {r['max_score']}" for r in repeats))
    return {
        "id": record["id"], "source": source, "type": record["type"], "expected": record["expected"], "outcome": o.band,
        "ok": is_ok(record["type"], record["expected"], o.band), "max_score": o.max_score,
        "final_score": o.final_score, "final_band": o.final_band, "alert": o.alert, "caution": o.caution,
        "trigger_s": o.trigger_s, "alert_s": o.alert_s, "decision_ms": o.decision_ms, "e2e_ms": o.e2e_ms,
        "stimulus": o.stimulus, "reasons": " ".join(o.reasons[:6]), "page": record.get("page") or "",
        "remote": record.get("remote_used") or record.get("remote") or "none", "call": record.get("call") or "",
        "duration_s": record.get("duration_s"), "notes": "; ".join(notes),
    }


def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Short hash of everything that shapes a live result: the config (minus its path) and the lexicons.
    A recorded run counts as live for the report only while this matches; otherwise it is replayed."""
    h = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k != "config_path"}, sort_keys=True,
                                default=str).encode())
    for p in sorted((ROOT / "detect" / "lexicons").glob("*.yaml")):
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def worst_run(type_: str, runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Of repeated live runs: the highest band / score for a normal scenario, the lowest for a scam."""
    key = lambda r: (RANK[r["live"]["band"]], r["live"]["max_score"])  # noqa: E731
    return max(runs, key=key) if type_ == "normal" else min(runs, key=key)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scam = [r for r in rows if r["type"] == "scam"]
    normal = [r for r in rows if r["type"] == "normal"]
    caught = sum(1 for r in scam if r["ok"])
    e2e = [r["e2e_ms"] / 1000.0 for r in scam if r["alert"] and r["e2e_ms"] is not None]
    dec = [r["decision_ms"] / 1000.0 for r in scam if r["alert"] and r["decision_ms"] is not None]
    s = {
        "n_scam": len(scam), "n_normal": len(normal),
        "scam_caught": caught, "scam_caught_pct": round(100.0 * caught / len(scam), 1) if scam else None,
        "scam_alerted": sum(1 for r in scam if r["alert"]),
        "normal_alerts": sum(1 for r in normal if r["alert"]),
        # a caution only counts as a false alarm where the scenario expects quiet (not where it is by design)
        "normal_cautions": sum(1 for r in normal if r["outcome"] == CAUTION and r["expected"] == QUIET),
        "tta_n": len(e2e),
        "tta_p50_s": _r(percentile(e2e, 0.5)), "tta_p95_s": _r(percentile(e2e, 0.95)),
        "decision_p50_s": _r(percentile(dec, 0.5)), "decision_p95_s": _r(percentile(dec, 0.95)),
    }
    s["pass_caught"] = s["scam_caught_pct"] is not None and s["scam_caught_pct"] >= TARGETS["scam_caught_pct"]
    s["pass_false_alarms"] = (s["normal_alerts"] <= TARGETS["normal_alerts"]
                              and s["normal_cautions"] <= TARGETS["normal_cautions"])
    s["pass_tta"] = s["tta_p95_s"] is not None and s["tta_p95_s"] <= TARGETS["tta_p95_s"]
    s["pass_all"] = s["pass_caught"] and s["pass_false_alarms"] and s["pass_tta"]
    return s


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None or x == "":
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def summary_lines(s: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    """(metric, value, target, status) rows for the summary table."""
    ok = lambda b: "OK" if b else "MISS"  # noqa: E731
    tta = (f"p50 {_fmt(s['tta_p50_s'])} s, p95 {_fmt(s['tta_p95_s'])} s (n={s['tta_n']})"
           if s["tta_n"] else "no alerts")
    return [
        ("scam caught (outcome >= expected)", f"{s['scam_caught']}/{s['n_scam']} ({_fmt(s['scam_caught_pct'])}%)",
         f">= {TARGETS['scam_caught_pct']:.0f}%", ok(s["pass_caught"])),
        ("scam alerted", f"{s['scam_alerted']}/{s['n_scam']}", "-", ""),
        ("false alarms among normal", f"{s['normal_alerts']} alerts, {s['normal_cautions']} false cautions of {s['n_normal']}",
         f"{TARGETS['normal_alerts']} alerts, <= {TARGETS['normal_cautions']} caution", ok(s["pass_false_alarms"])),
        ("time to alert, stimulus -> alert", tta, f"p95 <= {TARGETS['tta_p95_s']:.0f} s", ok(s["pass_tta"])),
        ("time to alert, trigger -> decision",
         f"p50 {_fmt(s['decision_p50_s'], 2)} s, p95 {_fmt(s['decision_p95_s'], 2)} s" if s["tta_n"] else "-", "-", ""),
    ]


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return out


def write_markdown(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], meta: Mapping[str, Any],
                   path: Path) -> None:
    lines = ["# Kavach scenario results", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines += ["", "## Summary", ""]
    lines += _md_table(("metric", "value", "target", "status"), summary_lines(summary))
    lines += ["", f"Overall: **{'all targets met' if summary['pass_all'] else 'targets missed'}**", "",
              "## Per scenario", ""]
    lines += _md_table(
        ("id", "source", "type", "page", "remote", "call", "expected", "outcome", "ok", "max", "final", "alert @s",
         "stimulus -> alert s", "trigger -> decision ms", "reason ids at max", "notes"),
        ([r["id"], r["source"], r["type"], r["page"] or "-", r["remote"], r["call"] or "-", r["expected"], r["outcome"],
          "yes" if r["ok"] else "**NO**", r["max_score"], f"{r['final_score']} {r['final_band']}", _fmt(r["alert_s"]),
          _fmt(None if r["e2e_ms"] is None else r["e2e_ms"] / 1000.0) + (f" ({r['stimulus']})" if r["stimulus"] else ""),
          _fmt(r["decision_ms"], 0), r["reasons"] or "-", r["notes"] or ""] for r in rows))
    lines += ["", "## Definitions", "",
              "- source: live = measured live under the current config and lexicons; replay = the scenario's "
              "recorded signal stream (from an earlier live run) re-scored by the fusion engine under the current "
              "config. Replay assumes the detectors produce the same labels/tactics; the lexicon and guard changes "
              "since those runs were checked to leave these scenarios' pages and scripts unchanged.",
              "- outcome: alert if an alert fired, else caution if the caution band was reached, else quiet. "
              "ok = scam outcome >= expected, normal outcome <= expected. A caution counts as a false alarm only "
              "where the scenario expects quiet (anydesk_idle_bank expects caution by design: tactic gate).",
              "- stimulus -> alert: when the alert's trigger is speech, from the moment the tipping phrase was "
              "spoken (speech:<tactic>; found on the script text, placed within its line by character position) "
              "to the alert decision; otherwise from the latest runner action before the trigger (page on screen, "
              "remote tool started, spoken line finished).",
              "- trigger -> decision: AlertEvent.decision_ms, from the observation time of the signal that crossed "
              "the alert band (frame capture, audio chunk end, process poll) to the fusion decision.",
              "- Headless: no overlay, so the time for the warning window to appear (typically < 0.3 s) is not included.",
              "- All pages and calls are fictional DEMO assets (demo/test_pages, demo/scripts); calls are spoken by "
              "SAPI TTS and captured through WASAPI loopback. Rows hold ids and numbers only.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def save_records(records: Mapping[str, Mapping[str, Any]], order: Sequence[str], path: Path) -> None:
    ids = [i for i in order if i in records] + [i for i in records if i not in order]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(records[i], separators=(",", ":")) + "\n" for i in ids), encoding="utf-8")


def fusion_summary(cfg: Mapping[str, Any]) -> str:
    f = cfg.get("fusion", {}) or {}
    sig = ", ".join(f"{k} {v.get('weight')}/{v.get('decay_s')}/{v.get('cap')}" for k, v in (f.get("signals") or {}).items())
    bands = f.get("bands") or {}
    return (f"bands caution {bands.get('caution')} / alert {bands.get('alert')}, no_tactic_max {f.get('no_tactic_max')}, "
            f"signals weight/decay_s/cap: {sig}")


def write_reports(records: Mapping[str, Mapping[str, Any]], order: Sequence[str], out_dir: Path,
                  cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """scenarios.md + scenarios.csv for every recorded scenario, in spec order: the live outcome where the
    run was recorded under the current config + lexicons, else a replay of its signals under `cfg`."""
    fp = config_fingerprint(cfg)
    recs = [records[i] for i in order if i in records]
    rows = [make_row(r, Outcome(**r["live"]), "live") if r.get("fingerprint") == fp
            else make_row(r, replay(r, cfg), "replay") for r in recs]
    summary = summarize(rows)
    runs = sorted(r["run_at"] for r in recs)
    providers = sorted({f"{k} {v}" for r in recs for k, v in (r.get("providers") or {}).items() if v})
    n_live = sum(1 for r in rows if r["source"] == "live")
    meta = {
        "runs": f"{runs[0]} .. {runs[-1]}" if runs else "-",
        "scenarios": f"{len(recs)} ({summary['n_scam']} scam, {summary['n_normal']} normal); "
                     f"{n_live} live under this config, {len(rows) - n_live} replayed (see Definitions)",
        "machine": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}",
        "models": ", ".join(providers) or "-",
        "fusion config": fusion_summary(cfg),
    }
    write_csv(rows, out_dir / "scenarios.csv")
    write_markdown(rows, summary, meta, out_dir / "scenarios.md")
    return rows, summary


def print_rows(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    print(f"{'id':<24}{'source':<8}{'type':<8}{'expected':<9}{'outcome':<9}{'ok':<4}{'max':>4}  {'e2e s':>6}  "
          f"reason ids at max")
    for r in rows:
        e2e = "-" if r["e2e_ms"] is None else f"{r['e2e_ms'] / 1000:.1f}"
        print(f"{r['id']:<24}{r.get('source', 'live'):<8}{r['type']:<8}{r['expected']:<9}{r['outcome']:<9}"
              f"{'yes' if r['ok'] else 'NO':<4}{r['max_score']:>4}  {e2e:>6}  {r['reasons'] or '-'}")
    print("--- summary")
    for metric, value, target, status in summary_lines(summary):
        print(f"  {metric:<36} {value:<42} target {target:<26} {status}")


# ---------------------------------------------------------------- tuning: before / after by replay

def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, Mapping):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: d}


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """base with overlay's keys replaced recursively (a tuning overlay only lists what it changes)."""
    out = copy.deepcopy(dict(base))
    for k, v in (overlay or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else copy.deepcopy(v)
    return out


def config_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    b, a = _flatten(before), _flatten(after)
    return [(k, b.get(k), a.get(k)) for k in sorted(set(b) | set(a))
            if b.get(k) != a.get(k) and k != "config_path"]


def compare(records: Mapping[str, Mapping[str, Any]], order: Sequence[str], before: Mapping[str, Any],
            after: Mapping[str, Any], path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Replay every recorded scenario under both configs; optional tuning.md. Returns summaries + md lines."""
    recs = [records[i] for i in order if i in records]
    rows_b = [make_row(r, replay(r, before)) for r in recs]
    rows_a = [make_row(r, replay(r, after)) for r in recs]
    sb, sa = summarize(rows_b), summarize(rows_a)
    changes = config_changes(before, after)
    lines = ["# Kavach fusion tuning: before / after (replay of recorded scenario runs)", "",
             "Every recorded scenario is re-scored from its derived signal stream (bench/results/scenario_runs.jsonl) "
             "under both configs; only the fusion section is re-evaluated.", "", "## Config changes", ""]
    lines += _md_table(("key", "before", "after"), ([k, b, a] for k, b, a in changes)) if changes else ["(none)"]
    other = [k for k, _, _ in changes if not k.startswith("fusion.")]
    if other:
        lines += ["", f"Not re-evaluated by replay (needs a live run): {', '.join(other)}"]
    lines += ["", "## Summary", ""]
    lines += _md_table(("metric", "before", "after", "target"),
                       ([mb, vb, va, tb] for (mb, vb, tb, _), (_, va, _, _) in zip(summary_lines(sb), summary_lines(sa))))
    lines += ["", "## Per scenario", ""]
    lines += _md_table(("id", "type", "expected", "before", "after", "ok before", "ok after", "e2e s before", "e2e s after",
                        "reason ids at max (after)"),
                       ([b["id"], b["type"], b["expected"], f"{b['outcome']} {b['max_score']}", f"{a['outcome']} {a['max_score']}",
                         "yes" if b["ok"] else "NO", "yes" if a["ok"] else "NO",
                         _fmt(None if b["e2e_ms"] is None else b["e2e_ms"] / 1000),
                         _fmt(None if a["e2e_ms"] is None else a["e2e_ms"] / 1000), a["reasons"] or "-"]
                        for b, a in zip(rows_b, rows_a)))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sb, sa, lines


# ---------------------------------------------------------------- live harness (Windows)

class RecordingPipeline(Pipeline):
    """Pipeline that hands every signal to `record(stage, payload, ts, t_post)` before fusion gets it."""

    def __init__(self, *args: Any, record: Callable[[str, Any, float, float], None], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._record = record

    def post(self, stage: str, payload: Any, ts: float | None = None) -> None:
        ts = self.clock() if ts is None else ts
        try:
            self._record(stage, payload, ts, self.clock())
        except Exception:  # noqa: BLE001
            log.exception("signal recorder failed")
        super().post(stage, payload, ts)


def find_browser() -> str | None:
    """Chrome or Edge from the App Paths registry (so pages open in their own, closable window)."""
    if sys.platform != "win32":
        return None
    import winreg

    for exe in ("chrome.exe", "msedge.exe"):
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                    path = winreg.QueryValue(k, None)
                if path and Path(path).is_file():
                    return path
            except OSError:
                continue
    return None


def find_anydesk(explicit: str | None = None) -> str | None:
    for p in ([explicit] if explicit else []) + list(ANYDESK_PATHS):
        if p and Path(p).is_file():
            return p
    return None


def _windows_with_title(prefix: str) -> set[int]:
    import win32gui

    found: set[int] = set()

    def cb(hwnd: int, _: Any) -> bool:
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd).startswith(prefix):
            found.add(hwnd)
        return True

    win32gui.EnumWindows(cb, None)
    return found


def foreground_process() -> tuple[int, str] | None:
    """(hwnd, process name) of the foreground window; None if there is none. Never the title."""
    import ctypes

    import psutil

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    try:
        name = psutil.Process(pid.value).name()
    except psutil.Error:
        name = "?"
    return hwnd, name


def bring_to_front(hwnd: int, maximize: bool = True) -> bool:
    """Show + focus a window. Windows limits focus stealing: attach to the foreground thread's input
    first, and fall back to a synthetic Alt tap (the documented way to unlock SetForegroundWindow)."""
    import ctypes

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    if not user32.IsWindow(hwnd):
        return False
    user32.ShowWindow(hwnd, 3 if maximize else 9)            # SW_MAXIMIZE / SW_RESTORE
    fg = user32.GetForegroundWindow()
    me = kernel32.GetCurrentThreadId()
    fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = bool(fg_tid and fg_tid != me and user32.AttachThreadInput(me, fg_tid, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(me, fg_tid, False)
    if user32.GetForegroundWindow() != hwnd:
        user32.keybd_event(0x12, 0, 0, 0)                     # VK_MENU down / up
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SetForegroundWindow(hwnd)
    return user32.GetForegroundWindow() == hwnd


class Browser:
    """Opens demo pages with the webbrowser module; with Chrome/Edge found, each page gets its own window
    (--new-window), found by its <title>, maximized, focused and closed again after the run."""
    NAME = "kavach-bench"

    def __init__(self) -> None:
        import webbrowser

        exe = find_browser()
        self.own_windows = exe is not None
        if exe:
            webbrowser.register(self.NAME, None, webbrowser.BackgroundBrowser([exe, "--new-window", "%s"]))
            self.controller = webbrowser.get(self.NAME)
        else:
            self.controller = webbrowser.get()
        self.name = Path(exe).name if exe else "default browser (tabs, not closed)"

    def open(self, path: Path, timeout: float = 15.0) -> int | None:
        title = page_title(path)
        before = _windows_with_title(title)
        self.controller.open(path.as_uri(), new=1)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            new = _windows_with_title(title) - before
            if new:
                hwnd = min(new)
                time.sleep(0.5)                                # let the page paint before it is shown
                bring_to_front(hwnd)
                return hwnd
            time.sleep(0.2)
        return None

    def close(self, hwnd: int | None) -> None:
        """WM_CLOSE to a window this runner opened (never with the tab fallback: that would close the
        user's own browser window)."""
        if hwnd and self.own_windows:
            import ctypes

            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)


class RemoteTool:
    """none | fake | fake_tray | anydesk | anydesk_idle.

    anydesk       AnyDesk.exe started (user process: remote access counts as live)
    anydesk_idle  the same - AnyDesk open, no incoming session. The bench has no remote party, so the two
                  differ only in intent; processes.py counts a user-role AnyDesk as live either way
    fake          hidden waitfor.exe as a user-role process (live)
    fake_tray     waitfor.exe with the tool's idle arg: role "tray" (installed / tray only, NOT live)
    anydesk* fall back to the fake (user process) when AnyDesk.exe is not installed; the row says so."""

    def __init__(self, mode: str, anydesk: str | None = None):
        self.mode = mode
        self.tray = mode == "fake_tray"
        self.anydesk = anydesk if mode.startswith("anydesk") else None
        self.proc: subprocess.Popen | None = None
        if mode == "none":
            self.used, self.note = "none", ""
        elif self.anydesk:
            self.used, self.note = mode, ""
        else:
            self.used = "fake_tray" if self.tray else "fake"
            self.note = ANYDESK_NOTE if mode.startswith("anydesk") else ""

    def start(self, seconds: float) -> bool:
        """True if a process was started (a stimulus)."""
        if self.mode == "none":
            return False
        if self.anydesk:
            self.proc = subprocess.Popen([self.anydesk])
        else:
            self.proc = subprocess.Popen(
                ["waitfor.exe", "/T", str(int(seconds) + 60), FAKE_IDLE_ARG if self.tray else FAKE_LIVE_ARG],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True

    def stop(self) -> None:
        if self.proc is None:
            return
        import psutil

        try:
            parent = psutil.Process(self.proc.pid)
            for p in [*parent.children(recursive=True), parent]:
                p.kill()
        except psutil.Error:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None


class Speaker(threading.Thread):
    """Speaks a call script with SAPI (default voice) line by line; on_line(kind, i, t) with kind
    "line_start" before and "line" after each line."""

    def __init__(self, lines: Sequence[str], delay_s: float, on_line: Callable[[str, int, float], None],
                 pause_s: float = 0.6, rate: int = 0):
        super().__init__(name="bench-speaker", daemon=True)
        self.lines, self.delay_s, self.pause_s, self.rate = list(lines), delay_s, pause_s, rate
        self.on_line = on_line
        self.finished = False
        self.error = ""
        self._stop_ev = threading.Event()

    def run(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Rate = self.rate
            if self._stop_ev.wait(self.delay_s):
                return
            for i, line in enumerate(self.lines, 1):
                self.on_line("line_start", i, time.time())
                voice.Speak(line, 1)                           # SVSFlagsAsync
                while not voice.WaitUntilDone(100):
                    if self._stop_ev.is_set():
                        voice.Speak("", 3)                     # async + purge: stop mid-line
                        return
                self.on_line("line", i, time.time())
                if self._stop_ev.wait(self.pause_s):
                    return
            self.finished = True
        except Exception as e:  # noqa: BLE001
            self.error = type(e).__name__
        finally:
            pythoncom.CoUninitialize()

    def stop(self) -> None:
        self._stop_ev.set()
        if self.is_alive():
            self.join(5)


def _warm_up(ocr: Any, asr: Any) -> None:
    """First inference on DirectML is slow; run both models once so scenario 1 is not penalised."""
    import numpy as np

    try:
        ocr(np.full((256, 512, 3), 255, np.uint8))
    except Exception as e:  # noqa: BLE001
        log.warning("OCR warm-up failed: %s", type(e).__name__)
    try:
        asr.transcribe_array(np.zeros(16000 * 2, np.float32))
    except Exception as e:  # noqa: BLE001
        log.warning("ASR warm-up failed: %s", type(e).__name__)
    asr.rolling.clear()


def run_scenario(sc: Scenario, cfg: Mapping[str, Any], ocr: Any, asr: Any, browser: Browser, neutral: int | None,
                 anydesk: str | None, label: str) -> dict[str, Any]:
    scfg = copy.deepcopy(dict(cfg))
    scfg["processes"] = {**scfg["processes"], "known_tools": [*scfg["processes"].get("known_tools", []), FAKE_TOOL]}
    asr.rolling.clear()
    tool = RemoteTool(sc.remote, anydesk)
    notes = [tool.note] if tool.note else []
    stimuli: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    if neutral:
        bring_to_front(neutral)
    time.sleep(1.0)

    t0 = time.time()
    tracker = Tracker(t0)

    def record(stage: str, payload: Any, ts: float, t_post: float) -> None:
        d = encode_signal(stage, payload, ts, t_post, t0)
        if d is not None:
            signals.append(d)

    def stim(kind: str, t: float, **extra: Any) -> None:
        stimuli.append({"t": round(t - t0, 3), "kind": kind, **extra})

    pipe = RecordingPipeline(scfg, incident_log=False, ocr=ocr, asr=asr, record=record)
    pipe.subscribe(on_state=tracker.state, on_event=tracker.event)
    hwnd, speaker = None, None
    stolen: dict[str, int] = {}                                # process name -> times it took the foreground
    off_since: float | None = None
    pipe.start()
    try:
        if tool.start(sc.duration_s):
            stim("tool", time.time())
        if sc.page_path:
            hwnd = browser.open(sc.page_path)
            if hwnd is None:
                notes.append("page window not found")
            stim("page", time.time())
        if sc.call_path:
            speaker = Speaker(script_lines(sc.call_path), sc.call_delay_s, lambda kind, i, t: stim(kind, t, i=i))
            speaker.start()
        target = hwnd or neutral
        end = t0 + sc.duration_s
        next_print = t0 + 5.0
        while time.time() < end:
            time.sleep(0.25)
            # Focus guard: the scenario's window must stay in front (what the user would be looking at).
            # Anything else that takes the foreground is put back after 0.5 s and named in the notes.
            fg = foreground_process() if target else None
            if fg is not None and fg[0] != target:
                off_since = off_since or time.time()
                if time.time() - off_since >= 0.5:
                    stolen[fg[1]] = stolen.get(fg[1], 0) + 1
                    if bring_to_front(target) and target == hwnd:
                        stim("page", time.time())               # the page is back on screen
                    off_since = None
            else:
                off_since = None
            if time.time() >= next_print:
                next_print += 5.0
                st = pipe.state
                print(f"  {label} {sc.id:<24} t={time.time() - t0:4.0f}s  score {st.score if st else '-':>3}  "
                      f"{st.band if st else '-'}", flush=True)
    finally:
        if speaker is not None:
            speaker.stop()
        tool.stop()
        pipe.stop()
        browser.close(hwnd)
        if neutral:
            bring_to_front(neutral)

    health = pipe.health()
    for stage, h in health.items():
        if h["errors"]:
            notes.append(f"{stage}: {h['errors']} errors ({h['last_error']})")
    if speaker is not None:
        if speaker.error:
            notes.append(f"speaker failed: {speaker.error}")
        elif not speaker.finished:
            notes.append("call cut off at duration_s")
        if pipe.chunk_counts["speech"] == 0:
            notes.append("no speech chunks captured (system audio muted?)")
    if stolen:
        notes.append("focus taken back from " + ", ".join(f"{p} x{n}" for p, n in sorted(stolen.items())))
    providers = {s: health[s]["info"].split(" ")[0] for s in ("ocr", "asr") if health[s]["info"]}
    onsets: dict[str, float] = {}
    if sc.call_path:
        starts = [s["t"] for s in stimuli if s["kind"] == "line_start"]
        ends = [s["t"] for s in stimuli if s["kind"] == "line"]
        onsets = speech_onsets(script_lines(sc.call_path), starts, ends, scfg.get("intent"))
    outcome = tracker.finish(stimuli, signals, onsets)
    rec = {
        "id": sc.id, "type": sc.type, "expected": sc.expected, "page": sc.page, "remote": sc.remote,
        "remote_used": tool.used, "call": sc.call, "duration_s": sc.duration_s,
        "run_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(t0)), "providers": providers,
        "speech_chunks": pipe.chunk_counts["speech"], "focus_steals": sum(stolen.values()), "notes": notes,
        "stimuli": stimuli, "onsets": onsets, "signals": signals, "live": asdict(outcome),
    }
    e2e = "-" if outcome.e2e_ms is None else f"{outcome.e2e_ms / 1000:.1f}s ({outcome.stimulus})"
    print(f"  {label} {sc.id:<24} -> {outcome.band:<7} (expected {sc.expected}) max {outcome.max_score:>3}  "
          f"stimulus->alert {e2e}{'  | ' + '; '.join(notes) if notes else ''}", flush=True)
    return rec


def run_live(scenarios: Sequence[Scenario], cfg: Mapping[str, Any], out_dir: Path, order: Sequence[str],
             anydesk: str | None, settle_s: float, repeat: int = 1) -> dict[str, dict[str, Any]]:
    """Runs each scenario `repeat` times; the stored record is the worst run (worst_run), with every run's
    outcome under "repeats"."""
    from models.asr import ASR
    from models.ocr import create_backend

    fp = config_fingerprint(cfg)
    runs_path = out_dir / RUNS_FILE
    records = load_records(runs_path)
    print("loading models ...", flush=True)
    t = time.perf_counter()
    ocr, asr = create_backend(cfg), ASR(cfg)
    _warm_up(ocr, asr)
    print(f"models ready in {time.perf_counter() - t:.1f} s (OCR {ocr.name}:{ocr.provider}, "
          f"ASR {asr.backend_name}:{asr.provider})", flush=True)
    browser = Browser()
    neutral = browser.open(PAGES_DIR / NEUTRAL_PAGE)
    print(f"browser: {browser.name}; neutral page {'open' if neutral else 'NOT found'}", flush=True)
    try:
        for n, sc in enumerate(scenarios, 1):
            runs = []
            for k in range(1, repeat + 1):
                label = f"[{n}/{len(scenarios)}]" + (f" run {k}/{repeat}" if repeat > 1 else "")
                runs.append(run_scenario(sc, cfg, ocr, asr, browser, neutral, anydesk, label))
                time.sleep(settle_s)
            rec = dict(worst_run(sc.type, runs))
            rec["fingerprint"] = fp
            rec["repeats"] = [{k: r["live"][k] for k in ("band", "max_score", "reasons", "e2e_ms", "stimulus")}
                              | {"run_at": r["run_at"]} for r in runs]
            records[sc.id] = rec
            save_records(records, order, runs_path)           # keep progress if a later run fails
    finally:
        browser.close(neutral)
    return records


# ---------------------------------------------------------------- CLI

def dry_run(scenarios: Sequence[Scenario], anydesk: str | None) -> int:
    browser = find_browser()
    print(f"browser: {browser or 'not found (default browser tabs; pages cannot be closed)'}")
    print(f"AnyDesk: {anydesk or 'not installed (fake tool stands in for anydesk scenarios)'}")
    print(f"{'id':<24}{'type':<8}{'page':<28}{'remote':<20}{'call':<26}{'lines':>5}{'dur s':>6}  expected")
    total = 0.0
    for sc in scenarios:
        lines = len(script_lines(sc.call_path)) if sc.call_path else 0
        remote = sc.remote if not sc.remote.startswith("anydesk") or anydesk else f"{sc.remote}->fake"
        print(f"{sc.id:<24}{sc.type:<8}{sc.page or '-':<28}{remote:<20}{sc.call or '-':<26}{lines:>5}"
              f"{sc.duration_s:>6.0f}  {sc.expected}")
        total += sc.duration_s + 6
    n_scam = sum(1 for s in scenarios if s.type == "scam")
    print(f"{len(scenarios)} scenarios ({n_scam} scam, {len(scenarios) - n_scam} normal), "
          f"about {total / 60:.0f} min live (plus model loading). Nothing was run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.scenarios", description="Kavach end-to-end scenario runner")
    p.add_argument("--only", metavar="IDS", help="comma-separated scenario ids (other recorded rows are kept)")
    p.add_argument("--dry-run", action="store_true", help="validate the spec and assets, print the plan, run nothing")
    p.add_argument("--replay", action="store_true", help="re-score recorded runs with the current config (no hardware)")
    p.add_argument("--compare", metavar="OVERLAY",
                   help="before (current config) / after (current config + OVERLAY yaml) replay -> tuning.md")
    p.add_argument("--spec", type=Path, default=SPEC_PATH)
    p.add_argument("--results", type=Path, default=RESULTS_DIR, help="output directory")
    p.add_argument("--anydesk", help="path to AnyDesk.exe (default: standard install locations)")
    p.add_argument("--settle", type=float, default=3.0, help="seconds between scenarios")
    p.add_argument("--repeat", type=int, default=1, help="live runs per scenario (the report row is the worst run)")
    p.add_argument("--report", action="store_true", help="rewrite scenarios.md/.csv from recorded runs, run nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from kavach_config import AttrDict, ConfigError, get_config, validate

    try:
        spec = load_spec(args.spec)
        scenarios = select(spec, args.only)
    except SpecError as e:
        print(f"spec error: {e}")
        return 2
    order = [s.id for s in spec]
    cfg = get_config().to_dict()
    anydesk = find_anydesk(args.anydesk)

    if args.dry_run:
        return dry_run(scenarios, anydesk)
    if args.report:
        records = load_records(args.results / RUNS_FILE)
        if not records:
            print(f"no recorded runs in {args.results / RUNS_FILE}")
            return 1
        rows, summary = write_reports(records, order, args.results, cfg)
        print_rows(rows, summary)
        print(f"written: {args.results / 'scenarios.md'}, {args.results / 'scenarios.csv'}")
        return 0
    if args.replay or args.compare:
        records = load_records(args.results / RUNS_FILE)
        records = {i: r for i, r in records.items() if i in {s.id for s in scenarios}}
        if not records:
            print(f"no recorded runs in {args.results / RUNS_FILE}; run the scenarios live first")
            return 1
        if args.compare:
            overlay = yaml.safe_load(Path(args.compare).read_text(encoding="utf-8")) or {}
            after = deep_merge(cfg, overlay)
            try:
                validate(AttrDict(after))
            except ConfigError as e:
                print(f"invalid overlay {args.compare}: {e}")
                return 2
            sb, sa, _ = compare(records, order, cfg, after, args.results / "tuning.md")
            for label, s in (("before (current config)", sb), (f"after ({args.compare})", sa)):
                print(f"--- {label}")
                for metric, value, target, status in summary_lines(s):
                    print(f"  {metric:<36} {value:<42} {status}")
            print(f"written: {args.results / 'tuning.md'}")
            return 0
        rows = [make_row(r, replay(r, cfg)) for r in (records[i] for i in order if i in records)]
        print_rows(rows, summarize(rows))
        return 0

    if sys.platform != "win32":
        print("live scenario runs need Windows (screen capture, WASAPI loopback, SAPI)")
        return 1
    n = len(scenarios)
    print(f"running {n} scenario(s) x{max(1, args.repeat)} live, "
          f"about {max(1, args.repeat) * sum(s.duration_s + 6 for s in scenarios) / 60:.0f} min. "
          f"The desktop and speakers are in use; keep the PC idle.", flush=True)
    records = run_live(scenarios, cfg, args.results, order, anydesk, args.settle, max(1, args.repeat))
    rows, summary = write_reports(records, order, args.results, cfg)
    print_rows(rows, summary)
    print(f"written: {args.results / 'scenarios.md'}, {args.results / 'scenarios.csv'}, {args.results / RUNS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
