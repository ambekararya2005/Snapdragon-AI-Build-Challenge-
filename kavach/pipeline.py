"""Kavach pipeline (F6 wiring): capture -> models -> detect -> fusion, one thread per stage.

Threads, all posting SignalEvents to one queue:
  - processes : ProcessMonitor.poll() every processes.poll_interval_s -> RemoteToolSignal per tool
  - screen    : ScreenSampler (own thread) -> newest changed frame only -> ocr thread:
                OCR -> screen_classifier -> ScreenLabel (ts = frame capture time)
  - audio     : AudioCapture (own threads) -> speech chunks (backlog of 3) -> asr thread:
                ASR -> rolling transcript -> detect.intent -> IntentResult (ts = chunk end)
  - fusion    : consumes the queue, RiskEngine.update() per event and score() per event and every
                fusion.tick_s (decay), publishes RiskState + Alert/Caution/Override events to subscribers.
OCR and ASR share the GPU through models.runtime's DirectML run lock (DML runs are serialized there).

    p = Pipeline()
    p.subscribe(on_state=lambda state: ..., on_event=lambda ev: ...)
    p.start(); ...; p.override(); ...; p.stop()
    p.health()        # per stage: alive, last_event, events, errors, last_error, info

Privacy: frames, OCR text and transcripts stay in RAM inside their stage; only labels, tactic ids,
scores and timestamps cross into fusion. The window shown on the status line is the process name +
redact_title(). Incidents go to fusion.incident_log (ids and numbers only).

Headless:  python -m kavach.pipeline --plain [--seconds 30] [--no-audio] [--no-screen] [--ocr-backend native]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.incidents import IncidentLog  # noqa: E402
from fusion.risk import RemoteToolSignal, RiskEngine, RiskState  # noqa: E402

log = logging.getLogger("kavach.pipeline")
STAGES = ("processes", "screen", "ocr", "audio", "asr", "fusion")
_OVERRIDE = object()


@dataclass(frozen=True)
class SignalEvent:
    stage: str
    ts: float                          # when the signal was observed (capture time)
    payload: Any                       # RemoteToolSignal | ScreenLabel | IntentResult


@dataclass
class StageHealth:
    name: str
    enabled: bool = True
    alive: bool = False
    last_event: float | None = None
    events: int = 0
    errors: int = 0
    last_error: str | None = None
    dropped: int = 0
    info: str = ""                     # provider / latency summary, ids and numbers only

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _latest_put(q: queue.Queue, item: Any) -> bool:
    """Put, dropping the oldest item when full. Returns False if something was dropped."""
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return not dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass


class Pipeline:
    def __init__(self, config: Mapping[str, Any] | None = None, *, screen: bool = True, audio: bool = True,
                 processes: bool = True, incident_log: IncidentLog | bool = True,
                 clock: Callable[[], float] = time.time):
        if config is None:
            from kavach_config import get_config
            config = get_config()
        self.cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        self.clock = clock
        self.engine = RiskEngine(self.cfg)
        self.tick_s = float(self.cfg.get("fusion", {}).get("tick_s") or 0.5)
        self.incident_log = (IncidentLog(config=self.cfg) if incident_log is True
                             else incident_log if isinstance(incident_log, IncidentLog) else None)
        self._enabled = {"processes": processes, "screen": screen, "ocr": screen, "audio": audio, "asr": audio,
                         "fusion": True}
        self._health = {s: StageHealth(s, self._enabled[s]) for s in STAGES}
        self._hlock = threading.Lock()
        self._q: queue.Queue[Any] = queue.Queue(maxsize=1000)
        self._frames: queue.Queue[Any] = queue.Queue(maxsize=1)      # newest screen frame only
        self._chunks: queue.Queue[Any] = queue.Queue(maxsize=3)      # ASR backlog, oldest dropped
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._state_subs: list[Callable[[RiskState], None]] = []
        self._event_subs: list[Callable[[Any], None]] = []
        self._sub_lock = threading.Lock()
        self._state: RiskState | None = None
        self.window: str = ""                                       # "[proc] <title hidden>" of the last OCR'd frame
        self.monitor = self.sampler = self.audio = self.ocr = self.asr = None
        self.started = False

    # -- public API

    def subscribe(self, on_state: Callable[[RiskState], None] | None = None,
                  on_event: Callable[[Any], None] | None = None) -> None:
        """on_state: every RiskState (each event + every tick_s). on_event: AlertEvent / CautionEvent / OverrideEvent."""
        with self._sub_lock:
            if on_state:
                self._state_subs.append(on_state)
            if on_event:
                self._event_subs.append(on_event)

    def post(self, stage: str, payload: Any, ts: float | None = None) -> None:
        """Hand a signal to fusion (used by the stage threads; also for tests and replays)."""
        ts = self.clock() if ts is None else ts
        self._mark(stage, ts)
        try:
            self._q.put_nowait(SignalEvent(stage, ts, payload))
        except queue.Full:
            self._error("fusion", "queue full, event dropped")

    def override(self) -> None:
        """"I'm safe, continue" from the UI (applied on the fusion thread)."""
        self._q.put(_OVERRIDE)

    @property
    def state(self) -> RiskState | None:
        return self._state

    def health(self) -> dict[str, dict[str, Any]]:
        alive = {t.name: t.is_alive() for t in self._threads}
        with self._hlock:
            out = {}
            for name, h in self._health.items():
                h.alive = h.enabled and self.started and alive.get(f"kavach-{name}", self._stage_alive(name))
                out[name] = h.as_dict()
            return out

    # -- start / stop

    def start(self) -> None:
        if self.started:
            return
        self._stop.clear()
        self.started = True
        self._spawn("fusion", self._fusion_loop)
        if self._enabled["processes"]:
            self._guard("processes", self._start_processes)
        if self._enabled["screen"]:
            self._guard("ocr", self._start_screen)
        if self._enabled["audio"]:
            self._guard("asr", self._start_audio)

    def stop(self, timeout: float = 10.0) -> None:
        if not self.started:
            return
        self._stop.set()
        for comp in (self.audio, self.sampler):
            if comp is not None:
                try:
                    comp.stop()
                except Exception as e:  # noqa: BLE001
                    log.warning("stop failed: %s", type(e).__name__)
        for t in self._threads:
            t.join(timeout)
        self._threads.clear()
        self.started = False
        self._drain_frames()

    def _spawn(self, name: str, target: Callable[[], None]) -> None:
        t = threading.Thread(target=target, name=f"kavach-{name}", daemon=True)
        self._threads.append(t)
        t.start()

    def _guard(self, stage: str, fn: Callable[[], None]) -> None:
        """Start one stage; if its models / devices fail, disable it and keep the rest running."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self._error(stage, f"start failed: {type(e).__name__}: {e}")
            log.warning("stage %s disabled: %s", stage, type(e).__name__)

    def _stage_alive(self, name: str) -> bool:
        if name == "screen" and self.sampler is not None:
            th = self.sampler._thread
            return bool(th and th.is_alive())
        if name == "audio" and self.audio is not None:
            return any(t.is_alive() for t in self.audio._threads)
        return False

    # -- health helpers

    def _mark(self, stage: str, ts: float | None = None, info: str | None = None) -> None:
        with self._hlock:
            h = self._health[stage]
            h.events += 1
            h.last_event = self.clock() if ts is None else ts
            if info is not None:
                h.info = info

    def _error(self, stage: str, msg: str) -> None:
        with self._hlock:
            h = self._health[stage]
            h.errors += 1
            h.last_error = msg[:160]

    def _dropped(self, stage: str) -> None:
        with self._hlock:
            self._health[stage].dropped += 1

    # -- processes

    def _start_processes(self) -> None:
        from capture.processes import ProcessMonitor

        self.monitor = ProcessMonitor(self.cfg)
        self._spawn("processes", self._processes_loop)

    def _processes_loop(self) -> None:
        interval = float(self.cfg["processes"].get("poll_interval_s", 2.0))
        known: set[str] = set()
        while not self._stop.is_set():
            try:
                self.monitor.poll()
                ts = self.clock()
                now_tools = set()
                for s in self.monitor.current():
                    now_tools.add(s.tool)
                    self.post("processes", RemoteToolSignal(s.tool, s.has_user_process, s.active_session), ts)
                for tool in known - now_tools:
                    self.post("processes", RemoteToolSignal(tool, False, False), ts)
                if not now_tools and not known:
                    self._mark("processes", ts)                 # a poll with nothing to report
                known = now_tools
                live = sorted(s.tool for s in self.monitor.current() if s.has_user_process)
                with self._hlock:
                    self._health["processes"].info = f"live: {', '.join(live) or 'none'}"
            except Exception as e:  # noqa: BLE001
                self._error("processes", f"{type(e).__name__}: {e}")
            self._stop.wait(interval)

    # -- screen -> OCR -> classifier

    def _start_screen(self) -> None:
        from capture.screen import ScreenSampler
        from models.ocr import create_backend

        self.ocr = create_backend(self.cfg)
        with self._hlock:
            self._health["ocr"].info = f"{self.ocr.name}:{self.ocr.provider}"
        self.sampler = ScreenSampler(self.cfg, on_frame=self._on_frame)
        self._spawn("ocr", self._ocr_loop)
        self.sampler.start()

    def _on_frame(self, frame: Any) -> None:
        # The sampler clears frame.image after this returns; keep our own reference for the OCR thread.
        self._mark("screen", frame.ts)
        if not _latest_put(self._frames, dataclasses.replace(frame)):
            self._dropped("screen")

    def _ocr_loop(self) -> None:
        from detect.screen_classifier import classify
        from kavach_privacy import redact_title

        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.ocr(frame.image)
                frame.image = None
                label = classify(result, frame.window, self.cfg.get("screen_classifier"))
                n_lines, ocr_ms = len(result.lines), result.total_ms
                del result
                self.window = f"[{frame.window.process_name or '?'}] {redact_title(frame.window.title)}"
                self.post("screen", label, frame.ts)
                scores = " ".join(f"{k}={v:.2f}" for k, v in label.labels.items() if k != "normal" and v >= 0.1)
                self._mark("ocr", info=f"{self.ocr.name}:{self.ocr.provider} {ocr_ms:.0f} ms, {n_lines} lines, "
                                       f"top={label.top_label} {scores}".rstrip())
            except Exception as e:  # noqa: BLE001
                self._error("ocr", f"{type(e).__name__}: {e}")
            finally:
                frame = None

    def _drain_frames(self) -> None:
        for q in (self._frames, self._chunks):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    # -- audio -> ASR -> intent

    def _start_audio(self) -> None:
        from capture.audio import AudioCapture
        from models.asr import ASR

        self.asr = ASR(self.cfg)
        with self._hlock:
            self._health["asr"].info = f"{self.asr.backend_name}:{self.asr.provider}"
        self.audio = AudioCapture(self.cfg, self._on_chunk)
        self._spawn("asr", self._asr_loop)
        self.audio.start()

    def _on_chunk(self, chunk: Any) -> None:
        self._mark("audio", chunk.ts_end)
        if not chunk.is_speech:
            return
        if not _latest_put(self._chunks, chunk):
            self._dropped("audio")

    def _asr_loop(self) -> None:
        from detect.intent import detect

        while not self._stop.is_set():
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                t = self.asr.transcribe(chunk)
                res = detect(self.asr.rolling, self.cfg.get("intent"))
                self.post("asr", res, chunk.ts_end)
                ms = f"{t.total_ms:.0f} ms" if t is not None else "-"
                self._mark("asr", chunk.ts_end, info=f"{self.asr.backend_name}:{self.asr.provider} {ms}, "
                                                      f"{res.distinct_tactics} tactics")
            except Exception as e:  # noqa: BLE001
                self._error("asr", f"{type(e).__name__}: {e}")
            finally:
                chunk = None

    # -- fusion

    def _fusion_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=max(0.0, next_tick - time.monotonic()))
            except queue.Empty:
                item = None
            now = self.clock()
            try:
                events: list[Any] = []
                if item is _OVERRIDE:
                    events.append(self.engine.user_override(now))
                elif item is not None:
                    self.engine.update(item.payload, now, item.ts)
                tick = time.monotonic() >= next_tick
                if item is not None or tick:
                    state = self.engine.score(now)
                    self._state = state
                    self._mark("fusion", now, info=f"score {state.score} {state.band}")
                    self._publish(state, events + state.events)
                if tick:
                    next_tick = time.monotonic() + self.tick_s
            except Exception as e:  # noqa: BLE001
                self._error("fusion", f"{type(e).__name__}: {e}")
                log.exception("fusion step failed")

    def _publish(self, state: RiskState, events: list[Any]) -> None:
        with self._sub_lock:
            state_subs, event_subs = list(self._state_subs), list(self._event_subs)
        for ev in events:
            if self.incident_log is not None:
                try:
                    self.incident_log(ev)
                except OSError as e:
                    self._error("fusion", f"incident log: {type(e).__name__}")
            for cb in event_subs:
                try:
                    cb(ev)
                except Exception:
                    log.exception("event subscriber failed")
        for cb in state_subs:
            try:
                cb(state)
            except Exception:
                log.exception("state subscriber failed")


# ---------------------------------------------------------------- headless CLI

def status_line(state: RiskState | None, window: str = "") -> str:
    """One line, ids and numbers only (plus the already-redacted window)."""
    if state is None:
        return f"{time.strftime('%H:%M:%S')}  score   -  starting"
    top = sorted(state.contributions, key=lambda c: (-c["points"], c["signal"]))
    reasons = " ".join(f"{c['signal']}({c['points']:g})" for c in top) or "-"
    flag = " [override]" if state.override_active else ""
    return f"{time.strftime('%H:%M:%S')}  score {state.score:3d}  {state.band:<7}{flag}  {reasons}" + (
        f"  | {window}" if window else "")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kavach.pipeline", description="Kavach pipeline, headless")
    p.add_argument("--plain", action="store_true", help="one line per second (default: one refreshing line)")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (default: until Ctrl+C)")
    p.add_argument("--no-screen", action="store_true", help="disable screen capture + OCR")
    p.add_argument("--no-audio", action="store_true", help="disable audio capture + ASR")
    p.add_argument("--no-processes", action="store_true", help="disable the remote-tool monitor")
    p.add_argument("--no-incident-log", action="store_true", help="do not write fusion.incident_log")
    p.add_argument("--ocr-backend", choices=["rapidocr", "native"], help="override config ocr.backend")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from kavach_config import get_config

    cfg = get_config().to_dict()
    if args.ocr_backend:
        cfg["ocr"]["backend"] = args.ocr_backend
    pipe = Pipeline(cfg, screen=not args.no_screen, audio=not args.no_audio, processes=not args.no_processes,
                    incident_log=not args.no_incident_log)
    alerts: list[Any] = []

    def on_event(ev: Any) -> None:
        if ev.kind == "alert":
            alerts.append(ev)
            print(f"{time.strftime('%H:%M:%S')}  >> ALERT #{ev.incident} score {ev.score}, trigger -> decision "
                  f"{ev.time_to_alert_ms:.0f} ms: {', '.join(ev.reason_ids)}", flush=True)
        elif ev.kind == "caution":
            print(f"{time.strftime('%H:%M:%S')}  >> caution score {ev.score}: {', '.join(ev.reason_ids)}", flush=True)

    pipe.subscribe(on_event=on_event)
    print("starting (loading models) ...", flush=True)
    t_start = time.monotonic()
    pipe.start()
    h = pipe.health()
    print(f"started in {time.monotonic() - t_start:.1f} s | OCR {h['ocr']['info'] or '-'} | ASR {h['asr']['info'] or '-'}"
          f" | incident log {pipe.incident_log.path if pipe.incident_log else 'off'}", flush=True)
    for name, st in h.items():
        if st["last_error"]:
            print(f"  stage {name}: {st['last_error']}", flush=True)

    max_score, bands = 0, {}
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(1.0)
            st = pipe.state
            if st is not None:
                max_score = max(max_score, st.score)
                bands[st.band] = bands.get(st.band, 0) + 1
            line = status_line(st, pipe.window)
            print(line if args.plain else f"\r{line[:160]:<160}", end="\n" if args.plain else "", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
    if not args.plain:
        print()
    print("--- summary ---")
    print(f"max score {max_score}, seconds per band {bands}, alerts {len(alerts)}"
          + (f" (first trigger -> decision {alerts[0].time_to_alert_ms:.0f} ms)" if alerts else ""))
    for name, st in pipe.health().items():
        if not st["enabled"]:
            print(f"  {name:<9} disabled")
            continue
        last = f"{max(0.0, time.time() - st['last_event']):.1f}s ago" if st["last_event"] else "never"
        err = f", errors {st['errors']} (last: {st['last_error']})" if st["errors"] else ""
        drop = f", dropped {st['dropped']}" if st["dropped"] else ""
        print(f"  {name:<9} events {st['events']:<5} last {last:<10} {st['info']}{drop}{err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
