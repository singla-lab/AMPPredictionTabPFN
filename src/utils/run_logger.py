"""
run_logger.py — lightweight resource monitor for long-running scripts.

Usage (in a script entry-point):
    from src.utils.run_logger import RunLogger

    with RunLogger(script_name="train.py", log_dir=REPO_ROOT / "logs"):
        main()

What gets logged:
  <stem>_<ts>_pid<N>.json  — resource stats + exit status (JSON)
  <stem>_<ts>_pid<N>.log   — full stdout + stderr transcript (plain text)

JSON fields:
    - script_name, pid, argv, start/end timestamps, wall_time_s
    - exit_status: "completed" | "exception" | "signal" | "oom"
    - exit_code, signal_name, exception_type, exception_msg, traceback
    - cpu_percent: {mean, max, samples}
    - ram_gb:      {mean, max, samples}
    - gpu_util_pct: {mean, max, samples}   (per device if ≥1 GPU present)
    - gpu_mem_gb:   {mean, max, samples}   (per device if ≥1 GPU present)

Signals handled: SIGTERM, SIGHUP.  SIGKILL cannot be intercepted by any
Python code; the log will still contain everything collected up to that point
because we flush periodically to a checkpoint file.
"""
import datetime
import gc
import json
import os
import pathlib
import signal
import sys
import threading
import time
import traceback


# ── optional deps (gracefully absent) ─────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import pynvml
    pynvml.nvmlInit()
    _N_GPU = pynvml.nvmlDeviceGetCount()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
    _N_GPU = 0


# ── stdout / stderr tee ───────────────────────────────────────────────────────
class _TeeStream:
    """
    Replaces sys.stdout (or sys.stderr) with an object that writes to both
    the original stream and a log file simultaneously.

    Usage:
        tee = _TeeStream(sys.stdout, log_path)
        sys.stdout = tee
        ...run code...
        sys.stdout = tee.restore()  # restores original stream, closes file
    """

    def __init__(self, original_stream, log_path: pathlib.Path):
        self._orig = original_stream
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self._orig.write(data)
            self._file.write(data)
        return len(data)

    def flush(self):
        self._orig.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    # Forward all other attribute access to the underlying stream
    def __getattr__(self, name):
        return getattr(self._orig, name)

    def restore(self):
        """Close the log file and return the original stream."""
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass
        return self._orig


# ── internal helpers ───────────────────────────────────────────────────────────
def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _collect_gpu_samples() -> dict:
    """Return {util: [float,...], mem_gb: [float,...]} per GPU index, or {}."""
    if not _HAS_NVML or _N_GPU == 0:
        return {}
    samples = {}
    for i in range(_N_GPU):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            key   = f"gpu{i}"
            if key not in samples:
                samples[key] = {"util": [], "mem_gb": []}
            samples[key]["util"].append(float(rates.gpu))
            samples[key]["mem_gb"].append(mem.used / 1e9)
        except Exception:
            pass
    return samples


def _aggregate(values: list) -> dict:
    if not values:
        return {"mean": None, "max": None, "n_samples": 0}
    return {
        "mean":      round(sum(values) / len(values), 4),
        "max":       round(max(values), 4),
        "n_samples": len(values),
    }


# ── background sampler ─────────────────────────────────────────────────────────
class _ResourceSampler(threading.Thread):
    """Daemon thread that polls CPU/RAM/GPU every `interval_s` seconds."""

    def __init__(self, interval_s: float = 5.0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()

        # accumulated raw lists
        self.cpu_pct:  list[float] = []
        self.ram_gb:   list[float] = []
        self.gpu_data: dict        = {}   # {gpu0: {util:[..], mem_gb:[..]}, ...}
        self._lock = threading.Lock()

    def run(self):
        proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        while not self._stop_event.wait(self.interval_s):
            self._sample(proc)

    def _sample(self, proc):
        with self._lock:
            # CPU
            if _HAS_PSUTIL:
                self.cpu_pct.append(psutil.cpu_percent(interval=None))
                try:
                    mem = proc.memory_info()
                    self.ram_gb.append(mem.rss / 1e9)
                except Exception:
                    pass

            # GPU (pynvml path)
            gpu_snap = _collect_gpu_samples()
            for key, data in gpu_snap.items():
                if key not in self.gpu_data:
                    self.gpu_data[key] = {"util": [], "mem_gb": []}
                self.gpu_data[key]["util"].extend(data["util"])
                self.gpu_data[key]["mem_gb"].extend(data["mem_gb"])

    def stop(self):
        self._stop_event.set()

    def summary(self) -> dict:
        with self._lock:
            out = {
                "cpu_percent": _aggregate(self.cpu_pct),
                "ram_gb":      _aggregate(self.ram_gb),
            }
            for key, data in self.gpu_data.items():
                out[f"{key}_util_pct"] = _aggregate(data["util"])
                out[f"{key}_mem_gb"]   = _aggregate(data["mem_gb"])
            return out


# ── main context manager ───────────────────────────────────────────────────────
class RunLogger:
    """
    Context manager that monitors resources and writes a run log on exit.

    Parameters
    ----------
    script_name : str
        Human-readable name shown in the log (e.g. "train.py").
    log_dir     : path-like
        Directory where logs are saved.  Created if absent.
    sample_interval_s : float
        How often (seconds) to poll CPU / RAM / GPU.
    extra_meta  : dict | None
        Any extra key-value pairs to embed in the log.
    """

    def __init__(
        self,
        script_name: str,
        log_dir,
        sample_interval_s: float = 5.0,
        extra_meta: dict | None = None,
        tee_output: bool = True,
    ):
        self.script_name = script_name
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.extra_meta = extra_meta or {}
        self.sample_interval_s = sample_interval_s
        self.tee_output = tee_output

        self._start_ts: str = ""
        self._sampler: _ResourceSampler | None = None
        self._log_path: pathlib.Path | None = None
        self._stdout_tee: _TeeStream | None = None
        self._stderr_tee: _TeeStream | None = None
        self._orig_sigterm = signal.SIG_DFL
        self._orig_sighup  = signal.SIG_DFL
        self._record: dict = {}

    # ── internal ──────────────────────────────────────────────────────────────
    def _make_log_path(self) -> pathlib.Path:
        """Returns the .json path; the .log file shares the same stem."""
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        stem = self.script_name.replace(".py", "").replace("/", "_").replace("\\", "_")
        return self.log_dir / f"{stem}_{ts}_pid{os.getpid()}.json"

    def _build_base_record(self) -> dict:
        return {
            "script_name": self.script_name,
            "pid":         os.getpid(),
            "argv":        sys.argv,
            "start_time":  self._start_ts,
            "end_time":    None,
            "wall_time_s": None,
            "exit_status": "running",
            "exit_code":   None,
            "signal_name": None,
            "exception_type": None,
            "exception_msg":  None,
            "traceback":      None,
            "rng_states":     None,   # populated in __enter__
            **self.extra_meta,
        }

    def _flush(self, extra: dict | None = None):
        """Write current state to disk (called periodically and on exit)."""
        if self._log_path is None:
            return
        rec = {**self._record}
        if extra:
            rec.update(extra)
        if self._sampler is not None:
            rec["resources"] = self._sampler.summary()
        try:
            # Atomic-ish write via temp file
            tmp = self._log_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=2, default=str)
            tmp.replace(self._log_path)
        except Exception:
            pass   # never raise inside a logger

    def _finalize(self, exit_status: str, **kwargs):
        now = _iso_now()
        if self._sampler is not None:
            self._sampler.stop()
        elapsed = (
            time.time() - self._t_start
            if hasattr(self, "_t_start")
            else None
        )
        self._record.update({
            "end_time":    now,
            "wall_time_s": round(elapsed, 3) if elapsed is not None else None,
            "exit_status": exit_status,
            **kwargs,
        })
        self._flush()
        # Stop tee AFTER final flush so the status line is also captured
        if self._stdout_tee is not None:
            sys.stdout = self._stdout_tee.restore()
            self._stdout_tee = None
        if self._stderr_tee is not None:
            sys.stderr = self._stderr_tee.restore()
            self._stderr_tee = None
        print(
            f"[RunLogger] JSON log → {self._log_path}  "
            f"Terminal log → {self._log_path.with_suffix('.log')}  "
            f"(status={exit_status})",
            file=sys.stderr,
            flush=True,
        )

    def _make_signal_handler(self, signum):
        def _handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(
                f"\n[RunLogger] Caught signal {sig_name} ({signum}) — "
                "saving log and re-raising ...",
                file=sys.stderr,
                flush=True,
            )
            self._finalize("signal", exit_code=128 + signum, signal_name=sig_name)
            # Restore original handler and re-raise so the process exits properly
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        return _handler

    # ── context manager API ───────────────────────────────────────────────────
    def __enter__(self):
        self._start_ts = _iso_now()
        self._t_start  = time.time()
        self._log_path = self._make_log_path()
        self._record   = self._build_base_record()

        # Snapshot RNG states for reproducibility
        try:
            from src.utils.seed_utils import get_rng_states
            self._record["rng_states"] = get_rng_states()
        except Exception:
            pass  # never block on logger failure

        # Tee stdout + stderr to <stem>.log
        if self.tee_output:
            term_log = self._log_path.with_suffix(".log")
            header = (
                f"{'='*72}\n"
                f"Script : {self.script_name}\n"
                f"PID    : {os.getpid()}\n"
                f"Started: {self._start_ts}\n"
                f"{'='*72}\n"
            )
            term_log.write_text(header, encoding="utf-8")
            self._stdout_tee = _TeeStream(sys.stdout, term_log)
            self._stderr_tee = _TeeStream(sys.stderr, term_log)
            sys.stdout = self._stdout_tee
            sys.stderr = self._stderr_tee

        # Start background sampler
        self._sampler = _ResourceSampler(interval_s=self.sample_interval_s)
        self._sampler.start()

        # Install signal handlers
        self._orig_sigterm = signal.signal(signal.SIGTERM, self._make_signal_handler(signal.SIGTERM))
        self._orig_sighup  = signal.signal(signal.SIGHUP,  self._make_signal_handler(signal.SIGHUP))

        # Flush initial record immediately
        self._flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        signal.signal(signal.SIGHUP,  self._orig_sighup)

        if exc_type is None:
            # Clean completion
            self._finalize("completed", exit_code=0)
        elif exc_type is MemoryError:
            self._finalize(
                "oom",
                exit_code=1,
                exception_type=exc_type.__name__,
                exception_msg=str(exc_val),
                traceback="".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
            )
        else:
            self._finalize(
                "exception",
                exit_code=1,
                exception_type=exc_type.__name__,
                exception_msg=str(exc_val),
                traceback="".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
            )

        # Do NOT suppress the exception
        return False
