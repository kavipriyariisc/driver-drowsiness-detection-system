"""
Telemetry Replay Streamer  —  UL-DD Live Demo
==============================================
Reads a UL-DD Telemetry CSV (60 Hz) and re-emits samples at 4 Hz in a
background thread, exactly as if the data were arriving from a live OBD-II /
CAN bus source.  Feed the output into your ring buffer and BiLSTM inference
the same way you would with real vehicle signals.

Typical usage
-------------
    replayer = TelemetryReplayer(
        csv_path="C:/Users/.../UL-DD/CSV_Files/.../B_Telemetry_A.csv",
        loop=True,          # restart when file ends
        speed_factor=1.0,   # 1.0 = real-time, 2.0 = double speed
    )

    with replayer:                      # starts + stops automatically
        for _ in range(500):
            sample = replayer.get_latest(timeout=0.5)
            if sample:
                print(sample)
            time.sleep(0.25)            # 4 Hz consumer
"""

import time
import threading
import queue
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column names as they appear in the UL-DD Telemetry CSV
# (the dataset uses 'pitch[deg]', 'roll[deg]', 'speed[m/s]', 'rpm', 'gear')
# ---------------------------------------------------------------------------
_TELE_COLS_PREFERRED = ['pitch[deg]', 'roll[deg]', 'speed[m/s]', 'rpm', 'gear']
_TELE_KEYS_SHORT     = ['pitch',      'roll',      'speed',       'rpm', 'gear']


class TelemetryReplayer:
    """
    Streams UL-DD telemetry data from a CSV file at 4 Hz in real-time.

    The UL-DD CSV is recorded at 60 Hz.  This class downsamples to 4 Hz
    (every 15th row) and emits one dict per 0.25 s via an internal thread.

    Parameters
    ----------
    csv_path : str | Path
        Path to ``{Subject}_Telemetry_{Session}.csv`` inside the UL-DD
        ``CSV_Files`` folder tree.
    loop : bool
        If True (default), restart replay from the beginning when the file
        ends — useful for demo sessions longer than 40 min.
    speed_factor : float
        Playback speed multiplier.  1.0 = real-time.  Use 2.0 to race
        through 40 min of data in 20 min during a short demo.
    subject_label : str
        Informational label printed in log messages (e.g., ``"B_Alert"``).
    """

    ORIG_HZ   = 60
    TARGET_HZ = 4
    _DOWNSAMPLE = ORIG_HZ // TARGET_HZ   # 15

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        csv_path: str,
        loop: bool = True,
        speed_factor: float = 1.0,
        subject_label: str = "",
    ):
        self.csv_path      = Path(csv_path)
        self.loop          = loop
        self.speed_factor  = max(speed_factor, 0.01)
        self.subject_label = subject_label or self.csv_path.stem

        self._data: np.ndarray = np.empty((0, 5), dtype=np.float32)
        self._col_keys: List[str] = _TELE_KEYS_SHORT[:]

        self._queue: "queue.Queue[Dict[str, float]]" = queue.Queue(maxsize=40)
        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started      = False

        # Replay position accessible from outside (read-only via property)
        self._current_idx = 0
        self._total_replayed = 0   # ever-increasing sample counter

        self._load()

    # --------------------------------------------------------------- loading

    def _load(self) -> None:
        """Parse and downsample the telemetry CSV."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"[TelemetryReplayer] CSV not found: {self.csv_path}\n"
                f"  Make sure the UL-DD dataset is accessible and the path "
                f"points to the Telemetry file for a subject/session."
            )

        df = pd.read_csv(str(self.csv_path))
        avail = df.columns.tolist()

        # --- pick columns by preferred name, then fallback partial-match ---
        selected_raw: List[str] = []
        selected_keys: List[str] = []
        for raw_col, short_key in zip(_TELE_COLS_PREFERRED, _TELE_KEYS_SHORT):
            if raw_col in avail:
                selected_raw.append(raw_col)
                selected_keys.append(short_key)
            else:
                # Try to match the base name without units (e.g., 'pitch')
                base = raw_col.split('[')[0]
                matches = [c for c in avail if base in c.lower()]
                if matches:
                    selected_raw.append(matches[0])
                    selected_keys.append(short_key)

        if not selected_raw:
            raise ValueError(
                f"[TelemetryReplayer] Could not find any expected columns in "
                f"{self.csv_path.name}.\n"
                f"  Expected (any of): {_TELE_COLS_PREFERRED}\n"
                f"  Found            : {avail[:15]}"
            )

        data = df[selected_raw].values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Downsample 60 Hz → 4 Hz (keep every 15th row)
        self._data      = data[:: self._DOWNSAMPLE]
        self._col_keys  = selected_keys

        dur = len(self._data) / self.TARGET_HZ
        print(
            f"[TelemetryReplayer] {self.subject_label} — "
            f"{len(self._data)} samples @ {self.TARGET_HZ} Hz  "
            f"({dur:.0f} s = {dur/60:.1f} min)  |  "
            f"cols={selected_keys}"
        )

    # -------------------------------------------------------- background I/O

    def _stream_worker(self) -> None:
        """Emit one sample every (1 / TARGET_HZ / speed_factor) seconds."""
        interval = (1.0 / self.TARGET_HZ) / self.speed_factor
        idx = 0

        while not self._stop_event.is_set():
            row   = self._data[idx]
            sample: Dict[str, float] = {
                k: float(v) for k, v in zip(self._col_keys, row)
            }
            # Guarantee all 5 standard keys exist even if CSV was missing some
            for k in _TELE_KEYS_SHORT:
                sample.setdefault(k, 0.0)

            # Store replay position
            self._current_idx    = idx
            self._total_replayed += 1

            # Push to queue; drop oldest item if consumer is lagging
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass

            idx += 1
            if idx >= len(self._data):
                if self.loop:
                    idx = 0          # wrap around
                else:
                    break            # file exhausted, stop thread

            time.sleep(interval)

    # ------------------------------------------------------ public interface

    def start(self) -> "TelemetryReplayer":
        """Start the background streaming thread. Returns self for chaining."""
        if self._started:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._stream_worker,
            name="TelemetryReplayThread",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        print(f"[TelemetryReplayer] Streaming started  "
              f"(loop={self.loop}, speed={self.speed_factor}x)")
        return self

    def stop(self) -> None:
        """Stop the background thread gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._started = False
        print("[TelemetryReplayer] Stopped.")

    def get_latest(self, timeout: float = 0.0) -> Optional[Dict[str, float]]:
        """
        Fetch the next telemetry sample from the queue.

        Parameters
        ----------
        timeout : float
            Seconds to block if queue is empty.  0 = non-blocking.

        Returns
        -------
        dict with keys ``pitch, roll, speed, rpm, gear``  — or ``None`` if empty.
        """
        try:
            return self._queue.get(block=(timeout > 0), timeout=timeout)
        except queue.Empty:
            return None

    def drain_to_array(self) -> np.ndarray:
        """
        Drain every queued sample into a ``(N, 5)`` float32 array.

        Call this once per inference step to collect all samples that arrived
        since the last call, then push them into your ring buffer.

        Returns
        -------
        np.ndarray  shape (N, 5) — columns: pitch, roll, speed, rpm, gear.
                    Shape (0, 5) if queue was empty.
        """
        rows: List[List[float]] = []
        while True:
            try:
                s = self._queue.get_nowait()
                rows.append([s['pitch'], s['roll'], s['speed'], s['rpm'], s['gear']])
            except queue.Empty:
                break
        if rows:
            return np.array(rows, dtype=np.float32)
        return np.empty((0, 5), dtype=np.float32)

    # -------------------------------------------- context-manager support

    def __enter__(self) -> "TelemetryReplayer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ---------------------------------------------------------- properties

    @property
    def n_samples(self) -> int:
        """Total 4 Hz samples available in the file."""
        return len(self._data)

    @property
    def duration_seconds(self) -> float:
        """Duration of the loaded telemetry in seconds."""
        return self.n_samples / self.TARGET_HZ

    @property
    def progress_pct(self) -> float:
        """Replay progress 0–100% (based on last emitted index)."""
        if self.n_samples == 0:
            return 0.0
        return (self._current_idx / self.n_samples) * 100.0


# ---------------------------------------------------------------------------
# RingBuffer — fixed-length deque over 2D feature arrays
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    Fixed-length sliding window over sequential feature vectors.

    Designed to accumulate ``(n_features,)`` rows until the buffer is
    full (``capacity`` timesteps), then return a ``(capacity, n_features)``
    numpy array ready to be fed into a BiLSTM.

    Parameters
    ----------
    capacity : int
        Number of timesteps to keep.  For 60 s @ 4 Hz: capacity=240.
    n_features : int
        Feature dimensionality of each row.
    fill_value : float
        Value used to initialise empty slots (default 0.0).
    """

    def __init__(self, capacity: int, n_features: int, fill_value: float = 0.0):
        self.capacity   = capacity
        self.n_features = n_features
        self._buf: deque = deque(
            [np.full(n_features, fill_value, dtype=np.float32)] * capacity,
            maxlen=capacity,
        )

    def push(self, row: np.ndarray) -> None:
        """Append one ``(n_features,)`` vector, dropping the oldest entry."""
        assert len(row) == self.n_features, (
            f"RingBuffer expects {self.n_features}-d row, got {len(row)}"
        )
        self._buf.append(row.astype(np.float32))

    def push_batch(self, rows: np.ndarray) -> None:
        """Push multiple rows ``(N, n_features)`` at once."""
        for row in rows:
            self.push(row)

    def get(self) -> np.ndarray:
        """Return the current window as ``(capacity, n_features)`` float32."""
        return np.stack(list(self._buf), axis=0)

    def is_ready(self) -> bool:
        """Always True once the buffer has been filled for the first time."""
        return len(self._buf) == self.capacity

    def reset(self, fill_value: float = 0.0) -> None:
        """Clear buffer and re-fill with ``fill_value``."""
        self._buf = deque(
            [np.full(self.n_features, fill_value, dtype=np.float32)] * self.capacity,
            maxlen=self.capacity,
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="TelemetryReplayer self-test")
    ap.add_argument("csv", help="Path to a UL-DD Telemetry CSV file")
    ap.add_argument("--speed", type=float, default=4.0,
                    help="Playback speed factor (default 4 = 4× real-time)")
    ap.add_argument("--samples", type=int, default=20,
                    help="Number of 4-Hz samples to print")
    args = ap.parse_args()

    replayer = TelemetryReplayer(args.csv, loop=False, speed_factor=args.speed)
    replayer.start()

    buf = RingBuffer(capacity=240, n_features=5)

    printed = 0
    while printed < args.samples:
        s = replayer.get_latest(timeout=0.5)
        if s is None:
            print("[self-test] Queue empty — file may have ended.")
            break
        row = np.array([s['pitch'], s['roll'], s['speed'], s['rpm'], s['gear']])
        buf.push(row)
        print(f"[{printed:03d}]  {s}  | buf_shape={buf.get().shape}")
        printed += 1

    replayer.stop()
    print(f"\nRingBuffer snapshot (last 3 rows):\n{buf.get()[-3:]}")
