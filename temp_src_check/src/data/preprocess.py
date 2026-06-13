"""
Data Preprocessing Pipeline — UL-DD Multimodal Dataset
University of Louisiana Drowsiness Detection (UL-DD)

Modalities : Facial Action Units (FAU, 30 features @ 60 Hz)
             Driving Telemetry (pitch, roll, speed, rpm, gear @ 60 Hz)
Labels     : KSS 1-9 -> 3 classes (Alert=0, Low Vigilant=1, Drowsy=2)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# UL-DD Dataset Processor  (Main thesis dataset)
# ─────────────────────────────────────────────

class ULDDProcessor:
    """
    Preprocessor for the University of Louisiana Drowsiness Detection (UL-DD) Dataset.

    Reference: "Multimodal Driver Drowsiness Detection" — UL-DD paper (cite in thesis).
    Usage   : Research / academic use — cite the dataset.

    Dataset layout (triple-nested archive structure):
        UL-DD/
        ├── Labels.csv                                 ← KSS self-ratings
        ├── Extracted_Features/Extracted_Features/Extracted_Features/
        │       └── {Subject}/{Session}/
        │               ├── {S}_FAU_{session}.csv      ← 30 Facial Action Units @ 60 Hz
        │               └── ...
        └── CSV_Files/CSV_Files/CSV_Files/
                └── {Subject}/{Session}/
                        ├── {S}_Telemetry_{session}.csv ← CAN signals @ 60 Hz
                        └── ...

    Subjects   : A – S (19 subjects)
    Sessions   : A = Awake/Alert  |  D = Drowsy
    Awake-only : C, F, L  (no Drowsy session)
    No telemetry: A       (Telemetry CSV missing for subject A)

    Processing:
        1. Downsample both signals 60 Hz → 4 Hz  (keep every 15th sample)
        2. Sliding windows  : WINDOW_SEC  = 60 s  → 240 timesteps
           Stride           : STRIDE_SEC  = 15 s  →  60 samples
        3. Label each window from its centre-time KSS epoch (1 label per 4 min)
           KSS 1–3 → 0 (Alert)  |  KSS 4–6 → 1 (Low Vigilant)  |  KSS 7–9 → 2 (Drowsy)
        4. Subject-independent 5-fold CV  (FOLDS attribute)

    Output (.npz per fold, saved to processed_dir/):
        M1  (FAU-only)  : X_fau_train, X_fau_test, y_train, y_test
        M2/M3 (aligned) : mm_fau_train, mm_tele_train, mm_y_train,
                          mm_fau_test,  mm_tele_test,  mm_y_test
        Scalers         : fau_mean, fau_std, tele_mean, tele_std
    """

    # ── Dataset constants ────────────────────────────────────────────────────

    SUBJECTS    = list('ABCDEFGHIJKLMNOPQRS')      # 19 subjects
    SESSION_MAP = {'Alert': 'A', 'Drowsy': 'D'}    # Labels.csv key → folder
    AWAKE_ONLY  = frozenset({'C', 'F', 'L'})        # Only Alert session exists
    NO_TELEMETRY = frozenset({'A'})                 # Telemetry CSV missing

    FAU_COLS = [
        'inner_brow_raiser', 'outer_brow_raiser', 'brow_lowerer',
        'upper_lid_raiser', 'cheek_raiser_feature', 'lid_tightener',
        'nose_wrinkler', 'upper_lip_raiser', 'nasolabial_furrow_deepener',
        'lip_corner_puller', 'cheek_puffer', 'dimpler', 'lip_corner_depressor',
        'lower_lip_depressor', 'chin_raiser', 'lip_puckerer', 'lip_stretcher',
        'lip_funneler', 'lip_tightener', 'lip_pressor', 'lips_part', 'jaw_drop',
        'mouth_stretch', 'lip_suck', 'lid_droop', 'slit', 'eye_closed',
        'squint', 'blink', 'wink',
    ]   # 30 Facial Action Unit features

    TELE_COLS = ['pitch[deg]', 'roll[deg]', 'speed[m/s]', 'rpm', 'gear']  # 5 CAN features

    # ── Processing hyperparameters ───────────────────────────────────────────

    ORIG_HZ           = 60   # Raw sampling rate of both signals (fps)
    TARGET_HZ         = 4    # Target rate after downsampling
    DOWNSAMPLE_FACTOR = 15   # ORIG_HZ // TARGET_HZ

    WINDOW_SEC = 60   # Window length  →  60 × 4 = 240 timesteps
    STRIDE_SEC = 15   # Window stride  →  15 × 4 =  60 samples
    EPOCH_SEC  = 240  # KSS label epoch duration  (4 minutes)

    N_CLASSES      = 3
    KSS_THRESHOLDS = (3, 6)  # ≤3 → Alert, ≤6 → Low Vigilant, >6 → Drowsy

    # Subject-independent 5-fold split  (held-out test subjects per fold)
    FOLDS = [
        list('ABCD'),   # Fold 0
        list('EFGH'),   # Fold 1
        list('IJK'),    # Fold 2
        list('LMNO'),   # Fold 3
        list('PQRS'),   # Fold 4
    ]

    CLASS_NAMES = ['Alert', 'LowVigilant', 'Drowsy']

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self, dataset_root: str, processed_dir: str):
        """
        Args:
            dataset_root  : Path to UL-DD root folder.
                            e.g. r'C:/Users/.../UL-DD'
            processed_dir : Output directory for fold .npz files.
                            e.g. 'datasets/processed/ul_dd'
        """
        self.root          = Path(dataset_root)
        self.processed_dir = Path(processed_dir)

        self.fau_root  = (self.root / 'Extracted_Features'
                                    / 'Extracted_Features'
                                    / 'Extracted_Features')
        self.tele_root = (self.root / 'CSV_Files'
                                    / 'CSV_Files'
                                    / 'CSV_Files')
        self.labels_path = self.root / 'Labels.csv'

        # Pre-compute window / stride in samples at TARGET_HZ
        self._win    = self.WINDOW_SEC * self.TARGET_HZ   # 240 samples
        self._stride = self.STRIDE_SEC * self.TARGET_HZ   #  60 samples

        self._labels_cache: dict = {}

    # ── Label utilities ───────────────────────────────────────────────────────

    def _kss_to_class(self, kss: float) -> int:
        """Map KSS score (1–9) to 3-class integer label."""
        if kss <= self.KSS_THRESHOLDS[0]:
            return 0   # Alert
        elif kss <= self.KSS_THRESHOLDS[1]:
            return 1   # Low Vigilant
        else:
            return 2   # Drowsy

    def load_labels(self) -> dict:
        """
        Parse Labels.csv (no header) into a nested dict.

        File format:
            A_Alert,4,4,4,5,5,5,5,6,6,6
            A_Drowsy,6,7,7,7,8,8,7,7,8,8
            ...

        Returns:
            {subject: {session_folder: [class_0, …, class_9]}}
            e.g. {'A': {'A': [0,0,0,1,1,1,1,1,1,1], 'D': [1,2,2,2,2,2,2,2,2,2]}, …}
        """
        if self._labels_cache:
            return self._labels_cache

        labels: dict = {}
        with open(self.labels_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts       = line.split(',')
                key         = parts[0]                     # 'A_Alert'
                kss_vals    = [float(v) for v in parts[1:]]
                classes     = [self._kss_to_class(k) for k in kss_vals]
                subj, sess_name = key.split('_', 1)        # 'A', 'Alert'
                sess_folder     = self.SESSION_MAP[sess_name]  # 'A' or 'D'
                labels.setdefault(subj, {})[sess_folder] = classes

        self._labels_cache = labels
        return labels

    # ── Modality loaders ──────────────────────────────────────────────────────

    def _load_fau(self, subject: str, session: str):
        """
        Load FAU CSV and downsample to TARGET_HZ.

        Returns:
            np.ndarray  shape (N_ds, 30)  or  None if file is missing.
        """
        path = (self.fau_root / subject / session
                / f"{subject}_FAU_{session}.csv")
        if not path.exists():
            return None
        df   = pd.read_csv(str(path), usecols=self.FAU_COLS)
        data = df.values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data[::self.DOWNSAMPLE_FACTOR]          # 60 Hz → 4 Hz

    def _load_telemetry(self, subject: str, session: str):
        """
        Load Telemetry CSV and downsample to TARGET_HZ.

        Returns:
            np.ndarray  shape (N_ds, 5)  or  None if unavailable.
        """
        if subject in self.NO_TELEMETRY:
            return None
        path = (self.tele_root / subject / session
                / f"{subject}_Telemetry_{session}.csv")
        if not path.exists():
            return None
        df   = pd.read_csv(str(path), usecols=self.TELE_COLS)
        data = df.values.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data[::self.DOWNSAMPLE_FACTOR]          # 60 Hz → 4 Hz

    # ── Windowing ────────────────────────────────────────────────────────────

    def _epoch_label(self, start_sample: int, epoch_classes: list) -> int:
        """Return the class label for the window whose start is `start_sample`."""
        centre_sec = (start_sample + self._win // 2) / self.TARGET_HZ
        epoch_idx  = int(centre_sec // self.EPOCH_SEC)
        epoch_idx  = min(epoch_idx, len(epoch_classes) - 1)
        return epoch_classes[epoch_idx]

    def _sliding_windows(self, fau_4hz: np.ndarray, epoch_classes: list,
                         tele_4hz=None):
        """
        Apply sliding window over 4-Hz downsampled arrays.

        Args:
            fau_4hz      : (T, 30)  FAU signal at 4 Hz
            epoch_classes: list of 10 class labels (one per KSS epoch)
            tele_4hz     : (T, 5)   Telemetry at 4 Hz, or None

        Returns:
            fau_wins  : (W, window, 30)  float32
            tele_wins : (W, window,  5)  float32   or  None
            labels    : (W,)             int32
        """
        n_samples = fau_4hz.shape[0]
        starts    = range(0, n_samples - self._win + 1, self._stride)

        fau_list, tele_list, lbl_list = [], [], []
        for s in starts:
            e = s + self._win
            fau_list.append(fau_4hz[s:e])
            lbl_list.append(self._epoch_label(s, epoch_classes))
            if tele_4hz is not None:
                tele_list.append(tele_4hz[s:e])

        fau_arr  = np.stack(fau_list).astype(np.float32)   # (W, T, 30)
        tele_arr = (np.stack(tele_list).astype(np.float32)
                    if tele_list else None)                  # (W, T,  5)
        lbl_arr  = np.array(lbl_list, dtype=np.int32)      # (W,)
        return fau_arr, tele_arr, lbl_arr

    # ── Per-session processing ────────────────────────────────────────────────

    def process_session(self, subject: str, session: str,
                        epoch_classes: list):
        """
        Load + window one (subject, session) pair.

        Returns:
            dict with keys:
                'fau'    : (W, 240, 30)
                'tele'   : (W, 240,  5)  or  None
                'labels' : (W,)
                'subject': str
                'session': str
            or None if the FAU file is missing.
        """
        fau_4hz = self._load_fau(subject, session)
        if fau_4hz is None:
            return None

        tele_4hz = self._load_telemetry(subject, session)

        # Align lengths in case of off-by-one after integer downsampling
        if tele_4hz is not None:
            n = min(len(fau_4hz), len(tele_4hz))
            fau_4hz  = fau_4hz[:n]
            tele_4hz = tele_4hz[:n]

        fau_w, tele_w, labels = self._sliding_windows(
            fau_4hz, epoch_classes, tele_4hz
        )
        return {
            'fau':     fau_w,
            'tele':    tele_w,
            'labels':  labels,
            'subject': subject,
            'session': session,
        }

    # ── Full dataset scan ─────────────────────────────────────────────────────

    def process_all(self, verbose: bool = True) -> list:
        """
        Process every available (subject, session) pair.

        Returns:
            List of session dicts (one per processed pair).
        """
        all_labels = self.load_labels()
        results    = []

        for subj in self.SUBJECTS:
            if subj not in all_labels:
                if verbose:
                    print(f"  [SKIP] {subj}: not in Labels.csv")
                continue

            for sess_name, sess_folder in self.SESSION_MAP.items():
                if subj in self.AWAKE_ONLY and sess_folder == 'D':
                    continue                      # No Drowsy session for this subject
                if sess_folder not in all_labels[subj]:
                    continue

                epoch_cls = all_labels[subj][sess_folder]

                if verbose:
                    print(f"  {subj}/{sess_folder} ({sess_name:6s}) ... ",
                          end='', flush=True)

                rec = self.process_session(subj, sess_folder, epoch_cls)

                if rec is None:
                    if verbose:
                        print("FAU missing — skipped")
                    continue

                results.append(rec)

                if verbose:
                    W      = len(rec['labels'])
                    dist   = np.bincount(rec['labels'], minlength=3).tolist()
                    tflag  = '✓' if rec['tele'] is not None else '✗'
                    print(f"{W:3d} windows | tele={tflag} | "
                          f"cls(0/1/2)={dist}")

        return results

    # ── Normalisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _fit_scaler(arrays: list):
        """
        Fit per-feature z-score scaler from a list of (W, T, F) arrays.

        Returns:
            mu  : (F,)  mean
            sig : (F,)  std + ε
        """
        flat = np.concatenate(
            [a.reshape(-1, a.shape[-1]) for a in arrays], axis=0
        )
        mu  = flat.mean(axis=0)
        sig = flat.std(axis=0) + 1e-8
        return mu, sig

    @staticmethod
    def _apply_scaler(arr: np.ndarray, mu: np.ndarray,
                      sig: np.ndarray) -> np.ndarray:
        """Apply z-score normalisation in-place and return float32."""
        return ((arr - mu) / sig).astype(np.float32)

    # ── 5-fold assembly ───────────────────────────────────────────────────────

    def build_folds(self, all_results: list, verbose: bool = True) -> list:
        """
        Assemble 5 subject-independent folds from the processed session list.

        For each fold k:
            - Test  subjects : FOLDS[k]
            - Train subjects : everyone else

        Scalers are fitted only on training data.

        Each returned fold dict contains:
            M1 (FAU-only, all sessions):
                X_fau_train, X_fau_test, y_train, y_test
            M2/M3 (aligned FAU + Telemetry, sessions that have both):
                mm_fau_train, mm_tele_train, mm_y_train
                mm_fau_test,  mm_tele_test,  mm_y_test
            Scalers (fit on train):
                fau_mean, fau_std, tele_mean, tele_std   (or None if no tele)

        Returns list of 5 fold dicts.
        """
        folds = []

        for k, test_subjs in enumerate(self.FOLDS):
            test_set  = frozenset(test_subjs)
            train_res = [r for r in all_results if r['subject'] not in test_set]
            test_res  = [r for r in all_results if r['subject'] in test_set]

            # ── M1: FAU-only (use ALL sessions regardless of telemetry) ──────
            X_fau_tr = np.concatenate([r['fau'] for r in train_res], axis=0)
            X_fau_te = np.concatenate([r['fau'] for r in test_res],  axis=0)
            y_tr     = np.concatenate([r['labels'] for r in train_res], axis=0)
            y_te     = np.concatenate([r['labels'] for r in test_res],  axis=0)

            fau_mu, fau_sig = self._fit_scaler([r['fau'] for r in train_res])
            X_fau_tr = self._apply_scaler(X_fau_tr, fau_mu, fau_sig)
            X_fau_te = self._apply_scaler(X_fau_te, fau_mu, fau_sig)

            # ── M2/M3: sessions that have BOTH FAU and Telemetry ─────────────
            mm_tr = [r for r in train_res if r['tele'] is not None]
            mm_te = [r for r in test_res  if r['tele'] is not None]

            if mm_tr and mm_te:
                mm_fau_tr  = np.concatenate([r['fau']    for r in mm_tr], axis=0)
                mm_tele_tr = np.concatenate([r['tele']   for r in mm_tr], axis=0)
                mm_y_tr    = np.concatenate([r['labels'] for r in mm_tr], axis=0)
                mm_fau_te  = np.concatenate([r['fau']    for r in mm_te], axis=0)
                mm_tele_te = np.concatenate([r['tele']   for r in mm_te], axis=0)
                mm_y_te    = np.concatenate([r['labels'] for r in mm_te], axis=0)

                # FAU normalised with the same scaler fitted on all train FAU
                mm_fau_tr = self._apply_scaler(mm_fau_tr, fau_mu, fau_sig)
                mm_fau_te = self._apply_scaler(mm_fau_te, fau_mu, fau_sig)

                # Telemetry scaler fitted only on MM train data
                tele_mu, tele_sig = self._fit_scaler([r['tele'] for r in mm_tr])
                mm_tele_tr = self._apply_scaler(mm_tele_tr, tele_mu, tele_sig)
                mm_tele_te = self._apply_scaler(mm_tele_te, tele_mu, tele_sig)
            else:
                mm_fau_tr = mm_tele_tr = mm_y_tr = None
                mm_fau_te = mm_tele_te = mm_y_te = None
                tele_mu = tele_sig = None

            fold = {
                'fold':          k,
                'test_subjects': test_subjs,
                # M1
                'X_fau_train':   X_fau_tr,
                'X_fau_test':    X_fau_te,
                'y_train':       y_tr,
                'y_test':        y_te,
                # M2/M3 (aligned)
                'mm_fau_train':  mm_fau_tr,
                'mm_tele_train': mm_tele_tr,
                'mm_y_train':    mm_y_tr,
                'mm_fau_test':   mm_fau_te,
                'mm_tele_test':  mm_tele_te,
                'mm_y_test':     mm_y_te,
                # Scalers
                'fau_mean':      fau_mu,
                'fau_std':       fau_sig,
                'tele_mean':     tele_mu,
                'tele_std':      tele_sig,
            }
            folds.append(fold)

            if verbose:
                mm_n_tr = len(mm_y_tr) if mm_y_tr is not None else 0
                mm_n_te = len(mm_y_te) if mm_y_te is not None else 0
                print(f"  Fold {k} | test={test_subjs} | "
                      f"M1 train={len(y_tr):4d} test={len(y_te):4d} | "
                      f"MM train={mm_n_tr:4d} test={mm_n_te:4d}")

        return folds

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_folds(self, folds: list) -> None:
        """Save each fold to a compressed .npz in processed_dir."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for fold in folds:
            k    = fold['fold']
            save = {}
            for key, val in fold.items():
                if isinstance(val, np.ndarray):
                    save[key] = val
                elif key == 'test_subjects' and isinstance(val, list):
                    save[key] = np.array(val, dtype='U1')   # unicode char array
            path = self.processed_dir / f"fold_{k}.npz"
            np.savez_compressed(str(path), **save)

            n_m1 = len(fold['y_train'])
            n_mm = (len(fold['mm_y_train'])
                    if fold['mm_y_train'] is not None else 0)
            print(f"  Saved fold_{k}.npz  "
                  f"(M1_train={n_m1} | MM_train={n_mm})")

    @staticmethod
    def load_fold(processed_dir: str, fold_idx: int) -> dict:
        """Load a fold .npz and return as a plain dict of numpy arrays."""
        path = Path(processed_dir) / f"fold_{fold_idx}.npz"
        return dict(np.load(str(path), allow_pickle=False))

    # ── Convenience: class-distribution summary ───────────────────────────────

    @staticmethod
    def class_distribution(y: np.ndarray) -> str:
        """Return a formatted string of class counts."""
        counts = np.bincount(y, minlength=3)
        total  = counts.sum()
        parts  = [f"cls{i}={counts[i]:4d} ({counts[i]/total*100:4.1f}%)"
                  for i in range(3)]
        return '  '.join(parts)

    # ── High-level entry point ────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> list:
        """
        Full preprocessing pipeline:
            1. Load & window all (subject, session) pairs
            2. Assemble 5 subject-independent folds
            3. Save to .npz files in processed_dir

        Returns list of fold dicts.
        """
        sep = '─' * 60
        print(f"\n{sep}")
        print("  UL-DD Preprocessing Pipeline")
        print(f"  Root : {self.root}")
        print(f"  Out  : {self.processed_dir}")
        print(f"  Win  : {self.WINDOW_SEC}s  Stride: {self.STRIDE_SEC}s  "
              f"Rate: {self.TARGET_HZ} Hz")
        print(sep)

        print("\n[1/3]  Processing sessions …")
        all_results = self.process_all(verbose=verbose)

        n_sess = len(all_results)
        n_win  = sum(len(r['labels']) for r in all_results)
        n_mm   = sum(1 for r in all_results if r['tele'] is not None)
        print(f"\n       {n_sess} sessions | {n_win} windows | "
              f"{n_mm} sessions with telemetry\n")

        print("[2/3]  Building 5 subject-independent folds …")
        folds = self.build_folds(all_results, verbose=verbose)

        print(f"\n[3/3]  Saving folds to {self.processed_dir} …")
        self.save_folds(folds)

        print(f"\n✓  Done — fold_0.npz … fold_4.npz written.\n")
        return folds