"""
Data Preprocessing Pipeline for Driver Drowsiness Detection
Primary Dataset: MRL Eye (talhabhatti7262/drivers-drowsiness-detection — CC0)
                 ~170k eye-crop images, 37 subject dirs, label encoded in filename.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE   = (64, 64)    # Eye crops are small; 64×64 is sufficient and fast
VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

# MRL Eye label convention (encoded in filename last token):
#   0 = closed eye  →  drowsy indicator
#   1 = open eye    →  alert indicator
MRL_LABEL_MAP  = {0: 'Closed', 1: 'Open'}
MRL_CLASS_NAMES = ['Closed', 'Open']   # index matches label value


# ─────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────

def normalize_data(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def augment_data(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a single image.
    Augmentations: brightness, horizontal flip, slight rotation, occlusion patch.
    """
    # Random brightness ±30%
    factor = np.random.uniform(0.7, 1.3)
    image = np.clip(image * factor, 0.0, 1.0)

    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    # Random rotation ±15°
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    # Random occlusion patch (simulates glasses/shadow)
    if np.random.rand() > 0.7:
        px = np.random.randint(0, w - w // 4)
        py = np.random.randint(0, h - h // 4)
        pw = np.random.randint(w // 8, w // 4)
        ph = np.random.randint(h // 8, h // 4)
        image[py:py + ph, px:px + pw] = 0.0

    return image.astype(np.float32)


def preprocess_data(image: np.ndarray, augment: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for one image.
    1. Resize to IMG_SIZE
    2. Normalize to [0, 1]
    3. Optionally augment
    """
    image = cv2.resize(image, IMG_SIZE)
    image = normalize_data(image)
    if augment:
        image = augment_data(image)
    return image


def split_data(X, y, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split arrays into train / val / test sets.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    # Split the remainder equally between val and test
    val_ratio = val_size / (1.0 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_ratio, random_state=random_state, stratify=y_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─────────────────────────────────────────────
# MRL Eye Dataset Processor  (Primary)
# ─────────────────────────────────────────────

class MRLEyeProcessor:
    """
    Processor for the MRL Eye Dataset.

    License  : CC0 Public Domain — safe for MTech thesis.
    Download : https://www.kaggle.com/datasets/talhabhatti7262/drivers-drowsiness-detection

    Expected raw structure (your layout):
        raw_dir/                  e.g.  datasets/raw/mrleye/
        ├── open/                 ← open-eye images
        └── closed/               ← closed-eye images

    Class mapping:
        0 → Closed  (drowsy indicator)
        1 → Open    (alert indicator)
    """

    # Maps folder name → integer label
    FOLDER_LABEL_MAP = {'closed': 0, 'open': 1}

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir       = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    # ── Internal helpers ──────────────────────

    def _load_folder(self, folder: Path, label: int,
                     augment: bool = False, max_count: int = None):
        """Load all images from a single class folder."""
        images, labels = [], []
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            if max_count and len(images) >= max_count:
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_data(img, augment=augment)
            images.append(img)
            labels.append(label)
        return images, labels

    # ── Public API ────────────────────────────

    def load_dataset(self, augment_closed: bool = True, max_per_class: int = None):
        """
        Load images from open/ and closed/ sub-folders.

        Args:
            augment_closed : Augment closed-eye samples to help balance classes.
            max_per_class  : Cap images per class (use ~5000 for a quick test).

        Returns:
            X : np.ndarray  shape (N, 64, 64, 3)
            y : np.ndarray  shape (N,)  — 0=Closed, 1=Open
        """
        print(f"Loading MRL Eye dataset from: {self.raw_dir}")
        all_images, all_labels = [], []

        for folder_name, label in self.FOLDER_LABEL_MAP.items():
            folder = self.raw_dir / folder_name
            if not folder.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {folder}\n"
                    f"Make sure your structure is:\n"
                    f"  {self.raw_dir}/open/   and   {self.raw_dir}/closed/"
                )
            augment = augment_closed and (label == 0)  # only augment closed
            imgs, lbls = self._load_folder(folder, label,
                                           augment=augment,
                                           max_count=max_per_class)
            print(f"  {folder_name:8s} (label={label}): {len(imgs)} images loaded")
            all_images.extend(imgs)
            all_labels.extend(lbls)

        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        print(f"  Total: {len(X)} images  |  Shape: {X.shape}")
        return X, y

    def process_and_save(self, augment_closed: bool = True,
                         max_per_class: int = None):
        """
        Load → preprocess → split → save as .npy files.

        Saved files:
            datasets/processed/mrl_eye/
                X_train.npy, y_train.npy
                X_val.npy,   y_val.npy
                X_test.npy,  y_test.npy
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_dataset(augment_closed=augment_closed,
                                 max_per_class=max_per_class)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_data(X, y)

        splits = {
            'X_train': X_tr, 'y_train': y_tr,
            'X_val':   X_v,  'y_val':   y_v,
            'X_test':  X_te, 'y_test':  y_te,
        }
        for name, arr in splits.items():
            np.save(str(self.processed_dir / f"{name}.npy"), arr)
            print(f"  Saved {name}.npy  {arr.shape}")

        print(f"\n✓ MRL Eye preprocessing complete → {self.processed_dir}")
        return splits

    @staticmethod
    def load_processed(processed_dir: str):
        """Load pre-saved .npy splits from disk."""
        d = Path(processed_dir)
        X_tr = np.load(d / 'X_train.npy')
        y_tr = np.load(d / 'y_train.npy')
        X_v  = np.load(d / 'X_val.npy')
        y_v  = np.load(d / 'y_val.npy')
        X_te = np.load(d / 'X_test.npy')
        y_te = np.load(d / 'y_test.npy')
        print(f"Loaded MRL splits — "
              f"train:{len(X_tr)}  val:{len(X_v)}  test:{len(X_te)}")
        return (X_tr, y_tr), (X_v, y_v), (X_te, y_te)

    @staticmethod
    def to_tf_dataset(X: np.ndarray, y: np.ndarray,
                      batch_size: int = 64,
                      shuffle: bool = True) -> tf.data.Dataset:
        """Convert numpy arrays to a tf.data.Dataset for training."""
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(X), 10_000))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# Legacy / supplementary processors
# ─────────────────────────────────────────────

class DDDKaggleProcessor:
    """
    Processor for the Kaggle Driver Drowsiness Dataset (DDD).

    Expected raw structure:
        raw_dir/
        ├── Drowsy/          (22,300 images)
        └── Non Drowsy/      (19,400 images)

    Download from:
        https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
    """

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

    # ── Internal helpers ──────────────────────

    def _load_images_from_folder(self, folder: Path, label: int,
                                  augment: bool = False):
        """Load, preprocess, and label all images in a folder."""
        images, labels = [], []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_data(img, augment=augment)
            images.append(img)
            labels.append(label)

        return images, labels

    # ── Public API ────────────────────────────

    def load_dataset(self, augment_drowsy: bool = True):
        """
        Load the full DDD dataset into numpy arrays.

        Args:
            augment_drowsy: Whether to augment Drowsy samples to balance classes.

        Returns:
            X (np.ndarray): shape (N, 224, 224, 3)
            y (np.ndarray): shape (N,) — 0=Non Drowsy, 1=Drowsy
        """
        print("Loading Kaggle DDD dataset...")

        all_images, all_labels = [], []

        for class_name, label in LABEL_MAP.items():
            folder = self.raw_dir / class_name
            if not folder.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {folder}\n"
                    f"Make sure the dataset is extracted into: {self.raw_dir}"
                )
            # Apply augmentation to Drowsy class if requested (helps balance)
            aug = augment_drowsy and (label == 1)
            imgs, lbls = self._load_images_from_folder(folder, label, augment=aug)
            print(f"  {class_name}: {len(imgs)} images loaded")
            all_images.extend(imgs)
            all_labels.extend(lbls)

        X = np.array(all_images, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        print(f"  Total: {len(X)} images  |  Shape: {X.shape}")
        return X, y

    def process_and_save(self, augment_drowsy: bool = True):
        """
        Load → preprocess → split → save as .npy files in processed_dir.

        Saved files:
            X_train.npy, y_train.npy
            X_val.npy,   y_val.npy
            X_test.npy,  y_test.npy
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        X, y = self.load_dataset(augment_drowsy=augment_drowsy)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_data(X, y)

        splits = {
            'X_train': X_tr, 'y_train': y_tr,
            'X_val':   X_v,  'y_val':   y_v,
            'X_test':  X_te, 'y_test':  y_te,
        }
        for name, arr in splits.items():
            path = self.processed_dir / f"{name}.npy"
            np.save(str(path), arr)
            print(f"  Saved {name}.npy  {arr.shape}")

        print(f"\n✓ Preprocessing complete → {self.processed_dir}")
        return splits

    @staticmethod
    def load_processed(processed_dir: str):
        """
        Load pre-saved .npy splits from disk.

        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        d = Path(processed_dir)
        X_tr = np.load(d / 'X_train.npy')
        y_tr = np.load(d / 'y_train.npy')
        X_v  = np.load(d / 'X_val.npy')
        y_v  = np.load(d / 'y_val.npy')
        X_te = np.load(d / 'X_test.npy')
        y_te = np.load(d / 'y_test.npy')
        print(f"Loaded splits — train:{len(X_tr)}  val:{len(X_v)}  test:{len(X_te)}")
        return (X_tr, y_tr), (X_v, y_v), (X_te, y_te)

    @staticmethod
    def to_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 32,
                      shuffle: bool = True) -> tf.data.Dataset:
        """Convert numpy arrays to a tf.data.Dataset for training."""
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


# ─────────────────────────────────────────────
# MRL Eye & CEW processors (lightweight wrappers)
# ─────────────────────────────────────────────

class DatasetProcessor:
    """
    Unified processor. Delegates to the appropriate class-specific processor.
    """

    @staticmethod
    def process_ddd_kaggle_dataset(raw_dir: str, processed_dir: str,
                                    augment_drowsy: bool = True):
        """Process Kaggle DDD dataset (primary dataset for this project)."""
        proc = DDDKaggleProcessor(raw_dir, processed_dir)
        return proc.process_and_save(augment_drowsy=augment_drowsy)

    @staticmethod
    def process_mrl_eye_dataset(raw_dir: str, processed_dir: str):
        """
        Process the official MRL Eye dataset (mrl.cs.vsb.cz/eyedataset).
        84,898 eye images across 37 subject directories.

        File naming: s<subject>_<seq>_<frame>_<state>_<...>.png
            state in filename: last numeric token — 0=closed, 1=open

        Download: https://mrl.cs.vsb.cz/eyedataset
        """
        out = Path(processed_dir)
        out.mkdir(parents=True, exist_ok=True)
        root = Path(raw_dir)
        images, labels = [], []
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        skipped = 0

        # Handle both flat structure and subject-subdirectory structure
        all_img_paths = []
        for item in root.iterdir():
            if item.is_dir():
                all_img_paths.extend(
                    p for p in item.iterdir() if p.suffix.lower() in valid_exts
                )
            elif item.suffix.lower() in valid_exts:
                all_img_paths.append(item)

        for img_path in all_img_paths:
            stem_parts = img_path.stem.split('_')
            try:
                label = int(stem_parts[-1])
                if label not in (0, 1):
                    skipped += 1
                    continue
            except (ValueError, IndexError):
                skipped += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_data(img)
            images.append(img)
            labels.append(label)

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        np.save(out / 'X_mrl.npy', X)
        np.save(out / 'y_mrl.npy', y)
        print(f"✓ MRL Eye: {len(X)} images | open:{np.sum(y==1)}  closed:{np.sum(y==0)}")
        if skipped:
            print(f"   Skipped {skipped} files")
        return X, y

    @staticmethod
    def process_cew_dataset(raw_dir: str, processed_dir: str):
        """
        Process CEW (Closed Eyes in the Wild) dataset.

        Supports two folder naming conventions automatically:
          - Kaggle version (ahamedfarouk/cew-dataset):  open/ and closed/
          - Original CEW:                               open_eyes/ and closed_eyes/

        Download (Kaggle): https://www.kaggle.com/datasets/ahamedfarouk/cew-dataset
        NOTE: License is 'Unknown' on Kaggle — for MTech thesis use the
              MRL Eye dataset (CC0) or the original CEW from the paper instead.
        """
        out = Path(processed_dir)
        out.mkdir(parents=True, exist_ok=True)
        raw = Path(raw_dir)

        # Auto-detect folder naming convention
        if (raw / 'open').exists():
            folder_map = {'open': 0, 'closed': 1}          # Kaggle version
        else:
            folder_map = {'open_eyes': 0, 'closed_eyes': 1} # Original CEW

        print(f"CEW folder convention detected: {list(folder_map.keys())}")
        images, labels = [], []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for folder_name, label in folder_map.items():
            folder = raw / folder_name
            if not folder.exists():
                print(f"  [skip] {folder} not found")
                continue
            count = 0
            for img_path in folder.iterdir():
                if img_path.suffix.lower() not in valid_exts:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_data(img)
                images.append(img)
                labels.append(label)
                count += 1
            print(f"  {folder_name}: {count} images")
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        np.save(out / 'X_cew.npy', X)
        np.save(out / 'y_cew.npy', y)
        print(f"✓ CEW: {len(X)} images saved to {out}")
        return X, y

    @staticmethod
    def process_mrl_eye_kaggle_dataset(raw_dir: str, processed_dir: str):
        """
        Process the MRL Eye dataset from Kaggle (talhabhatti7262/drivers-drowsiness-detection).
        License: CC0 Public Domain — safe for MTech thesis.

        Download: https://www.kaggle.com/datasets/talhabhatti7262/drivers-drowsiness-detection
        Extract so that the path looks like:
            raw_dir/mrlEyes_2018_01/<subject_dirs>/

        MRL Eye file naming convention:
            s<subject>_<session>_<frame>_<label>.png
            where label: 0 = closed eye, 1 = open eye
        """
        out = Path(processed_dir)
        out.mkdir(parents=True, exist_ok=True)

        mrl_root = Path(raw_dir) / 'mrlEyes_2018_01'
        if not mrl_root.exists():
            # fallback: treat raw_dir itself as the root
            mrl_root = Path(raw_dir)

        images, labels = [], []
        valid_exts = {'.png', '.jpg', '.jpeg'}
        skipped = 0

        for subject_dir in sorted(mrl_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for img_path in subject_dir.iterdir():
                if img_path.suffix.lower() not in valid_exts:
                    continue
                # Extract label from filename: last part before extension
                # e.g. s0001_00001_0_1_01.png → label = parts[-1]
                stem_parts = img_path.stem.split('_')
                try:
                    label = int(stem_parts[-1])   # 0=closed, 1=open
                    if label not in (0, 1):
                        skipped += 1
                        continue
                except (ValueError, IndexError):
                    skipped += 1
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_data(img)
                images.append(img)
                labels.append(label)

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        np.save(out / 'X_mrl_kaggle.npy', X)
        np.save(out / 'y_mrl_kaggle.npy', y)
        print(f"✓ MRL Eye (Kaggle): {len(X)} images | open:{np.sum(y==1)}  closed:{np.sum(y==0)}")
        if skipped:
            print(f"   Skipped {skipped} files (unrecognised label format)")
        return X, y