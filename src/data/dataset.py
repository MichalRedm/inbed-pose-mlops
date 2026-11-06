"""Dataset implementation for the Simultaneously-collected Multimodal Lying Pose (SLP) dataset."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset, default_collate

from src.data.augmentations import PoseAugmenter
from src.utils.heatmaps import generate_gaussian_heatmaps


def collate_pose(
    batch: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Custom collate function for labeled pose batches, filtering out None targets."""
    clean = [
        item for item in batch if item is not None and item.get("target") is not None
    ]
    if not clean:
        return None
    return cast(dict[str, Any], default_collate(clean))


def collate_unlabeled(
    batch: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Custom collate function for unlabeled/covered batches, stripping None entries."""
    clean = []
    for item in batch:
        if item is None:
            continue
        filtered = {k: v for k, v in item.items() if v is not None}
        clean.append(filtered)
    if not clean:
        return None
    return cast(dict[str, Any], default_collate(clean))


class SLPDataset(Dataset[dict[str, Any]]):
    """Simultaneously-collected Multimodal Lying Pose (SLP) dataset.

    Supports multimodal inputs (RGB, IR) and blanket cover types (uncover, cover1, cover2).
    """

    def __init__(
        self,
        root: str | Path,
        subjects: Iterable[int] = range(1, 31),
        modalities: list[str] | None = None,
        covers: list[str] | None = None,
        split: str = "train",
        augmenter: PoseAugmenter | None = None,
        image_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        return_heatmaps: bool = True,
    ) -> None:
        self.root = Path(root)
        self.subjects = list(subjects)
        self.modalities = modalities or ["IR"]
        self.covers = covers or ["uncover"]
        self.split = split
        self.augmenter = augmenter
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.return_heatmaps = return_heatmaps

        # Compute expected input channels: 3 for RGB, 1 for IR, 4 for combined
        self.in_channels = sum(3 if m.upper() == "RGB" else 1 for m in self.modalities)

        self.samples = self._prepare_samples()

    def set_sigma(self, sigma: float) -> None:
        """Dynamically update Gaussian standard deviation."""
        self.sigma = sigma

    def _prepare_samples(self) -> list[dict[str, Any]]:
        """Scans directory hierarchy and collects aligned multimodal image paths and annotations."""
        samples: list[dict[str, Any]] = []
        split_candidates = (
            ["valid", "train", ""] if self.split == "valid" else ["train", "valid", ""]
        )

        for subject_id in self.subjects:
            subject_names = [f"{subject_id:05d}", f"Subject_{subject_id:02d}"]
            subj_dir: Path | None = None

            for sn in subject_names:
                for candidate in split_candidates:
                    p = self.root / candidate / sn if candidate else self.root / sn
                    if p.exists() and any(p.iterdir()):
                        subj_dir = p
                        break
                if subj_dir:
                    break

            if not subj_dir:
                continue

            # Load annotations if available
            annotations: dict[str, np.ndarray | None] = {}
            for mod in self.modalities:
                mat_path = subj_dir / f"joints_gt_{mod}.mat"
                if mat_path.exists():
                    try:
                        mat_data = sio.loadmat(str(mat_path))
                        # joints_gt is (3, 14, N)
                        ann = cast(np.ndarray, mat_data["joints_gt"]).copy()
                        # SLP 1-based indexing correction
                        ann[:2, :, :] -= 1.0
                        annotations[mod] = ann
                    except Exception:
                        annotations[mod] = None
                else:
                    annotations[mod] = None

            # Collect image paths per cover
            for cover in self.covers:
                mod_paths: dict[str, list[Path]] = {}
                for mod in self.modalities:
                    possible_dirs = [
                        subj_dir / mod / cover,
                        subj_dir / cover,
                        subj_dir / mod,
                    ]
                    found_dir = None
                    for d in possible_dirs:
                        if d.exists() and d.is_dir():
                            found_dir = d
                            break

                    if found_dir:
                        images = sorted(
                            list(found_dir.glob("*.jpg"))
                            + list(found_dir.glob("*.png"))
                        )
                        if images:
                            mod_paths[mod] = images

                if not mod_paths:
                    continue

                num_frames = min(len(paths) for paths in mod_paths.values())
                for i in range(num_frames):
                    sample: dict[str, Any] = {
                        "subject": subject_id,
                        "cover": cover,
                        "index": i,
                        "image_paths": {
                            mod: mod_paths[mod][i] for mod in self.modalities
                        },
                        "joints": {},
                    }

                    for mod in self.modalities:
                        ann_mod = annotations.get(mod)
                        if ann_mod is not None and i < ann_mod.shape[2]:
                            sample["joints"][mod] = ann_mod[:, :, i]
                        else:
                            sample["joints"][mod] = None

                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # 1. Load and align modalities
        mod_tensors: list[torch.Tensor] = []
        primary_joints: np.ndarray | None = None

        for mod in self.modalities:
            img_path = sample["image_paths"][mod]
            with Image.open(img_path) as img:
                if mod.upper() == "RGB":
                    img_loaded = img.convert("RGB")
                else:
                    img_loaded = img.convert("L")

                joints = sample["joints"].get(mod)
                if joints is not None and primary_joints is None:
                    primary_joints = joints.copy()

                if self.augmenter:
                    t_img, t_joints = self.augmenter(
                        img_loaded, joints, image_size=self.image_size
                    )
                    if t_joints is not None:
                        primary_joints = t_joints
                else:
                    w_orig, h_orig = img_loaded.size
                    h_target, w_target = self.image_size
                    img_resized = img_loaded.resize(
                        (w_target, h_target), Image.Resampling.BILINEAR
                    )
                    t_img = torch.from_numpy(
                        np.array(img_resized, dtype=np.float32) / 255.0
                    )
                    if t_img.ndim == 2:
                        t_img = t_img.unsqueeze(0)  # (1, H, W)
                    else:
                        t_img = t_img.permute(2, 0, 1)  # (C, H, W)

                    if primary_joints is not None:
                        primary_joints[0, :] *= w_target / w_orig
                        primary_joints[1, :] *= h_target / h_orig

                mod_tensors.append(t_img)

        # Concatenate modalities across channels
        combined_img = torch.cat(mod_tensors, dim=0)

        result: dict[str, Any] = {
            "image": combined_img,
            "subject": sample["subject"],
            "cover": sample["cover"],
            "index": sample["index"],
            "target": None,
            "joints": None,
        }

        if primary_joints is not None:
            joints_tensor = torch.from_numpy(primary_joints).float()
            result["joints"] = joints_tensor

            if self.return_heatmaps:
                # Add batch dim for vectorized generator: (1, 3, 14) -> (1, 14, H, W) -> squeeze(0)
                heatmaps = generate_gaussian_heatmaps(
                    joints=joints_tensor.unsqueeze(0),
                    heatmap_size=self.heatmap_size,
                    image_size=self.image_size,
                    sigma=self.sigma,
                ).squeeze(0)
                result["target"] = heatmaps

        return result
