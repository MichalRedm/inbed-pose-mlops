"""Pytest fixtures for inbed-pose-mlops test suite."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from PIL import Image


@pytest.fixture
def mock_slp_root(tmp_path: Path) -> Generator[Path, None, None]:
    """Generates a minimal mock SLP dataset filesystem for testing."""
    root = tmp_path / "slp_mock"
    root.mkdir(parents=True, exist_ok=True)

    # Subjects: 1 (train), 31 (val), 41 (test)
    subjects = [1, 31, 41]
    covers = ["uncover", "cover1", "cover2"]
    modalities = ["IR", "RGB"]
    num_frames = 3

    for subj_id in subjects:
        subj_dir = root / f"{subj_id:05d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Synthetic ground truth annotations: (3, 14, num_frames)
        # coords in range [20, 200], visibility = 1
        gt_coords = np.random.uniform(30.0, 200.0, size=(2, 14, num_frames))
        gt_vis = np.ones((1, 14, num_frames), dtype=np.float64)
        joints_gt = np.concatenate([gt_coords, gt_vis], axis=0)

        for mod in modalities:
            # Save MATLAB ground truth annotations
            mat_path = subj_dir / f"joints_gt_{mod}.mat"
            sio.savemat(str(mat_path), {"joints_gt": joints_gt})

            for cover in covers:
                cover_dir = subj_dir / mod / cover
                cover_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx in range(num_frames):
                    img_name = f"{frame_idx + 1:06d}.{'jpg' if mod == 'RGB' else 'png'}"
                    img_path = cover_dir / img_name

                    if mod == "RGB":
                        arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
                        im = Image.fromarray(arr, mode="RGB")
                    else:
                        arr = (np.random.rand(100, 100) * 255).astype(np.uint8)
                        im = Image.fromarray(arr, mode="L")
                    im.save(img_path)

    yield root
