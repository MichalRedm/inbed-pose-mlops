"""PyTorch Lightning DataModule for SLP in-bed pose estimation."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from src.data.augmentations import PoseAugmenter
from src.data.dataset import SLPDataset, collate_pose, collate_unlabeled


class SLPDataModule(pl.LightningDataModule):
    """LightningDataModule encapsulating SLP dataset preparation and loaders.

    Supports standard supervised training as well as semi-supervised dual-stream
    training utilizing unlabeled covered poses.
    """

    def __init__(
        self,
        data_root: str | Path = "data/raw",
        batch_size: int = 16,
        num_workers: int = 2,
        modalities: list[str] | None = None,
        train_covers: list[str] | None = None,
        val_covers: list[str] | None = None,
        test_covers: list[str] | None = None,
        train_subjects: Iterable[int] | None = None,
        val_subjects: Iterable[int] | None = None,
        test_subjects: Iterable[int] | None = None,
        image_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        semi_supervised: bool = False,
        unlabeled_covers: list[str] | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.modalities = modalities or ["IR"]
        self.train_covers = train_covers or ["uncover"]
        self.val_covers = val_covers or ["uncover", "cover1", "cover2"]
        self.test_covers = test_covers or ["cover1", "cover2"]

        existing_train = [
            s for s in range(1, 25) if (self.data_root / f"{s:05d}").exists()
        ]
        existing_val = [
            s for s in range(25, 31) if (self.data_root / f"{s:05d}").exists()
        ]
        existing_test = [
            s for s in range(31, 81) if (self.data_root / f"{s:05d}").exists()
        ]

        self.train_subjects = list(
            train_subjects
            if train_subjects is not None
            else (existing_train or range(1, 31))
        )
        self.val_subjects = list(
            val_subjects
            if val_subjects is not None
            else (existing_val or range(31, 41))
        )
        self.test_subjects = list(
            test_subjects
            if test_subjects is not None
            else (existing_test or range(41, 46))
        )

        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.semi_supervised = semi_supervised
        self.unlabeled_covers = unlabeled_covers or ["cover1", "cover2"]
        self.pin_memory = pin_memory

        self.train_dataset_labeled: SLPDataset | None = None
        self.train_dataset_unlabeled: SLPDataset | None = None
        self.val_dataset: SLPDataset | None = None
        self.test_dataset: SLPDataset | None = None

    def prepare_data(self) -> None:
        """Verifies that the SLP dataset directory exists."""
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"SLP dataset directory not found at: {self.data_root.resolve()}. "
                "Please place the SLP dataset in data/raw or specify --data_root."
            )

    def setup(self, stage: str | None = None) -> None:
        """Instantiates datasets for train, val, and test splits."""
        train_augmenter = PoseAugmenter(
            flip_prob=0.5,
            rotation_degrees=20.0,
            scale_range=(0.85, 1.15),
            is_training=True,
        )

        if stage in (None, "fit"):
            # Labeled training set
            self.train_dataset_labeled = SLPDataset(
                root=self.data_root,
                subjects=self.train_subjects,
                modalities=self.modalities,
                covers=self.train_covers,
                split="train",
                augmenter=train_augmenter,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
                return_heatmaps=True,
            )

            # Unlabeled training set for semi-supervised training
            if self.semi_supervised:
                self.train_dataset_unlabeled = SLPDataset(
                    root=self.data_root,
                    subjects=self.train_subjects,
                    modalities=self.modalities,
                    covers=self.unlabeled_covers,
                    split="train",
                    augmenter=train_augmenter,
                    image_size=self.image_size,
                    heatmap_size=self.heatmap_size,
                    sigma=self.sigma,
                    return_heatmaps=False,
                )

            # Validation set
            self.val_dataset = SLPDataset(
                root=self.data_root,
                subjects=self.val_subjects,
                modalities=self.modalities,
                covers=self.val_covers,
                split="valid",
                augmenter=None,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
                return_heatmaps=True,
            )

        if stage in (None, "test"):
            # Test set
            self.test_dataset = SLPDataset(
                root=self.data_root,
                subjects=self.test_subjects,
                modalities=self.modalities,
                covers=self.test_covers,
                split="valid",
                augmenter=None,
                image_size=self.image_size,
                heatmap_size=self.heatmap_size,
                sigma=self.sigma,
                return_heatmaps=True,
            )

    def train_dataloader(self) -> Any:
        """Constructs the training dataloader (single-stream or combined)."""
        if self.train_dataset_labeled is None:
            raise RuntimeError(
                "DataModule setup() must be called before train_dataloader()"
            )

        if not self.semi_supervised:
            return DataLoader(
                self.train_dataset_labeled,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate_pose,
            )

        if self.train_dataset_unlabeled is None:
            raise RuntimeError(
                "Unlabeled dataset was not initialized for semi-supervised mode."
            )

        # In semi-supervised mode, allocate half batch to labeled and half to unlabeled
        half_batch = max(1, self.batch_size // 2)

        loader_labeled = DataLoader(
            self.train_dataset_labeled,
            batch_size=half_batch,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_pose,
        )
        loader_unlabeled = DataLoader(
            self.train_dataset_unlabeled,
            batch_size=half_batch,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_unlabeled,
        )

        return CombinedLoader(
            {"labeled": loader_labeled, "unlabeled": loader_unlabeled},
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Constructs the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "DataModule setup() must be called before val_dataloader()"
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_pose,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Constructs the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError(
                "DataModule setup() must be called before test_dataloader()"
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_pose,
        )
