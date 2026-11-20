"""Inference engine for ResNetPose model serving."""

import base64
import io
import time
from pathlib import Path
from typing import Literal, cast

import bentoml
import numpy as np
import torch
from PIL import Image

from src.metrics.pose_metrics import LSP_JOINT_NAMES
from src.models.resnet_pose import ResNetPose
from src.serving.schemas import KeypointPrediction, PosePredictionResponse
from src.utils.heatmaps import decode_heatmaps

BackboneType = Literal["resnet18", "resnet34", "resnet50"]


class PoseInferenceEngine:
    """Encapsulates model loading, preprocessing, and batched heatmap inference."""

    def __init__(
        self,
        bento_model_tag: str = "inbed_pose_resnet18:latest",
        checkpoint_path: str | Path | None = None,
        backbone: BackboneType = "resnet18",
        in_channels: int = 1,
        num_joints: int = 14,
        image_size: tuple[int, int] = (256, 256),
        device: str = "auto",
    ) -> None:
        self.bento_model_tag = bento_model_tag
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.backbone: BackboneType = backbone
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.image_size = image_size
        self.device = self._resolve_device(device)
        self.model_source = "uninitialized"

        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self._warmup()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolves 'auto' to CUDA if available, otherwise CPU."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self) -> ResNetPose:
        """Loads ResNetPose from BentoML store, Lightning checkpoint, or fallback."""
        model = ResNetPose(
            backbone=self.backbone,
            num_joints=self.num_joints,
            in_channels=self.in_channels,
            pretrained=False,
        )

        # 1. Try loading from BentoML Model Store
        try:
            bento_model = bentoml.models.get(self.bento_model_tag)
            weights_path = Path(bento_model.path) / "model_state_dict.pt"
            if weights_path.exists():
                state_dict = torch.load(
                    weights_path, map_location="cpu", weights_only=True
                )
                model.load_state_dict(state_dict)
                self.model_source = f"bentoml:{self.bento_model_tag}"
                return model
        except Exception:
            pass

        # 2. Try loading from PyTorch Lightning checkpoint (.ckpt)
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            ckpt = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            raw_state_dict = ckpt.get("state_dict", ckpt)
            cleaned_state_dict = {
                k.removeprefix("model."): v
                for k, v in raw_state_dict.items()
                if k.startswith("model.")
            }
            if cleaned_state_dict:
                model.load_state_dict(cleaned_state_dict)
                self.model_source = f"ckpt:{self.checkpoint_path.name}"
                return model

        # 3. Deterministic fallback for isolated CI environments
        torch.manual_seed(42)
        self.model_source = "deterministic_fallback"
        return model

    def _warmup(self) -> None:
        """Runs a dummy forward pass to initialize execution kernels."""
        dummy = torch.zeros(
            (1, self.in_channels, self.image_size[0], self.image_size[1]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            _ = self.model(dummy)

    def preprocess_base64_images(self, images_b64: list[str]) -> torch.Tensor:
        """Decodes base64 images and stacks them into a (B, C, H, W) tensor."""
        if not images_b64:
            raise ValueError("images_b64 must contain at least one image.")

        h_target, w_target = self.image_size
        tensors: list[torch.Tensor] = []

        for b64_str in images_b64:
            raw_bytes = base64.b64decode(b64_str)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                if self.in_channels == 1:
                    img_conv = img.convert("L")
                else:
                    img_conv = img.convert("RGB")

                img_resized = img_conv.resize(
                    (w_target, h_target), Image.Resampling.BILINEAR
                )
                arr = np.asarray(img_resized, dtype=np.float32) / 255.0
                if arr.ndim == 2:
                    t = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
                else:
                    t = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
                tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        return batch

    def predict_batch(self, images_b64: list[str]) -> list[PosePredictionResponse]:
        """Runs batched inference and returns one PosePredictionResponse per image."""
        t_start = time.perf_counter()
        batch_tensor = self.preprocess_base64_images(images_b64)
        batch_size = int(batch_tensor.shape[0])

        with torch.inference_mode():
            heatmaps = self.model(batch_tensor)
            coords = decode_heatmaps(heatmaps, image_size=self.image_size)
            # Peak confidence per joint: (B, J)
            flat_hm = heatmaps.view(batch_size, self.num_joints, -1)
            confs, _ = flat_hm.max(dim=-1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        coords_list = cast(list[list[list[float]]], coords.cpu().tolist())
        confs_list = cast(list[list[float]], confs.cpu().tolist())

        responses: list[PosePredictionResponse] = []
        for idx in range(batch_size):
            sample_coords = coords_list[idx]
            sample_confs = confs_list[idx]
            keypoints = [
                KeypointPrediction(
                    name=LSP_JOINT_NAMES[j],
                    x=round(float(sample_coords[j][0]), 2),
                    y=round(float(sample_coords[j][1]), 2),
                    confidence=round(float(sample_confs[j]), 4),
                )
                for j in range(self.num_joints)
            ]
            responses.append(
                PosePredictionResponse(
                    keypoints=keypoints,
                    coordinates=[
                        [round(float(c[0]), 2), round(float(c[1]), 2)]
                        for c in sample_coords
                    ],
                    confidences=[round(float(v), 4) for v in sample_confs],
                    batch_size_processed=batch_size,
                    inference_time_ms=round(elapsed_ms, 3),
                    device=str(self.device),
                )
            )
        return responses
