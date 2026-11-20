"""BentoML REST API service for In-Bed Human Pose Estimation."""

import os
from pathlib import Path
from typing import Any, cast

import bentoml
import yaml

from src.serving.predictor import BackboneType, PoseInferenceEngine
from src.serving.schemas import PosePredictionResponse, ServiceMetadataResponse


def _load_serving_config() -> dict[str, Any]:
    """Loads serving configuration from configs/serving.yaml."""
    config_path = Path(os.getenv("POSE_SERVING_CONFIG", "configs/serving.yaml"))
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return cast(dict[str, Any], yaml.safe_load(f))
    return {}


_CFG = _load_serving_config()
_MODEL_CFG = _CFG.get("model", {})
_BATCH_CFG = _CFG.get("serving", {}).get("batching", {})

ENABLE_BATCHING: bool = (
    os.getenv(
        "POSE_ENABLE_BATCHING",
        "1" if _BATCH_CFG.get("enabled", True) else "0",
    )
    == "1"
)
MAX_BATCH_SIZE: int = int(
    os.getenv("POSE_MAX_BATCH_SIZE", str(_BATCH_CFG.get("max_batch_size", 32)))
)
MAX_LATENCY_MS: int = int(
    os.getenv("POSE_MAX_LATENCY_MS", str(_BATCH_CFG.get("max_latency_ms", 50)))
)
CHECKPOINT_PATH: str = os.getenv(
    "POSE_CHECKPOINT_PATH",
    str(_MODEL_CFG.get("checkpoint_path", "")),
)
DEVICE_OVERRIDE: str = os.getenv(
    "POSE_DEVICE",
    str(_MODEL_CFG.get("device", "auto")),
)


@bentoml.service(
    name="inbed_pose_service",
    traffic={"timeout": 120},
)
class InBedPoseService:
    """BentoML service exposing ResNetPose 2D keypoint estimation."""

    def __init__(self) -> None:
        img_size_cfg = _MODEL_CFG.get("image_size", [256, 256])
        image_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))
        backbone = cast(BackboneType, _MODEL_CFG.get("backbone", "resnet18"))

        self.engine = PoseInferenceEngine(
            bento_model_tag=str(
                _MODEL_CFG.get("bento_model_tag", "inbed_pose_resnet18:latest")
            ),
            checkpoint_path=CHECKPOINT_PATH or None,
            backbone=backbone,
            in_channels=int(_MODEL_CFG.get("in_channels", 1)),
            num_joints=int(_MODEL_CFG.get("num_joints", 14)),
            image_size=image_size,
            device=DEVICE_OVERRIDE,
        )

    @bentoml.api(
        batchable=ENABLE_BATCHING,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def predict(self, images_b64: list[str]) -> list[PosePredictionResponse]:
        """Predicts 14 LSP keypoints for a batch of base64-encoded images."""
        return self.engine.predict_batch(images_b64)

    @bentoml.api(batchable=False)
    def predict_single(self, image_b64: str) -> PosePredictionResponse:
        """Convenience unbatched endpoint for a single base64-encoded image."""
        return self.engine.predict_batch([image_b64])[0]

    @bentoml.api(batchable=False)
    def metadata(self) -> ServiceMetadataResponse:
        """Returns active model, device, and dynamic batching configuration."""
        return ServiceMetadataResponse(
            model_name=self.engine.bento_model_tag,
            backbone=self.engine.backbone,
            device=str(self.engine.device),
            batching_enabled=ENABLE_BATCHING,
            max_batch_size=MAX_BATCH_SIZE,
            max_latency_ms=MAX_LATENCY_MS,
            image_size=self.engine.image_size,
        )
