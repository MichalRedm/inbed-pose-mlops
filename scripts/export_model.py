"""Package the trained HW1 ResNetPose checkpoint into the BentoML Model Store."""

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Any, cast

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import bentoml
import torch
import yaml
from PIL import Image

from src.serving.predictor import BackboneType, PoseInferenceEngine


def export_checkpoint_to_bentoml(
    checkpoint_path: str | Path,
    bento_model_name: str = "inbed_pose_resnet18",
    backbone: BackboneType = "resnet18",
    in_channels: int = 1,
    num_joints: int = 14,
    image_size: tuple[int, int] = (256, 256),
    heatmap_size: tuple[int, int] = (64, 64),
) -> str:
    """Extracts ResNetPose state_dict from a Lightning checkpoint and saves to BentoML."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    raw_state_dict = ckpt.get("state_dict", ckpt)
    cleaned_state_dict = {
        k.removeprefix("model."): v
        for k, v in raw_state_dict.items()
        if k.startswith("model.")
    }
    if not cleaned_state_dict:
        cleaned_state_dict = raw_state_dict

    metadata: dict[str, Any] = {
        "backbone": backbone,
        "in_channels": in_channels,
        "num_joints": num_joints,
        "image_size": list(image_size),
        "heatmap_size": list(heatmap_size),
        "source_checkpoint": str(ckpt_file),
        "epoch": int(ckpt.get("epoch", -1)),
    }

    with bentoml.models.create(
        name=bento_model_name,
        module="bentoml.pytorch",
        metadata=metadata,
        labels={"framework": "pytorch", "task": "pose-estimation", "dataset": "SLP"},
    ) as model_ref:
        torch.save(cleaned_state_dict, Path(model_ref.path) / "model_state_dict.pt")

    tag = str(model_ref.tag)

    # Sanity-check: reload from BentoML store and run a dummy prediction
    engine = PoseInferenceEngine(
        bento_model_tag=tag,
        backbone=backbone,
        in_channels=in_channels,
        num_joints=num_joints,
        image_size=image_size,
        device="cpu",
    )
    dummy_img = Image.new("L", (image_size[1], image_size[0]), color=128)
    buf = io.BytesIO()
    dummy_img.save(buf, format="PNG")
    dummy_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    preds = engine.predict_batch([dummy_b64])
    assert len(preds) == 1 and len(preds[0].keypoints) == num_joints

    return tag


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for model export."""
    parser = argparse.ArgumentParser(
        description="Export trained HW1 Lightning checkpoint to BentoML Model Store."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/serving.yaml",
        help="Path to serving configuration YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override path to Lightning .ckpt file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))

    model_cfg = cfg.get("model", {})
    ckpt_path = args.checkpoint or model_cfg["checkpoint_path"]
    bento_tag_base = str(
        model_cfg.get("bento_model_tag", "inbed_pose_resnet18:latest")
    ).split(":")[0]
    img_sz = model_cfg.get("image_size", [256, 256])
    hm_sz = model_cfg.get("heatmap_size", [64, 64])

    tag = export_checkpoint_to_bentoml(
        checkpoint_path=ckpt_path,
        bento_model_name=bento_tag_base,
        backbone=cast(BackboneType, model_cfg.get("backbone", "resnet18")),
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_joints=int(model_cfg.get("num_joints", 14)),
        image_size=(int(img_sz[0]), int(img_sz[1])),
        heatmap_size=(int(hm_sz[0]), int(hm_sz[1])),
    )
    print(f"Successfully exported and verified BentoML model: {tag}")


if __name__ == "__main__":
    main()
