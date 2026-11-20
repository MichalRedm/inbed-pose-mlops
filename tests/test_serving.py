"""Unit and integration tests for BentoML model serving and benchmarking."""

import base64
import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from scripts.benchmark_serving import (
    aggregate_repeats,
    compute_run_metrics,
    format_markdown_table,
)
from scripts.export_model import export_checkpoint_to_bentoml
from src.models.resnet_pose import ResNetPose
from src.serving.predictor import PoseInferenceEngine
from src.serving.service import InBedPoseService


def _make_b64_ir_image(width: int = 128, height: int = 128) -> str:
    """Creates a base64-encoded synthetic grayscale IR image."""
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_pose_inference_engine_single_and_batch() -> None:
    """Verifies PoseInferenceEngine.predict_batch on batch sizes 1, 4, and 8."""
    engine = PoseInferenceEngine(
        bento_model_tag="nonexistent_tag:latest",
        checkpoint_path=None,
        backbone="resnet18",
        in_channels=1,
        num_joints=14,
        image_size=(256, 256),
        device="cpu",
    )

    for bs in [1, 4, 8]:
        images_b64 = [_make_b64_ir_image() for _ in range(bs)]
        preds = engine.predict_batch(images_b64)
        assert len(preds) == bs
        for pred in preds:
            assert len(pred.keypoints) == 14
            assert len(pred.coordinates) == 14
            assert len(pred.confidences) == 14
            assert pred.batch_size_processed == bs
            assert pred.inference_time_ms > 0.0
            for kp in pred.keypoints:
                assert 0.0 <= kp.x <= 256.0
                assert 0.0 <= kp.y <= 256.0


def test_export_model_to_bentoml_store(tmp_path: Path) -> None:
    """Verifies exporting a Lightning checkpoint to BentoML and reloading it."""
    model = ResNetPose(backbone="resnet18", num_joints=14, in_channels=1)
    ckpt_path = tmp_path / "test_pose.ckpt"
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "epoch": 1}, ckpt_path)

    tag = export_checkpoint_to_bentoml(
        checkpoint_path=ckpt_path,
        bento_model_name="test_inbed_pose_resnet18",
        backbone="resnet18",
        in_channels=1,
        num_joints=14,
        image_size=(256, 256),
        heatmap_size=(64, 64),
    )
    assert tag.startswith("test_inbed_pose_resnet18:")

    engine = PoseInferenceEngine(
        bento_model_tag=tag,
        backbone="resnet18",
        in_channels=1,
        num_joints=14,
        image_size=(256, 256),
        device="cpu",
    )
    assert engine.model_source == f"bentoml:{tag}"


def test_bentoml_service_endpoints() -> None:
    """Tests InBedPoseService predict, predict_single, and metadata methods."""
    service = InBedPoseService()
    b64_img = _make_b64_ir_image()

    meta = service.metadata()
    assert meta.backbone == "resnet18"
    assert meta.image_size == (256, 256)

    single = service.predict_single(image_b64=b64_img)
    assert len(single.keypoints) == 14
    assert single.batch_size_processed == 1

    batched = service.predict(images_b64=[b64_img, b64_img])
    assert len(batched) == 2
    assert all(p.batch_size_processed == 2 for p in batched)


def test_benchmark_metrics_aggregation() -> None:
    """Tests throughput, latency percentile, repeat aggregation, and Markdown table."""
    run1 = compute_run_metrics(
        latencies_ms=[10.0, 20.0, 30.0, 40.0],
        batch_sizes=[1, 1, 1, 1],
        elapsed_s=2.0,
    )
    assert run1["throughput_samples_per_s"] == pytest.approx(2.0)
    assert run1["p50_ms"] == pytest.approx(25.0)
    assert run1["p95_ms"] > 35.0
    assert run1["mean_batch_size"] == pytest.approx(1.0)

    run2 = compute_run_metrics(
        latencies_ms=[12.0, 22.0, 32.0, 42.0],
        batch_sizes=[2, 2, 2, 2],
        elapsed_s=2.0,
    )
    agg = aggregate_repeats([run1, run2])
    assert agg["throughput_samples_per_s_mean"] == pytest.approx(2.0)
    assert agg["throughput_samples_per_s_std"] == pytest.approx(0.0)

    summary = {
        "no_batching": {1: agg},
        "adaptive_batching": {1: agg},
    }
    md = format_markdown_table(summary, [1])
    assert "| Concurrency |" in md
    assert "| 1 |" in md
