"""Example HTTP client for querying the BentoML In-Bed Pose Estimation API."""

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, cast

# Ensure repository root is on sys.path for direct CLI execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import httpx
import yaml
from PIL import Image


def load_image_as_base64(image_path: str | Path | None, synthetic: bool = False) -> str:
    """Loads an image from disk or generates a synthetic 256x256 IR test frame."""
    if not synthetic and image_path is not None and Path(image_path).exists():
        raw = Path(image_path).read_bytes()
        return base64.b64encode(raw).decode("ascii")

    img = Image.new("L", (256, 256), color=96)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments for the example client."""
    parser = argparse.ArgumentParser(
        description="Send a sample inference request to the BentoML Pose API."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/serving.yaml",
        help="Path to serving YAML config.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Base URL of the running BentoML service (e.g. http://127.0.0.1:3000).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic 256x256 grayscale test image.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = cast(dict[str, Any], yaml.safe_load(f))

    serving_cfg = cfg.get("serving", {})
    bench_cfg = cfg.get("benchmark", {})
    base_url = args.server_url or (
        f"http://{serving_cfg.get('host', '127.0.0.1')}:{serving_cfg.get('port', 3000)}"
    )
    img_path = args.image or bench_cfg.get("sample_image_path")
    img_b64 = load_image_as_base64(img_path, synthetic=args.synthetic)

    with httpx.Client(timeout=30.0) as client:
        meta_resp = client.post(f"{base_url}/metadata", json={})
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        pred_resp = client.post(
            f"{base_url}/predict",
            json={"images_b64": [img_b64]},
        )
        pred_resp.raise_for_status()
        preds = pred_resp.json()

    result = preds[0]
    print("=" * 65)
    print("BentoML In-Bed Pose Estimation — Client Response")
    print("=" * 65)
    print(f"Service URL          : {base_url}")
    print(f"Model / Backbone     : {meta['model_name']} ({meta['backbone']})")
    print(f"Device               : {result['device']}")
    print(f"Batching Enabled     : {meta['batching_enabled']}")
    print(f"Batch Size Processed : {result['batch_size_processed']}")
    print(f"Inference Time       : {result['inference_time_ms']:.2f} ms")
    print("-" * 65)
    print(f"{'Joint':<14} | {'X (px)':>8} | {'Y (px)':>8} | {'Confidence':>10}")
    print("-" * 65)
    for kp in result["keypoints"]:
        print(
            f"{kp['name']:<14} | {kp['x']:8.2f} | {kp['y']:8.2f} | {kp['confidence']:10.4f}"
        )
    print("=" * 65)
    print("\nRaw JSON snippet (first 2 keypoints):")
    print(json.dumps(result["keypoints"][:2], indent=2))


if __name__ == "__main__":
    main()
