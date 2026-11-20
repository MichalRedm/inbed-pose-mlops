"""Pydantic v2 request and response schemas for the BentoML pose serving API."""

from pydantic import BaseModel, Field


class KeypointPrediction(BaseModel):
    """Single predicted 2D body keypoint."""

    name: str = Field(..., description="LSP joint name (e.g., R_Ankle, Head).")
    x: float = Field(..., description="Predicted X coordinate in input image space.")
    y: float = Field(..., description="Predicted Y coordinate in input image space.")
    confidence: float = Field(
        ..., description="Peak heatmap confidence score for this joint."
    )


class PosePredictionResponse(BaseModel):
    """Structured 14-keypoint pose estimation response for a single image."""

    keypoints: list[KeypointPrediction] = Field(
        ..., description="List of 14 predicted LSP keypoints with confidence."
    )
    coordinates: list[list[float]] = Field(
        ..., description="14x2 list of [x, y] coordinates in image space."
    )
    confidences: list[float] = Field(
        ..., description="14 per-joint peak heatmap confidence values."
    )
    batch_size_processed: int = Field(
        ..., description="Actual batch size processed by the inference engine."
    )
    inference_time_ms: float = Field(
        ..., description="Server-side batch inference time in milliseconds."
    )
    device: str = Field(..., description="Execution device ('cpu' or 'cuda').")


class ServiceMetadataResponse(BaseModel):
    """Service configuration and batching metadata."""

    model_name: str
    backbone: str
    device: str
    batching_enabled: bool
    max_batch_size: int
    max_latency_ms: int
    image_size: tuple[int, int]
