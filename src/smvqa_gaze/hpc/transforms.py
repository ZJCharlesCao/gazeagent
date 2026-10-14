from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    DUAL_PROMPT,
    FRAME_SIZE,
    GAZE_PROMPT_M,
    GAZE_PROMPT_T,
    GAZE_PROMPT_V,
    MASK_SIDE,
    TIER_BOX_LINE_PX,
    TIER_DUAL_SIDE,
    TIER_SIDE,
    Condition,
)
from .gaze_source import GazeSources


@dataclass
class RenderedFrame:
    """The model-facing images and per-frame text for one sampled frame."""

    images: list[np.ndarray]
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def resize(image: np.ndarray, side: int) -> np.ndarray:
    import cv2

    return cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2]
    if patch.shape[0] != y2 - y1 or patch.shape[1] != x2 - x1:
        raise ValueError(f"Crop {box} leaves the {image.shape} frame")
    return patch


def grounding_coordinates(box: tuple[int, int, int, int], point: tuple[float, float]) -> str:
    """Qwen3-VL grounding format: JSON with relative 0-1000 integer coordinates."""
    scale = 1000 / FRAME_SIZE
    x1, y1, x2, y2 = (int(round(value * scale)) for value in box)
    px, py = (int(round(value * scale)) for value in point)
    return f'{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}], "point_2d": [{px}, {py}]}}'


def draw_gaze_box(image: np.ndarray, box: tuple[int, int, int, int], line: int) -> np.ndarray:
    import cv2

    side = image.shape[0]
    scale = side / FRAME_SIZE
    x1, y1, x2, y2 = (int(round(value * scale)) for value in box)
    output = np.ascontiguousarray(image.copy())
    cv2.rectangle(output, (x1, y1), (min(x2, side - 1), min(y2, side - 1)), (255, 0, 0), line)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.circle(output, center, max(2, line + 1), (255, 0, 0), -1)
    return output


def gaze_mask(box: tuple[int, int, int, int], side: int = MASK_SIDE) -> np.ndarray:
    scale = side / FRAME_SIZE
    x1, y1, x2, y2 = (int(round(value * scale)) for value in box)
    mask = np.zeros((side, side, 3), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def condition_prompt(condition: Condition) -> str:
    if condition.visual == "dual":
        return DUAL_PROMPT
    return {"T": GAZE_PROMPT_T, "V": GAZE_PROMPT_V, "M": GAZE_PROMPT_M}.get(condition.hint or "", "")


def render_frame(
    condition: Condition,
    tier: str,
    image: np.ndarray,
    video_id: str,
    t: float,
    sources: GazeSources | None,
) -> RenderedFrame:
    """Apply README section 7.3-7.5 to one 1408x1408 RGB frame."""
    if image.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Expected {FRAME_SIZE}px frames, got {image.shape}")
    side = TIER_SIDE[tier]
    meta: dict[str, Any] = {"t": t}
    box = point = None
    if condition.needs_gaze and condition.gaze_source is not None:
        if sources is None:
            raise ValueError(f"{condition.name} needs gaze sources")
        box_side = TIER_DUAL_SIDE[tier] if condition.visual == "dual" else side
        box, gaze = sources.box(condition.gaze_source, video_id, t, box_side)
        point = (gaze.u, gaze.v)
        meta.update(
            {
                "gaze_box": list(box),
                "gaze_point": [round(gaze.u, 1), round(gaze.v, 1)],
                "gaze_status": gaze.status,
            }
        )

    if condition.visual == "fovea":
        assert box is not None
        return RenderedFrame([crop(image, box)], meta=meta)
    if condition.visual == "dual":
        assert box is not None
        dual = TIER_DUAL_SIDE[tier]
        return RenderedFrame([resize(image, dual), crop(image, box)], meta=meta)

    full = resize(image, side)
    if condition.hint is None:
        return RenderedFrame([full], meta=meta)
    assert box is not None and point is not None
    if condition.hint == "T":
        return RenderedFrame([full], text=grounding_coordinates(box, point), meta=meta)
    if condition.hint == "V":
        return RenderedFrame([draw_gaze_box(full, box, TIER_BOX_LINE_PX[tier])], meta=meta)
    if condition.hint == "M":
        return RenderedFrame([full, gaze_mask(box)], meta=meta)
    raise ValueError(f"Unknown hint {condition.hint!r}")


def encode_image(image: np.ndarray, quality: int = 95) -> str:
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[0] <= MASK_SIDE:
        ok, data = cv2.imencode(".png", bgr)
        mime = "image/png"
    else:
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    if not ok:
        raise RuntimeError("Image encoding failed")
    return f"data:{mime};base64,{base64.b64encode(data.tobytes()).decode('ascii')}"


def frames_to_content(
    frames: list[RenderedFrame], labels: list[str]
) -> list[dict[str, Any]]:
    """OpenAI-style multimodal content: label, image(s), then per-frame text."""
    content: list[dict[str, Any]] = []
    for index, (frame, label) in enumerate(zip(frames, labels, strict=True), start=1):
        content.append({"type": "text", "text": f"Frame {index} ({label}):"})
        for image in frame.images:
            content.append({"type": "image_url", "image_url": {"url": encode_image(image)}})
        if frame.text:
            content.append({"type": "text", "text": f"Frame {index} gaze: {frame.text}"})
    return content


def expected_visual_tokens(frames: list[RenderedFrame]) -> int:
    """Qwen3-VL: 16 px patches merged 2x2, i.e. one token per 32x32 pixels.

    The processor-measured count (token_audit) is authoritative; this is only
    used for planning.
    """
    return sum((image.shape[0] // 32) * (image.shape[1] // 32) for frame in frames for image in frame.images)
