from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


FRAME_SIZE = 1408
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
RETRIEVER_ID = "facebook/contriever-msmarco"
COVERAGE_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_ID = "openai/clip-vit-large-patch14"
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEED = 0

# README section 7.2: budget is visual tokens per frame (~32x32 px per token).
TIER_SIDE = {"B1": 512, "B2": 1024}
TIER_TARGET_TOKENS = {"B1": 256, "B2": 1024}
# README section 7.4: C4 uses the multiple of 32 closest to S / sqrt(2).
TIER_DUAL_SIDE = {"B1": 352, "B2": 736}
TIER_BOX_LINE_PX = {"B1": 2, "B2": 3}
MASK_SIDE = 128
# Qwen3-VL's processor upsamples images below 65,536 px (256x256), which turned
# the 128x128 C3-M mask into 64 tokens instead of ~16. Lowering the minimum to
# 128x128 changes nothing else (every other image is >= 352x352). The vLLM
# server must use the same value (scripts/start_vllm.sh).
MM_PROCESSOR_KWARGS = {"size": {"shortest_edge": MASK_SIDE * MASK_SIDE, "longest_edge": 16777216}}

CLIP_SECONDS = 30
SHARD_SECONDS = 30 * 60
VIDEORAG_ANSWER_FRAMES = 32

GAZE_PROMPT_T = (
    "For each frame, the user's gaze region is given as a box in Qwen3-VL "
    "grounding coordinates (relative, 0-1000), followed by the gaze point. "
    "Pay more attention to the content inside the gaze region."
)
GAZE_PROMPT_V = (
    "The red box in each frame marks where the user was looking. Pay more "
    "attention to the content inside the red box."
)
GAZE_PROMPT_M = (
    "Each frame is followed by a black-and-white mask of the same view. White "
    "marks where the user was looking. Pay more attention to the corresponding "
    "region of the frame."
)
DUAL_PROMPT = (
    "Image A is the full view at low resolution. Image B is a high-resolution "
    "crop around where the user was looking."
)


@dataclass(frozen=True)
class Condition:
    """One row of README section 7.3 (plus its section 7.8/7.9 variants)."""

    name: str
    visual: str  # full | fovea | dual
    hint: str | None = None  # T | V | M (section 7.5)
    gaze_source: str | None = None  # own | shift | shift2s | cross | saliency
    gaze_meta: bool = False  # C5
    alpha: float | None = None  # C6 segmentation + frame weighting
    caption_from: str | None = None  # reuse another condition's captions
    priority: str = "P0"
    tiers: tuple[str, ...] = ("B1", "B2")

    @property
    def needs_gaze(self) -> bool:
        return self.gaze_source is not None


def _hint_conditions(prefix: str, source: str, priority: str, tiers=("B1", "B2")):
    return [
        Condition(f"{prefix}-{hint}", "full", hint, source, priority=priority, tiers=tiers)
        for hint in ("T", "V", "M")
    ]


CONDITIONS: dict[str, Condition] = {
    condition.name: condition
    for condition in [
        Condition("C0", "full"),
        Condition("C1", "fovea", gaze_source="own"),
        *_hint_conditions("C3", "own", "P0"),
        Condition("C5", "full", gaze_source="own", gaze_meta=True, caption_from="C0"),
        *_hint_conditions("C8shift", "shift", "P1"),
        *_hint_conditions("C8cross", "cross", "P1"),
        *_hint_conditions("C9", "saliency", "P1"),
        Condition("C4", "dual", gaze_source="own", priority="P2"),
        Condition("C6a0", "full", gaze_source="own", alpha=0.0, priority="P2"),
        Condition("C6a0.5", "full", gaze_source="own", alpha=0.5, priority="P2"),
        Condition("C6a1", "full", gaze_source="own", alpha=1.0, priority="P2"),
        # P3 robustness (optional; README section 8).
        Condition("C1-C8shift", "fovea", gaze_source="shift", priority="P3", tiers=("B1",)),
        Condition(
            "C5-C8shift",
            "full",
            gaze_source="shift",
            gaze_meta=True,
            caption_from="C0",
            priority="P3",
            tiers=("B1",),
        ),
        *_hint_conditions("C8shift2s", "shift2s", "P3"),
    ]
}
PIPELINES = ("videorag", "egobutler")
PRIORITY_ORDER = ("P0", "P1", "P2", "P3")


@dataclass
class RunConfig:
    """Fixed hyper-parameters shared by every condition (README section 1.4)."""

    model: str = MODEL_ID
    base_url: str = "http://127.0.0.1:8000/v1"
    seed: int = SEED
    temperature: float = 0.0
    caption_max_tokens: int = 256
    summary_max_tokens: int = 384
    answer_max_tokens: int = 32
    request_concurrency: int = 6
    request_timeout_s: float = 900.0
    # Must match vLLM --max-model-len; bounds frames per C6 unit (README 5.2).
    max_model_len: int = 49152
    text_margin_tokens: int = 4096
    egorag_days_k: int = 2
    egorag_hours_k: int = 3
    egorag_clips_k: int = 5
    videorag_asr_k: int = 10
    videorag_ocr_k: int = 10
    videorag_gaze_k: int = 10
    videorag_keyframes: int = 8
    ocr_fps: float = 0.2
    ocr_min_confidence: float = 0.5
    detector_box_threshold: float = 0.35
    detector_text_threshold: float = 0.25
    beta: float = 0.5
    shift_seconds: float = 300.0
    shift_fallback_seconds: float = 420.0
    shift2s_seconds: float = 2.0
    coverage_threshold: float = 0.7
    ans_f1_positive: str = "answerable"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | None) -> "RunConfig":
        if path is None or not path.is_file():
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        known = {key: values.pop(key) for key in list(values) if key in cls.__dataclass_fields__}
        config = cls(**known)
        config.extra.update(values)
        return config


@dataclass(frozen=True)
class Paths:
    """Data (unpacked transfer package, read-only) and work roots on HPC."""

    data: Path
    work: Path

    @property
    def qa_subset(self) -> Path:
        return self.data / "qa/qa_subset.jsonl"

    @property
    def fe_labels(self) -> Path:
        return self.data / "qa/qa_fixation_labels.jsonl"

    @property
    def visual_targets(self) -> Path:
        return self.data / "qa/visual_targets.jsonl"

    @property
    def sessions(self) -> Path:
        return self.data / "docs/selected_sessions.json"

    def video(self, video_id: str) -> Path:
        return self.data / "videos" / f"{video_id}.mp4"

    def gaze_dir(self, video_id: str) -> Path:
        return self.data / "gaze" / video_id

    def calib(self, video_id: str) -> Path:
        return self.data / "calib" / f"{video_id}.json"

    def saliency_dir(self, video_id: str) -> Path:
        return self.data / "saliency" / video_id

    def transcript(self, participant: int, video_id: str) -> Path:
        stem = video_id.replace(f"Person_{participant}_", f"person_{participant}_", 1)
        return (
            self.data
            / "transcripts"
            / f"person_{participant}"
            / f"{stem}_whisper_transcript.json"
        )

    @property
    def frame_cache(self) -> Path:
        return self.work / "cache/frames_1fps"

    @property
    def feature_cache(self) -> Path:
        return self.work / "cache/features"

    def memory_dir(self, pipeline: str, tier: str, condition: str) -> Path:
        return self.work / "memory" / pipeline / tier / condition

    def pred_dir(self, pipeline: str, tier: str, condition: str) -> Path:
        return self.work / "preds" / pipeline / tier / condition

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def logs(self) -> Path:
        return self.work / "logs"


def code_version() -> dict[str, Any]:
    """Git revision when available, plus a hash of the package source."""
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(path.read_bytes())
    revision = None
    dirty = None
    try:
        repo = package_root.parents[1]
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"git_revision": revision, "git_dirty": dirty, "source_sha256": digest.hexdigest()}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_manifest(
    directory: Path,
    stage: str,
    inputs: list[Path],
    parameters: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> Path:
    """README section 1.5: every stage output carries a manifest.json."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "inputs": {
            str(path): file_digest(path) for path in inputs if path.is_file()
        },
        "parameters": parameters,
        "code_version": code_version(),
        "outputs": outputs or {},
    }
    destination = directory / "manifest.json"
    temporary = destination.with_name("manifest.json.part")
    temporary.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def atomic_save_npy(path: Path, array: Any) -> None:
    """Caches are shared by parallel condition processes: write then rename,
    so a reader never sees a half-written file."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part.npy")
    np.save(temporary, array)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
