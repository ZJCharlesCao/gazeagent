import numpy as np

from smvqa_gaze.h1 import (
    classify_fixation_state,
    crop_fixation_frame,
    expand_detection_box,
    extract_visual_targets,
    merge_gaze_episodes,
    parse_projection_judgment,
    require_video_time_fixations,
    sample_unique_projection_frames,
    _nearest_frame_boxes,
)
from smvqa_gaze.saliency import _extract_one_fps_frames, normalize_saliency_map


def test_fixation_crop_is_centered_and_padded_at_frame_edges():
    frame = np.full((300, 300, 3), 100, dtype=np.uint8)

    crop, box = crop_fixation_frame(frame, 0, 0)

    assert crop.shape == (256, 256, 3)
    assert np.all(crop[:128, :] == 0)
    assert np.all(crop[:, :128] == 0)
    assert np.all(crop[128:, 128:] == 100)
    assert box == [-128, -128, 128, 128]


def test_gaze_episodes_merge_only_adjacent_similar_labels():
    labels = [
        {
            "fix_id": fix_id,
            "t_on": float(fix_id),
            "t_off": float(fix_id) + 0.2,
            "dur_ms": 200.0,
            "label": "cup",
            "box_px": [fix_id, fix_id, fix_id + 1, fix_id + 1],
        }
        for fix_id in (0, 1, 3)
    ]

    episodes = merge_gaze_episodes(
        labels,
        [[1.0, 0.0], [0.9, 0.1], [1.0, 0.0]],
    )

    assert len(episodes) == 2
    assert episodes[0]["n_fixations"] == 2
    assert episodes[0]["total_dwell_ms"] == 400.0
    assert episodes[0]["box_px"] == [0, 0, 2, 2]
    assert episodes[1]["n_fixations"] == 1


def test_fixations_must_use_shared_video_time_axis():
    import pandas as pd
    import pytest
    from pathlib import Path

    legacy = pd.DataFrame(columns=["fix_id", "t_on", "t_off", "dur_ms", "u", "v"])
    with pytest.raises(ValueError, match="video time"):
        require_video_time_fixations(legacy, Path("fixations.parquet"))
    current = pd.DataFrame(
        columns=["fix_id", "t_on", "t_off", "t_on_us", "t_off_us", "dur_ms", "u", "v"]
    )
    require_video_time_fixations(current, Path("fixations.parquet"))


def test_h1_uses_distinct_fixation_statuses():
    assert classify_fixation_state(False, 0, 0) == "undetected"
    assert classify_fixation_state(True, 200, 200) == "fixated"
    assert classify_fixation_state(True, 199, 1) == "glanced"
    assert classify_fixation_state(True, 0, 0) == "never"


def test_detection_boxes_expand_two_degrees_and_stay_in_frame():
    box = expand_detection_box(
        [10, 10, 100, 100],
        [600, 600],
        image_size=(1408, 1408),
    )

    assert box[0] == 0
    assert box[1] == 0
    assert 120 < box[2] < 122
    assert 120 < box[3] < 122


def test_gaze_matches_nearest_two_fps_detection_frame():
    boxes = _nearest_frame_boxes(1.24, [1.0, 1.5], [[[1, 2, 3, 4]], [[5, 6, 7, 8]]])
    assert boxes == [[1, 2, 3, 4]]
    assert _nearest_frame_boxes(1.26, [1.0, 1.5], [[], [[5, 6, 7, 8]]]) == [
        [5, 6, 7, 8]
    ]
    assert _nearest_frame_boxes(1.3, [0.0], [[]]) == []


def test_unisal_saliency_maps_are_normalized_to_uint8():
    saliency = normalize_saliency_map(np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert saliency.dtype == np.uint8
    assert saliency.tolist() == [[0, 255], [127, 63]]


def test_unisal_input_extraction_samples_one_frame_per_second(tmp_path):
    import pytest

    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV has no MJPG video writer")
    for value in range(10):
        writer.write(np.full((24, 32, 3), value * 10, dtype=np.uint8))
    writer.release()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    count = _extract_one_fps_frames(video_path, images_dir)

    assert count == 3
    assert sorted(path.name for path in images_dir.iterdir()) == [
        "frame_00000000.jpg",
        "frame_00000001.jpg",
        "frame_00000002.jpg",
    ]


def test_projection_qc_requires_unambiguous_yes_or_no():
    assert parse_projection_judgment("Yes, it is on the cup.") == "yes"
    assert parse_projection_judgment("No.") == "no"
    import pytest

    with pytest.raises(ValueError, match="unambiguous"):
        parse_projection_judgment("yes, but no")


def test_projection_qc_samples_distinct_video_frames():
    rows = [
        {"video_id": "session-a", "frame_index": 10, "sample": 1},
        {"video_id": "session-a", "frame_index": 10, "sample": 2},
        {"video_id": "session-a", "frame_index": 11, "sample": 3},
        {"video_id": "session-b", "frame_index": 10, "sample": 4},
    ]

    chosen = sample_unique_projection_frames(rows, sample_count=50, seed=2026)

    assert len(chosen) == 3
    assert len({(row["video_id"], row["frame_index"]) for row in chosen}) == 3


def test_visual_targets_retry_an_invalid_word_level_array():
    class Labeler:
        def __init__(self):
            self.responses = iter(
                ['["car", "keys", "on", "the", "table"]', '["car keys"]']
            )

        def generate_text(self, prompt):
            return next(self.responses)

    assert extract_visual_targets("Where are the keys?", "On the table", Labeler()) == [
        "car keys"
    ]


def test_visual_targets_preserve_strict_retry_errors():
    class Labeler:
        def generate_text(self, prompt):
            return "not a JSON array"

    import pytest

    with pytest.raises(ValueError, match="after one constrained retry"):
        extract_visual_targets("What is shown?", "A cup", Labeler())


def test_visual_targets_cap_persistent_overflow_after_retry():
    class Labeler:
        def generate_text(self, prompt):
            return '["Sydney", "Taipei", "Osaka", "Tokyo"]'

    import pytest

    with pytest.warns(RuntimeWarning, match="retaining the first 3"):
        targets = extract_visual_targets(
            "Which cities are visible?", "Sydney, Taipei, Osaka, Tokyo", Labeler()
        )

    assert targets == ["Sydney", "Taipei", "Osaka"]


def test_context_crop_marks_fixation_and_none_labels_are_detected():
    from smvqa_gaze.h1 import _is_none_label, context_crop_with_marker

    frame = np.full((600, 600, 3), 50, dtype=np.uint8)
    crop = context_crop_with_marker(frame, 10, 10)
    assert crop.shape == (512, 512, 3)
    assert tuple(crop[256, 256]) == (0, 0, 255)
    assert _is_none_label(" None.") and _is_none_label("'none'") and not _is_none_label("cup")
