import numpy as np
import pytest

from smvqa_gaze.hpc.answering import option_order, parse_ranking, score_ranking
from smvqa_gaze.hpc.config import CONDITIONS, TIER_DUAL_SIDE, TIER_SIDE
from smvqa_gaze.hpc.data import Cutoff, SessionInfo
from smvqa_gaze.hpc.gaze_source import GazeTrack, square_box
from smvqa_gaze.hpc.gazemeta import allocate_frames, constrain_segments
from smvqa_gaze.hpc.metrics import answerability_f1, summarize
from smvqa_gaze.hpc.transforms import expected_visual_tokens, grounding_coordinates, render_frame


def question(answerable=True):
    choices = [
        {"text": "This question can not be answered.", "type": "unanswerable"},
        {"text": "on the table", "type": "correct" if answerable else "wrong"},
        {"text": "somewhere", "type": "vague" if answerable else "wrong"},
        {"text": "in the car", "type": "wrong"},
    ]
    return {
        "question_id": "7",
        "choices": choices,
        "correct_option_index": 1 if answerable else 0,
        "abstain_option_index": 0,
    }


def test_option_order_is_fixed_per_question():
    assert option_order(question(), 0) == option_order(question(), 0)
    assert sorted(option_order(question(), 0)) == [0, 1, 2, 3]


def test_ranking_maps_back_to_original_options():
    q = question()
    order = [3, 1, 0, 2]  # letters A..D show options 3,1,0,2
    ranking, status = parse_ranking("B>C>A>D")
    scored = score_ranking(q, ranking, order)
    assert status == "ok"
    assert scored["top1"] == 1 and scored["correct"] and scored["rank_of_correct"] == 1
    scored = score_ranking(q, parse_ranking("C>B>A>D")[0], order)
    assert scored["abstained"] and scored["rank_of_correct"] == 2


def test_unanswerable_question_counts_abstention_as_correct():
    q = question(answerable=False)
    scored = score_ranking(q, [0, 1, 2, 3], [0, 1, 2, 3])
    assert scored["correct"] and scored["abstained"]


def test_answerability_f1_and_rates():
    rows = [
        {"is_answerable": True, "abstained": False, "correct": True, "rank_of_correct": 1, "top1_type": "correct", "parse_status": "ok", "category": "a"},
        {"is_answerable": True, "abstained": True, "correct": False, "rank_of_correct": 2, "top1_type": "unanswerable", "parse_status": "ok", "category": "a"},
        {"is_answerable": False, "abstained": True, "correct": True, "rank_of_correct": 1, "top1_type": "unanswerable", "parse_status": "ok", "category": "a"},
    ]
    f1 = answerability_f1(rows)
    assert f1["ans_f1_answerable"] == pytest.approx(2 / 3)
    assert f1["ans_f1_unanswerable"] == pytest.approx(2 / 3)
    summary = summarize(rows)
    assert summary["qa_acc"] == pytest.approx(2 / 3)
    assert summary["false_abstention_rate"] == pytest.approx(0.5)
    assert summary["qa_mrr"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_causal_cutoff():
    session = SessionInfo(3, 2, "2026-02-23", "v", 100.0)
    earlier = SessionInfo(3, 1, "2026-02-23", "u", 100.0)
    cutoff = Cutoff(3, 2, 50.0)
    assert cutoff.allows(earlier, 100.0)
    assert cutoff.allows(session, 50.0)
    assert not cutoff.allows(session, 50.5)
    assert not cutoff.allows(SessionInfo(8, 1, "d", "w", 10.0), 1.0)


def test_gaze_box_is_translated_into_frame():
    assert square_box(10, 10, 512) == (0, 0, 512, 512)
    assert square_box(1400, 700, 512) == (896, 444, 1408, 956)


def test_gaze_track_hold_and_fallback_rules():
    t = np.arange(0, 3, 1 / 60)
    u = np.full_like(t, 300.0)
    v = np.full_like(t, 400.0)
    valid = np.ones_like(t, dtype=bool)
    valid[(t > 1.0) & (t < 1.2)] = False  # 200 ms blink: held
    valid[(t > 2.0)] = False  # long gap
    track = GazeTrack(t, u, v, valid)
    assert track.sample(0.5).status == "valid"
    assert track.sample(1.1).status == "held"
    assert track.sample(2.5) is None
    assert track.box_point(2.5).status == "held_1s"
    track_far = GazeTrack(np.array([0.0, 0.1, 5.0]), np.array([1.0, 1.0, np.nan]), np.array([1.0, 1.0, np.nan]), np.array([True, True, False]))
    assert track_far.box_point(4.0).status == "center"


def test_c6_allocation_sums_and_respects_minimum():
    segments = [(0, 10), (10, 70), (70, 75)]
    counts = allocate_frames(segments, [0.0, 1.0, 9.0], 75, alpha=1.0)
    assert sum(counts) == 75 and min(counts) >= 1 and max(counts) <= 60
    uniform = allocate_frames(segments, [0.0, 1.0, 9.0], 75, alpha=0.0)
    assert uniform == [10, 60, 5]


def test_segments_are_between_5_and_60_seconds():
    segments = constrain_segments([1.0, 3.0, 20.0, 22.0, 150.0], 200.0)
    lengths = [end - start for start, end in segments]
    assert segments[0][0] == 0 and segments[-1][1] == 200
    assert all(5 <= length <= 60 for length in lengths)


class FakeSources:
    def box(self, source, video_id, t, side):
        from smvqa_gaze.hpc.gaze_source import GazePoint

        box = square_box(700, 700, side)
        return box, GazePoint(700, 700, "valid")


@pytest.mark.parametrize("name", ["C0", "C1", "C3-T", "C3-V", "C3-M", "C4", "C9-V"])
@pytest.mark.parametrize("tier", ["B1", "B2"])
def test_condition_rendering_matches_token_budget(name, tier):
    frame = np.zeros((1408, 1408, 3), dtype=np.uint8)
    rendered = render_frame(CONDITIONS[name], tier, frame, "v", 1.0, FakeSources())
    side = TIER_SIDE[tier]
    target = (side // 32) ** 2
    tokens = expected_visual_tokens([rendered])
    if name == "C4":
        assert [img.shape[0] for img in rendered.images] == [TIER_DUAL_SIDE[tier]] * 2
        assert abs(tokens / target - 1) < 0.06
    elif name.endswith("-M"):
        assert rendered.images[1].shape[:2] == (128, 128)
        assert tokens == target + 16
    else:
        assert tokens == target
    if name == "C3-T":
        assert '"bbox_2d"' in rendered.text


def test_grounding_coordinates_are_relative_0_1000():
    text = grounding_coordinates((0, 0, 704, 1408), (352.0, 704.0))
    assert text == '{"bbox_2d": [0, 0, 500, 1000], "point_2d": [250, 500]}'
