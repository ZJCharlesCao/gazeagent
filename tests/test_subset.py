import pytest

from smvqa_gaze.subset import Session, choose_prefix_subset


def make_qa(participant: int, session: int, counts: dict[str, int]) -> list[dict]:
    rows = []
    for category, count in counts.items():
        rows.extend(
            {
                "participant": participant,
                "primary_session": session,
                "category_key": category,
            }
            for _ in range(count)
        )
    return rows


def test_subset_uses_shortest_prefix_with_multiday_coverage():
    categories = (
        "object_location_memory",
        "visual_recall",
        "intent_recall",
        "conversational_memory",
    )
    sessions = [
        Session(1, 1, "2026-01-01", 300, 0.99),
        Session(1, 2, "2026-01-01", 300, 0.99),
        Session(1, 3, "2026-01-02", 300, 0.99),
    ]
    rows = make_qa(1, 1, {category: 40 for category in categories})
    rows += make_qa(1, 2, {category: 40 for category in categories})
    rows += make_qa(1, 3, {category: 40 for category in categories})
    candidate = choose_prefix_subset(rows, sessions)
    assert candidate.cutoffs == ((1, 3),)
    assert candidate.duration_s == 900
    assert candidate.counts == (120, 120, 120, 120)


def test_subset_rejects_session_gaps_and_low_coverage():
    categories = (
        "object_location_memory",
        "visual_recall",
        "intent_recall",
        "conversational_memory",
    )
    rows = make_qa(1, 1, {category: 120 for category in categories})
    sessions = [
        Session(1, 1, "2026-01-01", 300, 0.99),
        Session(1, 3, "2026-01-02", 300, 0.99),
    ]
    with pytest.raises(ValueError, match="No valid subset"):
        choose_prefix_subset(rows, sessions)


def test_subset_rejects_prefixes_with_unavailable_rgb_video():
    categories = (
        "object_location_memory",
        "visual_recall",
        "intent_recall",
        "conversational_memory",
    )
    rows = make_qa(1, 1, {category: 120 for category in categories})
    rows += make_qa(1, 2, {category: 120 for category in categories})
    sessions = [
        Session(1, 1, "2026-01-01", 300, 0.99, video_available=True),
        Session(1, 2, "2026-01-02", 300, 0.99, video_available=False),
    ]

    with pytest.raises(ValueError, match="No valid subset"):
        choose_prefix_subset(rows, sessions)
