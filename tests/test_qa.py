from smvqa_gaze.qa import normalize_qa
from smvqa_gaze.subset import assign_splits, filter_subset


def sample_row(question_id: int, session: int, category: str) -> dict:
    return {
        "question_id": question_id,
        "question": "Where is the object?",
        "choices": [
            "This question can not be answered.",
            "on the table",
            "vague",
            "wrong",
        ],
        "correct_answer": "on the table",
        "correct_option_index": 1,
        "choice_types": ["incorrect", "correct", "vague", "incorrect"],
        "subject": 2,
        "metadata": {
            "skill": category,
            "primary_video_id": f"Person_2_session_{session}_03102026_glasses_1284",
        },
        "video_ids": [],
        "start_time": 1770000000,
        "question_evidence": {
            "time_spans": [
                {
                    "video_id": "Person_2_session_3_03102026_glasses_1284",
                    "start_time": 9,
                    "end_time": 12,
                }
            ]
        },
        "is_answerable": True,
        "answer_evidence": {
            "evidence_list": [
                {
                    "video_id": "Person_2_session_1_03102026_glasses_1284",
                    "time_span": {"start_time": 10, "end_time": 12},
                    "modalities": ["Video"],
                }
            ]
        },
    }


def test_normalize_qa_maps_schema_and_evidence():
    row = normalize_qa(sample_row(1, 3, "object_location_memory"))
    assert row["category"] == "Object & Location Memory"
    assert row["participant"] == 2
    assert row["primary_session"] == 3
    assert row["choices"][0]["type"] == "unanswerable"
    assert row["choices"][0]["is_abstain"] and not row["choices"][0]["is_correct"]
    assert row["choices"][1]["type"] == "correct"
    assert row["choices"][1]["is_correct"]
    assert row["abstain_option_index"] == 0
    assert row["answer_evidence"][0]["t_start"] == 10
    assert row["question_time_s"] == 12
    assert row["question_time_unix"] == 1770000012


def test_unanswerable_question_marks_abstention_as_correct_option():
    raw = sample_row(2, 3, "intent_recall")
    raw["choices"] = [
        "This question can not be answered.",
        "This question cannot be answered because B left early.",
        "on the table",
        "in the drawer",
    ]
    raw["choice_types"] = ["correct", "incorrect", "incorrect", "incorrect"]
    raw["correct_option_index"] = 0
    raw["is_answerable"] = False
    row = normalize_qa(raw)
    assert [choice["type"] for choice in row["choices"]] == [
        "unanswerable",
        "wrong",
        "wrong",
        "wrong",
    ]
    assert row["choices"][0]["is_correct"] and row["choices"][0]["is_abstain"]
    assert not row["choices"][1]["is_abstain"]


def test_subset_filter_and_split_are_deterministic():
    raw = [
        normalize_qa(sample_row(i, i % 3 + 1, "visual_recall"))
        for i in range(1, 31)
    ]
    selected = filter_subset(raw, {2: 2})
    first = assign_splits(selected, seed=33)
    second = assign_splits(selected, seed=33)
    assert first == second
    assert len(first) == 20
    assert sum(row["split"] == "dev" for row in first) == 3
