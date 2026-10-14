import json
from pathlib import Path

import pytest

# H4 runs in the HPC environment; the local environment has no statsmodels.
pytest.importorskip("statsmodels")

import cv2
import numpy as np
import pandas as pd
import pytest

from smvqa_gaze.hpc import analysis
from smvqa_gaze.hpc.analysis import (
    cluster_bootstrap,
    compare_runs,
    fit_logit_model,
    holm,
    hypothesis_tests,
    hypothesis_verdicts,
    load_questions,
    load_runs,
    main_model,
    mcnemar_test,
    question_nss,
    render_mosaic,
    run_analysis,
    run_metrics,
    table_t1,
    write_failure_cases,
)
from smvqa_gaze.hpc.config import Paths, RunConfig

PARTICIPANTS = (1, 2, 3)
CATEGORIES = ("object_location_memory", "visual_recall", "intent_recall", "conversational_memory")
FE_CYCLE = ("fixated", "never", "glanced", "fixated", "never", "undetected")
CONDITIONS = ("C0", "C1", "C3-V", "C8shift-V", "C9-V", "C5", "C6a0", "C6a0.5", "C6a1")
DROPPED = {("egobutler", "B2", "C6a1"), ("egobutler", "B2", "C9-V")}
N_QUESTIONS = 60
N_TEST = 54


def video_id(participant: int) -> str:
    return f"Person_{participant}_s1"


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_questions():
    rows = []
    for i in range(N_QUESTIONS):
        participant = PARTICIPANTS[i % 3]
        answerable = i % 10 != 9
        t0 = 5.0 + (i // 3) * 4.0
        rows.append(
            {
                "question_id": f"q{i:03d}",
                "participant": participant,
                "primary_session": 1,
                "question_time_s": 200.0,
                "category_key": CATEGORIES[i % 4],
                "split": "test" if i < N_TEST else "dev",
                "question": f"Where did I put item {i}?",
                "choices": [
                    {"text": "on the table", "type": "correct"},
                    {"text": "somewhere", "type": "vague"},
                    {"text": "in the car", "type": "wrong"},
                    {"text": "cannot be determined", "type": "unanswerable"},
                ],
                "correct_option_index": 0 if answerable else 3,
                "abstain_option_index": 3,
                "is_answerable": answerable,
                "answer_evidence": [
                    {"video_id": video_id(participant), "t_start": t0, "t_end": t0 + 4.0, "modalities": ["visual"]}
                ],
            }
        )
    return rows


def p_correct(condition: str, fe: str, tier: str) -> float:
    base = 0.55 + (0.05 if tier == "B2" else 0.0)
    if condition == "C1":
        return {"fixated": 0.75, "never": 0.1}.get(fe, 0.4)
    if condition in ("C5", "C3-V", "C6a1") and fe == "fixated":
        return 0.85
    if condition in ("C8shift-V", "C9-V") and fe == "fixated":
        return 0.45
    return base


def make_prediction(question, pipeline, tier, condition, fe, rng):
    answerable = question["is_answerable"]
    correct = bool(rng.random() < p_correct(condition, fe, tier))
    abstain = (not answerable and correct) or (answerable and not correct and condition == "C1" and rng.random() < 0.5)
    correct_index = question["correct_option_index"]
    if correct:
        top1 = correct_index
    elif abstain:
        top1 = 3
    else:
        top1 = int(rng.choice([i for i in (1, 2, 3, 0) if i != correct_index][:2]))
    ranking = [top1] + [i for i in range(4) if i != top1]
    vid = video_id(question["participant"])
    gaze = condition not in ("C0",)
    row = {
        "key": question["question_id"],
        "question_id": question["question_id"],
        "participant": question["participant"],
        "session": 1,
        "category": question["category_key"],
        "split": question["split"],
        "fe_status": fe,
        "is_answerable": answerable,
        "option_order": [0, 1, 2, 3],
        "raw_output": "ABCD"[ranking[0]] + ">" + ">".join("ABCD"[i] for i in ranking[1:]),
        "parse_status": "ok",
        "ranking": ranking,
        "top1": top1,
        "top1_type": question["choices"][top1]["type"],
        "correct": correct,
        "rank_of_correct": ranking.index(correct_index) + 1,
        "abstained": top1 == 3,
        "has_vague": True,
        "prompt_tokens": 1000,
        "latency_s": 1.0,
    }
    span = question["answer_evidence"][0]
    if pipeline == "videorag":
        times = [float(t) for t in range(0, 32, 4)]
        row["retrieved"] = [{"video_id": vid, "start": 0.0, "end": 1800.0, "score": 1.0, "db": "SHARD"}]
        row["retrieval_hit"] = bool(rng.random() < (0.8 if condition == "C5" else 0.6))
        row["k"] = 1
        row["frame_times"] = times
        row["frame_meta"] = (
            [{"t": t, "gaze_box": [400, 400, 912, 912], "gaze_point": [656.0, 656.0], "gaze_status": "valid"} for t in times]
            if gaze else None
        )
    else:
        starts = [0.0, 30.0, 60.0]
        row["retrieved"] = [{"video_id": vid, "start": s, "end": s + 30.0, "score": 0.5} for s in starts]
        row["retrieval_hit"] = any(s < span["t_end"] and s + 30 > span["t_start"] for s in starts)
        row["k"] = 5
        row["frame_meta"] = None
    row["target_coverage"] = bool(rng.random() < 0.5)
    return row


def build_workdir(root: Path, runs=None, dev_run: bool = True) -> Paths:
    data, work = root / "data", root / "work"
    paths = Paths(data=data, work=work)
    questions = make_questions()
    write_jsonl(paths.qa_subset, questions)
    labels = [
        {"question_id": q["question_id"], "category_key": q["category_key"], "status": FE_CYCLE[i % len(FE_CYCLE)]}
        for i, q in enumerate(questions)
    ]
    write_jsonl(paths.fe_labels, labels)
    paths.sessions.parent.mkdir(parents=True, exist_ok=True)
    paths.sessions.write_text(
        json.dumps(
            [
                {"participant": p, "session": 1, "date": "2024-01-01", "video_id": video_id(p), "duration_s": 240.0}
                for p in PARTICIPANTS
            ]
        )
    )
    rng = np.random.default_rng(1)
    for p in PARTICIPANTS:
        t = np.arange(0, 240, 1 / 30)
        gaze_dir = paths.gaze_dir(video_id(p))
        gaze_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "t_us": (t * 1e6).astype(np.int64),
                "t_video_s": t,
                "u": 704 + 100 * p * np.sin(t),
                "v": 704 + 80 * np.cos(t / 2),
                "valid": rng.random(t.size) > 0.05,
            }
        ).to_parquet(gaze_dir / "gaze_px.parquet")
        n_fix = 40 * p
        pd.DataFrame(
            {
                "fix_id": np.arange(n_fix),
                "t_on": np.linspace(0, 230, n_fix),
                "t_off": np.linspace(0, 230, n_fix) + 0.3,
                "dur_ms": 300.0,
                "u": 704 + rng.normal(0, 60 * p, n_fix),
                "v": 704 + rng.normal(0, 40, n_fix),
            }
        ).to_parquet(gaze_dir / "fixations.parquet")
        saliency_dir = paths.saliency_dir(video_id(p))
        saliency_dir.mkdir(parents=True, exist_ok=True)
        for second in range(0, 100):
            image = (rng.random((32, 32)) * 255).astype(np.uint8)
            cv2.imwrite(str(saliency_dir / f"frame_{second:08d}.jpg"), image)
    frame_dir = paths.frame_cache / video_id(1)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for second in (0, 4, 8):
        cv2.imwrite(str(frame_dir / f"{second:06d}.jpg"), np.full((64, 64, 3), 120, np.uint8))

    if runs is None:
        runs = [
            (pipeline, tier, condition)
            for pipeline in ("videorag", "egobutler")
            for tier in ("B1", "B2")
            for condition in CONDITIONS
            if (pipeline, tier, condition) not in DROPPED
        ]
    fe_of = {label["question_id"]: label["status"] for label in labels}
    for index, (pipeline, tier, condition) in enumerate(runs):
        run_rng = np.random.default_rng(100 + index)
        rows = [make_prediction(q, pipeline, tier, condition, fe_of[q["question_id"]], run_rng) for q in questions]
        write_jsonl(paths.pred_dir(pipeline, tier, condition) / "predictions.jsonl", rows)
        if pipeline == "egobutler":
            units = [
                {
                    "video_id": video_id(p),
                    "start": s,
                    "end": s + 30.0,
                    "caption": "a kitchen",
                    "frame_times": [s + k for k in range(0, 30, 4)],
                    "frame_meta": (
                        [{"t": s + k, "gaze_box": [100, 100, 612, 612]} for k in range(0, 30, 4)]
                        if condition != "C0" else None
                    ),
                }
                for p in PARTICIPANTS
                for s in (0.0, 30.0, 60.0)
            ]
            write_jsonl(paths.memory_dir(pipeline, tier, condition) / "units.jsonl", units)
    if dev_run:
        dev_rows = [make_prediction(q, "videorag", "B1", "C0", "fixated", rng) for q in questions[:5]]
        write_jsonl(paths.pred_dir("videorag", "B1", "C0_dev_n20") / "predictions.jsonl", dev_rows)
    return paths


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    paths = build_workdir(tmp_path_factory.mktemp("full"))
    summary = run_analysis(paths, RunConfig(), split="test", make_figures=True, n_boot=200)
    return paths, summary


def test_run_analysis_writes_tables_figures_and_report(analysed):
    paths, summary = analysed
    results = paths.results
    expected_runs = 2 * 2 * len(CONDITIONS) - len(DROPPED)
    assert summary["n_runs"] == expected_runs
    assert summary["errors"] == []
    assert summary["n_rows"] == expected_runs * N_TEST  # dev split and dev run excluded
    for name in (
        "T0_fe_by_category", "T1_main", "T2_by_category", "T3_category_by_fe",
        "T4_budget", "T5_personalization", "T6_answerability", "T7_retrieval_memory",
    ):
        assert (results / f"{name}.csv").is_file()
        assert (results / f"{name}.md").read_text().startswith("### T")
    metrics = pd.read_csv(results / "metrics_all.csv")
    assert len(metrics) == expected_runs
    assert set(metrics["n"]) == {N_TEST}
    assert "C0_dev_n20" not in set(metrics["condition"])

    figs = results / "figs"
    for name in ("fig1_fe_by_category", "fig3_budget_slopes", "fig4_personalization",
                 "fig5_pg_vs_gaze_behaviour", "fig6_c6_alpha_dose"):
        assert (figs / f"{name}.png").is_file(), name
    assert len(list(figs.glob("fig2_condition_x_fe_*.png"))) == 4

    for name in ("bootstrap_acc", "pairwise_vs_c0", "hypothesis_tests", "lopo", "nss_by_question",
                 "participant_gaze_stats"):
        assert (results / "stats" / f"{name}.csv").is_file(), name
    nss = pd.read_csv(results / "stats" / "nss_by_question.csv")
    assert len(nss) == N_TEST
    assert set(nss["nss_group"].dropna()) == {"conventional", "idiosyncratic"}

    assert set(summary["verdicts"]) == {"H1", "H2", "H3", "H4", "H5", "H6"}
    assert set(summary["verdicts"].values()) <= {"supported", "not supported", "inconclusive"}
    # C8cross was never run, so H4 (which requires own > cross) cannot be decided.
    assert summary["verdicts"]["H4"] == "inconclusive"

    report = (results / "REPORT.md").read_text()
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6. Known limitations"):
        assert heading in report
    assert "TODO(human)" in report
    assert "C6a1: egobutler B2" in report
    assert "fig1_fe_by_category.png" in report
    assert summary["n_failure_cases"] > 0
    assert (results / "failure_cases" / "INDEX.md").is_file()
    assert list((results / "failure_cases").glob("videorag_B1_C1/*.png"))
    assert (results / "manifest.json").is_file()


def test_t1_shape_and_missing_cells(analysed):
    paths, _ = analysed
    questions = load_questions(paths)
    df = load_runs(paths, "test", questions)
    table = table_t1(run_metrics(df))
    assert table.display.shape == (len(CONDITIONS), 4)
    assert list(table.display.columns) == ["videorag B1", "videorag B2", "egobutler B1", "egobutler B2"]
    assert table.display.loc["C6a1", "egobutler B2"] == ""
    assert table.display.loc["C0", "videorag B1"].count("/") == 2


def test_pairwise_p_values_are_probabilities(analysed):
    paths, _ = analysed
    pairwise = pd.read_csv(paths.results / "stats" / "pairwise_vs_c0.csv")
    assert len(pairwise)
    assert pairwise["p"].between(0, 1).all()
    assert pairwise["p_holm"].between(0, 1).all()
    assert (pairwise["p_holm"] >= pairwise["p"] - 1e-12).all()
    tests = pd.read_csv(paths.results / "stats" / "hypothesis_tests.csv")
    decided = tests.dropna(subset=["p"])
    assert decided["p"].between(0, 1).all() and decided["p_holm"].between(0, 1).all()


def test_mcnemar_and_holm():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    b = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1])
    result = mcnemar_test(a, b)
    assert result["n10"] == 3 and result["n01"] == 0 and result["exact"]
    assert result["p"] == pytest.approx(0.25)
    assert mcnemar_test(a, a)["p"] == 1.0
    big = mcnemar_test(np.r_[np.ones(40), np.zeros(10)], np.r_[np.zeros(30), np.ones(20)])
    assert not big["exact"] and 0 <= big["p"] <= 1
    adjusted = holm([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(adjusted[2])
    assert adjusted[0] == pytest.approx(0.03)
    assert np.all(adjusted[[0, 1, 3]] <= 1) and np.all(adjusted[[0, 1, 3]] >= [0.01, 0.04, 0.03])
    from statsmodels.stats.multitest import multipletests

    raw = np.random.default_rng(3).random(17) ** 3
    np.testing.assert_allclose(holm(raw), multipletests(raw, method="holm")[1])


def test_cluster_bootstrap_is_deterministic():
    rng = np.random.default_rng(5)
    values = rng.integers(0, 2, 90).astype(float)
    clusters = np.repeat([1, 2, 3], 30)
    first = cluster_bootstrap(values, clusters, n_boot=1000, seed=0)
    second = cluster_bootstrap(values, clusters, n_boot=1000, seed=0)
    other = cluster_bootstrap(values, clusters, n_boot=1000, seed=1)
    np.testing.assert_array_equal(first["boot"], second["boot"])
    assert first["ci_lo"] == second["ci_lo"] and first["ci_hi"] == second["ci_hi"]
    assert not np.array_equal(first["boot"], other["boot"])
    assert first["ci_lo"] <= first["estimate"] <= first["ci_hi"]
    assert first["estimate"] == pytest.approx(values.mean())
    empty = cluster_bootstrap(np.array([np.nan]), np.array([1]))
    assert empty["n"] == 0 and np.isnan(empty["estimate"])


def test_model_fallback_path(analysed, monkeypatch):
    paths, _ = analysed
    df = load_runs(paths, "test")
    fixed = main_model(df, "videorag", prefer_mixed=False)
    assert fixed["model_used"].startswith("GLM")
    assert "C(participant)" in fixed["formula"]
    assert any("fe[T.never]" in term for term in fixed["key_terms"]["term"])

    from statsmodels.genmod import bayes_mixed_glm

    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(bayes_mixed_glm.BinomialBayesMixedGLM, "from_formula", classmethod(broken))
    data = pd.DataFrame(
        {
            "correct_i": [0, 1, 1, 0, 1, 0, 1, 1],
            "x": ["a", "b"] * 4,
            "participant": ["1", "1", "2", "2", "3", "3", "1", "2"],
        }
    )
    result = fit_logit_model(data, "correct_i ~ x", {"participant": "0 + C(participant)"})
    assert result["model_used"].startswith("GLM")
    assert "mixed model failed" in result["reason"] and "singular" in result["reason"]
    assert len(result["terms"]) >= 2


def test_mixed_model_path_runs(analysed):
    paths, _ = analysed
    df = load_runs(paths, "test")
    result = main_model(df, "egobutler", prefer_mixed=True)
    assert result["model_used"] in ("BinomialBayesMixedGLM (variational Bayes)", "GLM Binomial with participant as fixed effect")
    assert not result["terms"].empty


def test_missing_runs_do_not_crash(tmp_path):
    paths = build_workdir(tmp_path, runs=[("videorag", "B1", "C0"), ("videorag", "B1", "C1")], dev_run=False)
    summary = run_analysis(paths, RunConfig(), make_figures=True, n_boot=50)
    assert summary["n_runs"] == 2
    assert summary["errors"] == []
    assert summary["verdicts"]["H4"] == "inconclusive"
    assert summary["verdicts"]["H5"] == "inconclusive"
    report = (paths.results / "REPORT.md").read_text()
    assert "C3-T: videorag B1, videorag B2, egobutler B1, egobutler B2" in report
    t1 = (paths.results / "T1_main.md").read_text()
    assert "egobutler B2" in t1


def test_no_runs_at_all(tmp_path):
    paths = build_workdir(tmp_path, runs=[], dev_run=False)
    summary = run_analysis(paths, RunConfig(), make_figures=False, n_boot=10)
    assert summary["n_runs"] == 0
    assert set(summary["verdicts"].values()) == {"inconclusive"}
    assert (paths.results / "REPORT.md").is_file()


def test_failure_cases_tolerate_missing_frames(analysed, tmp_path):
    paths, _ = analysed
    output = render_mosaic(
        Paths(data=tmp_path, work=tmp_path),
        [{"video_id": "nope", "t": 3.0, "box": [0, 0, 512, 512]}],
        ["question?", "chosen: x", "correct: y"],
        tmp_path / "case.png",
    )
    assert cv2.imread(str(output)) is not None
    text_only = render_mosaic(paths, [], ["no frames at all"], tmp_path / "text.png")
    assert cv2.imread(str(text_only)) is not None

    # Same predictions, but without any cached frames or memory units.
    empty_work = Paths(data=paths.data, work=tmp_path / "work")
    df = load_runs(paths, "test")
    cases = write_failure_cases(empty_work, df, load_questions(paths))
    assert cases and all((empty_work.results / case["image"]).is_file() for case in cases)
    ego = [case for case in cases if case["pipeline"] == "egobutler"]
    assert ego and all(case["n_frames"] == 0 for case in ego)


def test_nss_is_high_when_gaze_hits_saliency_peak(tmp_path):
    paths = Paths(data=tmp_path / "data", work=tmp_path / "work")
    gaze_dir = paths.gaze_dir("vid")
    gaze_dir.mkdir(parents=True)
    t = np.arange(0, 5, 1 / 100)
    pd.DataFrame({"t_video_s": t, "u": np.full(t.size, 1000.0), "v": np.full(t.size, 300.0), "valid": True}).to_parquet(
        gaze_dir / "gaze_px.parquet"
    )
    saliency_dir = paths.saliency_dir("vid")
    saliency_dir.mkdir(parents=True)
    image = np.zeros((88, 88), np.uint8)
    image[300 * 88 // 1408 - 3: 300 * 88 // 1408 + 4, 1000 * 88 // 1408 - 3: 1000 * 88 // 1408 + 4] = 255
    for second in range(6):
        cv2.imwrite(str(saliency_dir / f"frame_{second:08d}.jpg"), image)
    nss, n_samples, n_maps = question_nss(paths, [{"video_id": "vid", "t_start": 1.0, "t_end": 3.0}], {}, max_hz=10)
    assert nss > 3
    assert n_samples <= 21  # subsampled to <= 10 Hz
    assert n_maps == 3  # seconds 1, 2, 3 each loaded once


def test_hypothesis_rules_on_constructed_effect():
    rows = []
    for pipeline in ("videorag",):
        for tier in ("B1",):
            for condition in ("C0", "C1"):
                for i in range(120):
                    fe = "never" if i % 2 else "fixated"
                    correct = True if condition == "C0" else fe == "fixated"
                    rows.append({"pipeline": pipeline, "tier": tier, "condition": condition, "question_id": f"q{i}",
                                 "participant": i % 3, "fe": fe, "correct": correct, "is_answerable": True,
                                 "abstained": False, "retrieval_hit": np.nan, "target_coverage": np.nan})
    df = pd.DataFrame(rows)
    contrast = compare_runs(df, "videorag", "B1", "C1", "C0", subset="never", n_boot=200)
    assert contrast["diff"] == pytest.approx(-1.0) and contrast["p"] < 1e-6
    verdicts = {v["hypothesis"]: v for v in hypothesis_verdicts(hypothesis_tests(df, n_boot=200))}
    assert verdicts["H2"]["verdict"] == "supported"
    assert verdicts["H1"]["verdict"] == "inconclusive"
    assert analysis.DECISION_RULES.count("\n") > 5
