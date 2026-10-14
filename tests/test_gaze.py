from smvqa_gaze.gaze import GazeSample, detect_ivt_fixations, gaze_coverage


def test_gaze_coverage_uses_sample_rate_and_validity():
    samples = [
        GazeSample(i * 16_667, 0.1, 0.2, 1.0, 0.1, True)
        for i in range(61)
    ]
    assert 0.99 <= gaze_coverage(samples) <= 1


def test_ivt_detects_long_stable_period():
    samples = [
        GazeSample(i * 16_667, 0.1 + (i % 2) * 1e-5, 0.2, 1.0, 0.1, True)
        for i in range(30)
    ]
    fixations = detect_ivt_fixations(samples)
    assert len(fixations) == 1
    assert fixations[0].dur_ms >= 100
