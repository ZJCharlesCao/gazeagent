from smvqa_gaze.download import _required_paths


def test_required_paths_include_published_rgb_video():
    required, optional = _required_paths(
        {
            "participant": 3,
            "video_id": "Person_3_session_5_03242026_glasses_1283",
        }
    )

    assert (
        "data/video/Person_3/Person_3_session_5_03242026_glasses_1283.mp4"
        in required
    )
    assert (
        "data/raw/person_3/person_3_session_5_03242026_glasses_1283.vrs"
        in required
    )
    assert not any(path.startswith("data/video/") for path in optional)
