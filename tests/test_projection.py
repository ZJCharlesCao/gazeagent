import pytest

from smvqa_gaze.projection import _scale_pixel_coordinates


def test_static_calibration_pixels_scale_to_released_video_size():
    scaled = _scale_pixel_coordinates(
        [1818.886, 1538.34],
        calibration_size=(2880, 2880),
    )

    assert scaled[0] == pytest.approx(889.2332)
    assert scaled[1] == pytest.approx(752.0773)


def test_pixel_scaling_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        _scale_pixel_coordinates([10, 10], calibration_size=(0, 2880))
