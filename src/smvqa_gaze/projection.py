from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .gaze import GazeSample


TIME_AXIS_NOTE = (
    "t_video_s, t_on and t_off are seconds on the released MP4 time axis, where "
    "frame 0 is aligned to the first general_eye_gaze.csv sample; t_us, t_on_us "
    "and t_off_us are raw Aria device timestamps."
)
FIXATION_COLUMNS = [
    "fix_id", "t_on", "t_off", "t_on_us", "t_off_us", "dur_ms", "u", "v"
]


def _scale_pixel_coordinates(
    pixel: Any,
    calibration_size: tuple[int, int],
    output_size: tuple[int, int] = (1408, 1408),
) -> tuple[float, float]:
    calibration_width, calibration_height = calibration_size
    output_width, output_height = output_size
    if min(calibration_width, calibration_height, output_width, output_height) <= 0:
        raise ValueError("Calibration and output dimensions must be positive")
    return (
        float(pixel[0]) * output_width / calibration_width,
        float(pixel[1]) * output_height / calibration_height,
    )


def project_gaze_to_rgb(
    gaze_csv: Path,
    vrs_path: Path,
    output_parquet: Path,
    make_upright: bool = True,
) -> dict[str, Any]:
    """Project CPF gaze rays through the RGB calibration supplied by the VRS."""
    try:
        import numpy as np
        import pandas as pd
        from projectaria_tools.core import data_provider
        from projectaria_tools.core import mps
        from projectaria_tools.core.mps import get_eyegaze_point_at_depth
        from projectaria_tools.core.mps import utils as mps_utils
        from projectaria_tools.core.sophus import SE3
    except ImportError as exc:
        raise RuntimeError(
            "Install projectaria-tools and pandas/pyarrow in the local environment"
        ) from exc

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not open VRS file: {vrs_path}")
    device_calibration = provider.get_device_calibration()
    if device_calibration is None:
        raise RuntimeError(f"No device calibration found in {vrs_path}")
    camera_calibration = device_calibration.get_camera_calib("camera-rgb")
    if camera_calibration is None:
        raise RuntimeError(f"No camera-rgb calibration found in {vrs_path}")

    mps_session_dir = gaze_csv.parent.parent
    online_calibration_provider = None
    if (mps_session_dir / "slam" / "online_calibration.jsonl").is_file():
        paths = mps.MpsDataPathsProvider(str(mps_session_dir)).get_data_paths()
        online_calibration_provider = mps.MpsDataProvider(paths)
        if not online_calibration_provider.has_online_calibrations():
            online_calibration_provider = None
    gaze_objects = mps.read_eyegaze(str(gaze_csv))
    output_rows = []
    online_projection_count = 0
    with gaze_csv.open(encoding="utf-8", newline="") as stream:
        for row, gaze in zip(csv.DictReader(stream), gaze_objects, strict=True):
            yaw = (
                float(row["left_yaw_rads_cpf"])
                + float(row["right_yaw_rads_cpf"])
            ) / 2
            pitch = float(row["pitch_rads_cpf"])
            depth = float(row["depth_m"])
            bounds = [
                float(row[field])
                for field in (
                    "left_yaw_low_rads_cpf",
                    "right_yaw_low_rads_cpf",
                    "pitch_low_rads_cpf",
                    "left_yaw_high_rads_cpf",
                    "right_yaw_high_rads_cpf",
                    "pitch_high_rads_cpf",
                )
            ]
            valid = all(
                math.isfinite(value) for value in (yaw, pitch, depth, *bounds)
            ) and depth > 0
            pixel = None
            if valid:
                timestamp_ns = int(row["tracking_timestamp_us"]) * 1_000
                online_calibration = (
                    online_calibration_provider.get_online_calibration(timestamp_ns)
                    if online_calibration_provider is not None
                    else None
                )
                online_camera = (
                    online_calibration.get_camera_calib("camera-rgb")
                    if online_calibration is not None
                    else None
                )
                if online_camera is not None and list(online_camera.get_image_size()) == [
                    1408,
                    1408,
                ]:
                    transform_device_cpf = device_calibration.get_transform_device_cpf()
                    transform_device_camera = online_camera.get_transform_device_camera()
                    if make_upright:
                        transform_camera_cw90 = SE3.from_matrix(
                            np.array(
                                [
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                                ]
                            )
                        )
                        transform_device_camera = (
                            transform_device_camera @ transform_camera_cw90
                        )
                    transform_camera_cpf = (
                        transform_device_camera.inverse() @ transform_device_cpf
                    )
                    point_cpf = get_eyegaze_point_at_depth(gaze.yaw, gaze.pitch, depth)
                    pixel = online_camera.project(transform_camera_cpf @ point_cpf)
                    online_projection_count += 1
                else:
                    pixel = mps_utils.get_gaze_vector_reprojection(
                        gaze,
                        "camera-rgb",
                        device_calibration,
                        camera_calibration,
                        depth_m=depth,
                        make_upright=make_upright,
                    )
                    if pixel is not None:
                        calibration_size = tuple(
                            int(value) for value in camera_calibration.get_image_size()
                        )
                        pixel = _scale_pixel_coordinates(
                            pixel, calibration_size, (1408, 1408)
                        )
            if pixel is None:
                u = v = None
                valid = False
            else:
                u, v = float(pixel[0]), float(pixel[1])
                valid = math.isfinite(u) and math.isfinite(v)
                if not valid:
                    u = v = None
            yaw_half_width = (
                float(row["left_yaw_high_rads_cpf"])
                + float(row["right_yaw_high_rads_cpf"])
                - float(row["left_yaw_low_rads_cpf"])
                - float(row["right_yaw_low_rads_cpf"])
            ) / 4
            pitch_half_width = (
                float(row["pitch_high_rads_cpf"])
                - float(row["pitch_low_rads_cpf"])
            ) / 2
            output_rows.append(
                {
                    "t_us": int(row["tracking_timestamp_us"]),
                    "u": u,
                    "v": v,
                    "valid": valid,
                    "depth_m": depth,
                    "unc_deg": math.degrees(
                        math.hypot(yaw_half_width, pitch_half_width)
                    ),
                }
            )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    first_t_us = output_rows[0]["t_us"] if output_rows else 0
    for row in output_rows:
        # Shared session time axis: video frame 0 is the first gaze sample.
        row["t_video_s"] = (row["t_us"] - first_t_us) / 1_000_000
    pd.DataFrame(
        output_rows,
        columns=["t_us", "t_video_s", "u", "v", "valid", "depth_m", "unc_deg"],
    ).to_parquet(output_parquet, index=False)
    return {
        "samples": len(output_rows),
        "projected_fraction": sum(row["valid"] for row in output_rows)
        / max(1, len(output_rows)),
        "in_frame_fraction": sum(
            row["valid"]
            and row["u"] is not None
            and row["v"] is not None
            and 0 <= row["u"] < 1408
            and 0 <= row["v"] < 1408
            for row in output_rows
        )
        / max(1, len(output_rows)),
        "online_calibration_samples": online_projection_count,
        "calibration_source": "MPS online calibration, with VRS device calibration fallback",
        "make_upright": make_upright,
    }


def export_rgb_calibration(
    gaze_csv: Path,
    vrs_path: Path,
    output_json: Path,
    output_size: tuple[int, int] = (1408, 1408),
) -> dict[str, Any]:
    try:
        import numpy as np
        from projectaria_tools.core import data_provider, mps
    except ImportError as exc:
        raise RuntimeError("Install projectaria-tools to export RGB calibration") from exc

    with gaze_csv.open(encoding="utf-8", newline="") as stream:
        first_row = next(csv.DictReader(stream), None)
    if first_row is None:
        raise ValueError(f"Gaze CSV is empty: {gaze_csv}")
    timestamp_us = int(first_row["tracking_timestamp_us"])

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"Could not open VRS file: {vrs_path}")
    device_calibration = provider.get_device_calibration()
    if device_calibration is None:
        raise RuntimeError(f"No device calibration found in {vrs_path}")
    camera = device_calibration.get_camera_calib("camera-rgb")
    if camera is None:
        raise RuntimeError(f"No camera-rgb calibration found in {vrs_path}")

    source = "VRS device calibration"
    camera_timestamp_us = None
    mps_session_dir = gaze_csv.parent.parent
    calibration_path = mps_session_dir / "slam" / "online_calibration.jsonl"
    if calibration_path.is_file():
        paths = mps.MpsDataPathsProvider(str(mps_session_dir)).get_data_paths()
        online_provider = mps.MpsDataProvider(paths)
        if online_provider.has_online_calibrations():
            online_camera_calibration = online_provider.get_online_calibration(
                timestamp_us * 1_000
            )
            if online_camera_calibration is not None:
                online_camera = online_camera_calibration.get_camera_calib(
                    "camera-rgb"
                )
                if online_camera is not None:
                    camera = online_camera
                    source = "MPS online calibration at first gaze sample"
                    camera_timestamp_us = timestamp_us

    calibration_width, calibration_height = (
        int(value) for value in camera.get_image_size()
    )
    output_width, output_height = output_size
    if min(calibration_width, calibration_height, output_width, output_height) <= 0:
        raise ValueError("RGB calibration and output image sizes must be positive")
    scale_x = output_width / calibration_width
    scale_y = output_height / calibration_height
    focal_lengths = np.asarray(camera.get_focal_lengths(), dtype=float)
    principal_point = np.asarray(camera.get_principal_point(), dtype=float)
    payload = {
        "image_size_px": [output_width, output_height],
        "calibration_image_size_px": [calibration_width, calibration_height],
        "focal_lengths_px": [
            float(focal_lengths[0] * scale_x),
            float(focal_lengths[1] * scale_y),
        ],
        "principal_point_px": [
            float(principal_point[0] * scale_x),
            float(principal_point[1] * scale_y),
        ],
        "t_device_camera": np.asarray(
            camera.get_transform_device_camera().to_matrix(), dtype=float
        ).tolist(),
        "t_device_cpf": np.asarray(
            device_calibration.get_transform_device_cpf().to_matrix(), dtype=float
        ).tolist(),
        "source": source,
        "camera_calibration_timestamp_us": camera_timestamp_us,
        "coordinate_note": (
            "Intrinsics are scaled to the released RGB dimensions. The source "
            "extrinsics are T_device_camera and T_device_CPF, expressed in the "
            "Aria device frame."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_fixations_parquet(
    gaze_csv: Path, gaze_parquet: Path, output_parquet: Path
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas and pyarrow to write Parquet outputs") from exc
    from .gaze import detect_ivt_fixations, read_gaze_csv

    samples = read_gaze_csv(gaze_csv)
    projected = pd.read_parquet(gaze_parquet)
    if len(samples) != len(projected):
        raise ValueError("Projected gaze row count does not match the source CSV")
    projected_samples = [
        GazeSample(
            timestamp_us=sample.timestamp_us,
            yaw_rad=sample.yaw_rad,
            pitch_rad=sample.pitch_rad,
            depth_m=sample.depth_m,
            uncertainty_deg=sample.uncertainty_deg,
            valid=sample.valid and bool(projected.iloc[index]["valid"]),
            u=projected.iloc[index]["u"],
            v=projected.iloc[index]["v"],
        )
        for index, sample in enumerate(samples)
    ]
    fixations = detect_ivt_fixations(projected_samples)
    first_t_us = samples[0].timestamp_us if samples else 0
    rows = [
        {
            "fix_id": fixation.fix_id,
            # t_on/t_off share the video time axis used by every H1/HPC output;
            # the raw Aria device timestamps are kept in *_us columns.
            "t_on": (fixation.t_on_us - first_t_us) / 1_000_000,
            "t_off": (fixation.t_off_us - first_t_us) / 1_000_000,
            "t_on_us": fixation.t_on_us,
            "t_off_us": fixation.t_off_us,
            "dur_ms": fixation.dur_ms,
            "u": fixation.u,
            "v": fixation.v,
        }
        for fixation in fixations
    ]
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIXATION_COLUMNS).to_parquet(output_parquet, index=False)
    return {"fixation_count": len(rows), "time_axis": TIME_AXIS_NOTE}
