# Dataset inventory

## Source discovery

The Hugging Face dataset tree was scanned for all 82 released RGB MP4 sessions. RGB is stored independently under `data/video/Person_*/...mp4`; the VRS files are sensor recordings and do not contain the published RGB stream. No official frame timestamp sidecar was found for these MP4s. Therefore frame 0 is aligned to the first `general_eye_gaze.csv` sample, as allowed by the experiment instructions.

The complete inventory records the remote paths, file sizes, duration, dimensions, frame rate, gaze coverage, valid-sample fraction, and first/last gaze timestamps. It is stored outside the source repository at `/mnt/D/smvqa_gaze_data/proc/qa/session_inventory.json`. Of 82 video sessions, 81 have a gaze file. The selector additionally requires a downloadable VRS and gaze coverage of at least 80%.

## Final selected sessions

| Participant | Session | Date | Duration (h) | Gaze coverage | Gaze valid | Video |
|---|---:|---|---:|---:|---:|---|
| 3 | 1 | 2026-02-23 | 0.433 | 99.936% | 100% | 1408×1408, 30 fps |
| 3 | 2 | 2026-02-23 | 0.572 | 99.960% | 100% | 1408×1408, 30 fps |
| 3 | 3 | 2026-03-23 | 0.685 | 100.000% | 100% | 1408×1408, 30 fps |
| 3 | 4 | 2026-03-23 | 0.563 | 99.992% | 100% | 1408×1408, 30 fps |
| 8 | 1 | 2026-04-11 | 0.852 | 100.000% | 100% | 1408×1408, 30 fps |
| 8 | 2 | 2026-04-11 | 0.835 | 100.000% | 100% | 1408×1408, 30 fps |
| 8 | 3 | 2026-04-11 | 0.262 | 99.918% | 100% | 1408×1408, 30 fps |
| 8 | 4 | 2026-04-11 | 0.590 | 100.000% | 100% | 1408×1408, 30 fps |
| 8 | 5 | 2026-04-11 | 0.574 | 99.962% | 100% | 1408×1408, 30 fps |
| 10 | 1 | 2026-04-11 | 0.851 | 99.865% | 100% | 1408×1408, 30 fps |
| 10 | 2 | 2026-04-11 | 0.834 | 100.000% | 100% | 1408×1408, 30 fps |
| 10 | 3 | 2026-04-11 | 0.262 | 100.000% | 100% | 1408×1408, 30 fps |
| 10 | 4 | 2026-04-11 | 0.590 | 99.927% | 100% | 1408×1408, 30 fps |
| 10 | 5 | 2026-04-11 | 0.575 | 99.970% | 100% | 1408×1408, 30 fps |

## Download scope

The final download manifest contains 92 files: 14 MP4s, 14 VRS files, 14 gaze CSVs, and available calibration/transcript assets. It has no sessions outside Persons 3, 8, and 10. Earlier scouting data and unselected candidate sessions are not part of the final transfer package.

After `smvqa-gaze prepare-selected-sessions` completes, per-session projected gaze, fixations, derived frame timestamps, and RGB calibration JSON are summarized in `/mnt/D/smvqa_gaze_data/proc/reports/gaze_processing_report.json`. Calibration prefers the MPS online RGB calibration at the first gaze timestamp and falls back to the VRS device calibration; fallback provenance is retained in each JSON.

The completed projection pass covered all 14 selected sessions. Valid projected gaze fractions ranged from 99.947% to 100.000% (mean 99.985%), with 51,530 I-VT fixations total; 25,377 fixations meet the ≥200 ms H1.1 annotation threshold. Six sessions use VRS static RGB calibration because their MPS online calibration is unavailable: Person 3 session 2; Person 8 sessions 1–2; and Person 10 sessions 1, 2, and 4. Their projection coverage is at least 99.970%.

## Derived gaze files and time axis

All derived gaze files share the released MP4 time axis in seconds, with frame 0 aligned to the first `general_eye_gaze.csv` sample:

| File (per session, `proc/gaze/<video_id>/`) | Time columns | Content |
|---|---|---|
| `gaze_px.parquet` | `t_video_s` (+ device `t_us`) | projected gaze, 60 Hz, upright 1408×1408 pixel coordinates |
| `fixations.parquet` | `t_on`, `t_off` (+ device `t_on_us`, `t_off_us`) | I-VT fixations (main) |
| `fixations_idt.parquet` | same | I-DT fixations (robustness, README L3.5) |
| `fixation_labels.parquet` | `t_on`, `t_off` | H1.1 labels for fixations ≥ 200 ms |
| `episodes.parquet` | `t_on`, `t_off` | H1.1 gaze episodes |

Video and gaze durations agree within 0.13 s in every selected session, which supports the first-sample alignment. The pixel projection follows the Project Aria upright convention with the unrotated RGB intrinsics; compared with projecting through the rotated calibration this leaves a constant offset of roughly 8–18 px (about 1°) at 1408 px, which is kept as is and should be listed with the other gaze error sources in the report.

The selected-session list used by the HPC code is shipped as `docs/selected_sessions.json` in the transfer package.
