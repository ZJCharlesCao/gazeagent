# Local environment report

## Host and GPU

- OS: Linux
- Local GPUs: 2 × NVIDIA GeForce RTX 3090 (24 GiB each)
- GPU 0 is occupied by an unrelated Ollama process; H1 commands default to `CUDA_VISIBLE_DEVICES=1`.
- The available local H1 environment is `/mnt/D/smvqa_gaze_env_local`, created from `env/environment_local.yml`.

## Installed/verified components

- Python 3.11
- PyTorch 2.14.1+cu130; CUDA available and both GPUs are detected.
- Transformers 5.19.0; Qwen3-VL model classes and GroundingDINO auto classes import successfully.
- Accelerate 1.15.0; Project Aria Tools, OpenCV, pandas, pyarrow, and `qwen-vl-utils` are installed.
- GroundingDINO's current Transformers processor API is `post_process_grounded_object_detection`; H1.3 uses the `IDEA-Research/grounding-dino-tiny` checkpoint with 0.25 box/text thresholds.
- Projected-gaze and crop/classification helper tests pass. All 14 selected sessions have projected gaze, fixations, frame maps, and calibration JSON. Minimum projected-valid fraction is 99.947%; mean is 99.985%.
- Qwen3-VL weights are cached at `/mnt/D/smvqa_gaze_data/models/huggingface`; snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

## Model execution status

- **H1.1 fixation labeling: complete.** Qwen3-VL-8B processed 25,377 eligible fixations across 14 sessions and produced 16,511 gaze episodes. Runtime was approximately 4,352 s (72.5 min); peak GPU memory was about 16.37 GiB allocated / 16.39 GiB reserved. Model weights are cached locally; the Hugging Face snapshot is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- **H1.2 projection QC: complete.** Qwen reviewed 150 distinct video frames (50 per participant). The `yes` rates were P3 42%, P8 8%, and P10 18%. These automated judgments are not a substitute for the pending manual visual review; the overlay samples are in `results/qc/gaze_overlay/`.
- **H1.3 target extraction and fixation-state labeling: re-run on 2026-10-13 after a review.** Changes against the first run: OCR/Gaze/Depth evidence now counts as visual (OCR-only Visual Scene Recall questions were previously `NA`); `NA`, `unanswerable` and `no_target` are separate statuses; per frame only the highest-scoring box per target is kept and boxes covering > 25% of the frame are discarded (previously 180 of 282 spans had a box covering more than half the frame, which made nearly every question `fixated`); GroundingDINO box threshold 0.35, text threshold 0.25; `fixated` uses the cumulative dwell of all fixations inside the box (>= 200 ms). Result over 564 questions: 155 `NA`, 191 `unanswerable`, 2 `no_target`, 127 `fixated`, 44 `glanced`, 21 `never`, 24 `undetected` (first run: 236 / 21 / 2 / 2). Per-category counts: `results/T0_fe_distribution.csv`.
- **H1.3 QC (README H1.3 step 4):** 150 random labeled questions exported as mosaics to `results/qc/fe_labels/` (green = target box, red = gaze samples, blue = fixations). Qwen3-VL second opinion agrees on 35.3%, Cohen's kappa = 0.02; Qwen answers `glanced` for 110 of 150 mosaics, i.e. it cannot judge dwell time from six still frames, so this kappa says little about label quality. Spot checks of the mosaics looked consistent with the automatic labels. **Human review of the mosaics is pending.**
- **L3.5/L3.6 robustness and overlays:** I-DT fixations (100 ms window, 1.0 deg) written as `fixations_idt.parquet`; I-DT finds 65-90% as many fixations as I-VT per session (`proc/reports/fixation_robustness.json`). The world-frame variant is not computed because `closed_loop_trajectory.csv` was not downloaded (optional). 20 overlay frames per participant and five 10 s gaze-trail clips are in `results/qc/l3_gaze_overlay/` for visual review.
- **H1.1 'none' labels and supplementary pass (2026-10-13):** with the README prompt 70.6% of the 25,377 labeled fixations were 'none', mostly because the exact crop centre fell between objects (gaze error ~1-2 deg; examples in `results/qc/h1_none_examples/`). Those 17,916 fixations were re-asked with a 512 px context crop and a red dot at the gaze point ("name the object closest to the red dot"; `smvqa-gaze relabel-none-fixations`, ~95 min on GPU 1). 'none' fell to 36.6%; gaze episodes were rebuilt with the same merging rule (16,511 -> 21,186). `fixation_labels.parquet` keeps the README label in `label_strict` and records `label_source` (strict/context); the pre-relabel files are backed up in `/mnt/D/smvqa_gaze_data/backup_h1_labels_strict/`. Remaining 'none' labels are treated as unnamed gaze by the HPC code (not in C5 focus lists or GazeDB, no C6 boundaries, dwell still weighted in C6).
- **H1.4 UNISAL saliency: complete.** An actual one-frame CUDA smoke test succeeded, followed by full inference over 14 sessions: 30,527 one-fps maps in 2,148 s (35.8 min). Outputs are uint8 JPEGs, quality 95, independently normalized per frame. UNISAL repository revision: `7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9`; checkpoint SHA-256: `4a9157411f1741d588b15670d15295e998805648b8b6348599fe447298338481`. The input-frame extraction uses OpenCV and does not require an external FFmpeg executable.

The selected-session gaze projection and fixation preparation is complete: all 14 sessions have projected gaze, fixations, frame maps, and calibration JSON. The minimum projected-valid fraction is 99.947%; the mean is 99.985%. H1 runtime is local and is excluded from the 22 GPU-hour HPC budget. No model cache belongs in Git or the transfer package.

## Time axis (changed 2026-10-13)

`fixations.parquet` `t_on`/`t_off`, `fixation_labels.parquet`, `episodes.parquet` and `gaze_px.parquet` `t_video_s` all use the released MP4 time axis in seconds (frame 0 = first gaze sample). Raw Aria device timestamps remain in `t_us`, `t_on_us`, `t_off_us`.

## QA option types (fixed 2026-10-13)

The released `choice_types` label the canonical abstention option "This question can not be answered." as `incorrect` (answerable questions) or `correct` (unanswerable questions). `qa_clean.jsonl` now marks that option with `type: unanswerable` and `is_abstain: true` in every question, and stores correctness separately in `is_correct`; distractors that merely start with "This question cannot be answered because ..." stay `wrong`. All 4,853 questions have exactly one abstention option; subset membership and dev/test splits are unchanged.

## Local verification of the HPC code (2026-10-13)

- HPC environment built from `env/environment_hpc.yml` at `/mnt/D/smvqa_gaze_env_hpc`: vLLM 0.31.0, torch 2.13.0 (CUDA 13.0), transformers 5.17.0; exact versions in `env/requirements_hpc.lock`. The yml previously referenced `-e ".[hpc]"`, which conda resolves relative to `env/` and could not install; it now uses `-e ..[hpc,test]`.
- Unit tests: 65 pass in the HPC environment (`pytest -q`).
- Processor-measured visual tokens per frame (`hpc-token-audit`): C0/C1/C3-T/C3-V/C5/C6/C8/C9 hit 256 (B1) and 1024 (B2) exactly; C4 is 242 (−5.5%) on B1 and 1058 (+3.3%) on B2; mask variants (C3-M/C8-M/C9-M) are 272 (+6.25%) on B1 and 1040 (+1.6%) on B2. This requires lowering the Qwen3-VL processor minimum to 128×128 px (`--mm-processor-kwargs` in `scripts/start_vllm.sh`); with the default 256×256 minimum the mask cost 64 tokens (+25% on B1). Verified against the live vLLM server: adding one mask adds 18 prompt tokens (16 image + 2 markers).
- C3-T coordinate format verified: Qwen3-VL's own grounding output is `{"bbox_2d": [x1, y1, x2, y2]}` with relative 0–1000 coordinates, which is what C3-T injects.
- C8 shift validity (README 7.8): with Δ = 5 min the same-label fraction is ≤ 15.6% in every session (< 20%), so Δ = 5 min is used throughout.
- C6 on the real data: 791 gaze-event segments (5.2–59.6 s), exactly the C0 frame total (30,527), caption token budget 99.4% of C0; frames per segment are capped at 60 (B1) and 44 (B2) so a segment fits the 49,152-token context.
- End-to-end smoke test with the real model on GPU 1 (RTX 3090, so B1 only and a reduced local server: 12,288-token context, 512 px max image side, enforce-eager): both pipelines × 11 conditions (C0, C1, C3-T/V/M, C4, C5, C6α0.5, C8shift-V, C8cross-M, C9-T) on 4 dev questions with memory built in a 300 s window before each question — 22/22 runs finished, 0 failed requests, 0 unparseable answers. B2 is exercised for the first time by the HPC dev check (`scripts/hpc_dev_check.sbatch`).
- The H4 analysis ran on these 22 real prediction files (tables, figures, statistics, failure-case mosaics, REPORT.md) without errors.
- Problems found and fixed during the smoke test: concurrent lazy model loads in worker threads (transformers meta-tensor error), C6 B2 segments exceeding the context window, a stray `}` in `start_vllm.sh` whenever `VLLM_LIMIT_MM_PER_PROMPT` was set, dev runs writing `DONE` markers into test-run directories, and single-character OCR noise.
