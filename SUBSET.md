# Selected experiment subset

## Selection result

The subset was selected by the prefix-constrained duration minimization over the complete video/gaze inventory. It contains all QA from four target categories in the selected session prefixes, subject to at least 120 questions per category, at least one multi-day participant, and valid video, VRS, and gaze coverage of at least 80%.

| Participant | Included sessions | Recording dates | RGB duration |
|---|---:|---|---:|
| Person 3 | 1–4 | 2026-02-23, 2026-03-23 | 2.253 h |
| Person 8 | 1–5 | 2026-04-11 | 3.113 h |
| Person 10 | 1–5 | 2026-04-11 | 3.112 h |
| **Total** | **14 sessions** | **3 participants; Person 3 spans multiple days** | **8.478 h** |

This is the minimum-duration eligible prefix selection found by the inventory selector for the four 120-question category quotas. The RGB assets total approximately 39.71 GB. All selected videos are 1408×1408 at 30 fps. The chosen sessions each have a VRS file and released RGB MP4.

## QA counts

| Category | Full dataset | Full share | Subset | Subset share | Dev | Test |
|---|---:|---:|---:|---:|---:|---:|
| Object & Location Memory | 933 | 19.2% | 144 | 25.5% | 22 | 122 |
| Visual Scene Recall | 850 | 17.5% | 139 | 24.6% | 21 | 118 |
| Intent Recall | 657 | 13.5% | 120 | 21.3% | 18 | 102 |
| Conversational Memory | 932 | 19.2% | 161 | 28.5% | 24 | 137 |
| Timeline Reconstruction | 750 | 15.5% | — | — | — | — |
| In-Context Retrieval | 731 | 15.1% | — | — | — | — |
| **Total** | **4853** | **100%** | **564** | **100%** | **85** | **479** |

The four chosen categories are intentionally oversampled to satisfy the per-category minimum. Timeline Reconstruction and In-Context Retrieval are excluded as specified in the experiment plan. Dev/test assignment is stratified by category with seed 2026 and a 15% dev fraction; all final statistical results use test only.

## Representation and quality

For the four selected categories, the full dataset has 3320 evidence spans and the selected subset has 509. Median answer-evidence duration is 32 s in the full data and 33 s in the subset. The median interval between evidence end and question time is 3892 s in both datasets. The subset therefore preserves these two evidence-timing summaries, while not claiming to represent the intentionally excluded categories.

Across the 14 selected sessions:

- Gaze coverage ranges from 99.865% to 100.000% (mean 99.966%).
- Gaze valid-sample fraction is 100% in the inventory.
- RGB format is 1408×1408, 30 fps; the selected MP4 files total 39.71 GB.
- Gaze-to-video timing uses the documented fallback: video frame 0 is aligned to the first gaze sample because no released per-frame timestamp sidecar was found. The generated frame map is a derived mapping, not an official timestamp source.
- The released VRS files contain sensor streams but not the RGB video; use the independent released MP4 files.

Exact per-session metadata is in `/mnt/D/smvqa_gaze_data/proc/qa/selected_sessions.json`; the full 82-session inventory is in `/mnt/D/smvqa_gaze_data/proc/qa/session_inventory.json`. These data-root paths are intentionally outside the source repository.
