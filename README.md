# Gaze × Egocentric Memory QA：实验执行手册（给执行 agent）

> 本文件是**唯一的任务说明**。工作分两地完成：
> - **本地（CPU + 可用 GPU）**：选定代表性子集 → 只下载子集数据 → QA 解析、gaze 投影与 fixation 检测 → H1 注视标注 → 打包元数据与视频上传 HPC。
> - **HPC（1 张 80 GB GPU）**：部署 vLLM → **实测吞吐并重新做算力预估** → 两条记忆 pipeline × 全部条件 × 两档 budget → 评测与统计 → 结果报告。
>
> 研究背景与文献见上级目录 [`../gaze_memory_survey.md`](../gaze_memory_survey.md)。本手册取代其中 §4.2 的条件表。

---

## 0. 研究问题与要验证的假设

**问题**：佩戴者的 gaze 能否帮助 egocentric 记忆 agent 更好地组织记忆，以及个性化的 gaze 注意是否会因为"个体只看到局部"（盲人摸象）而损害问答。

| 编号 | 假设 | 主要对应条件 |
|---|---|---|
| H1 组织有益 | gaze 作为元数据/索引（不删除视觉信息）能提升证据检索与"注视目标相关"题目的准确率 | C5, C6 |
| H2 过滤有害 | 只保留 gaze 区域时，"证据从未被注视"的题目显著下降，"证据被注视"的题目持平或上升 | C1 vs C0 |
| H3 提示的作用 | 在不删除信息的前提下，用 prompt 引导模型关注 gaze 区域，会让记忆偏向注视目标；对非注视证据有轻度损害 | C3-T/V/M |
| H4 个性化 | 本人、时间对齐的 gaze 优于错位 gaze 和通用显著性模型；其优势主要体现在 fixated 子集 | C3 vs C8 vs C9 |
| H5 budget | budget 增大时，整帧条件主要获得分辨率收益，gaze crop 条件主要获得覆盖面积收益；两者差距随 budget 增大而缩小 | 所有条件 × B1/B2 |
| H6 可答性 | 过滤型条件会抬高错误弃答率（选 unanswerable） | C1, C4 |

---

## 1. 总体约束（agent 必须遵守）

1. **模型统一用 Qwen**：
   - 视觉 caption 与回答：`Qwen/Qwen3-VL-8B-Instruct`（bf16，主模型）。不使用 FP8 等量化版本，以免精度差异影响条件间比较。
   - EgoButler 的小时级摘要（论文用的是 Gemini-3-Flash）：改用同一个 `Qwen3-VL-8B-Instruct` 做纯文本推理，所有条件保持一致。
   - 显著性模型（仅 C9）：见 §7.9。
2. **推理框架**：vLLM ≥ 0.11.0，transformers ≥ 4.57.0（Qwen3-VL 官方要求）。贪心解码 `temperature=0`，固定 `seed=0`。
3. **时间因果性**：回答每道题时，记忆中只能包含该题 `start_time` 之前的内容（论文协议：视频在提问时刻截断）。检索时按时间戳过滤，**不得泄漏未来信息**。
4. **除被操控的变量外，其他一切保持不变**：同一套 prompt 模板、检索器、top-k、caption 长度上限、帧数。每个条件只改动 §7 中说明的部分。
5. **可恢复**：每个阶段的产物写盘并带 `manifest.json`（输入哈希、参数、代码版本）。HPC 作业可能被时限中断，所有预测按题追加写入，重启后跳过已完成的题。
6. **不改 QA 文件**。子集的选取规则见 §3，所有过滤（如 F(E) 标签缺失）都要记录数量与原因。
7. **算力上限：HPC 上计划用时 22 GPU 小时，硬上限 23.5 小时（1 张 80 GB GPU）**，包括 H2 基准测试和 H3 全部运行。H1 在本地 GPU 执行并单独记录耗时，不计入 HPC 预算。计划按 22 小时编排，超出时按 H2 第 4 步的顺序削减；实际运行无论如何不得超过 23.5 小时。
8. **质量检查不阻断流程**：各处的检查结果（gaze 投影、时间对齐、token 匹配、标签一致率等）如实写入报告；检查结果明显异常时，在报告中标注受影响的条件，并继续执行其余部分。

---

## 2. 两地分工与目录结构

| 阶段 | 地点 | 内容 | 需要 GPU |
|---|---|---|---|
| L0 | 本地 | 轻量环境（无 GPU 依赖） | 否 |
| L1 | 本地 | 下载 QA，选定代表性子集（§3） | 否 |
| L2 | 本地 | 只下载子集 session 的视频、gaze、标定、转写 | 否 |
| L3 | 本地 | QA 解析、gaze 投影、时间对齐、fixation 检测、可视化检查 | 否 |
| L4 | 本地 | 打包上传 HPC | 否 |
| H0 | HPC | 完整环境、vLLM 服务 | 是 |
| H1 | 本地 GPU | 注视目标标注、F(E) 标签、显著性图 | 是 |
| H2 | HPC | **实测吞吐，重新做算力预估**，生成运行计划 | 是 |
| H3 | HPC | 运行实验矩阵（§8） | 是 |
| H4 | HPC | 评测、统计、报告（§9–§10） | 少量 |

```
smvqa_gaze/
  env/                    # environment_local.yml / environment_hpc.yml / requirements.lock
  data/raw/               # 子集原始数据（只读）
  data/proc/
    qa/                   # qa_clean.jsonl, qa_subset.jsonl, qa_fixation_labels.jsonl
    gaze/                 # 每个 session 的 gaze_px.parquet, fixations.parquet, episodes.parquet
    saliency/             # C9 显著性图（压缩存储）
  transfer/               # L4 打包目录（上传 HPC 的全部内容）
  memory/{pipeline}/{tier}/{condition}/   # 记忆库（captions、索引、元数据）
  preds/{pipeline}/{tier}/{condition}/predictions.jsonl
  results/                # metrics、表格、图、COMPUTE_PLAN.md、REPORT.md
  logs/
  src/                    # 代码（agent 编写，本地与 HPC 共用一份，用 git 同步）
```

---

## 3. 代表性子集（L1）

**原则**：记忆构建的成本由**视频时长**决定，答题成本相对很小。所以先选 session 控制视频时长，再在选中的 session 里取指定题型的**全部**题目。

### 3.1 题型：选 4 类，覆盖假设的各个方向

| 入选题型 | 理由 |
|---|---|
| Object & Location Memory | 预计受 gaze 过滤伤害最大（随手放置，常不注视） |
| Visual Scene Recall | 预计受 gaze 帮助最大（读屏幕、说明书时通常注视） |
| Intent Recall | 检验 gaze 作为"组织者"的作用（EgoGazeVQA 中意图类增益最大） |
| Conversational Memory | 负对照：主要依赖转写，gaze 条件不应影响它 |

暂不纳入 Timeline Reconstruction 与 In-Context Retrieval（多证据、跨度长、构建记忆的成本高）。之后如需补充，只要在已选 session 里加题即可，不必重建记忆。

### 3.2 选 session

1. 下载 `data/json/*`（约 13 MB），解析出每道题的参与者、题型、所需 session（`video_ids`）和提问时刻。
2. **记忆范围**：回答一道题时，记忆包含该参与者在提问时刻之前的全部录制。所以选中一道题，就要下载该参与者从第一个 session 到该题所在 session 的全部视频。
3. **选择方法**：在候选参与者组合中，选出满足下列条件且总视频时长最短的组合（参与者只有 10 人，直接枚举）：
   - 4 个入选题型各至少 120 道题；
   - 至少包含 1 名多日录制的参与者；
   - 每个入选 session 都有 eye gaze 文件，且 gaze 有效率 ≥ 80%（读 CSV 即可算出）。
   目标总时长 6–10 小时（受 22 GPU 小时计划用时约束，最终以 H2 的预估为准）。如果达不到每类 120 题，允许降到 80 题，并在 `SUBSET.md` 中说明。
4. 多日参与者的录制可能很长。允许只取其前若干天，但截断点之后的题一律不用。
5. **开发集**：从子集中按题型分层随机抽 15%（固定种子）作为 dev 集，只用于调参（C5 的 β、prompt 措辞）。dev 集不进入最终统计，报告中所有数字都来自其余 85% 的 test 集。
6. 输出 `data/proc/qa/qa_subset.jsonl`（带 `split: dev|test` 字段）和 `SUBSET.md`：参与者、session、总时长、各题型 × split 的题数，以及子集与全量在题型比例、证据时长、问题与证据时间间隔上的对比（说明代表性）。

---

## 4. 本地阶段（L0–L4，纯 CPU）

### L0 本地环境

conda（Python 3.11）：`huggingface_hub`、`projectaria_tools`、`pandas`、`pyarrow`、`numpy`、`opencv-python`、`av`、`ffmpeg`、`matplotlib`。不安装 torch/vLLM。

### L2 下载子集

1. 只下载 §3 选中 session 的以下文件：
   - `data/mps/<person>/<session>/eye_gaze/*`
   - `data/mps/<person>/<session>/slam/online_calibration.jsonl`
   - `data/mps/<person>/<session>/slam/closed_loop_trajectory.csv`（仅做头动校正时需要，可选）
   - 该 session 的转写文件
   - 该 session 的 RGB 视频
2. **不要下载** `semidense_observations.csv.gz` 和 `semidense_points.csv.gz`（单个 2–5 GB，用不到）。
3. **RGB 视频的位置**：论文和数据卡没有写明。先列出 HF 仓库的完整文件树查找视频（mp4 或 VRS）；如果没有，再查官方代码仓库 `AIoT-MLSys-Lab/supermemory-vqa` 的 README 和项目页 `supermemory-vqa.github.io`。仍然找不到时，在 `INVENTORY.md` 中写明查过的位置，并告知用户。gaze 相关的 L3 步骤不依赖视频，可以先做。
4. 如果视频以 VRS 提供，用 `projectaria_tools` 导出 RGB 流为 mp4（保持 1408×1408、30 fps，H.264 CRF 18），并保留帧时间戳表。
5. 克隆官方代码仓库，记录其中是否有评测脚本、Video-RAG / EgoButler 实现、答题 prompt（HPC 阶段优先复用）。
6. 写仓库根目录 `INVENTORY.md`：session 列表、视频时长与分辨率、gaze 采样率与缺失率、`video_ids` 与 MPS 目录名的对应关系。

### L3 QA 解析与 gaze 预处理

**L3.1 QA 解析**：HF viewer 显示的列为 `question_id, question, choices, correct_answer, correct_option_index, choice_types, subject, metadata, video_ids, start_time, question_evidence, is_answerable, answer_evidence`。统一成 `qa_clean.jsonl`：每题包含题型类别、提问时刻（秒）、答案证据列表 `[video_id, t_start, t_end, modalities]`、四个选项及其类型（correct/vague/wrong/unanswerable）。

**L3.2 读取 gaze**：`general_eye_gaze.csv` 每行是一条视线：`yaw_rads_cpf`、`pitch_rads_cpf`、`depth_m`，以及置信区间 `yaw_low/high`、`pitch_low/high`、左右眼分量等（参考系为 CPF，即两眼中点的头部坐标系）。如存在 `personalized_eye_gaze.csv` 则记录下来，但主实验统一用 general 版本（本数据集预计只有 general 版本）。

**L3.3 投影到 RGB 像素**：
- 用 `projectaria_tools` 的 MPS 工具读取 gaze，按 `depth_m` 取三维注视点，再变换到 RGB 相机坐标并投影（官方文档 "Eye Gaze Code Snippets" 中的 reprojection 流程）。
- 需要的标定：RGB 相机内外参（`online_calibration.jsonl` 中按时间戳给出的 `camera-rgb`）和 `T_device_CPF`（通常来自 VRS 的 device calibration）。如果没有 VRS，先在 MPS 目录里找 device calibration；仍没有就用 Aria Gen1 的标称 `T_device_CPF`（可从任一公开 Aria 样例 VRS 读取），并在报告中注明误差来源。
- 注意 RGB 图像方向：Aria 原始 RGB 帧是旋转 90° 的，发布的视频可能已转正。必须确认投影坐标与视频帧方向一致。
- 输出 `gaze_px.parquet`：`t_us, u, v, valid, depth_m, unc_deg`（`unc_deg` 由 low/high 置信区间换算）。

**L3.4 时间对齐**：确定视频第 0 帧对应的设备时间戳（从视频元数据、VRS 时间戳表或数据集 manifest 获取）。如果没有显式时间戳，再按 gaze 首条记录对齐，并用 L3.6 的可视化检查验证。每一帧取时间最近的有效 gaze 样本（容差 ≤ 1/2 个 gaze 采样间隔）；无效样本（眨眼、缺失）在 300 ms 内沿用上一个有效值，超过 300 ms 记为缺失。

**L3.5 Fixation 检测（I-VT 为主，I-DT 为稳健性检验）**：
- 在角度空间（yaw/pitch，单位度）计算角速度；先做 3 点中值滤波。
- I-VT：角速度 < 30°/s 记为 fixation 样本；合并间隔 < 75 ms 且位置差 < 0.5° 的相邻 fixation；丢弃时长 < 100 ms 的 fixation。
- I-DT（稳健性）：窗口最短 100 ms，离散度阈值 1.0°。
- 头动问题：角度是相对头部的，转头时盯住物体会表现为眼动补偿。如果下载了 `closed_loop_trajectory.csv`，就把视线方向换到世界坐标系后再算角速度，作为对照版本；主实验用头部坐标版本，报告两者的 fixation 统计差异。
- 输出 `fixations.parquet`：`fix_id, t_on, t_off, dur_ms, u, v`（中位数位置）。

**L3.6 可视化检查**：每个参与者随机抽 20 帧画出 gaze 点，保存到 `results/qc/gaze_overlay/`；再抽 5 段 10 秒视频，叠加 gaze 轨迹导出 mp4。把检查图交给用户目视确认投影方向和时间对齐（HPC 上还会做一次自动检查，见 H1）。

对已下载的选中 session，`smvqa-gaze prepare-selected-sessions --data-root /mnt/D/smvqa_gaze_data` 会批量生成投影、fixation、以首条 gaze 对齐的视频帧时间表、每 session 标定 JSON 和处理报告。其默认输入是 `proc/qa/selected_sessions.json`，只处理该清单中的 session。

### L4 打包上传

`transfer/` 内容：
- `videos/`：子集 session 的 RGB mp4 与帧时间戳表
- `gaze/`：`gaze_px.parquet`、`fixations.parquet`
- `calib/`：每个 session 的 RGB 内参与 `T_device_CPF`（导出为 JSON，HPC 上不必再装投影依赖）
- `transcripts/`
- `qa/`：`qa_clean.jsonl`、`qa_subset.jsonl`
- `docs/`：`SUBSET.md`、`INVENTORY.md`、`results/qc/` 的检查图
- `MANIFEST.json`：每个文件的大小与 sha256

上传方式由用户决定（rsync/scp/Globus）。HPC 端第一步先校验 sha256。

每 session 标定 JSON 可用 `smvqa-gaze export-calibration --gaze-csv <general_eye_gaze.csv> --vrs <session.vrs> --output <calib.json>` 生成；优先采用首条 gaze 时间对应的 MPS online RGB 标定，缺失时回退 VRS device calibration，并按发布视频分辨率缩放内参。

全部准备完成后，`smvqa-gaze assemble-transfer --data-root /mnt/D/smvqa_gaze_data --project-root <repository> --output /mnt/D/smvqa_gaze_data/transfer_ready` 会只复制最终选中 session 的 MP4、派生 gaze/fixation、标定、可用转写、QA 和文档，写入逐文件 SHA-256 `MANIFEST.json`。命令不会打包原始 VRS 或未选中的候选数据；目标目录若已存在会拒绝覆盖。随后用 `smvqa-gaze verify-manifest --transfer-dir /mnt/D/smvqa_gaze_data/transfer_ready` 校验，并用 `smvqa-gaze archive --transfer-dir /mnt/D/smvqa_gaze_data/transfer_ready --output /mnt/D/smvqa_gaze_data/transfer_ready.tar` 生成未压缩归档（避免对 MP4 做无效压缩）。

---

## 5. HPC 阶段：环境与并行方案（H0）

### 5.1 环境

conda（Python 3.11）：`torch`（与 CUDA 匹配）、`vllm>=0.11.0`、`transformers>=4.57.0`、`qwen-vl-utils`、`decord` 或 `av`、`opencv-python`、`faiss-gpu`、`sentence-transformers`（检索编码器 `facebook/contriever-msmarco`，与论文 Video-RAG 一致）、`easyocr`、开放词汇检测器（论文用 APE；若安装困难换 `GroundingDINO`，并在报告中注明与论文的差异）、UNISAL（GitHub `rdroste/unisal`）、`statsmodels`、`scipy`、`scikit-learn`。

在仓库根目录执行 `bash scripts/deploy_hpc.sh` 可创建或更新 `smvqa-gaze-hpc` 环境并启动 vLLM 服务。首次运行会按需从 Hugging Face 下载模型；默认缓存位置是 `$SCRATCH/.cache/huggingface`（若未设置 `SCRATCH`，则为 `$HOME/.cache/huggingface`）。可通过 `HF_HOME` 指定持久化缓存目录，并通过 `VLLM_MODEL`、`VLLM_HOST`、`VLLM_PORT`、`VLLM_GPU_MEMORY_UTILIZATION` 等环境变量覆盖服务参数。该环境同时包含 H2–H4 所需的 EasyOCR、GroundingDINO（transformers）、Contriever、FAISS 与统计依赖；`bash scripts/deploy_hpc.sh --no-serve` 只建环境不启动服务。UNISAL 不需要在 HPC 安装（显著性图已在本地 H1.4 生成并随传输包上传）。HPC 阶段（H0 dev 检查 → H2 → H3/H4）的完整操作步骤见仓库根目录 `HPC_RUNBOOK.md`。

smoke test：一张图 + 8 帧图像列表，确认 Qwen3-VL-8B 能输出，并**用 processor 实际统计视觉 token 数**（约 32×32 像素一个 token，以 processor 输出为准）。结果写入 `env/ENV_REPORT.md`。

### 5.2 在一张 80 GB GPU 上并行

**不要起 4 个独立的 8B 模型实例。** Qwen3-VL-8B 的 bf16 权重约 17–18 GB，4 个实例光权重就占约 70 GB，几乎不剩 KV cache 和视觉编码器的激活显存，会频繁 OOM 或吞吐极低。

**推荐做法：一个 vLLM 服务 + 多个实验进程作为客户端。** vLLM 的连续批处理会把多个任务的请求合在一起算，GPU 利用率比多个实例高得多。这样就能同时推进 4 个（或更多）实验条件。

- 服务端（示例参数，H2 中按实测调整）：
  ```
  vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 49152 --max-num-seqs 16 \
    --max-num-batched-tokens 32768 \
    --limit-mm-per-prompt '{"image": 70}' \
    --enable-prefix-caching --seed 0
  ```
  - `max-model-len`：B2 下 Video-RAG 答题约 32 帧 × 1024 token ≈ 33k 视觉 token，加检索文本，留到 48k。
  - `limit-mm-per-prompt`：C4 与 C3-M 每帧有两张图，32 帧需要 64 张以上。
  - `--enable-prefix-caching`：同一套 prompt 模板的前缀可复用。
- 客户端：每个实验条件一个进程，通过 OpenAI 兼容接口异步发送请求，每个进程并发 4–8 个请求。同时运行 4 个条件进程，总并发约 16–32。
- **其他 GPU 模型**（检测器、UNISAL、Contriever、EasyOCR，各自都在几 GB 以内）：放在单独的阶段批量跑完并缓存结果；如需与 vLLM 同时运行，把 vLLM 的 `gpu-memory-utilization` 降到 0.70。
- 作业调度：vLLM 服务与客户端放在同一个作业里；作业开头等服务就绪（健康检查接口）后再启动客户端。

---

## 6. 本地 GPU 阶段 H1 与 HPC 阶段 H2

本地 H1 使用 `env/environment_local.yml` 中的 `local-gpu` 依赖。若多卡工作站有其他进程占用 GPU，应通过 `CUDA_VISIBLE_DEVICES` 仅暴露空闲卡；H1 不应抢占其他用户的 GPU。模型缓存和 H1 产物应放在数据盘，不要加入 Git。

### H1.1 注视目标标注（gaze episodes）

对每个时长 ≥ 200 ms 的 fixation，以注视点为中心在原分辨率帧上裁 256×256，用 Qwen3-VL-8B 回答：
`"Name the single object or piece of text at the center of this image in at most 6 words. If nothing specific, answer 'none'."`
相邻且标签语义相同（文本嵌入余弦相似度 > 0.85）的 fixation 合并为一个 gaze episode。输出 `episodes.parquet`：`ep_id, t_on, t_off, total_dwell_ms, n_fixations, label, box_px`。

`smvqa-gaze annotate-selected-gaze-episodes --data-root /mnt/D/smvqa_gaze_data` 会按选中清单完成全部 session，逐 fixation checkpoint 并生成汇总报告；`--limit 1` 可先对每段做单 fixation smoke test。

### H1.2 投影自动检查

每个参与者随机抽 50 帧画上 gaze 点，用 Qwen3-VL 判断："Is the red dot on a meaningful object the person is likely interacting with? yes/no"。把 yes 比例与 L3.6 的目视结论一起写入报告。

### H1.3 每道题的注视状态标签 F(E)

1. **确定证据目标**：用 Qwen3-VL-8B（纯文本）从题目和正确答案中抽取 1–3 个视觉目标名词短语（如 "car keys"、"the recipe on the tablet"）。Conversational Memory 中证据模态不含视觉的题目，标为 `NA`。
2. **定位目标**：在每个答案证据时间段内按 2 fps 取帧，用开放词汇检测器检测目标，得到每帧的 box；检测置信度低于阈值的帧忽略。
3. **判定**：把 box 向外扩 2° 视角（像素数由 RGB 内参换算）。在证据时间段内（两侧各扩 2 s）：
   - fixation 落在 box 内、累计时长 ≥ 200 ms → `fixated`
   - 有 gaze 样本落在 box 内但累计 < 200 ms → `glanced`
   - 目标被检测到，但 gaze 从未落在 box 内 → `never`
   - 目标在证据时间段内一次都没被检测到 → `undetected`（单独统计，不进入 F(E) 分层分析）
4. **人工抽检替代**：随机抽 150 题，导出带 box 和 gaze 的帧拼图到 `results/qc/fe_labels/`，供用户抽查。再用 Qwen3-VL-8B 看拼图做第二意见判定，报告两者的一致率（Cohen's κ）。
5. 输出 `qa_fixation_labels.jsonl`，并在 `results/T0_fe_distribution.csv` 中给出**各题型 × F(E) 的题数与比例**。这是论文的第一张表：有多少记忆问题的证据用户从未看过。

执行入口：先用 `smvqa-gaze extract-visual-targets --input data/proc/qa/qa_subset.jsonl --output data/proc/qa/visual_targets.jsonl` 抽取目标，再用 `smvqa-gaze label-fixation-states --qa data/proc/qa/qa_subset.jsonl --targets data/proc/qa/visual_targets.jsonl --output data/proc/qa/qa_fixation_labels.jsonl --distribution results/T0_fe_distribution.csv` 完成检测和分类。定位器默认使用 `IDEA-Research/grounding-dino-tiny`（Transformers GroundingDINO），box/text 阈值均为 0.25；按每帧 0.5 秒采样，并由该 session 导出的 RGB 焦距将 box 扩展 2°。两条命令都可断点续跑；如需与论文 APE 检测器严格一致，须替换定位器并记录差异。

### H1.4 显著性图

用 UNISAL 对子集视频按 1 fps 逐帧计算显著性图（C9 使用），压缩存储。

本地入口为 `smvqa-gaze generate-saliency --data-root /mnt/D/smvqa_gaze_data --unisal-root /mnt/D/smvqa_gaze_data/models/unisal --cuda-visible-devices 1`。它使用 `rdroste/unisal` 官方 `pretrained_unisal` 权重，临时提取 1 fps 帧、保留跨 chunk 的循环状态，并把逐帧归一化的 uint8 JPEG（质量 95）写入 `proc/saliency/<video_id>/`；模型代码与临时帧不会进入 Git 或传输包。官方仓库已归档，模型原环境较旧；我们使用其检查点与接口，并在本地 PyTorch 环境验证兼容性后再记录状态。

### H2 实测吞吐，重新做算力预估

1. 用 §5.2 的服务配置跑基准测试（都只用 dev 集或少量片段）：
   - EgoButler caption：B1、B2 各 100 个 30 秒片段（C0 一种，加 C4 一种以覆盖两张图的情况）
   - Video-RAG 答题：B1、B2 各 100 题
   - 记录：每秒视觉 token 数、每片段/每题耗时、峰值显存、OOM 情况；调整 `max-num-seqs` 与 `max-num-batched-tokens`，取吞吐最高且稳定的配置。
2. 根据子集实际时长、题数和 §8 的运行矩阵，逐个条件 × 档位 × pipeline 计算 GPU 小时，并区分：
   - 需要重新生成 caption 的条件（C1、C3 三变体、C4、C6、C8、C9）
   - 只在答题阶段变化的条件（Video-RAG 的大部分条件）
   - 可以复用已有结果的条件（C5 复用 C0 的 caption）
3. 预算核算：H2 已用掉的 GPU 小时 + H3 全部任务的预计 GPU 小时，再乘以 1.1 的余量系数，总数必须 ≤ 22 小时；另在报告中记录本地 H1 的 GPU 耗时。
4. **超出 22 小时时，按以下顺序逐步削减，直到满足上限**：
   1. 去掉 P3（全部稳健性实验）
   2. P2 只保留 C6(α=0) 和 C6(α=1)，去掉 C4
   3. C8 与 C9 只做 C3 三个变体中在 P0 里效果最好的一个（按 dev 集选）
   4. P1、P2 只跑 B1；B2 只保留 P0
   5. 去掉 P2
   6. 缩小子集：去掉视频时长占比最大、但对题量贡献最小的参与者，重新生成 `SUBSET.md`

   每一步削减都写进 `COMPUTE_PLAN.md`，并说明对应的假设是否因此无法检验。
5. 写 `results/COMPUTE_PLAN.md`：最终的服务参数、每项任务的预计耗时、预算核算表、削减记录、按优先级排好的执行顺序和并行分组（每组 4 个条件同时跑），以及作业脚本。**写完后交给用户确认再开始 H3。**
6. H3 运行过程中持续记录实际用时。累计用时达到 22 小时后，不再启动新任务，只让正在运行的任务跑完；达到 23.5 小时时立即停止所有任务，把已完成的结果整理进报告（预测按题追加写入，已完成的部分不会丢失）。

---

## 7. 记忆 pipeline 与实验条件

### 7.1 两条 pipeline（按 SuperMemory-VQA 论文复现）

**Video-RAG**（信息损失发生在读取时）
- 会话按 30 分钟切片，每个分片建三个检索库：ASR（发布的 WhisperX 转写）、OCR（EasyOCR，作用于采样帧）、物体检测（APE/GroundingDINO，作用于 CLIP 选出的关键帧）。用 Contriever 编码，FAISS 索引。
- 回答：检索最相关分片 → 从该分片均匀取 32 帧 + 合并检索到的文本 → Qwen3-VL 作答。

**EgoButler**（信息损失发生在写入时）
- 每 30 秒片段按 1 fps 取帧，结合转写生成 caption；再由 caption 汇总成小时级和天级摘要。
- 回答：EgoRAG 从天到小时再到片段逐级检索 → top-k 片段 caption 交给 Qwen3-VL 作答。top-k 以官方实现为准，没有则取 k=5。

**答题与评测协议**：沿用官方评测脚本。如果没有，就按论文实现：四个选项（correct/vague/wrong/unanswerable）以固定种子打乱顺序后给模型，要求模型输出四个选项的完整排序；第一名用于计算 QA-Acc，排序用于计算 QA-MRR，"是否选 unanswerable"用于计算 Ans-F1。

### 7.2 两档 budget

budget 定义为**每帧视觉 token 数**；帧数在两档之间保持不变（Video-RAG 答题 32 帧；EgoButler 每 30 秒 30 帧）。这样两档之间只有空间信息量不同。

| 档位 | 每帧 token | 对应正方形边长 S（Qwen3-VL，约 32 px/token） |
|---|---|---|
| **B1（低）** | 256 | 512 × 512 |
| **B2（高）** | 1024 | 1024 × 1024 |

- 把帧作为**图像列表**输入并显式设置 `resized_height/resized_width`，避免视频模式的 2× 时间压缩造成不同条件之间 token 数不一致。
- 每个条件实际消耗的 token 数要从 processor 统计出来，与目标值偏差须在 ±5% 以内，写入 `manifest.json`。两个已知例外：C4 在 B1 下为 −5.5%（2 × 352²），C3-M/C9-M 在 B1 下因掩码多出约 +6%。这两项允许，但要在报告里注明。
- 原始帧为 1408 × 1408。

### 7.3 条件总表

| 条件 | 视觉输入（每帧，档位边长 S） | prompt / 记忆中额外内容 | 检验 |
|---|---|---|---|
| **C0 Full** | 整帧缩放到 S×S | 无 | 基线 |
| **C1 Fovea-only** | 以 gaze 为中心、**原分辨率**裁 S×S（B1 覆盖约 13% 画面，B2 约 53%） | 无 | H2、H5 |
| **C3-T 坐标注入** | 同 C0（图像不变） | prompt 注入每帧 gaze 框坐标 | H3 |
| **C3-V 画框** | 同 C0，在图上画出 gaze 框 | prompt 说明框的含义，要求更加注意框内内容 | H3 |
| **C3-M 掩码** | 同 C0，另附一张低分辨率黑白掩码 | prompt 说明掩码的含义 | H3 |
| **C4 Fovea+Periphery** | 低清整帧 + 原分辨率 gaze crop，两者各约占一半 token | prompt 说明两张图的关系 | 兼得？ |
| **C5 Gaze 元数据** | 同 C0 | 记忆中写入结构化 gaze 元数据，参与检索与答题 | H1 |
| **C6(α) Gaze 事件切分 + 加权** | 同 C0，但片段边界和帧分配由 gaze 决定，α∈{0, 0.5, 1} | 无 | H1、剂量 |
| **C8 错位 gaze** | 与 C3 相同，但 gaze 换成错位版本 | 同 C3 | H4 |
| **C9 通用显著性** | 与 C3 相同，但框/掩码来自显著性模型 | 同 C3 | H4 |

> 去掉原先的 C2（中心/随机 crop）。原先的 C6 与 C7 合并为 C6(α)：α=0 即纯 gaze 事件切分。原先的 C8 改为"用错位 gaze 重做 C3"。

### 7.4 C1 与 C4 细节

- **gaze 框**：以当前帧 gaze 像素点为中心、边长为 S 的正方形，越界时平移到画面内（不缩小）。gaze 缺失的帧沿用最近 1 s 内的有效 gaze；仍缺失则用画面中心，并统计这类帧的比例。
- **C1**：直接裁出该框，已经是 S×S，不再缩放。
- **C4**：边长取最接近 S/√2 的 32 的倍数（B1 为 352，B2 为 736）。图 1 为整帧缩放到该边长，图 2 为以 gaze 为中心、原分辨率裁同样边长。prompt：`"Image A is the full view at low resolution. Image B is a high-resolution crop around where the user was looking."`
- **在两条 pipeline 中的作用位置**：
  - EgoButler：caption 模型看到的就是变换后的帧。
  - Video-RAG：答题的 32 帧使用变换后的帧。C1 条件下，OCR 与检测库也只在 crop 上构建（系统从未看到外周）。C4 条件下，OCR 与检测库在整帧上构建（低清 + crop 合起来覆盖了全画面）。

### 7.5 C3：三种 gaze prompt（参考已发表工作）

三种方式的框都取 §7.4 中的 gaze 框（换算到缩放后 S×S 图像上的坐标），这样 C3 与 C1 关注的是同一块区域：C1 是只保留这块区域，C3 是保留全部、只做提示。

| 变体 | 图像 | prompt 模板（英文给模型，示例） | 参考工作 |
|---|---|---|---|
| **C3-T** | 不变 | `"For each frame, the user's gaze region is given as a box in the format of Qwen3-VL grounding coordinates. Frame 3: <box>(x1,y1),(x2,y2)</box>, gaze point (x,y). Pay more attention to the content inside the gaze region."` | EgoGazeVQA 的 GazeT（文字坐标）；StreamGaze 的文字 fixation 坐标 |
| **C3-V** | 画红色矩形框（B1 线宽 2 px，B2 线宽 3 px）+ 中心小圆点 | `"The red box in each frame marks where the user was looking. Pay more attention to the content inside the red box."` | EgoGazeVQA 的 GazeV（画圈）；StreamGaze 的点 + 视野圈；Shtedritski et al. ICCV 2023（红圈提示）；ViP-LLaVA、Set-of-Mark 等视觉标记提示 |
| **C3-M** | 原图不变，另附一张 128×128 黑白掩码（白色 = gaze 框区域，约 16 token） | `"Each frame is followed by a black-and-white mask of the same view. White marks where the user was looking. Pay more attention to the corresponding region of the frame."` | EgoGazeVQA 的 GazeS 与 StreamGaze 的 salience 图（二者为软热力图，这里用二值掩码，便于与 C1/C3 的框对齐）；Voila-A；FGVP（NeurIPS 2023，掩码式视觉提示） |

- **坐标格式**：C3-T 必须使用 Qwen3-VL 原生的 grounding 坐标格式。官方说明是"相对坐标"，agent 需用官方 grounding 示例确认具体范围（例如 0–1000）与标记写法后再定稿模板。
- **掩码 token 开销**：C3-M 每帧多约 16 个 token（B1 约 +6%，B2 约 +1.6%），记录实际数值，作为可接受偏差。
- **作用位置**：
  - EgoButler：caption prompt 在原模板后追加上表中的提示句，要求 caption 中优先描述 gaze 区域内的物体和文字，但不得省略其他重要内容。答题阶段不再加 gaze 提示（答题时只能看到 caption）。
  - Video-RAG：答题时对 32 帧施加同样的提示；OCR 与检测库不变。
- **gaze 提示的元数据化**：每个 C3 变体都把逐帧的 `gaze_box`、`gaze_point` 以及 S2.5 的注视目标标签以结构化字段写入记忆（不参与检索，仅存档）。这样后续可以在 C5 中复用，并且能分析"caption 中有多少内容来自框内"。

### 7.6 C5：Gaze 元数据（组织者角色）

视觉输入与 C0 完全相同，caption 也直接复用 C0 的结果；只在记忆中增加 gaze 元数据。

**元数据结构**（每个 30 秒片段一条，存为 JSON 字段 `gaze_meta`）：

```json
{
  "clip": ["00:12:30", "00:13:00"],
  "focus": [
    {"label": "cutting board with onions", "dwell_s": 6.2, "n_fix": 9, "first_seen": false, "box_norm": [0.31,0.42,0.58,0.70]},
    {"label": "phone screen: timer 05:00", "dwell_s": 2.1, "n_fix": 3, "first_seen": true, "box_norm": [0.62,0.18,0.80,0.39]}
  ],
  "gaze_shift_count": 14,
  "summary": "User mainly looked at: cutting board with onions (6.2s); phone screen: timer 05:00 (2.1s)."
}
```

- `first_seen`：该标签在过去 10 分钟内是否出现过（用来区分新目标与反复查看的目标）。
- `focus` 按 dwell 降序排列，保留 dwell ≥ 0.5 s 的前 5 个。

**使用方式**：
- **写入**：EgoButler 中每个片段 caption 后附 `summary` 文本；生成小时/天摘要时，在 prompt 中要求"保留用户反复注视的物体和文字"。Video-RAG 中新增第四个检索库 GazeDB，每个 gaze episode 一条，文本为 `label + 时间`。
- **检索**：得分 = `sim(q, 内容) + β · sim(q, gaze_meta.summary)`。主实验取 β = 0.5；在 dev 集上扫 β ∈ {0, 0.25, 0.5, 1}，选出后固定。β = 0 即退化为 C0 + 存档元数据。
- **答题**：检索到的片段连同其 `gaze_meta.summary` 一起提供给模型。

**你的假设的直接检验**：
- 答题层面：C5 相对 C0 的提升应集中在 `fixated` 子集，`never` 子集持平。
- 记忆层面：计算**目标覆盖率** = 记忆文本中（caption + 元数据 + 摘要）是否提到该题证据目标（目标短语与记忆句子的嵌入相似度 > 0.7，或字符串命中），分别统计 `fixated` 与 `never` 子集。预期 C5 与 C3 中 fixated 目标的覆盖率上升。该指标对**所有条件**都计算。

### 7.7 C6(α)：Gaze 事件切分 + 帧预算加权（仅改变"怎么切、帧放哪"）

- **切分**：当 gaze episode 的标签发生变化且新标签持续 ≥ 2 s 时，设为候选边界；相邻边界间隔限制在 5–60 s 之间（过短则合并，过长则在中点强制切分）。
- **帧分配**：每小时总帧数 N 与 C0 相同（1 fps）。第 i 段分得
  `n_i = N · [(1 − α) · d_i / D + α · w_i / W]`
  其中 `d_i` 为段长，`w_i` 为该段内 fixation 的总 dwell 乘以新颖度系数（`first_seen` 为 1，否则为 0.3），D、W 为对应总和。每段至少 1 帧，按四舍五入后调整到总数 N。
- **α ∈ {0, 0.5, 1}**：α = 0 就是纯 gaze 事件切分（帧在时间上均匀分布）；α = 1 则帧全部按注视权重分配。
- **caption 长度**：每段 `max_new_tokens` 与该段帧数成正比，使总 caption 文本量与 C0 相近（偏差 ±10% 以内，报告实际值）。
- **在 Video-RAG 中**：分片仍为 30 分钟；答题的 32 帧以及检测用关键帧，按同一公式在分片内采样，而不是均匀采样。

### 7.8 C8：错位 gaze（个性化分析的核心）

对 **C3-T、C3-V、C3-M 三个变体全部重做**，只替换 gaze 来源：

| 子条件 | 构造 | 保留了什么 | 破坏了什么 |
|---|---|---|---|
| **C8-shift** | 同一人、同一 session，gaze 序列循环平移 Δ = 5 分钟 | 个人的 gaze 统计特性（中心偏好、扫视频率） | gaze 与画面内容的对齐 |
| **C8-cross** | 另一位参与者的 gaze 序列（优先选活动类型相近的 session），按相对时间对齐并循环补齐 | 人群层面的 gaze 统计 | 个人特性与内容对齐 |
| C8-shift-2s（可选） | 平移 Δ = 2 s | 大致内容 | 时间精度（检验对 gaze 延迟的敏感度） |

- 检查平移后的有效性：平移后 gaze 框内的注视目标标签与原标签相同的帧，比例应 < 20%；否则改用 Δ = 7 分钟。
- 建议额外对 **C1 和 C5** 也做 C8-shift（仅 B1），用于区分"过滤"与"组织"两种角色下个性化的作用。

### 7.9 C9：通用显著性（personalized gaze vs. general saliency）

- **模型**：UNISAL（视频显著性模型，参数量小，在人群注视数据上训练），在原分辨率帧上逐帧输出显著性图。可选第二个模型：GLC（在 Ego4D/EGTEA 上训练的自我中心 gaze 估计模型），代表"通用的自我中心 gaze"。
- **构造框**：显著性图做高斯平滑后，取使框内显著性质量最大的 S×S 窗口（与 gaze 框同大小）。C9-M 的掩码就是这个框。
- 用这个框完全按 C3 的三种方式重做：**C9-T / C9-V / C9-M**。
- 额外计算每题证据时段内**真实 gaze 与显著性图的一致性**（NSS），按中位数把题目分为"gaze 符合常规"和"gaze 很个人化"两组，供 §7 的个性化分析使用。

---

## 8. 运行矩阵与优先级

两条 pipeline × 两档 budget，全部在 §3 子集的 test 集上评测。每一格是一次完整运行（构建记忆 + 答题 + 评测）。

| 优先级 | 条件 | 运行数（×2 pipeline × 2 档） | 说明 |
|---|---|---|---|
| **P0** | C0, C1, C3-T, C3-V, C3-M, C5 | 6 × 4 = 24 | 主结论所需。C5 复用 C0 caption，成本低 |
| **P1** | C8-shift、C8-cross（各 × C3 三变体）, C9-T/V/M | 9 × 4 = 36 | 个性化分析 |
| **P2** | C4, C6(α=0, 0.5, 1) | 4 × 4 = 16 | 兼得与组织结构 |
| P3（可选） | C8 用于 C1/C5、C8-shift-2s、I-DT 版 F(E)、世界坐标 fixation | — | 稳健性 |

**执行顺序**（具体分组与耗时以 `COMPUTE_PLAN.md` 为准）：
1. 先跑 C0 的两条 pipeline、两档 budget，作为基线。论文中 Qwen-3-VL-8B 在全量数据上的结果为 Video-RAG Ans-F1 75.0 / Acc 41.8 / MRR 63.8、EgoButler 44.5 / 38.8 / 61.0。由于这里只用了子集和 4 个题型，且 EgoButler 的摘要模型换成了 Qwen，数字不必与论文一致，只在报告中并列作参考。
2. 按 P0 → P1 → P2 推进，每组 4 个条件并行（§5.2）。每个条件先跑 B1，确认输出正常后再跑 B2。
3. dev 集只用于 C5 的 β 和 prompt 措辞的调参，调好后固定，再在 test 集上跑。

---

## 9. 评测与统计分析

### 9.1 每次运行输出的指标

`preds/.../predictions.jsonl`（每题：排序、top1、是否弃答、检索到的证据时间段）和 `metrics.json`：

- 官方指标：**Ans-F1、QA-Acc、QA-MRR**
- **错误弃答率**：可答题中选 unanswerable 的比例
- **vague 率**：top1 选中 vague 选项的比例
- **证据检索召回 Recall@k**：检索到的片段与任一答案证据时间段有重叠的题目比例
- **目标覆盖率**（§7.6），按 F(E) 分组

### 9.2 必须产出的表（`results/` 下的 CSV + Markdown）

| 表 | 内容 |
|---|---|
| **T0** | 各题型 × F(E) 的题数与比例（H1.3） |
| **T1 主表** | 行：条件；列：pipeline × 档位；单元格：Ans-F1 / Acc / MRR |
| **T2 分题型** | 每个 pipeline × 档位一张：行为条件，列为 4 个入选题型的 QA-Acc（附题数），外加 Overall |
| **T3 题型 × 注视状态** | 行为条件，列为 题型 × {fixated, glanced, never}；另附汇总的 fixated / never 两列及其差值 |
| **T4 budget** | 每个条件 B2 − B1 的差值，按题型与 F(E) 分组 |
| **T5 个性化** | C3 vs C8-shift vs C8-cross vs C9，分 T/V/M 三个变体；指标包括个性化增益 `PG = Acc(C3) − Acc(C8-shift)` 与 `Acc(C3) − Acc(C9)`，按题型、F(E)、参与者分组 |
| **T6 可答性** | 各条件的错误弃答率、vague 率、Ans-F1 |
| **T7 检索与记忆** | 各条件的 Recall@k 与目标覆盖率（分 fixated/never） |

### 9.3 必须产出的图（`results/figs/`）

1. 各题型中 F(E) 的分布（堆叠条形图）
2. 条件 × F(E) 的准确率交互图（每个 pipeline × 档位一张）
3. B1 → B2 的变化（斜率图，每个条件一条线）
4. 个性化：C3 / C8-shift / C8-cross / C9 的准确率，按 fixated/never 分面
5. 子集中各参与者的 PG，与其 gaze 行为统计（fixation 频率、离散度、中心偏好）的对照图（参与者只有 3–4 人，只作描述，不做相关性检验）
6. C6 的 α 剂量曲线

### 9.4 统计检验

- **置信区间**：按题目做 bootstrap（1,000 次），以参与者为簇重抽样。
- **条件间成对比较**：同一批题上做 McNemar 检验，按 Holm 方法校正多重比较。
- **主模型**（混合效应 logistic 回归，`statsmodels` 的 GLMM，或通过 rpy2 调用 R 的 `lme4`）：
  `correct ~ condition * F(E) * tier + category + (1 | participant) + (1 | session)`
  重点报告 `condition × F(E)=never` 与 `condition × tier` 两类交互项。
- **个性化**：`correct ~ gaze_source * F(E) + category + (1 | participant)`，其中 gaze_source ∈ {own, shift, cross, saliency}；另加 gaze 与显著性一致性分组（§7.9）的交互。
- **稳健性**：留一参与者重算主要效应，确认主结论不由单个参与者驱动。子集只有 3–4 名参与者，随机效应的估计不稳；如果混合模型不收敛，改用以参与者为固定效应的 logistic 回归，并在报告中注明。

---

## 10. 最终报告 `results/REPORT.md`

1. 环境、数据、与论文设置的差异（摘要模型、检测器、视频方向/投影处理等）
2. 所有质量检查结果（gaze 叠加、F(E) 一致率、token 匹配、基线复现）
3. T0–T7 表与 6 张图
4. 对 H1–H6 逐条给出"支持 / 不支持 / 不确定"，并附关键统计量（效应量、置信区间、p 值）
5. 失败案例：每个关键条件各挑 5 个典型错误（附帧拼图、gaze 框、模型输出）
6. 已知限制与未完成项

---
