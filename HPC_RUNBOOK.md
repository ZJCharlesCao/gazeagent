# HPC 运行手册（H0–H4）

本手册对应 README §5–§10。所有 HPC 阶段代码已在本仓库实现，并在本地做过冒烟测试（记录见 `env/ENV_REPORT.md` 的 “Local verification of the HPC code”）。到了 HPC 只需：搭环境 → 下载模型 → 解包校验 → 跑 dev 检查 → H2 基准与计划 → 确认计划 → 提交 H3。正常情况下不需要改代码；需要调的只有路径、SLURM 参数和（必要时）vLLM 的 CUDA 版本。

## 0. 前提检查

- GPU：1 张 80 GB（A100/H100）。`nvidia-smi` 的驱动需支持 CUDA 13（vLLM 0.31.0 的默认 wheel 基于 CUDA 13.0，驱动 ≥ 580）。驱动较旧时，按 vLLM 官方安装文档换成与集群 CUDA 匹配的 wheel，并在 `env/ENV_REPORT.md` 记录实际版本。
- 磁盘：传输包约 42 GB，工作目录约 15 GB（1 fps 帧缓存约 10 GB + 记忆、预测、特征），模型缓存约 20 GB。

## 1. 目录布局

代码和数据分开放。代码目录是 git 仓库，不要把数据放进去。

| 内容 | 位置（默认） | 说明 |
|---|---|---|
| 代码 | `~/gazeagent` | `git clone`；以后改代码只需 `git pull`（环境是 editable 安装，不用重装） |
| 数据包 | `$SCRATCH/smvqa_gaze_data/transfer_ready_v2` | 从开发机 rsync 上传一次，只读 |
| 运行产物 | `$SCRATCH/smvqa_gaze_data/work` | 帧缓存、特征、记忆、预测、结果、日志 |
| 模型缓存 | `$SCRATCH/.cache/huggingface` | `HF_HOME` |

集群没有 `$SCRATCH` 时，默认落到 `$HOME`；也可以在提交作业前导出 `SMVQA_TRANSFER`、`SMVQA_WORK`、`HF_HOME` 改到别处（见 `scripts/hpc_common.sh`）。

## 2. 一次性准备

**2.1 让 HPC 能 SSH 到开发机（一次性）**

开发机只允许密钥登录，所以要把 HPC 的公钥加到开发机的 `~/.ssh/authorized_keys`。下面的 `<开发机地址>`、`<开发机用户>`、`<开发机数据路径>` 换成实际值（不要写进仓库）。

```bash
# HPC 登录节点
ssh-keygen -t ed25519 -N "" -f ~/.ssh/devbox      # 已有可复用的密钥可跳过
cat ~/.ssh/devbox.pub                            # 把这一行加到开发机的 ~/.ssh/authorized_keys
cat >> ~/.ssh/config <<'CFG'
Host devbox
    HostName <开发机地址>
    User <开发机用户>
    IdentityFile ~/.ssh/devbox
    IdentitiesOnly yes
CFG
chmod 600 ~/.ssh/config
ssh devbox hostname                              # 能输出开发机的主机名即连通
```

如果 `ssh devbox` 超时，说明 HPC 到开发机的 22 端口被防火墙挡住了，就改回在开发机上用 rsync 向 HPC 推送（方向相反，命令相同）。

**2.2 登录节点：拉代码和数据**

```bash
# 代码（二选一）
cd ~ && git clone https://github.com/ZJCharlesCao/gazeagent.git   # 从 GitHub 拉
# cd ~ && git clone devbox:<开发机上的仓库路径> gazeagent            # 或从开发机拉，不需要 GitHub 凭据

# 数据（约 41.6 GB；中断后重跑同一条命令会续传）
mkdir -p ${SCRATCH:-$HOME}/smvqa_gaze_data
rsync -aH --partial --progress devbox:<开发机数据路径>/transfer_ready_v2/ \
    ${SCRATCH:-$HOME}/smvqa_gaze_data/transfer_ready_v2/
```

以后改了代码，在 HPC 上 `cd ~/gazeagent && git pull` 即可。数据只有变化时才需要重新 rsync，rsync 只会传有差异的文件。

**2.3 登录节点：建环境、下载模型、校验数据**

```bash
cd ~/gazeagent
module load anaconda 2>/dev/null || true       # 按集群实际的 conda 模块名；已有 conda 可跳过
bash scripts/deploy_hpc.sh --no-serve          # 按 env/environment_hpc.yml 建 smvqa-gaze-hpc 环境（约 10 GB）
conda activate smvqa-gaze-hpc
export HF_HOME=${SCRATCH:-$HOME}/.cache/huggingface
python -m smvqa_gaze hpc-fetch-models          # Qwen3-VL-8B、Contriever、MiniLM、CLIP、GroundingDINO、EasyOCR（约 20 GB）
python -m smvqa_gaze verify-manifest --transfer-dir ${SCRATCH:-$HOME}/smvqa_gaze_data/transfer_ready_v2
```

`hpc-fetch-models` 默认关闭 hf_xet 并逐个文件下载：登录节点通常限制单用户内存，并行下载器会报 `memory allocation ... failed` 后中止。中断后重跑同一条命令会跳过已完成的文件；内存充足的节点可加 `--fast`。`verify-manifest` 应输出 `Manifest verified`。`env/requirements_hpc.lock` 是本地验证通过的完整版本清单，安装出问题时用 `pip install -r env/requirements_hpc.lock` 复现。

**2.4 SLURM 参数**

按集群要求给 `scripts/` 下三个 `*.sbatch` 加上 `#SBATCH --partition=...`、`#SBATCH --account=...`，必要时把 `--gres=gpu:1` 写成具体型号（如 `gpu:a100:1`）。这类集群专属的改动建议只留在 HPC 本地，不提交回仓库。计算节点默认离线读取模型缓存（`HF_HUB_OFFLINE=1`）。

## 3. 三个作业（按顺序提交，都在仓库根目录 `~/gazeagent` 下执行 `sbatch`）

| 步骤 | 命令 | 内容 | 预计墙钟 | 成功标志 |
|---|---|---|---|---|
| dev 检查（H0） | `sbatch scripts/hpc_dev_check.sbatch` | 校验 sha256，缓存 1 fps 帧，C8 平移有效性检查，C9 显著性框，processor 实测 token（`results/token_audit.csv`），两条 pipeline × 11 个条件在 4 道 dev 题上端到端运行（含 B2 的 C0） | ~1–2 h | 日志末尾 `DEV CHECK PASSED` |
| H2 | `sbatch scripts/hpc_bench.sbatch` | 依次启动 3 组 vLLM 参数测吞吐（caption C0/C4 × B1/B2 各 100 段、Video-RAG 答题 × B1/B2 各 100 题、文本摘要），在停掉 vLLM 后预计算 OCR/CLIP 特征，生成 `results/COMPUTE_PLAN.md` | ~2–3 h | 生成 COMPUTE_PLAN.md |
| **人工确认** | 阅读 `results/COMPUTE_PLAN.md` | 预算核算、削减记录、执行顺序 | — | 你确认 |
| H3+H4 | `sbatch scripts/hpc_run.sbatch` | 按计划每 4 个任务并行；C0 先跑；每个条件 B1 合格后才跑 B2；C5 先在 dev 上扫 β 再跑 test；最后评测、统计、写 `results/REPORT.md` | ≤ 预算 | `results/REPORT.md` |

H3 作业被时限中断后**直接重新提交同一个脚本**：已完成的任务有 `DONE` 标记会跳过，未完成的任务按题续跑。

### GPU 小时预算

所有作业的墙钟时间都记在 `$SMVQA_WORK/logs/budget.json`（一张卡，GPU 小时 = 墙钟小时）。`smvqa-gaze hpc-budget --data ... --work ...` 查看累计用量。达到 22 h 后不再启动新任务；达到 23.5 h 时立即停止所有任务（README §1.7、H2.6）。dev 检查也计入预算。

## 4. 常用单条命令

```bash
S="python -m smvqa_gaze"; A="--data $SMVQA_TRANSFER --work $SMVQA_WORK"
$S hpc-run $A --pipeline egobutler --tier B1 --condition C3-V --split dev --limit 8   # 单条件 dev 运行
$S hpc-run $A --pipeline videorag  --tier B2 --condition C0                          # 单条件完整运行（可续跑）
$S hpc-evaluate $A        # 重算所有 metrics.json
$S hpc-analyze  $A        # 表 T0–T7、图 1–6、统计、REPORT.md
```

## 5. 产物位置（`$SMVQA_WORK`）

- `memory/{pipeline}/{tier}/{condition}/`：EgoButler 的 `units.jsonl`（caption、逐帧 gaze 框/点/标签存档、C5 的 gaze_meta）与 `summaries.jsonl`；每个目录带 `manifest.json`（输入哈希、参数、代码版本、processor 实测 token）。
- `preds/{pipeline}/{tier}/{condition}/predictions.jsonl`（按题追加）、`coverage.jsonl`、`metrics.json`、`DONE`。
- `results/`：`token_audit.csv`、`bench/`、`COMPUTE_PLAN.md`、T0–T7、`figs/`、`stats/`、`failure_cases/`、`REPORT.md`。
- `cache/`：帧缓存、特征、`shift_check.json`、`beta_{pipeline}.json`、`best_c3_variant.json`。

## 6. 实现中的设计决定（与论文或 README 的差异，报告中需注明）

1. **官方代码**：`AIoT-MLSys-Lab/supermemory-vqa` 只包含标注流水线，没有评测脚本和 Video-RAG/EgoButler 实现，所以两条 pipeline 和评测都按论文描述与 README 实现。
2. **答题协议**：四个选项按 `(seed=0, question_id)` 固定打乱，所有条件相同；模型输出完整排序 `X>Y>Z>W`。QA-Acc 看第一名；QA-MRR = 1/正确选项名次；选到 "This question can not be answered." 记为弃答。论文没有说明 Ans-F1 的正类，主指标以“可回答”为正类，同时报告“不可回答”为正类的 F1 和宏平均（`RunConfig.ans_f1_positive` 可改）。
3. **因果性**：只使用结束时间早于提问时刻的记忆单元。小时/天摘要如果覆盖了提问之后的时间就不使用，改用其中可用的子单元打分。
4. **EgoRAG**：天 top-2 → 小时 top-3 → 片段 top-5（官方没有给出 k，README 默认 k=5）。检索器统一用 Contriever 做精确内积检索（FAISS IndexFlatIP）。
5. **Video-RAG**：OCR 每 5 s 一帧（EasyOCR，置信度 ≥ 0.5）；检测用 GroundingDINO-tiny 替代 APE，在 CLIP 选出的 8 个关键帧上检测 Qwen 从问题中抽取的物体；分片打分 = 片段内 ASR/OCR 条目的最高相似度（C5 再加 β × gaze 摘要的最高相似度）。
6. **视觉 token**：图像以列表形式输入，尺寸预先缩放好（32 的倍数）。vLLM 与 processor 的最小像素都降到 128×128，使 C3-M 的掩码约为 16 token（默认值会被放大到 64 token，B1 下偏差会达到 +25%）。实测偏差见 `results/token_audit.csv`；C4 在 B1 下为 −5.5%，C3-M/C8-M/C9-M 在 B1 下为 +6.25%，都是 README 允许的例外。
7. **C3-T 坐标**：使用 Qwen3-VL 原生的相对坐标（0–1000），格式 `{"bbox_2d": [x1, y1, x2, y2], "point_2d": [x, y]}`。
8. **gaze 缺失**：逐帧取最近样本（容差半个采样间隔），眨眼时沿用 300 ms 内的上一个有效值；画 gaze 框时再放宽到 1 s 内，仍缺失则用画面中心，各情况的比例记录在 `frame_meta.gaze_status`。
9. **C8-cross 配对**：Person 8 与 Person 10 是同一时间录制的同一批活动，按 session 序号互相配对；Person 3 配对同序号、时长最接近的 session。
10. **`none` 注视标签**：按 README 原提示词标注时，70.6% 的 fixation 是 `none`，主要原因是注视点落在物体之间。因此在本地补了一轮标注：用 512 px 上下文图加红点，问"离红点最近的物体"，`none` 降到 36.6%。原标签保存在 `label_strict` 列，来源记录在 `label_source` 列。剩下的 `none` 不进 C5 的 focus 列表和 GazeDB，也不触发 C6 的切分边界，但注视时长仍计入 C6 的帧权重（新颖度系数按 0.3 计）。
11. **C6**：每段帧数上限 60（单次请求的图像上限为 70）；caption 的 `max_new_tokens` = 256 × 该段帧数 / 30。
12. **目标覆盖率**：只检查与证据时间段重叠的记忆文本（caption + gaze 元数据 + 小时摘要；Video-RAG 为 ASR/OCR/Gaze 条目）。判定标准为字符串命中，或 MiniLM 句向量余弦相似度 > 0.7。
13. **β 调参**：C5 在 dev 集上扫 β ∈ {0, 0.25, 0.5, 1}，每条 pipeline 单独选，B2 沿用 B1 的结果，选定后固定再跑 test。
