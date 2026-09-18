# Occupancy Probe → PI0.5 Ablation

这条线只做一件事：用 **自身占用（self-occupancy）探针** 找出与「自我表征」相关的层 × token-bin，再消融它们，看动作是否跟着变。

这里的 probe 的监督信号是机械臂在基座坐标系下的 3D 占用格。能较好还原占用的 bin，视为自我表征相关单元的候选。

```text
occupancy GT + 隐状态     occupancy probe              pi05_ablation
体素占用 + 层×token-bin  →  每格独立 MLP，IoU 定位神经元  →  高/低 IoU bin 置零，看动作 chunk
```

前置：Linux + GPU、`MUJOCO_GL=egl`、`$PI05_PATH`。产物放 `outputs/`。

占用网格：`[-0.8, 0.8]² × [0.0, 1.6]` m，`16³`。解码器：`Linear(D,64) → GELU → Linear(64,4096)`。损失默认 **正类加权 BCE + soft Dice**。每个 tower × layer × token-bin（Expert 还有 Euler 时刻）单独一个 MLP。

---



## 目标与意义


| 阶段                  | 问什么                   | 结论形态                                        |
| ------------------- | --------------------- | ------------------------------------------- |
| **Occupancy probe** | 哪些隐状态单元编码「我的身体占了哪些格子」 | 每格 probe 的 held-out soft IoU；高 IoU ≈ 自我表征候选 |
| **Ablation**        | 这些单元是否也参与出动作          | 同层 good vs bad bin 置零后，动作 chunk 的 L2 差      |


Ablation 只比较 **同层、深度铺开** 的高/低 IoU bin，避免把层深或某个固定 token 位置跟「自我表征」混在一起。默认离线 `predict_action_chunk`，每步清 cache，不用 `select_action`。

IoU 高仍可能来自像素/本体/时间泄漏，不等于独立的 3D 自我模型。可用 shuffle / CMI / NDS 压这个解释（文末对照脚本）。

---



## 1. Occupancy probe：数据与训练

两套管线；**下游 ablation 默认读 Spatial** `run_full.py` **的** `metrics.csv`**。**

### A. Spatial 端到端

均匀 20 帧/demo；token 等宽分箱；Expert 在 Euler `t=1.0, 0.5, 0.1` 取状态。可 `--stage gt|activations|train` 续跑。

```bash
python scripts/occupancy/run_full.py \
  --dataset-dir /path/to/libero_spatial \
  --pi05-path "$PI05_PATH" \
  --output-dir outputs/self_occupancy/pi05_libero_spatial_10k \
  --stage all \
  --max-tasks 10 --max-demos 500 --max-frames 20 --max-samples 10000 \
  --layers all --bins 96 --bin-indices all \
  --shard-size 8 --epochs 20 --batch-size 128 \
  --grid-size 16 --supersample 2

python scripts/occupancy/plot_results.py \
  --run-dir outputs/self_occupancy/pi05_libero_spatial_10k \
  --plots 1,2,3,4
```

产物：`metrics.csv`（层 × bin 的 probe IoU）、`probes/*/decoder.pt`。

### B. Libero-90 分步（规模更大，分箱与 A 不同）

1. **GT**：回放并把 Panda 碰撞体素化 → `occupancy.npy` `[20,16,16,16]` float16，附 RGB。
2. **激活**：同一 20 帧；PaliGemma 968 → **97** 个连续 10-token bin；Expert 50 → **5** bin；五个 Euler 时刻 `1.0,0.8,0.6,0.3,0.1`。`split.json` 每任务 50 demo 按 30/10/10 划 train/test/ablation，**只推理 train+test**。
3. **探针训练**：按 split 训 MLP；split 里的 ablation 子集此阶段不用。

```bash
python scripts/occupancy/extract_libero_gt.py --output-root /path/to/occupancy
python scripts/occupancy/extract_libero_activations.py \
  --occupancy-root /path/to/occupancy \
  --output-root /path/to/occupancy_activations \
  --pi05-path "$PI05_PATH"

python scripts/occupancy/train_libero_decoders.py \
  --activation-root /path/to/occupancy_activations \
  --occupancy-root /path/to/occupancy \
  --output-dir outputs/occupancy_decoders \
  --towers all --layers all --bin-indices all \
  --loss bce_dice --epochs 20
```

B 的 bin（10-token 连续、5 个 flow 时刻）与 A 的 `--bins 96` 不同，**不要混用两套** `metrics.csv` **做同一套 bin 选择。**  
损失对照：`--loss bce` vs 默认 `bce_dice`。

### 对照（可选，不重抽激活）

```bash
python scripts/occupancy/shuffle_controls.py --run-dir outputs/self_occupancy/pi05_libero_spatial_10k
python scripts/occupancy/conditional_mi.py --run-dir outputs/self_occupancy/pi05_libero_spatial_10k
python scripts/occupancy/nds_q_vs_o.py --run-dir outputs/self_occupancy/pi05_libero_spatial_10k --bottleneck 64
```

---



## 2. PI0.5 Ablation：干预自我表征候选 bin

从 probe 的 `metrics.csv` **按层**选 good/bad bin（每塔铺开若干层，每层各取若干高/低 IoU），在离线动作预测上分别置零，记 flattened chunk L2。

```bash
python scripts/pi05_ablation/select_bins.py \
  --metrics outputs/self_occupancy/pi05_libero_spatial_10k/metrics.csv \
  --output-dir outputs/ablation/pi05_probe_guided_layer_matched \
  --layers-per-tower 5 --bins-per-group 2 \
  --min-iou-gap 0.03 --max-good-bin-repeats 2 --expert-max-bin 48

python scripts/pi05_ablation/run_offline_ablation.py \
  --probe-run-dir outputs/self_occupancy/pi05_libero_spatial_10k \
  --selected-csv outputs/ablation/pi05_probe_guided_layer_matched/selected_conditions.csv \
  --pi05-path "$PI05_PATH" \
  --output-dir outputs/ablation/pi05_probe_guided_frames \
  --num-frames 1000 --sample-seed 0
```

`sampled_frames.csv` 的 `row_index` 与 `frame_deltas.npz` 的 `chunk_l2[i]` 对齐。只重画图：`--plot-only`。

帧级标量（运动、运动学、接近、接触、阶段）和 impact 分组直方图。Impact = `mean(Δ_good) − mean(Δ_bad)`；默认最高 20%、剩余中最接近 0 的 20%、最低 20%。

```bash
python scripts/pi05_ablation/extract_frame_stats.py \
  --sampled-csv outputs/ablation/pi05_probe_guided_frames/sampled_frames.csv \
  --output-dir outputs/ablation/pi05_probe_guided_frames/frame_stats \
  --simulator auto --replay-mode selected

python scripts/pi05_ablation/plot_frame_stat_groups.py \
  --stats-csv outputs/ablation/pi05_probe_guided_frames/frame_stats/frame_stats.csv \
  --impact-csv outputs/ablation/pi05_probe_guided_frames/frame_good_minus_bad.csv \
  --output-dir outputs/ablation/pi05_probe_guided_frames/frame_stat_groups
```

闭环 IoU-diff（旧 rollout，不是上面的离线帧任务）：`scripts/pi05_ablation/plot_closed_loop_iou_diff.py`。在线干预 tracer：`src/online_rollout.py`。

---



## 核心代码


| 作用                   | 路径                                                           |
| -------------------- | ------------------------------------------------------------ |
| 占用 MLP / BCE+Dice    | `src/occupancy_decoder.py`                                   |
| Spatial 全量与 Euler 捕获 | `scripts/occupancy/run_full.py`，`src/pi05_occupancy_full.py` |
| GT 体素化               | `src/libero_self_occupancy.py`                               |
| 层匹配选 bin             | `src/pi05_probe_ablation.py`                                 |
| 离线消融                 | `src/pi05_frame_ablation.py`                                 |


