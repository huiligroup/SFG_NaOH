# NaOH ViSNet-eIP 训练说明

本文档说明 `NaOH_7.5M/visnet` 中当前 ViSNet-eIP 训练管线的结构、数据格式、单位、PBC/邻居表设计，以及推荐运行方式。

## 1. 推荐入口

当前推荐入口是顶层 `TOML + .sh`：

```text
配置文件: NaOH_7.5M/config/visnet_initial_train.toml
启动脚本: NaOH_7.5M/scripts/train_visnet_initial.sh
```

这个入口会按顺序执行：

```text
preprocess -> precompute_edges -> train
```

推荐命令：

```bash
bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml
```

如需后台运行：

```bash
nohup bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml \
  > NaOH_7.5M/visnet/runs/naoh12_visnet/train.log 2>&1 &
```

如果只想临时覆盖少量训练参数，而不改 TOML，可以把额外参数放到 `--` 后面转发给 `train.py`：

```bash
bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml \
  -- --epochs 1 --limit-batches 4 --limit-val-batches 2 --device cpu
```

## 2. 数据来源

原始数据来自：

```text
NaOH_7.5M/data/NVE/NaOH12
```

其中至少包括：

```text
NaOH-12-pos-1.xyz   坐标和注释行势能
NaOH-12-frc-1.xyz   CP2K 力
NaOH-12-1.ener      能量表
NaOH12.inp          周期盒信息
```

初始训练脚本会先调用 `preprocess.py`，输出：

```text
NaOH_7.5M/data/visnet/naoh12.pkl
```

## 3. `visnet_initial_train.toml` 结构

训练配置统一放在：

[`NaOH_7.5M/config/visnet_initial_train.toml`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/config/visnet_initial_train.toml)

主要分为三段：

- `[preprocess]`
  - 原始 CP2K 轨迹位置
  - 输出 `.pkl` 路径
  - 使用哪些轨迹子目录
  - 是否做 `.ener` 对照检查
- `[edges]`
  - 是否预计算邻居表
  - 邻居表输出路径
  - `cutoff`、`max_num_neighbors`
  - `frame_stride`、`splits`、`num_workers`
- `[train]`
  - `train.py` 的核心超参数
  - 模型深度、batch size、学习率、早停、wandb、checkpoint 输出等

约束关系：

- `preprocess.output` 应与 `train.data` 相同
- 若 `edges.enabled = true`，则 `edges.output` 应与 `train.edge_data` 相同

脚本会在启动前检查这两条，避免预处理一个文件却训练另一个文件。

## 4. `.pkl` 数据结构

`naoh12.pkl` 是一个 Python pickle 字典，主要字段包括：

```text
z        原子序数，shape [480]
species  元素符号，shape [480]
pos      坐标，shape [n_frames, 480, 3]
forces   力，shape [n_frames, 480, 3]
energy   总势能，shape [n_frames]
cell     周期盒，shape [n_frames, 3, 3]
traj     轨迹标签
frame    原始 CP2K step
split    train / val / test
units    单位说明
metadata 数据统计
```

当前体系每帧为 480 原子，元素计数为：

```text
O: 160
H: 308
Na: 12
```

默认按时间顺序切分：

```text
80% train / 10% val / 10% test
```

训练默认 `frame_stride = 4`，也就是每 4 帧取 1 帧，减少轨迹强相关冗余。

## 5. 单位

数据单位如下：

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| `pos` | Angstrom | 原子坐标 |
| `cell` | Angstrom | 周期盒矢量 |
| `energy` | Hartree / frame | 每帧总势能 |
| `forces` | Hartree / Angstrom | 由 CP2K 原始 Hartree/Bohr 转换而来 |

常见换算：

```text
1 Hartree = 27.211386245988 eV
1 Hartree / Angstrom = 27.211386245988 eV / Angstrom
```

## 6. 网络结构

模型入口：

[`NaOH_7.5M/visnet/model.py`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/visnet/model.py)

核心类 `VisNetEIP` 由两部分组成：

1. `ViSNet` 主体
   - 使用 `torch_geometric.nn.models.ViSNet`
   - 从原子类型、坐标、邻居关系预测总能量
   - 力通过总能量对坐标求负梯度得到
2. evidential head
   - 接 ViSNet 的等变向量特征
   - 为每个原子每个力分量输出 `nu, alpha, beta`
   - `gamma` 使用 ViSNet 预测力本身

当前默认重要超参数：

```text
hidden_channels = 64
num_layers = 6
num_heads = 8
num_rbf = 32
cutoff = 6.0 Å
max_num_neighbors = 64
```

对 480 原子的 `NaOH12` 周期 box，真正决定显存峰值的主要是：

```text
batch_size
grad_accumulation_steps
gradient_checkpointing
max_num_neighbors
```

## 7. Loss 设计

loss 位于：

[`NaOH_7.5M/visnet/losses.py`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/visnet/losses.py)

总 loss 形式：

```text
L = energy_l1 + force_weight * (force_nll + reg_weight * evidence_reg)
```

其中：

| 项 | 说明 |
| --- | --- |
| `energy_l1` | 总能量 MAE，单位 Hartree/frame |
| `force_nll` | evidential quantile force loss |
| `evidence_reg` | 抑制错误高置信度的正则项 |
| `q` | quantile，默认 0.5 |

## 8. PBC 与邻居表

邻居表工具位于：

[`NaOH_7.5M/visnet/data/neighbors.py`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/visnet/data/neighbors.py)

当前实现采用 GPUMD 风格的邻居表：

```text
edge_index: [2, n_edges]
cell_shift: [n_edges, 3]
```

边向量在模型中按以下形式构造：

```text
edge_vec = pos[src] + cell_shift @ cell - pos[dst]
```

这保证：

- 邻居搜索和周期平移不进入 autograd
- `edge_vec` 仍对 `pos` 可导
- 因而仍可用 `-dE/dR` 得到力

## 9. 邻居表预计算

正式训练默认由 `train_visnet_initial.sh` 根据 `[edges]` 段自动处理。只有在单独调试时，才建议直接运行底层脚本：

```bash
python NaOH_7.5M/visnet/data/precompute_edges.py \
  --data NaOH_7.5M/data/visnet/naoh12.pkl \
  --cutoff 6.0 \
  --max-num-neighbors 64 \
  --frame-stride 4
```

默认输出：

```text
NaOH_7.5M/data/visnet/naoh12_edges_cutoff6.0.pt
```

当前已经加入自动缓存检测：

- 如果 `edge_data` 已存在，且数据文件、`cutoff`、`max_num_neighbors`、`frame_stride`、`splits`、`limit` 都匹配，脚本会直接复用缓存
- 如果数据文件时间戳或大小变化，或邻居表参数变化，则自动判定为 stale 并重建
- 对旧版不带完整签名信息的缓存，会退回到 frame index 校验；校验通过时仍可复用

## 10. `train.py --config`

训练脚本现在支持：

```bash
python NaOH_7.5M/visnet/train.py \
  --config NaOH_7.5M/config/visnet_initial_train.toml
```

优先级为：

```text
CLI 显式参数 > TOML [train] > 代码默认值
```

这意味着：

- 正式运行优先用 `.sh`
- 调试单独训练阶段时可以直接用 `train.py --config`
- 必要时再用额外 CLI 参数覆盖 TOML

## 11. 训练输出与指标单位

默认输出目录由 `[train].output_dir` 控制，典型内容包括：

```text
best.pt
visnet_eip_smoke.pt
metrics.json
high_uncertainty_frames.csv
```

主要指标单位：

| 指标 | 单位 | 说明 |
| --- | --- | --- |
| `train/loss` | 混合 loss | energy L1 + evidential force loss |
| `train/energy_mae` | Hartree/frame | 总势能 MAE |
| `train/force_mae` | Hartree/Angstrom | 力 MAE |
| `train/energy_rmse` | Hartree/frame | 总势能 RMSE |
| `train/force_rmse` | Hartree/Angstrom | 力 RMSE |
| `val/force_mae` | Hartree/Angstrom | 默认早停监控指标 |
| `val/uncertainty_error_spearman` | 无量纲 | 帧级不确定性与误差相关性 |

`high_uncertainty_frames.csv` 适合主动学习时回看高不确定帧。

## 12. 显存优化

如果训练报 `torch.OutOfMemoryError`，优先调整这几个量：

1. 降低 `batch_size`
2. 增大 `grad_accumulation_steps`
3. 打开 `gradient_checkpointing = true`
4. 必要时降低 `max_num_neighbors`

当前训练代码已经支持：

- activation checkpointing：降低表征层激活显存
- gradient accumulation：用小 micro-batch 保持较大的有效 batch size

推荐的保守组合：

```text
batch_size = 4
grad_accumulation_steps = 8
gradient_checkpointing = true
```

如果还不够，再降到：

```text
batch_size = 2
grad_accumulation_steps = 16
```

## 13. 与 GPUMD 的边界

当前模型不是 GPUMD 的 `nep.txt`，也不会导出为 GPUMD 原生势。当前对齐的是：

- 周期边界处理方式
- 邻居表工作流
- 推理阶段的工程结构

如果目标是直接在 GPUMD 中运行，需要重新训练 GPUMD 支持的势，或修改 GPUMD/LibTorch 接口去承载 ViSNet。这不属于当前实现范围。

## 14. 从零开始的最短顺序

1. 训练初始模型：

```bash
bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml
```

2. 确认生成：

```text
NaOH_7.5M/visnet/runs/naoh12_visnet/best.pt
```

3. 然后再启动主动学习：

```bash
bash NaOH_7.5M/scripts/run_visnet_active_learning.sh \
  --config NaOH_7.5M/config/visnet_active_learning.toml
```
