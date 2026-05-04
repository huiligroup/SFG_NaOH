# NaOH ViSNet-eIP 网络设计与训练说明

本文档说明 `NaOH_7.5M/visnet` 中当前 ViSNet-eIP 训练管线的设计、数据格式、周期边界条件处理、训练命令、指标单位，以及它与 GPUMD 的关系。

## 1. 目标与边界

当前实现的目标是训练一个用于 NaOH 体系的 `ViSNet + evidential learning` 势能模型。模型以原子种类、坐标和周期盒为输入，输出总势能、原子力以及力分量的不确定性参数。

需要明确的是：当前模型不是 GPUMD 的 NEP 势，也不会导出为 `nep.txt`。GPUMD 原生运行的是它已经实现的势函数类型，例如 NEP、DP、Tersoff、EAM 等。ViSNet 是 PyTorch/PyG message passing 网络，不能直接作为 GPUMD 的 `potential` 文件使用。

本项目对接 GPUMD 的方式是工程设计层面的对齐：采用 GPUMD 风格的周期边界和邻居表处理，即先构造邻居索引和周期平移，再在模型中用这些边计算相互作用。这可以减少 forward 中的 Python 邻居搜索和 autograd 图负担。

## 2. 数据来源与单位

原始数据来自：

```text
NaOH_7.5M/data/NVE/Na12/a
NaOH_7.5M/data/NVE/Na12/b
```

每条轨迹包含：

```text
NaOH04-pos-1.xyz   坐标和注释行势能
NaOH04-frc-1.xyz   CP2K 输出的力
NaOH04-1.ener      能量表
input.inp          周期盒信息
```

预处理脚本：

```bash
python NaOH_7.5M/visnet/data/preprocess.py
```

输出数据：

```text
NaOH_7.5M/data/visnet/na12_ab.pkl
```

数据单位如下：

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| `pos` | Angstrom | 原子坐标 |
| `cell` | Angstrom | 周期盒矢量 |
| `energy` | Hartree / frame | 每帧总势能 |
| `forces` | Hartree / Angstrom | CP2K 原始 Hartree/Bohr 已转换 |

换算到常见单位：

```text
1 Hartree = 27.211386245988 eV
1 Hartree / Angstrom = 27.211386245988 eV / Angstrom
```

## 3. pkl 数据结构

`na12_ab.pkl` 是一个 Python pickle 字典，主要字段包括：

```text
z        原子序数，shape [480]
species  元素符号，shape [480]
pos      坐标，shape [n_frames, 480, 3]
forces   力，shape [n_frames, 480, 3]
energy   总势能，shape [n_frames]
cell     周期盒，shape [n_frames, 3, 3]
traj     轨迹标签，a 或 b
frame    原始 CP2K step
split    train / val / test
units    单位说明
metadata 数据统计
```

当前每帧有 480 个原子，元素计数为：

```text
O: 160
H: 308
Na: 12
```

数据集默认按每条轨迹时间顺序切分：

```text
80% train / 10% val / 10% test
```

训练时默认 `--frame-stride 4`，也就是每 4 帧取 1 帧，减少强相关轨迹帧带来的样本冗余。

## 4. 网络结构

模型入口位于：

```text
NaOH_7.5M/visnet/model.py
```

核心类是 `VisNetEIP`，由两部分组成：

1. `ViSNet` 主体网络
   - 使用 PyTorch Geometric 的 `torch_geometric.nn.models.ViSNet`。
   - 输出原子局域表示和原子能量贡献。
   - 总能量由原子能量求和得到。
   - 力由总能量对坐标的负梯度得到：

```text
F = -dE/dR
```

2. evidential head
   - 接 ViSNet 的等变向量特征。
   - 对每个原子的 x/y/z 力分量输出 `nu, alpha, beta`。
   - `gamma` 使用 ViSNet 预测力。
   - epistemic uncertainty 使用：

```text
beta / (nu * (alpha - 1))
```

## 5. Loss 设计

loss 实现在：

```text
NaOH_7.5M/visnet/losses.py
```

总 loss 包含：

```text
L = energy_l1 + force_weight * (force_nll + reg_weight * evidence_reg)
```

其中：

| 项 | 说明 |
| --- | --- |
| `energy_l1` | 总能量 MAE，单位 Hartree/frame |
| `force_nll` | evidential quantile Student-t NLL |
| `evidence_reg` | 预测错误时降低 evidence 的正则项 |
| `q` | quantile，默认 0.5 |

训练中打印的 `loss` 是上述组合 loss，不是单纯 MAE。

## 6. GPUMD 风格 PBC 与邻居表

邻居表工具位于：

```text
NaOH_7.5M/visnet/data/neighbors.py
```

当前实现支持正交周期盒。对每条边保存：

```text
edge_index: [2, n_edges]
cell_shift: [n_edges, 3]
```

边向量在模型中按以下公式构造：

```text
edge_vec = pos[src] + cell_shift @ cell - pos[dst]
```

这与 minimum-image convention 等价，但 `edge_index` 和 `cell_shift` 不进入 autograd。只有 `edge_vec` 对 `pos` 可导，因此仍然可以通过 `-dE/dR` 得到力。

如果训练时没有提供预计算邻居表，模型会在 `torch.no_grad()` 中运行时构造邻居表。这比旧实现更省显存，因为邻居搜索本身不再进入 autograd 图。

## 7. 预计算邻居表

预计算脚本：

```bash
python NaOH_7.5M/visnet/data/precompute_edges.py \
  --data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --cutoff 6.0 \
  --max-num-neighbors 64 \
  --frame-stride 4
```

默认输出：

```text
NaOH_7.5M/data/visnet/na12_ab_edges_cutoff6.0.pt
```

训练时可以显式使用：

```bash
--edge-data NaOH_7.5M/data/visnet/na12_ab_edges_cutoff6.0.pt
```

也可以让训练脚本在缺失时自动生成：

```bash
--build-edges-if-missing
```

注意：完整轨迹的邻居表可能较大。首次测试建议先使用小规模参数检查流程，例如：

```bash
python NaOH_7.5M/visnet/data/precompute_edges.py \
  --cutoff 6.0 \
  --frame-stride 4 \
  --limit 100 \
  --output /tmp/na12_edges_test.pt
```

## 8. 训练命令

推荐长训练命令：

```bash
mkdir -p NaOH_7.5M/visnet/runs/visnet_1

nohup python NaOH_7.5M/visnet/train.py \
  --output-dir NaOH_7.5M/visnet/runs/visnet_1 \
  --epochs 1000 \
  --batch-size 4 \
  --lr 0.0001 \
  --num-layers 6 \
  --cutoff 6.0 \
  --max-num-neighbors 64 \
  --frame-stride 4 \
  --batches-per-step 100 \
  --val-interval-steps 10 \
  --early-stopping-monitor val/force_mae \
  --early-stopping-patience 20 \
  --device cuda \
  --wandb \
  --wandb-project naoh-visnet-eip \
  --wandb-name visnet_1 \
  > NaOH_7.5M/visnet/runs/visnet_1/train.log 2>&1 &
```

如果已经预计算邻居表：

```bash
--edge-data NaOH_7.5M/data/visnet/na12_ab_edges_cutoff6.0.pt
```

如果只想本地记录 wandb：

```bash
--wandb-mode offline
```

## 9. 训练日志与指标单位

训练每 `--batches-per-step` 个 batch 输出一次汇总。默认每 10 个训练 step 做一次验证。

主要指标：

| 指标 | 单位 | 说明 |
| --- | --- | --- |
| `train/loss` | 混合 loss | energy L1 + evidential force loss |
| `train/energy_mae` | Hartree/frame | 总势能 MAE |
| `train/force_mae` | Hartree/Angstrom | 所有原子所有分量的力 MAE |
| `train/energy_rmse` | Hartree/frame | 总势能 RMSE |
| `train/force_rmse` | Hartree/Angstrom | 力分量 RMSE |
| `val/force_mae` | Hartree/Angstrom | 默认早停监控指标 |
| `val/epistemic_mean` | Hartree/Angstrom 相关尺度 | epistemic uncertainty 均值 |
| `val/epistemic_p95` | Hartree/Angstrom 相关尺度 | epistemic uncertainty 第 95 百分位 |
| `val/epistemic_max` | Hartree/Angstrom 相关尺度 | epistemic uncertainty 最大值 |
| `val/aleatoric_mean` | Hartree/Angstrom 相关尺度 | aleatoric uncertainty 均值 |
| `val/uncertainty_error_spearman` | 无量纲 | `epistemic_p95` 与帧级 `force_mae` 的 Spearman 相关系数 |

checkpoint：

```text
best.pt              验证指标最好的模型
visnet_eip_smoke.pt  最后一次保存的模型
metrics.json         训练和验证指标历史
high_uncertainty_frames.csv  验证集中按 epistemic_p95 排序的高不确定性帧
```

`high_uncertainty_frames.csv` 每次验证后更新，主要列包括：

```text
traj, frame, frame_index, force_mae, force_rmse, energy_abs_error,
epistemic_mean, epistemic_p95, epistemic_max,
aleatoric_mean, aleatoric_p95, aleatoric_max
```

主动学习时优先查看 `epistemic_p95` 或 `epistemic_max` 高的帧，而不是只看平均不确定性。局部环境异常通常会被 mean 平滑掉。

## 10. GPUMD/NEP 边界说明

GPUMD 的 `potential` 文件不是通用神经网络模型格式。GPUMD 的 NEP 势使用专门的描述符和单隐层网络参数，训练后生成 `nep.txt`。当前 ViSNet-eIP 是 PyTorch/PyG message passing 网络，不能无损转换成 `nep.txt`。

如果目标是直接在 GPUMD 中运行 MD，有两条不同路线：

1. 使用 GPUMD 的 `nep` 工具重新训练 NEP 势。
2. 修改 GPUMD 源码或接入 LibTorch/TorchScript 来运行 ViSNet。

这两条都不是当前实现的范围。当前实现的范围是：保留 ViSNet-eIP，并把 PBC/邻居表处理做成接近 GPUMD 的高效方式。

## 11. 后续推理路线

后续如果要做 MD 推理，可以复用当前运行时邻居构造器：

1. 输入单帧 `z, pos, cell`。
2. 构造 `batch=0`。
3. 用 `build_neighbor_list` 生成 `edge_index, cell_shift`。
4. 调用 `VisNetEIP.forward(..., edge_index=edge_index, cell_shift=cell_shift)`。
5. 读取 `energy, force, epistemic`。

这样可以在 PyTorch/ASE 侧进行 MD 或不确定性驱动采样，同时保持与训练阶段一致的 PBC 语义。
