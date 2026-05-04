# ViSNet 分子特征向量与代表结构选择说明

本文档说明 `NaOH_7.5M/src/descriptor` 中的分子描述符提取与代表结构选择流程。这个流程用于从已经训练好的 ViSNet-eIP 网络中取出能量输出头之前的原子特征，并把指定分子的原子特征拼接成一个分子特征向量。

## 数据来源

原始轨迹位于：

```bash
NaOH_7.5M/data/NVE/Na12/a
NaOH_7.5M/data/NVE/Na12/b
```

训练和描述符脚本默认读取预处理后的 pkl：

```bash
NaOH_7.5M/data/visnet/na12_ab.pkl
```

描述符脚本把这个 pkl 当作已经采样好的数据源。如果 `NaOH_7.5M/data/visnet/na12_ab.pkl` 已经是每四帧提取一个结构，那么描述符脚本中的 `--frame-stride 1` 表示不再额外跳帧；`--frame-stride 2` 表示在这个 pkl 的基础上每隔一条记录再取一次。

脚本会在输出 pkl 中记录 `source_frame_step_by_traj` 和 `frame_stride_on_pkl`，用于追踪原始 MD step 与描述符抽样之间的关系。

## 分子编号规则

第一版采用固定原子编号追踪分子，也就是同一个分子 ID 在所有帧中始终对应同一组原子索引。默认分子编号来自原始 `NaOH04-pos-1.xyz` 的原子顺序：

- `0-147`：水分子，每个分子为 `O,H,H`。
- `148-159`：NaOH 单元，每个分子为 `Na,O,H`。

例如 `--mol-ids 148,149,150` 表示提取前三个 NaOH 单元；`--mol-ids 148-159` 表示提取全部 12 个 NaOH 单元；`--mol-ids all` 表示提取全部 160 个分子。

当前版本不根据 O-H 距离重新识别成键，也不处理分子交换或质子转移导致的拓扑变化。它追踪的是固定原子组。

## 特征来源

模型文件：

```bash
NaOH_7.5M/visnet/model.py
```

描述符脚本调用 `VisNetEIP.extract_atom_features(...)`，返回 ViSNet representation 的 scalar atom feature `x`。这个 `x` 是进入 `output_model.pre_reduce(x, v)` 能量输出头之前的原子特征。

对一个分子，脚本按该分子的原子顺序取出每个原子的 `x`，然后直接展平成一维向量：

```text
molecule_feature = concat(x_atom_1, x_atom_2, x_atom_3)
```

如果 checkpoint 的 `hidden_channels=64`，三原子分子的特征维度就是 `3 * 64 = 192`。

## 特征提取命令

推荐先用训练效果更好的 checkpoint：

```bash
python NaOH_7.5M/src/descriptor/extract_mol_features.py \
  --data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --mol-ids 148,149,150 \
  --frame-stride 1 \
  --device cuda \
  --output NaOH_7.5M/src/descriptor/mol_features.pkl
```

CPU 冒烟测试：

```bash
python NaOH_7.5M/src/descriptor/extract_mol_features.py \
  --data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --mol-ids 148 \
  --limit-frames 2 \
  --device cpu \
  --output NaOH_7.5M/src/descriptor/mol_features_smoke.pkl
```

输出 pkl 的主要字段：

- `features`：形状为 `[记录数, 特征维度]` 的分子特征矩阵。
- `mol_ids`：每条记录对应的分子 ID。
- `atom_indices`：每条记录对应的原子索引。
- `traj`、`frame`、`frame_index`：回溯到原始轨迹和 pkl 内部帧索引的信息。
- `pos`：每条记录中该分子的坐标。
- `cell`：每条记录对应的周期盒。
- `system_pos`：每个被提取帧的完整 480 原子体系坐标。
- `system_cell`：每个被提取帧的周期盒。
- `record_frame_pos_index`：每条记录对应 `system_pos` 的索引。
- `checkpoint`、`feature_type`、`source_data`、`model_args`：复现实验所需的模型和数据来源。

## 代表结构选择

代表结构选择脚本：

```bash
NaOH_7.5M/src/descriptor/select_representative_mol.py
```

选择算法是 farthest-point sampling：

1. 对所有候选特征做逐维标准化。
2. 默认先计算每个特征向量到其他所有向量的平均 Euclidean distance，并选平均距离最大的向量作为第一个代表结构。
3. 之后每一步选择“到已选集合的最近距离”最大的向量。
4. 重复直到达到指定的 `k`。

推荐命令：

```bash
python NaOH_7.5M/src/descriptor/select_representative_mol.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --k 20 \
  --output-dir NaOH_7.5M/src/descriptor/represent_mol
```

如果只想从某一个分子的特征中选择代表结构：

```bash
python NaOH_7.5M/src/descriptor/select_representative_mol.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --mol-id 148 \
  --k 20 \
  --output-dir NaOH_7.5M/src/descriptor/represent_mol
```

当候选记录非常多时，精确计算第一步平均距离会很慢。此时可以用快速近似初始化：

```bash
--initial-mode centroid
```

默认仍然是 `--initial-mode exact-mean`，对应“到其他所有特征向量平均距离最大”的定义。

## 输出结构

代表结构默认输出到：

```bash
NaOH_7.5M/src/descriptor/represent_mol
```

每个代表结构一个文件夹，例如：

```text
represent_mol/
  rank_0001_mol_148_traj_a_frame_00000000/
    system.xyz
    feature.npy
    metadata.json
  representatives.pkl
```

其中：

- `system.xyz` 是该代表分子所在帧的完整 480 原子体系结构。
- `feature.npy` 是该代表分子的原始 ViSNet 分子特征向量。
- `metadata.json` 记录 `rank`、`mol_id`、`atom_indices`、`traj`、`frame`、`frame_index`、checkpoint 和源 pkl。
- `representatives.pkl` 汇总全部代表结构、标准化参数和选出的特征矩阵。

## 代表结构分析与可视化

分析工具已放在：

```bash
NaOH_7.5M/utils
```

分析工具默认读取：

```bash
NaOH_7.5M/src/descriptor/mol_features.pkl
NaOH_7.5M/src/descriptor/represent_mol/representatives.pkl
```

默认输出目录为：

```bash
NaOH_7.5M/utils/descriptor_report
```

这些工具的目标不是重新选择代表结构，而是从特征空间和真实构型两个角度检查已选代表分子是否合理：

- 展示所有分子特征向量的 PCA 分布。
- 在 PCA 图中高亮代表结构，检查代表点是否覆盖点云边界和稀疏区域。
- 统计代表结构之间的两两距离，检查它们是否确实相互不同。
- 统计全体样本到最近代表点的距离，检查代表点对整个特征空间的覆盖程度。
- 对代表分子做键长、键角、最近邻距离、局部配位数等构型统计。
- 绘制代表结构的 3D 图，用完整体系作为背景并高亮代表分子。

一键生成分析报告：

```bash
python NaOH_7.5M/utils/build_descriptor_report.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --representatives NaOH_7.5M/src/descriptor/represent_mol/representatives.pkl \
  --output-dir NaOH_7.5M/utils/descriptor_report
```

单独运行 PCA 分析：

```bash
python NaOH_7.5M/utils/plot_descriptor_pca.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --representatives NaOH_7.5M/src/descriptor/represent_mol/representatives.pkl \
  --output-dir NaOH_7.5M/utils/descriptor_report/pca
```

单独运行代表点距离分析：

```bash
python NaOH_7.5M/utils/analyze_representatives.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --representatives NaOH_7.5M/src/descriptor/represent_mol/representatives.pkl \
  --output-dir NaOH_7.5M/utils/descriptor_report/distances
```

单独运行构型统计和 3D 绘图：

```bash
python NaOH_7.5M/utils/plot_representative_structures.py \
  --features NaOH_7.5M/src/descriptor/mol_features.pkl \
  --representatives NaOH_7.5M/src/descriptor/represent_mol/representatives.pkl \
  --output-dir NaOH_7.5M/utils/descriptor_report/structures
```

主要输出文件包括：

- `report.md`：汇总 PCA、距离统计、构型统计和代表结构列表。
- `summary.json`、`summary.csv`、`analysis.pkl`：机器可读的统计结果。
- `pca_2d.png`、`pca_3d.png`、`pca_variance.png`：特征空间 PCA 分布和主成分解释方差。
- `pca_by_traj.png`、`pca_by_mol_id.png`、`pca_by_frame.png`、`pca_coordinates.csv`：按轨迹、分子编号和帧编号检查 PCA 分布。
- `coverage_distance_hist.png`：全体样本到最近代表点的距离分布。
- `representative_pairwise_distance.png`：代表点之间的两两距离矩阵或分布。
- `rank_distance_curve.png`：按选择顺序记录的代表点距离曲线。
- `random_baseline_comparison.png`、`random_baseline.csv`：与随机选择代表点的基线对比。
- `structure_stats.csv`：代表分子的键长、键角、最近邻距离和局部配位数。
- `bond_length_hist.png`、`angle_hist.png`、`coordination_hist.png`：构型统计图。
- 每个代表结构的 3D PNG：完整体系半透明显示，代表分子高亮显示。

分析结果的解释标准：

- PCA 中，合理的代表点通常应覆盖主要点云的边界、稀疏区域或不同分支，而不是全部挤在中心区域。
- 代表点两两距离应整体高于随机样本基线；如果很多代表点彼此很近，说明选择结果存在冗余。
- 全体样本到最近代表点的距离分布越低，说明代表点对特征空间覆盖越充分；长尾越明显，说明仍有区域没有被代表结构覆盖。
- 构型统计用于判断特征空间差异是否对应真实物理结构差异。例如 Na-O 距离、O-H 距离、H-O-H 或 Na-O-H 角度、局部配位数是否随代表 rank 呈现明显变化。
- 如果 PCA 或距离统计显示代表点差异很大，但构型统计几乎没有变化，说明差异可能主要来自网络特征空间中的环境响应，而不是所选分子内部几何变化。

## 注意事项

- 特征向量不是能量、力或 eIP uncertainty，而是 ViSNet 网络内部学到的结构表示。
- 不同 checkpoint 的特征空间不应直接混合比较。
- 如果更改 `hidden_channels` 或网络层数，需要重新提取特征。
- 当前输出的 `.xyz` 坐标单位继承训练 pkl，单位为 Angstrom。
- 代表结构合理性需要同时看特征空间差异和真实构型差异，不能只依赖 PCA 图或单一距离指标。
