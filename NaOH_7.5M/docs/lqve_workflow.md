# LQVE 全流程使用文档

本文档说明如何使用 `NaOH_7.5M/src/lqve` 中的新 LQVE 模块完成从参考 DVR 到实时频移分析的完整流程。旧代码位于 `NaOH_7.5M/src/lqve_old`，只作为数值和算法参考；新的输入、处理中间文件和结果应尽量放在 `NaOH_7.5M/src/lqve` 的对应目录中。

## 1. 方法总览

LQVE 当前被拆成两个层次。

第一层是参考体系准备。对一组已经选定的参考分子或参考构型，先用高精度 PES 求解 DVR，得到参考振动能级、参考波函数和 DVR/PODVR 网格点。这个阶段只在参考库准备时运行，不在每个 AIMD frame 中重复求解。

第二层是逐帧实时环境计算。对 AIMD 轨迹中的每一帧，将已经准备好的 DVR 点嵌入当前环境，计算这些 DVR 构型的能量，再用有效 Hamiltonian 得到该帧下的瞬时振动能级、跃迁频率和频移。

当前实现还支持动态参考选择：如果已经预先准备了多个参考 DVR/PES/波函数，可以在每一帧先用 ViSNet 提取探针分子特征，与参考库特征比较，选择最相近的参考。注意，这一步只是在已有参考之间选择，不会重新生成 PES，也不会重新求 DVR。

整体数据流如下：

```text
reference PES
  -> 01_build_reference_dvr.py
  -> dvr_result.npz + metadata.json

AIMD frame + reference DVR
  -> 02_build_embedded_geometries.py
  -> embedded_geometries.npz + metadata.json

embedded DVR geometries
  -> 03_run_qc_energies.py
  -> qc_energies.npz + metadata.json

DVR result + QC/ViSNet energies + reference grid energies
  -> 04_compute_lqve_shift.py
  -> shift_result.npz + shifts_summary.csv

shift summary
  -> 05_analyze_results.py
  -> tables + figures + report.md
```

## 2. 目录结构

主要目录如下：

```text
NaOH_7.5M/src/lqve/
  configs/                    JSON 配置文件示例
  scripts/                    五个命令行入口
  lqve/                       可作为 Python 包调用的核心模块
    dvr/                      DVR/PODVR、PES 读取、波函数和能级
    embedding/                Kabsch 对齐、模式投影、DVR 点嵌入
    reference_selection/      ViSNet 特征匹配和动态参考选择
    qc/                       ViSNet/xTB/Gaussian/CP2K/mock 能量后端
    perturbation/             有效 Hamiltonian 与频移计算
    analysis/                 统计、绘图和报告
  data/
    raw/                      原始局部输入
    processed/                处理后的轨迹或结构
    dvr/                      参考 DVR 结果
    embedded_geometries/      每帧嵌入后的 DVR 构型
    qc_outputs/               每帧能量计算结果
    shifts/                   每帧频率和频移结果
  references/
    structures/               参考探针结构
    pes/                      参考 PES 或参考 DVR 网格能量
    modes/                    正则模式或局域模式
    dvr_points/               额外保存的 DVR 点文件
  results/
    tables/                   最终表格
    figures/                  最终图片
    logs/                     日志
```

已有配置示例：

```text
NaOH_7.5M/src/lqve/configs/dvr_example.json
NaOH_7.5M/src/lqve/configs/embedding_example.json
NaOH_7.5M/src/lqve/configs/qc_example_visnet.json
NaOH_7.5M/src/lqve/configs/qc_example_xtb.json
NaOH_7.5M/src/lqve/configs/shift_example.json
NaOH_7.5M/src/lqve/configs/analysis_example.json
NaOH_7.5M/src/lqve/configs/workflow_example.json
```

## 3. 单位约定

内部和输出单位需要保持一致，否则 shift 阶段很容易出错。

- Cartesian 坐标：Angstrom。
- `dvr_result.npz` 中的 DVR grid：Bohr。
- embedding 时 `--grid-unit` 用来声明 DVR grid 输入单位，常用 `bohr`。
- mode 文件单位由 `--mode-unit` 声明，可选 `dimensionless`、`angstrom_per_angstrom`、`angstrom_per_bohr`、`bohr_per_bohr`。
- 所有能量最终都应为 Hartree。
- `reference_energies` 是每个 DVR grid 点的参考能量，单位 Hartree。
- shift 输出的 transition 和 shift 单位为 `cm^-1`。
- ViSNet 后端预测的能量按训练代码约定输出 Hartree。

## 4. 第一步：构建参考 DVR

脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py --help
```

功能：

- 读取参考 PES 表格。
- 构建每个模式的一维 SINC-DVR。
- 选择 PODVR 点数。
- 组装 N 维 Hamiltonian。
- 求解最低若干个参考振动态。
- 输出参考能级、参考波函数和 DVR grid。

推荐命令：

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --pes NaOH_7.5M/src/lqve_old/get_DVR_points/energy_9115_minus_1_to_1_8000.dat \
  --dims 3 \
  --mode-names q1,q2,q3 \
  --ranges -0.7:0.7,-0.9:0.9,-0.6:0.6 \
  --sinc-points 200,200,200 \
  --podvr-points 7,10,7 \
  --states 80 \
  --coord-unit angstrom \
  --energy-unit cm-1 \
  --output-dir NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7
```

也可以使用配置文件：

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --config NaOH_7.5M/src/lqve/configs/dvr_example.json
```

关键参数：

- `--pes`：参考 PES 表格路径。
- `--dims`：DVR 维数。
- `--mode-names`：模式名，仅用于 metadata。
- `--ranges`：每个模式的坐标范围。
- `--sinc-points`：每维 SINC-DVR 点数。
- `--podvr-points`：每维 PODVR 点数。
- `--states`：求解的最低振动态数量。
- `--dense-threshold`：小矩阵使用 dense eigh，大矩阵使用 sparse eigsh。
- `--solver-tol`：稀疏求解容差。

输出目录包含：

```text
dvr_result.npz
metadata.json
```

`dvr_result.npz` 的主要数组：

- `levels_hartree`：参考能级，单位 Hartree。
- `wavefunctions`：参考波函数。
- `grid_0`, `grid_1`, ...：每一维 DVR grid，单位 Bohr。
- `basis_shape`：每维 PODVR 点数组成的 grid shape。
- `state_count`：求解态数。

`metadata.json` 记录输入 PES、单位、模式范围、点数、求解器、矩阵规模和运行时间。

## 5. 第二步：将 DVR 点嵌入实时环境

脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py --help
```

功能：

- 读取参考探针结构。
- 读取当前 AIMD frame 或 cluster/system XYZ。
- 按 `probe_indices` 找到当前帧中的探针分子。
- 对实时探针与参考探针做质量加权 Kabsch 对齐。
- 投影掉指定振动模式方向以得到环境扰动部分。
- 将所有 DVR grid 点沿模式方向重建为探针构型。
- 旋回实时坐标系，替换完整体系中的探针坐标。

显式参考模式的推荐命令：

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --reference-xyz NaOH_7.5M/src/lqve_old/ref/h2o_ref.xyz \
  --system-xyz NaOH_7.5M/src/lqve_old/ref/one_cl_9115_ref.xyz \
  --probe-indices 0,1,2 \
  --modes NaOH_7.5M/src/lqve/references/modes/h2o_modes.npy \
  --dvr-data NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7 \
  --grid-unit bohr \
  --mode-unit angstrom_per_bohr \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/one_cl_9115
```

如需写出每个 DVR 点对应的完整体系 XYZ，加上：

```bash
--write-xyz
```

关键参数：

- `--reference-xyz`：参考探针结构，只包含探针原子。
- `--system-xyz`：当前 frame 的完整体系或 cluster XYZ，可为多帧 XYZ。
- `--frame-index`：多帧 XYZ 中的帧编号，0-based。
- `--probe-indices`：完整体系中探针原子的 0-based 索引。
- `--modes`：模式矩阵，支持 `.npy/.npz/.csv/.txt/.dat`，形状可为 `(D,N,3)` 或 `(D,3N)`。
- `--dvr-data`：第一步输出的 DVR 目录。
- `--grid-unit`：DVR grid 的单位。
- `--mode-unit`：mode 向量单位。
- `--centroid-weight`：质心和对齐权重，默认 `mass`。

输出目录包含：

```text
embedded_geometries.npz
metadata.json
selected_reference.json       # 仅动态参考选择时存在
xyz/                          # 仅 --write-xyz 时存在
xyz_index.csv                 # 仅 --write-xyz 时存在
```

`embedded_geometries.npz` 的主要数组：

- `embedded_probe`：所有 DVR 点对应的探针坐标，shape 为 `(G,N_probe,3)`。
- `embedded_system`：所有 DVR 点对应的完整体系坐标，shape 为 `(G,N_total,3)`。
- `grid_points`：DVR grid 组合，shape 为 `(G,D)`。
- `probe_indices`：探针原子索引。
- `reference_positions`：参考探针坐标。
- `system_positions`：原始当前帧完整体系坐标。
- `rotation`、`translation`：对齐和旋回信息。

## 6. 动态参考选择

动态参考选择发生在 embedding 阶段内部。它不是额外的第 0 步，也不会重新生成参考 DVR。它的作用是：在每个 AIMD frame 中，根据 ViSNet 特征从已经准备好的参考库中选择最相近的一套参考 PES/DVR/波函数。

### 6.1 参考库准备

参考库文件建议放在：

```text
NaOH_7.5M/src/lqve/references/reference_library.json
```

示例文件：

```text
NaOH_7.5M/src/lqve/references/reference_library_example.json
```

每个 reference entry 至少需要：

```json
{
  "reference_id": "example_ref_0001",
  "feature_file": "NaOH_7.5M/src/descriptor/represent_mol/rank_0001_mol_001_traj_a_frame_00024524/feature.npy",
  "reference_xyz": "NaOH_7.5M/src/lqve/references/structures/example_ref_0001.xyz",
  "modes": "NaOH_7.5M/src/lqve/references/modes/example_ref_0001_modes.npy",
  "dvr_data": "NaOH_7.5M/src/lqve/data/dvr/example_ref_0001",
  "reference_energies": "NaOH_7.5M/src/lqve/references/pes/example_ref_0001_reference_energies_hartree.npy"
}
```

字段含义：

- `reference_id`：参考的唯一名字，会进入 `selected_reference.json` 和 `shifts_summary.csv`。
- `feature_file` 或 `feature`：参考分子的 ViSNet 特征。推荐使用 `feature.npy`。
- `reference_xyz`：该参考对应的探针结构。
- `modes`：该参考对应的模式文件。
- `dvr_data`：该参考对应的 DVR 结果目录。
- `reference_energies`：该参考在 DVR grid 上的参考势能，单位 Hartree，长度必须等于 DVR grid 点数。

参考特征应与当前 query 使用同一个 ViSNet checkpoint 和同一种特征定义：`visnet_scalar_x_before_energy_head`。如果特征维度不一致，程序会停止。

### 6.2 动态选择命令

如果当前 frame 来自 `NaOH_7.5M/data/visnet/na12_ab.pkl`，可以用 `--feature-data` 直接读取其中的 `pos/cell/z/species` 做 ViSNet 前向：

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --system-xyz NaOH_7.5M/src/lqve/data/processed/frame.xyz \
  --probe-indices 444,445,446 \
  --reference-library NaOH_7.5M/src/lqve/references/reference_library.json \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --feature-data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --frame-index 0 \
  --device cuda \
  --reference-top-k 5 \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000
```

也可以使用 Na12 默认分子编号规则：

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --system-xyz NaOH_7.5M/src/lqve/data/processed/frame.xyz \
  --mol-id 148 \
  --reference-library NaOH_7.5M/src/lqve/references/reference_library.json \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --feature-data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --frame-index 0 \
  --device cuda \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000
```

Na12 默认分子编号规则与 `src/descriptor` 保持一致：

- `0-147`：水分子，原子顺序 `O,H,H`。
- `148-159`：NaOH 单元，原子顺序 `Na,O,H`。

动态选择的输出：

- `query_feature.npy`：当前帧探针分子的 ViSNet 特征。
- `reference_match.npz`：query feature、候选 reference index 和距离。
- `reference_match.json`：top-k 候选、距离、query metadata。
- `selected_reference_config.json`：选中的 reference entry。
- `selected_reference.json`：embedding 输出目录中的同一选择结果。

后续 QC 会把 `selected_reference` 复制到 `qc_outputs/<frame>/metadata.json`。shift 阶段可以据此逐帧读取不同的 `dvr_data` 和 `reference_energies`。

## 7. 第三步：计算 DVR 构型能量

脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py --help
```

能量后端包括：

- `visnet`：本项目优先后端，加载 `NaOH_7.5M/visnet` checkpoint，对完整 box 的 DVR 构型批量预测能量。
- `xtb`：命令行 xTB 后端。
- `gaussian`：基于模板生成 Gaussian 输入并解析能量。
- `cp2k`：基于模板生成 CP2K 输入并解析能量。
- `mock`：确定性假能量，用于测试 I/O、并行和 resume。

ViSNet 推荐命令：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000 \
  --backend visnet \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --cell 16.63,16.63,44.10 \
  --device cuda \
  --batch-size 16 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/frame_000000_visnet \
  --resume
```

mock 测试命令：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000 \
  --backend mock \
  --batch-size 16 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/frame_000000_mock \
  --resume
```

xTB 示例：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000 \
  --backend xtb \
  --command "xtb --gfn 2" \
  --workers 4 \
  --batch-size 1 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/frame_000000_xtb \
  --keep-workdirs \
  --resume
```

关键参数：

- `--geometries`：embedding 输出目录、`embedded_geometries.npz`，或包含多个 frame 子目录的父目录。
- `--backend`：选择能量后端。
- `--batch-size`：每个 chunk 的 DVR 几何数量。ViSNet 可设大一些，外部 QC 通常设为 1。
- `--workers`：外部后端并行 worker 数。ViSNet 后端内部使用 GPU batch，不走多进程。
- `--resume`：已有 `qc_energies.npz` 的 frame 会跳过。
- `--limit-frames`：只处理前若干个 frame，用于测试。
- `--limit-geometries`：每个 frame 只处理前若干个 DVR grid，用于测试。
- `--dry-run`：只写 manifest，不运行能量计算。
- `--keep-workdirs`：保留外部 QC 临时目录；默认成功任务可清理，失败任务保留。

输出目录结构：

```text
qc_outputs/<run_name>/
  manifest.jsonl
  <frame_id>/
    qc_energies.npz
    metadata.json
```

`qc_energies.npz` 包含：

- `energies_hartree`：每个 DVR 构型的能量，单位 Hartree。
- `grid_points`：对应的 DVR grid 坐标。
- `success_mask`：每个 DVR 点是否成功。
- `source_indices`：原始 grid index，用于恢复顺序。

## 8. 第四步：计算 LQVE shift

脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py --help
```

功能：

- 读取 DVR 参考能级和波函数。
- 读取每帧 DVR grid 构型能量。
- 读取参考 DVR grid 能量。
- 构造扰动势：

```text
V_per(grid) = E_qc(grid) - E_ref(grid)
```

默认会减去扰动势均值，以消除环境整体能量平移对频率的影响。

- 构造有效 Hamiltonian：

```text
H_ij = E0_i delta_ij + sum_g wf_gi V_per_g wf_gj
```

- 对角化有效 Hamiltonian，得到瞬时能级。
- 输出相对基态的 transition 和相对参考 transition 的 shift。

显式参考命令：

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --dvr-data NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7 \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/frame_000000_visnet \
  --reference-energies NaOH_7.5M/src/lqve/references/pes/reference_grid_energies_hartree.npy \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name frame_000000_visnet \
  --n-contract 30 \
  --n-transitions 6
```

动态参考命令：

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name dynamic_visnet \
  --n-contract 30 \
  --n-transitions 6
```

动态参考模式下可以不传 `--dvr-data` 和 `--reference-energies`，因为它们会从每帧 `selected_reference` metadata 中读取。前提是 `reference_library.json` 的每个 entry 已经包含 `dvr_data` 和 `reference_energies`。

关键参数：

- `--dvr-data`：DVR 结果目录。显式参考模式需要。
- `--qc-outputs`：第三步输出目录。
- `--reference-energies`：参考 DVR grid 能量，单位 Hartree。
- `--reference-key`：当参考能量是 `.npz` 或带表头 CSV 时，指定 key 或列名。
- `--reference-mode first-qc`：只用于 smoke/debug，把第一个 QC frame 当参考能量，不建议正式使用。
- `--n-contract`：参与有效 Hamiltonian 的参考态数量。
- `--n-transitions`：输出的跃迁数量。
- `--keep-mean`：不减去扰动均值。

输出目录结构：

```text
data/shifts/<run_name>/
  metadata.json
  manifest.jsonl
  shifts_summary.csv
  shifts_summary.npz
  <frame_id>/
    shift_result.npz
    metadata.json
```

`shift_result.npz` 包含：

- `levels_hartree`：该 frame 的有效 Hamiltonian 能级。
- `transitions_cm1`：相对基态的跃迁频率，单位 `cm^-1`。
- `shifts_cm1`：相对参考 transition 的频移，单位 `cm^-1`。
- `success_mask`：DVR grid 点成功标记。
- `grid_points`：DVR grid 点。

`shifts_summary.csv` 至少包含：

- `frame_id`
- `success`
- `source_qc_dir`
- `reference_id`
- `dvr_data`
- `reference_energies`
- `transition_1_cm1`, `transition_2_cm1`, ...
- `shift_1_cm1`, `shift_2_cm1`, ...

## 9. 第五步：分析和可视化

脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py --help
```

推荐命令：

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --shifts NaOH_7.5M/src/lqve/data/shifts/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/results \
  --run-name dynamic_visnet \
  --plot-format png
```

只分析指定 transition：

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --shifts NaOH_7.5M/src/lqve/data/shifts/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/results \
  --run-name dynamic_visnet_t1_t3 \
  --transitions 1,3 \
  --plot-format png
```

输出目录结构：

```text
results/<run_name>/
  report.md
  analysis_metadata.json
  tables/
    transition_stats.csv
    shift_stats.csv
    frame_quality.csv
  figures/
    transitions_timeseries.png
    shifts_timeseries.png
    transition_histograms.png
    shift_histograms.png
    transition_correlation.png
    frame_quality.png
  logs/
    analysis_config.json
```

报告中会列出：

- 输入 shift 目录。
- frame 数量、成功数量、成功率。
- transition/shift 的 NaN 比例。
- 所选 transition。
- reference 使用统计。
- 所有表格和图片路径。
- 失败或不完整 frame 列表。

## 10. 配置文件方式

五个脚本都支持 JSON 配置。配置文件适合生产运行，因为命令短、参数可记录。

DVR：

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --config NaOH_7.5M/src/lqve/configs/dvr_example.json
```

Embedding：

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --config NaOH_7.5M/src/lqve/configs/embedding_example.json
```

ViSNet QC：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --config NaOH_7.5M/src/lqve/configs/qc_example_visnet.json
```

Shift：

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --config NaOH_7.5M/src/lqve/configs/shift_example.json
```

Analysis：

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --config NaOH_7.5M/src/lqve/configs/analysis_example.json
```

CLI 参数会覆盖配置文件中的部分字段。实际生产中建议为每个 run 保存一份独立配置，不直接改示例文件。

## 11. 总控 workflow

如果已经准备好一组参考 PES、参考振动模、参考分子构型和一条或多条轨迹，可以使用总控 workflow 一次完成：

```text
reference DVR preparation
  -> per-frame embedding
  -> per-frame energy calculation
  -> per-frame shift
  -> all_frequencies.csv
```

analysis 不包含在总控流程中。总控只负责计算并汇总频率；如果需要图片、统计表和报告，仍然在总控结束后单独运行 `05_analyze_results.py`。

总控脚本：

```bash
python NaOH_7.5M/src/lqve/scripts/run_lqve_workflow.py --help
```

推荐命令：

```bash
python NaOH_7.5M/src/lqve/scripts/run_lqve_workflow.py \
  --config NaOH_7.5M/src/lqve/configs/workflow_example.json
```

smoke 测试时限制 frame 数：

```bash
python NaOH_7.5M/src/lqve/scripts/run_lqve_workflow.py \
  --config NaOH_7.5M/src/lqve/configs/workflow_example.json \
  --run-name na12_lqve_smoke \
  --limit-frames 1
```

强制重算该 workflow 管理的 run 目录：

```bash
python NaOH_7.5M/src/lqve/scripts/run_lqve_workflow.py \
  --config NaOH_7.5M/src/lqve/configs/workflow_example.json \
  --overwrite
```

总控配置示例：

```text
NaOH_7.5M/src/lqve/configs/workflow_example.json
```

核心字段：

- `run_name`：本次 workflow 的名字，会用于组织输出目录。
- `lqve_root`：LQVE 根目录，默认 `NaOH_7.5M/src/lqve`。
- `dynamic_reference`：是否逐帧用 ViSNet 特征选择最近参考。
- `references`：参考列表。每个 reference 包含 `reference_id`、`pes`、`reference_xyz`、`modes` 和 `dvr` 参数；动态参考选择时还必须包含 `feature_file`。
- `trajectories`：轨迹列表。每条轨迹包含 `traj_id`、`path`、`frame_start`、`frame_stop`、`frame_stride`、`probe_indices` 或 `mol_id`。
- `reference_selection`：动态参考选择参数，包括 ViSNet checkpoint、device 和 top-k。
- `embedding`：DVR 点嵌入参数。
- `energy`：能量后端参数，支持 `visnet|mock|xtb|gaussian|cp2k`。
- `shift`：有效 Hamiltonian 的 `n_contract`、`n_transitions` 和是否减去扰动均值。

总控会为每个 reference 检查或生成：

```text
NaOH_7.5M/src/lqve/data/dvr/<reference_id>/dvr_result.npz
NaOH_7.5M/src/lqve/references/pes/<reference_id>_reference_energies_hartree.npy
```

其中 `reference_energies_hartree.npy` 是在该 reference 的 DVR grid 上重新评估参考 PES 得到的能量，单位 Hartree，供 shift 阶段使用。正式计算不使用 `reference-mode first-qc`。

总控运行时会自动生成 reference library：

```text
NaOH_7.5M/src/lqve/data/workflows/<run_name>/reference_library.json
```

逐帧中间结果默认保存在：

```text
NaOH_7.5M/src/lqve/data/embedded_geometries/<run_name>/<traj_id>_frame_XXXXXX/
NaOH_7.5M/src/lqve/data/qc_outputs/<run_name>/<traj_id>_frame_XXXXXX/
NaOH_7.5M/src/lqve/data/shifts/<run_name>/<traj_id>_frame_XXXXXX/
```

最终总频率文件：

```text
NaOH_7.5M/src/lqve/data/shifts/<run_name>/all_frequencies.csv
```

`all_frequencies.csv` 字段包括：

- `traj_id`
- `frame_index`
- `frame_id`
- `success`
- `source_qc_dir`
- `reference_id`
- `dvr_data`
- `reference_energies`
- `transition_1_cm1`, `transition_2_cm1`, ...
- `shift_1_cm1`, `shift_2_cm1`, ...

总控也会保存 frame registry：

```text
NaOH_7.5M/src/lqve/data/workflows/<run_name>/frame_registry.json
```

### 11.1 Python 包调用

总控 workflow 也可以作为 Python 包调用：

```python
from lqve.workflow import WorkflowConfig, run_lqve_workflow

config = WorkflowConfig.from_json("NaOH_7.5M/src/lqve/configs/workflow_example.json")
result = run_lqve_workflow(config)
print(result.all_frequencies_csv)
```

如果希望在其他项目脚本中直接 `import lqve`，可以安装为 editable 包：

```bash
python -m pip install -e NaOH_7.5M/src/lqve
```

安装后可直接：

```python
from lqve import WorkflowConfig, run_lqve_workflow
```

### 11.2 总控配置最小模板

下面是总控配置的结构示意。实际路径需要替换为已经存在的参考 PES、模式、参考结构、轨迹和 ViSNet checkpoint。

```json
{
  "run_name": "na12_lqve_dynamic_visnet",
  "lqve_root": "NaOH_7.5M/src/lqve",
  "dynamic_reference": true,
  "resume": true,
  "overwrite": false,
  "keep_intermediates": true,
  "limit_frames": null,
  "references": [
    {
      "reference_id": "ref_0001",
      "pes": "NaOH_7.5M/src/lqve/references/pes/ref_0001_pes.dat",
      "reference_xyz": "NaOH_7.5M/src/lqve/references/structures/ref_0001.xyz",
      "modes": "NaOH_7.5M/src/lqve/references/modes/ref_0001_modes.npy",
      "feature_file": "NaOH_7.5M/src/descriptor/represent_mol/rank_0001_mol_001_traj_a_frame_00024524/feature.npy",
      "grid_unit": "bohr",
      "mode_unit": "angstrom_per_bohr",
      "dvr": {
        "states": 80,
        "coord_unit": "angstrom",
        "energy_unit": "cm-1",
        "dense_threshold": 512,
        "solver_tol": 1.0e-10,
        "modes": [
          {"name": "q1", "lower": -0.7, "upper": 0.7, "sinc_points": 200, "podvr_points": 7},
          {"name": "q2", "lower": -0.9, "upper": 0.9, "sinc_points": 200, "podvr_points": 10},
          {"name": "q3", "lower": -0.6, "upper": 0.6, "sinc_points": 200, "podvr_points": 7}
        ]
      }
    }
  ],
  "trajectories": [
    {
      "traj_id": "na12_a",
      "path": "NaOH_7.5M/src/lqve/data/processed/na12_a.xyz",
      "probe_indices": [444, 445, 446],
      "mol_id": 148,
      "feature_data": "NaOH_7.5M/data/visnet/na12_ab.pkl",
      "frame_start": 0,
      "frame_stop": null,
      "frame_stride": 1,
      "cell": [16.63, 16.63, 44.10]
    }
  ],
  "reference_selection": {
    "enabled": true,
    "checkpoint": "NaOH_7.5M/visnet/runs/visnet_2/best.pt",
    "device": "cuda",
    "top_k": 5
  },
  "embedding": {
    "grid_unit": "bohr",
    "mode_unit": "angstrom_per_bohr",
    "centroid_weight": "mass",
    "write_xyz": false
  },
  "energy": {
    "backend": "visnet",
    "checkpoint": "NaOH_7.5M/visnet/runs/visnet_2/best.pt",
    "device": "cuda",
    "cell": [16.63, 16.63, 44.10],
    "batch_size": 16,
    "workers": 1,
    "limit_geometries": null
  },
  "shift": {
    "n_contract": 30,
    "n_transitions": 6,
    "subtract_mean": true
  }
}
```

### 11.3 固定参考模式

如果不需要逐帧动态选择参考，可以设置：

```json
{
  "dynamic_reference": false,
  "reference_selection": {"enabled": false}
}
```

此时每条 trajectory 必须显式给出 `probe_indices`。如果配置中有多个 reference，还需要给每条 trajectory 指定 `reference_id`：

```json
{
  "traj_id": "na12_a",
  "path": "NaOH_7.5M/src/lqve/data/processed/na12_a.xyz",
  "probe_indices": [444, 445, 446],
  "reference_id": "ref_0001",
  "frame_start": 0,
  "frame_stride": 1,
  "cell": [16.63, 16.63, 44.10]
}
```

### 11.4 总控与五步脚本的关系

总控 workflow 适合生产批量运行。五个分步脚本仍然保留，适合调试单个步骤、检查中间文件或替换某个后端。

推荐使用方式：

- 正式批量计算：优先使用 `run_lqve_workflow.py`。
- 调试 DVR、embedding、QC 或 shift 某一步：使用 `01-05` 分步脚本。
- 生成图和统计报告：总控结束后单独运行 `05_analyze_results.py`。

## 12. 逐帧并行策略

LQVE 第 2-4 步天然以 frame 为单位独立。

推荐生产组织方式：

1. 先准备所有参考 DVR 和 `reference_library.json`。
2. 将 AIMD 轨迹拆成 frame，或用多帧 XYZ 的 `--frame-index` 按帧读取。
3. 对每个 frame 单独运行 embedding。
4. 对每个 frame 或 frame 目录集合运行 QC/ViSNet energy。
5. 对所有 QC 输出统一运行 shift。
6. 对 shift summary 做 analysis。

ViSNet 后端建议：

- 通过 `--batch-size` 控制 GPU 显存。
- 如果显存不足，降低 `--batch-size`。
- 如果 GPU 利用率低但显存占用高，优先确认是否使用了预计算邻居表和合理 batch size。
- `--limit-geometries` 可用于先测单 frame 的前几个 DVR 点。

外部 QC 后端建议：

- `--workers` 控制并行进程数。
- `--batch-size 1` 通常更稳，因为每个 DVR 点是一套独立外部计算。
- `--resume` 用于断点续跑。
- `--keep-workdirs` 适合调试；生产中不一定需要保留所有成功 job 的临时文件。

## 13. 最小 smoke 流程

最小 smoke 目标是验证 I/O 和数据契约，不追求物理结果。

1. 构建一个小 DVR 或使用已有 DVR 目录。

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --config NaOH_7.5M/src/lqve/configs/dvr_example.json
```

2. 对单 frame 生成 embedding。

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --config NaOH_7.5M/src/lqve/configs/embedding_example.json
```

3. 用 mock 后端测试 QC 输出。

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/one_cl_9115 \
  --backend mock \
  --batch-size 8 \
  --limit-geometries 8 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/one_cl_9115_mock \
  --resume
```

4. 用 `first-qc` 做 shift smoke。

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --dvr-data NaOH_7.5M/src/lqve/data/dvr/naoh_3d_7_10_7 \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/one_cl_9115_mock \
  --reference-mode first-qc \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name one_cl_9115_mock_smoke \
  --n-contract 10 \
  --n-transitions 3
```

5. 生成 analysis 报告。

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --shifts NaOH_7.5M/src/lqve/data/shifts/one_cl_9115_mock_smoke \
  --output-dir NaOH_7.5M/src/lqve/results \
  --run-name one_cl_9115_mock_smoke
```

`--reference-mode first-qc` 只用于 smoke/debug。正式结果必须提供真实 `reference_energies`。

## 14. 正式生产流程模板

### 13.1 准备多个参考

对每个参考构型分别运行 DVR：

```bash
python NaOH_7.5M/src/lqve/scripts/01_build_reference_dvr.py \
  --pes <reference_pes.dat> \
  --dims 3 \
  --mode-names q1,q2,q3 \
  --ranges <ranges> \
  --sinc-points 200,200,200 \
  --podvr-points 7,10,7 \
  --states 80 \
  --coord-unit angstrom \
  --energy-unit cm-1 \
  --output-dir NaOH_7.5M/src/lqve/data/dvr/<reference_id>
```

为每个参考准备：

- `feature.npy`
- `reference_xyz`
- `modes`
- `dvr_data`
- `reference_energies`

然后写入：

```text
NaOH_7.5M/src/lqve/references/reference_library.json
```

### 13.2 对每个 frame 做动态参考 embedding

```bash
python NaOH_7.5M/src/lqve/scripts/02_build_embedded_geometries.py \
  --system-xyz NaOH_7.5M/src/lqve/data/processed/frame_000000.xyz \
  --probe-indices 444,445,446 \
  --reference-library NaOH_7.5M/src/lqve/references/reference_library.json \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --feature-data NaOH_7.5M/data/visnet/na12_ab.pkl \
  --frame-index 0 \
  --device cuda \
  --reference-top-k 5 \
  --output-dir NaOH_7.5M/src/lqve/data/embedded_geometries/frame_000000
```

对多个 frame 可以在 shell、SLURM 或 Python 调度器中并行提交。每个 frame 的输出目录应唯一。

### 13.3 用 ViSNet 批量计算能量

如果 `data/embedded_geometries/dynamic_run/` 下有多个 frame 子目录：

```bash
python NaOH_7.5M/src/lqve/scripts/03_run_qc_energies.py \
  --geometries NaOH_7.5M/src/lqve/data/embedded_geometries/dynamic_run \
  --backend visnet \
  --checkpoint NaOH_7.5M/visnet/runs/visnet_2/best.pt \
  --cell 16.63,16.63,44.10 \
  --device cuda \
  --batch-size 16 \
  --output-dir NaOH_7.5M/src/lqve/data/qc_outputs/dynamic_visnet \
  --resume
```

### 13.4 根据每帧 selected reference 计算 shift

```bash
python NaOH_7.5M/src/lqve/scripts/04_compute_lqve_shift.py \
  --qc-outputs NaOH_7.5M/src/lqve/data/qc_outputs/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/data/shifts \
  --run-name dynamic_visnet \
  --n-contract 30 \
  --n-transitions 6
```

### 13.5 汇总分析

```bash
python NaOH_7.5M/src/lqve/scripts/05_analyze_results.py \
  --shifts NaOH_7.5M/src/lqve/data/shifts/dynamic_visnet \
  --output-dir NaOH_7.5M/src/lqve/results \
  --run-name dynamic_visnet \
  --plot-format png
```

重点查看：

- `results/dynamic_visnet/report.md`
- `tables/frame_quality.csv`
- `tables/transition_stats.csv`
- `tables/shift_stats.csv`
- `figures/shifts_timeseries.png`
- 报告中的 reference usage 统计。

## 15. 常见问题和排查

### reference assets 缺失

报错通常类似：

```text
Selected reference is missing LQVE assets [...]
```

原因是 `reference_library.json` 中选中的 entry 缺少 `reference_xyz`、`modes`、`dvr_data` 或 `reference_energies`。补齐这些字段后重跑 embedding。

### feature 维度不匹配

如果 query feature 和 reference feature 维度不同，通常说明：

- 使用了不同的 ViSNet checkpoint。
- 使用了不同的 feature 类型。
- 探针原子数不同。

需要保证参考库和实时 query 都使用 `visnet_scalar_x_before_energy_head`，并且分子原子数和原子顺序一致。

### cell 缺失

从 `--system-xyz` 直接提取 ViSNet 特征时需要 `--cell`。如果使用 `--feature-data NaOH_7.5M/data/visnet/na12_ab.pkl`，cell 会从 pkl 中读取。

### probe species 顺序不一致

Embedding 会检查 `reference_xyz` 中的元素顺序是否与 `probe_indices` 指向的元素顺序一致。例如参考是 `Na,O,H`，当前 probe 也必须是 `Na,O,H`。如果顺序不一致，需要修正 `probe_indices` 或参考结构。

### reference energy 长度不匹配

shift 阶段要求：

```text
len(reference_energies) == prod(dvr_result.basis_shape)
```

如果长度不一致，说明参考能量不是同一个 DVR grid 上的能量，或者 grid 点顺序和数量被改过。

### QC grid 点顺序不一致

`qc_energies.npz` 中如果有 `source_indices`，shift 会按它恢复顺序。如果没有 `source_indices`，则 `grid_points` 必须与 DVR grid 顺序一致，且单位可识别为 Bohr 或 Angstrom。

### ViSNet GPU 显存过高

降低：

```bash
--batch-size
```

也可以先用：

```bash
--limit-geometries 4
```

确认单 frame 能跑通，再扩大到完整 DVR grid。

### 外部 QC 失败

检查：

- frame 目录下的 `metadata.json`
- root `manifest.jsonl`
- 后端 workdir 中的输入和输出文件

调试时建议加：

```bash
--keep-workdirs
```

生产时可以关闭该选项，减少小文件数量。

## 16. 与其他文档的关系

ViSNet-eIP 网络训练见：

```text
NaOH_7.5M/docs/visnet_eip_training.md
```

分子特征提取、代表结构选择和 descriptor 分析见：

```text
NaOH_7.5M/docs/visnet_descriptor_selection.md
```

本文档只说明 LQVE 模块如何使用这些特征和参考库，不重复 ViSNet 训练细节。
