# ViSNet-eIP 在线主动选点

本文档对应以下统一入口：

```text
配置文件: NaOH_7.5M/config/visnet_active_learning.toml
启动脚本: NaOH_7.5M/scripts/run_visnet_active_learning.sh
底层程序: NaOH_7.5M/visnet/run_active_learning.py
```

## 1. 目标

这套流程用于在 `ASE` 中用当前 `ViSNet-eIP` 势做固定盒 `NVT 300 K` 分子动力学，同时用单模型 `eIP epistemic` 做 uncertainty-driven dynamics (UDD) 偏置采样。每一轮完成：

1. 用当前 checkpoint 做无偏 warm-up
2. 统计 `u_ref` 和 `u_scale`
3. 切换到带 UDD bias 的动力学
4. 在线检测高不确定帧并构成候选池
5. 用 FPS 选出最不相同的代表帧
6. 对代表帧做 `CP2K energy + forces` 标注
7. 将新标注帧追加到训练集并 warm-start 重训练
8. 从本轮结束构型继续下一轮

第一版不接入 `AI2BMD` 主流程，也不做 committee，不做 `NPT`，不训练 stress/virial。

## 2. 运行入口

推荐命令：

```bash
bash NaOH_7.5M/scripts/run_visnet_active_learning.sh \
  --config NaOH_7.5M/config/visnet_active_learning.toml
```

这个 shell 脚本只做两件事：

- 对关键输入做 fail-fast 检查
- 调用底层 `run_active_learning.py --config ...`

它**不会**隐式替你训练初始模型。

## 3. 启动前必须存在的文件

脚本启动前会检查：

- `model.checkpoint`
- `retrain.base_data`
- `md.initial_xyz`
- `cp2k.template_source`

因此这条链不能从零开始直接跑。必须先完成初始监督训练，也就是先执行：

```bash
bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml
```

并确认至少已经生成：

```text
NaOH_7.5M/visnet/runs/naoh12_visnet/best.pt
NaOH_7.5M/data/visnet/naoh12.pkl
```

## 4. 配置文件

配置文件位置：

[`NaOH_7.5M/config/visnet_active_learning.toml`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/config/visnet_active_learning.toml)

主要配置块：

- `[model]`
  - ViSNet-eIP checkpoint、推理设备
- `[md]`
  - 初始结构、盒子、Langevin 参数、warm-up 步数、每轮步数
- `[udd]`
  - `lambda_bias_ev`、参考分位数、触发阈值、连续命中次数
- `[selection]`
  - 每轮代表结构个数、冷却步数、最小特征距离
- `[cp2k]`
  - `mock|direct|slurm`、模板、命令、MPI/SLURM 参数
- `[retrain]`
  - 基础数据集、重训练 epoch、batch size、lr、early stopping、是否预计算边
- `[output]`
  - 运行目录、resume/overwrite、是否保留中间结果

当前默认模板已经按服务器直连 CP2K 写好：

```toml
[cp2k]
execution = "direct"
command = "mpirun -np 8 cp2k.psmp"
nproc = 1
```

这里 `nproc = 1` 是故意的，因为 `command` 中已经显式包含了 `mpirun -np 8`。

## 5. 输出目录

运行目录：

```text
NaOH_7.5M/data/active_learning/<run_name>/
```

每一轮目录：

```text
round_0001/
round_0002/
...
```

每轮主要文件：

- `trajectory.traj`
- `md_log.csv`
- `candidate_pool.pkl/.csv`
- `selected_frames.pkl/.csv`
- `labeled_frames.pkl/.json`
- `cp2k_jobs/`
- `augmented_training.pkl`
- `retrain/`
- `restart_state/restart_state.traj`
- `round_manifest.json`

总控状态文件：

- `workflow_state.json`
- `summary.json`

## 6. CP2K 标注

CP2K 模板默认复用：

[`NaOH_7.5M/data/NVE/NaOH12/NaOH12.inp`](/Users/sii-haoyutang/Code/PycharmProjects/SFG_NaOH/NaOH_7.5M/data/NVE/NaOH12/NaOH12.inp)

工作流会自动改写：

- `PROJECT`
- `RUN_TYPE -> ENERGY_FORCE`
- `&CELL`
- `&COORD`

并从 CP2K 输出中解析：

- 总能量 `Hartree`
- 原子力 `Hartree/Bohr`

之后自动换算到训练数据使用的 `Hartree/Angstrom`。

## 7. smoke 与正式运行

如果要先做一轮 smoke，建议直接在 TOML 中临时下调：

- `md.warmup_steps`
- `md.steps_per_round`
- `selection.batch_query_size`
- `retrain.epochs`
- `retrain.limit_batches`
- `retrain.limit_val_batches`

如果要切到完全不调用 CP2K 的逻辑检查，也可以把：

```toml
[cp2k]
execution = "mock"
```

正式运行时，保持：

```toml
[cp2k]
execution = "direct"
command = "mpirun -np 8 cp2k.psmp"
nproc = 1
```

如果你的服务器以后改成队列提交，再切换到 `slurm` 模式。

## 8. 当前边界

- 目前只实现 `Langevin`
- 当前只支持全周期 box 的 `ASE + ViSNet-eIP` 在线采样
- `output.keep_intermediates=false` 现在只清理 `cp2k_jobs/` 下的作业子目录，不会删除 round manifest、候选文件和重训练结果
- 这套流程默认假定你已经有一个可用的初始 ViSNet-eIP checkpoint
- 旧 checkpoint 已被移除，因此现在必须先基于新的 `NaOH12` 数据重新训练

## 9. 最短顺序

1. 初始训练：

```bash
bash NaOH_7.5M/scripts/train_visnet_initial.sh \
  --config NaOH_7.5M/config/visnet_initial_train.toml
```

2. 检查输出：

```text
NaOH_7.5M/visnet/runs/naoh12_visnet/best.pt
NaOH_7.5M/data/visnet/naoh12.pkl
```

3. 启动主动学习：

```bash
bash NaOH_7.5M/scripts/run_visnet_active_learning.sh \
  --config NaOH_7.5M/config/visnet_active_learning.toml
```
