# MicroDuck × Motrix 行走强化学习工作坊

用 MotrixLab 训练 MicroDuck，在 MotrixSim 中观察行走。第一版提供两份教学模型、带中文注释的行走配置，以及修改参数、训练、回放和录制的统一入口。

## 先看它怎么走

在工程根目录打开 PowerShell。环境已在课前准备好的电脑，逐条执行：

```powershell
.\tools\lesson.cmd check --brief
.\tools\lesson.cmd play --case drift
.\tools\lesson.cmd play --case straight
```

播放完一个案例，关闭仿真窗口，再运行下一个。`drift` 是走偏示例；`straight` 是直行候选，也是课堂续训的起点。

## 改一个参数，再训练

```powershell
notepad .\configs\walking_constraints.yaml
```

找到 `task.rewards.penalty_heading_rate.weight`。先保留 `-20.0` 完成 A 组，再改为 `-25.0` 完成 B 组，其余参数保持相同。每组都逐条执行：

```powershell
.\tools\lesson.cmd train --steps 256 --constraints configs/walking_constraints.yaml --dry-run
.\tools\lesson.cmd train --steps 256 --constraints configs/walking_constraints.yaml --render
.\tools\lesson.cmd play-trained --latest
.\tools\lesson.cmd play-trained --latest --record --seconds 20
```

第一条核对参数，第二条训练，第三条回放，第四条保存视频。`256` 表示新增的并行采样循环数，用于先跑通流程。A 组完成后先保存结果，再改参数做 B 组；新结果在 `outputs/training_outputs/local_tutorial/`。

## 代码在哪里

| 想看什么 | 文件或目录 |
| --- | --- |
| 全部行走约束与中文注释 | `configs/walking_constraints.yaml` |
| YAML 怎样进入任务 | `src/microduck_course/constraints.py` |
| 教学任务与转向扣分 | `src/microduck_course/task.py`、`heading_rewards.py` |
| 训练和回放适配 | `src/microduck_course/runtime.py` |
| FastSAC 的 Actor、Critic 与经验池 | `upstream/MotrixLab/motrix_rl_builtin/src/motrix_rl_builtin/fastsac/` |
| MicroDuck 的 MJCF、URDF、网格与关节表 | `models/microduck/` |
| 两份课堂模型及对应配置 | `policies/source_runs/` |
| 契约检查 | `checks/` |

## 课前准备环境

需要 Windows、Git for Windows 和 NVIDIA CUDA 运行条件。仓库不包含 Python 虚拟环境。固定版上游提供安装入口，在工程根目录运行：

```powershell
Push-Location .\upstream\MotrixLab
.\install.ps1 --python 3.12 --gpu cuda
Pop-Location
uv pip install --python .\upstream\MotrixLab\.venv\Scripts\python.exe -e .
.\tools\lesson.cmd check --brief
```

安装只在课前进行。若已有可用环境，可把 `COURSE_PYTHON` 指向该环境的 Python；否则入口按本工程 `.venv`、`upstream/MotrixLab/.venv` 的顺序查找。依赖版本参考 `environment/requirements-observed.txt`，它是本次运行环境的版本记录。

## 第一版状态与来源

算法为 FastSAC，最终行走画面由 MotrixSim 展示。直行候选的已有严格测试为六次通过五次，课堂实验继续比较方向、迈步和稳定性。第二引擎 Sim2Sim 尚待实测；本课不做 Sim2Real。

MotrixLab 固定提交为 `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499`。上游代码、机器人模型各自的来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，发布快照说明见 [upstream/README.md](upstream/README.md)。本仓库未另行为用户自有代码指定开源许可证。
