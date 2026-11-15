# FastSAC 固定源码导航

源码版本：MotrixLab `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499`。以下路径都以 `upstream/MotrixLab/` 为根，可在编辑器中按文件名搜索。不要用另一个版本的教程行号套用。

## 建议阅读顺序

| 顺序 | 真实路径 | 读完应能回答 |
|---|---|---|
| 1 | `motrix_envs/src/motrix_envs/locomotion/humanoid/microduck.py` | 我们训练哪个任务？哪些奖励被小鸭子覆盖？ |
| 2 | `motrix_envs/src/motrix_envs/locomotion/humanoid/cfg.py` | Actor / Critic 分别看见什么？控制周期是多少？ |
| 3 | `motrix_robots/src/motrix_robots/microduck.py` | 哪个模型、默认姿态、机器人控制参数？ |
| 4 | `motrix_env_core/src/motrix_env_core/mdp/action.py` | 动作如何变成关节目标？ |
| 5 | `configs/task/microduck-walk-flat/motrix.fastsac.yaml` | 哪些算法默认值被任务覆盖？ |
| 6 | `motrix_rl/src/motrix_rl/configs/algo_base/motrix.fastsac.yaml` | Actor、Critic、经验池、优化器参数在哪里？ |
| 7 | `motrix_rl_builtin/src/motrix_rl_builtin/fastsac/networks.py` | 隐藏层、分布、tanh、Q 网络的形状？ |
| 8 | `motrix_rl_builtin/src/motrix_rl_builtin/fastsac/agent.py` | 两种观测如何构建、归一化和更新？ |
| 9 | `motrix_rl_builtin/src/motrix_rl_builtin/fastsac/buffer.py` | Replay 保存什么？n-step 与截断怎样处理？ |
| 10 | `motrix_rl_builtin/src/motrix_rl_builtin/fastsac/sync/train.py` 及 `async_impl/` | 采样、更新、保存的真实调用顺序？ |

FastSAC 自带 `export.py`，不代表任何导出路径或第二引擎部署都已对 MicroDuck 跑通。课程导出入口应如实说明产物类型，导出成功和 Sim2Sim 验收是两件事。

## 五个定位练习

**找动作维度。** 从机器人执行器顺序和观测声明出发核对 14 个关节，包含头颈 4 轴。不能根据“两条腿各五轴”把网络输出写成 10。

**找动作的单位。** `JointPositionActionTerm` 使用 `target_raw * action_scales + default_angles`。本任务统一 `action_scale=0.5`，含义为每单位动作对应的弧度残差。最终安全边界还由动作空间和执行器控制范围处理。

**找奖励生效位置。** 在 `microduck.py` 找 `_make_microduck_rewards()`，然后到课程 `src/microduck_course/task.py` 查实验倍数怎样进入 `rewards.penalty_action_rate.weight`。只改 YAML 文本，若没有进入传给环境的配置，就不算实验。

**找非对称观测。** `cfg.py` 的 `WalkObservationsCfg.PolicyCfg` 和 `ValueCfg` 分开，`agent.py` 也分别建立归一化器和网络。`networks.py` 顶部的早期“共享观测”说明不能覆盖真实调用。

**找目标熵。** 实际公式为 `target_entropy = -act_dim * target_entropy_ratio`。本任务 ratio 为 -0.1，14 维动作下结果为 +1.4。不要依照别的 SAC 实现习惯擅自修改符号。

## 主讲只需展开三处代码

第一处展示 `cfg.py` 的观测分组，让同学指出“机身倾斜”和“期望向前”分别来自哪里。第二处展示 `microduck.py` 的奖励配置，说明行走目标不只包含速度。第三处展示课程 `task.py` 与最终配置，把实验卡上的倍数对应到真实训练环境。

时间允许再看 `networks.py`。不建议在三小时课中现场逐行推导异步共享内存、CUDA IPC 和混合精度实现；这些属于课后选读。

## 来源

- [固定任务源码](https://github.com/Motphys/MotrixLab/blob/1ddb9743c220c3a1e234ea0237d5ce9cc56d0499/motrix_envs/src/motrix_envs/locomotion/humanoid/microduck.py)
- [固定 FastSAC 实现](https://github.com/Motphys/MotrixLab/tree/1ddb9743c220c3a1e234ea0237d5ce9cc56d0499/motrix_rl_builtin/src/motrix_rl_builtin/fastsac)
- [Pollen MicroDuck](https://github.com/pollen-robotics/microduck)：机器人源项目；与 MotrixLab 本课封装分别标识版本。
