# 课程配置与命令

本工程是一层课程适配代码，复用 `upstream/MotrixLab` 固定提交 `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499` 的 `microduck-walk-flat` 与内置 FastSAC。课程注册名为 `microduck-course-flat`，最终显示及策略回放显式选择 `motrixsim`。没有另写简化 SAC，也没有已完成的第二引擎迁移。

## 哪个文件由谁改

| 文件 | 使用者 | 内容 |
|---|---|---|
| `experiment.yaml` | 学员 | 只开放 `action_rate_multiplier` 一个实验因素 |
| `classroom.yaml` | 教师 | 环境数、续训步数、种子、线程、日志、设备、评估命令与输出位置 |
| `policies.yaml` | 教师 | 真实 reference / starter 的路径、SHA256、人工行走验收状态 |
| `policy_contract.yaml` | 教师 | 源码静态推导的观测/动作顺序，当前仍未做 SDK 运行验收 |

学员推荐比较 `0.5 / 1.0 / 2.0`。倍率作用于真实任务的 `rewards.penalty_action_rate.weight`，官方任务基准为 `-0.5`。代码在任务 factory 中用 dataclass replacement 构造新配置；每次运行都保存实际课程配置，回放旧 run 不读取后来修改的倍率。

课程配置使用同步 FastSAC，保留官方网络结构；小规模环境数和更新量只是镜像试配起点。它们没有通过云平台容量或行走效果验收，不代表 `32 env × 50 人` 可同时运行。

## 环境前提

工程根目录执行命令。镜像准备阶段安装好上游运行环境后，用该虚拟环境的 Python 安装课程包：`python -m pip install -e .`，或用 `uv pip install --python <上游虚拟环境Python> -e .`。镜像发布前完成，学生现场不安装。

若只检查离线文件，可设置 `PYTHONPATH=src` 并使用已有 PyYAML 的 Python。完整训练还需要固定版上游、真实模型网格、Torch、MotrixSim SDK、插件包及可用算力。`--dry-run` 不替代这些前提。

## 稳定命令

以下命令均支持 `--student s01`、`--config configs/experiment.yaml`、`--classroom configs/classroom.yaml`。学员代号只用匿名字母数字，不填姓名或联系方式。

```bash
python -m microduck_course config --student s01
python -m microduck_course check --student s01
python -m microduck_course check --student s01 --runtime
python -m microduck_course baseline --student s01 --dry-run
python -m microduck_course train --student s01 --dry-run
python -m microduck_course baseline --student s01
python -m microduck_course train --student s01
python -m microduck_course play --student s01 --run <run_id>
python -m microduck_course play --student s01 --run <run_id> --record --seconds 20
python -m microduck_course export --student s01 --run <run_id>
python -m microduck_course collect --student s01 --run <run_id>
```

`baseline/train/play/export` 支持 `--dry-run`。它只显示计划；训练 dry-run 不创建 run，不生成权重，也不标记训练成功。`play/export` 的 dry-run 仍要求对应 run 已真实完成且 checkpoint 文件匹配已记录哈希。`check` 发现前置条件缺失返回退出码 `2`；这是预期阻塞信息。

`check --runtime` 实际构建 1 个 MotrixSim 环境并 reset，核对观测维度、动作维度与 reward 注入。通过仅证明 reset/契约这一层，不证明窗口可见、鼠标交互、走路质量或 50 人容量。

`baseline/play` 默认打开 MotrixSim 原生交互窗口；云端必须有平台远程桌面/图形转发。`--record` 使用真正的 MotrixSim 离屏视频，不要求学生看到本地桌面窗口，但云端离屏图形后端仍需课前验收。浏览器访问 RoboGo 本身不等于它能转发 MotrixSim 窗口。

`baseline` 仅播放教师 reference 作为示范。受控实验的基线应从同一 starter、同一训练 seed 和预算开始，以倍率 `1.0` 真正续训一次，再与 `0.5` 或 `2.0` 运行比较。核对各次 manifest 的 `start.checkpoint_sha256`、训练配置和更新证据，不能直接把熟练参考策略与短训结果的差异归因于奖励倍率。

## 固定评估预设

`classroom.evaluation` 试配命令为 `[0.15, 0, 0]`，依次是 `vx(m/s)、vy(m/s)、wz(rad/s)`，评估 seed 为 `42`。它仅在 `baseline/play` 的评估配置快照中激活：实际 `WalkCommandCfg.vel_limit` 的上下界都设为此向量，`stand_prob=0`。训练仍使用官方随机命令分布。reference 来自官方原任务时，也通过官方推理 context 使用同一个课程评估 factory，不修改源运行的元数据或任务文件。

`0.15 m/s` 是指令值，不是实测速度。本节所述课堂 `baseline/play` 报告不自动计算速度跟踪误差、摔倒率或动作平滑度；环境噪声和物理随机化仍依官方任务设置。比较时核对两次保存的评估预设与 seed 一致，并由教师观看真实回放。每次回放在独立的 `evaluations/<evaluation_id>/` 下保存设置与报告，失败也记录原因。教师专项轨迹实测另见[直线行走实验](../outputs/straight_walking_experiment/README.md)。

## 教师直线行走实验配置

教师实验保留了走偏示例，并记录了新增转向成本及不同扣分系数的试验。当前课堂起步模型使用 `weight=-20`、`sigma=0.015`，由教程入口固定选择，不能随意换成某次训练的最后一个文件。具体来源和哈希保存在 `policies/source_runs/`。

直行候选开发测试通过 3/3、独立复测通过 2/3，严格合计 5/6，其中一次最大偏角略超门槛，仍未登记为课堂 ready。指标见 [历史验证结果](../../04_演示与验证素材/05_历史验证记录/final_candidate.json)；参数和归因限制见 [参数与算法说明](../../02_运行与实践手册/02_参数与算法说明.md)。

## 真实权重登记

`policies.yaml` 目前两项均为 `missing`，没有用空文件模拟权重。教师在已构建镜像中先运行：

```bash
python -m microduck_course train --student teacher --from-scratch
```

从零训练的时长和效果没有保证。要进行长时间教师训练，课前修改 `classroom.yaml` 的 `additional_iterations` 等教师参数，实际运行并验收。正式课堂必须准备可走参考策略和可续训起步状态。

验收策略后，在 `policies.yaml` 中填写 `status: ready`、`run_dir`、`checkpoint`、`sha256`、`verified_walking: true`。其中 `run_dir` 指向官方运行目录，包含 `metadata.json`、`task_config.yaml`、`checkpoints/`；`checkpoint` 相对该目录，SHA256 由真实文件计算。仅接受 `microduck-walk-flat` 或 `microduck-course-flat`、`motrix.fastsac`、MotrixSim 的 metadata-backed run。starter 还要求完整优化器和观测归一化状态。

算法按“官方 base → 官方 MicroDuck task recipe → starter 实际 algo（若续训）→ 课堂资源参数”合并；task 中的 `alpha_init` 和 `target_entropy_ratio` 等覆盖也会保留。`config` 显示从零训练将用的合并算法与静态预期 reward；续训的网络架构由 starter 元数据决定。官方 `num_learning_iterations` 是绝对目标；课程入口读取真实 `global_step`，在其上加 `additional_iterations`，避免把一个旧 checkpoint 原样保存却误称完成了新训练。

## 输出与验收口径

```text
outputs/s01/<run_id>/
  manifest.json                 # 状态、输出 checkpoint 哈希、完整起点身份与源 metadata
  course_config.json            # 本次教师与学员配置快照
  task_effective.json           # 真实任务 factory 的 reward、dt、关节顺序和模型
  motrix_runs/.../
    metadata.json
    task_config.yaml
    checkpoints/...pt
    ...                        # 官方日志/TensorBoard
  evaluations/<evaluation_id>/
    evaluation_settings.json   # 固定命令、seed、评估模式的实际快照
    evaluation_report.json     # 实际 factory 命令边界、局限及回放状态
    walking.mp4                # 可选真实录制
  export/policy.onnx            # 可选官方导出
  export/policy.report.json     # 数值 parity 结果
submissions/s01-<run_id>.zip
```

训练开始前即在 manifest 保存 starter 角色、运行目录、checkpoint 路径和 SHA256、原始 metadata 及 metadata/task_config 的 SHA256；训练失败仍保留。只有真实非空 checkpoint 可读取、global_step 完成新增步数、update_idx 实际增加并跨过至少一次 actor 更新门，才标记 `completed`。只采样不学习会失败，不能以 global_step 增加冒充策略更新。SDK/训练异常标记 `failed` 并保留原因。成功 manifest 另记 `initial/final_update_idx`、新增 critic 更新和 actor 更新门次数。完成训练不等于已经走稳；`walking_verified` 保持 `false`，由教师观看真实 MotrixSim 回放后单独记录。

`course_config.json` 保存用户配置层；最终 typed 算法配置见官方 `motrix_runs/.../task_config.yaml`；真实任务 factory 配置摘要见 `task_effective.json`。三者共同提供本次运行的可追溯信息。`task_effective.json` 写入时尚未创建仿真，文件存在本身不能证明模拟或训练完成。

ONNX 导出前后都核对官方 best policy 与本次 manifest 指定的 checkpoint 路径及 SHA256 一致，避免回放和导出选中不同策略；报告记录源 checkpoint 哈希，并保留官方数值 parity 验证。导出成功不代表 Sim2Sim 成功；第二引擎和迁移路线尚未确定。本阶段课程验收是“真实续训 → 学员独立 checkpoint → MotrixSim 回放 → 观察与提交”。
