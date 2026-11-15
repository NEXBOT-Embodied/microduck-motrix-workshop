# MicroDuck 教学模型资产

本目录使用 MotrixLab 已适配的 **Pollen Robotics MicroDuck**，作为课程统一模型。供应商改版未定型，本包不把未确认的改版参数写入模型。

学生最终在 **MotrixSim** 中观察仿真。训练和回放优先沿用 `mjcf/microduck.xml` 及官方任务；`urdf/microduck.urdf` 用于结构讲解、关节检查和后续模型交换，不能替代原生任务配置。

## 内容

| 路径 | 用途 |
|---|---|
| `mjcf/microduck.xml` | MotrixLab 原版模型，保留传感器、位置执行器、脚底碰撞体与躯干终止胶囊 |
| `mjcf/assets/` | 40 个真实 STL；不是 Git LFS 占位文本，共 21,625,760 字节 |
| `urdf/microduck.urdf` | 从同一版 MJCF 确定性导出的 URDF，15 个 link / 14 个 revolute joint |
| `joint_map.yaml` | 策略关节次序、关节上下限、STAND 初始姿态和坐标单位 |
| `CONVERSION_REPORT.md` | 转换保留项、损失项与验收范围 |
| `asset_manifest.json` | 文件 SHA-256、来源、版本和许可 |
| `licenses/` | 上游代码与模型网格许可、归属说明 |
| `tools/validation_report.json` | 实际运行的资源、惯量、加载与正运动学验收结果 |

## 来源锁定

- MotrixLab：<https://github.com/Motphys/MotrixLab/tree/1ddb9743c220c3a1e234ea0237d5ce9cc56d0499>
- 模型源路径：`motrix_robots/src/motrix_robots/assets/microduck/`
- MotrixLab 的模型说明注明移植自 <https://github.com/pollen-robotics/microduck_rl>，并加入了躯干终止胶囊与简化脚底碰撞网格。本目录保留这两项适配。
- Pollen 模型许可参考版本：`bae39f5900b318f7e87fc485c30a31213cfa14eb`；具体归属见 `licenses/ATTRIBUTION.md`。

## 关节与策略接口

模型有 14 个驱动关节：左右腿各 5 个，颈部/头部共 4 个。**策略次序并不是“左腿、右腿、头部”**；它与源模型执行器次序一致：

```text
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

训练、导出、回放都应从 `joint_map.yaml` 读取或核对同一组名字，不能仅靠数组长度判断兼容。14 维动作不等于 14 维观测；实际观测和动作缩放由教学任务配置定义。

根节点 `trunk_base` 在原生 MJCF 中是自由基座。左右脚 link 为 `ankle_left` / `ankle_right`。全部长度使用米、角度使用弧度；URDF 与 MJCF 每个 link 的质量与完整惯量保持一致，总质量为 **0.73724318 kg**（模型值，不是硬件称重结果）。

## 重建和验证

以下操作应在教师镜像构建阶段完成。现场不安装依赖。

```bash
python -m venv .model-validation
# Linux/macOS: .model-validation/bin/python
# Windows: .model-validation/Scripts/python.exe
python -m pip install -r models/microduck/tools/requirements-validation.txt
python models/microduck/tools/fetch_assets.py
python models/microduck/tools/convert_mjcf_to_urdf.py --export-velocity-rad-s 20
python models/microduck/tools/validate_model.py
```

先激活上面创建的虚拟环境，再运行 `python`；或直接使用该虚拟环境的 Python 绝对路径。转换参数 `20` 仅填充 URDF 格式所要求的速度字段，**不是电机实测速度或硬件安全上限**，也不参与本课程原生 MJCF 训练。

如果下载源码后 `upstream/MotrixLab` 内仍保留 LFS 指针，使用以下脚本。默认只检查；`--apply` 仅替换与固定版本 SHA 和大小一致的指针，遇到其他修改立即拒绝：

```bash
python models/microduck/tools/sync_upstream_assets.py --upstream-root upstream/MotrixLab
python models/microduck/tools/sync_upstream_assets.py --upstream-root upstream/MotrixLab --apply
```

## 当前验收结论

40 个网格 SHA / 大小与原始 LFS 声明一致；URDF 资源闭包、树结构与正定惯量检查通过。原生 MJCF 和生成的 URDF 已用 MuJoCo 3.3.7 加载，并对 12 组姿态的全部 15 个 link 做正运动学比对通过。这证明本次资产整理与坐标转换可复核，**尚不证明训练收敛、行走稳定或已在 MotrixSim 中完成实操验收**。这些工作属于教师镜像的后续联调。

## 许可

源代码和 MJCF 为 Apache-2.0；STL 网格为 Pollen Robotics 的 **CC BY-NC-SA 4.0（署名、非商业、相同方式共享）**。MotrixLab 的两个简化脚底网格是其派生资产。付费或带推广性质的活动应由主办方确认适用的模型网格授权；需要时取得 Pollen Robotics 或实际供应商的相应许可。活动脚本、运行通过和本次整理都不会改变上游网格许可。
