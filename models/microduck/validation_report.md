# MicroDuck 模型资产验收报告

验证日期：2026-11-08。结论：**模型资产、结构、惯量和静态正运动学验收通过**。本报告不构成行走策略、训练收敛或 MotrixSim 课程联调通过的证明。

## 验收对象与证据

- 原生模型：`mjcf/microduck.xml`，来自 MotrixLab 固定提交 `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499`。
- 导出模型：`urdf/microduck.urdf`；关节与姿态元数据为 `joint_map.yaml`。
- 模型网格：`mjcf/assets/`，40 个真实二进制 STL，共 **21,625,760 字节**。
- 可执行验证程序：[tools/validate_model.py](tools/validate_model.py)。
- 实际运行的完整机器结果：[tools/validation_report.json](tools/validation_report.json)，`status = PASS`。
- LFS 来源与校验：[tools/lfs_fetch_manifest.json](tools/lfs_fetch_manifest.json)。文件 SHA / 来源归档：[asset_manifest.json](asset_manifest.json)。
- 转换保留项与损失项的完整解释：[CONVERSION_REPORT.md](CONVERSION_REPORT.md)。

## 环境与实际命令

验证环境为 Windows / Python 3.12 的独立虚拟环境，加载引擎为 **MuJoCo 3.3.7**。它用于检查导出的 URDF 与原生 MJCF，不是最终授课展示引擎。学生面前的最终展示仍为 **MotrixSim**。

验证使用了以下工具。下面把当时的本机路径改为工程根目录下的相对路径，便于复现：

```powershell
python -m venv .\models\microduck\tools\.validation-venv

& './models/microduck/tools/.validation-venv/Scripts/python.exe' -m pip install 'mujoco==3.3.7' 'numpy>=2,<3' 'scipy>=1.14,<2' 'pyyaml>=6,<7'

& './models/microduck/tools/.validation-venv/Scripts/python.exe' './models/microduck/tools/convert_mjcf_to_urdf.py' --export-velocity-rad-s 20

& './models/microduck/tools/.validation-venv/Scripts/python.exe' './models/microduck/tools/validate_model.py'

& './models/microduck/tools/.validation-venv/Scripts/python.exe' './models/microduck/tools/build_manifest.py' --verify-remote
```

MuJoCo 的 Windows 原生文件读取器未能直接打开本项目的中文绝对路径。验证脚本因此使用 Python 读取同一份 XML / STL 字节，再通过 MuJoCo 的 VFS 传入。机器报告的 `checks.loading.resource_loader` 记录了此方式；验证没有替换或重建网格。教师 Linux 镜像应采用稳定路径，例如 `/opt/microduck-course`。

## 验收结果

| 项目 | 检查方法与结果 |
|---|---|
| 网格真文件 | 40 / 40 与固定版本 Git LFS 的 SHA-256 和声明字节数一致，非 LFS 指针文本；STL 面数与文件长度相符，顶点数值全部有限 |
| 资源闭包 | 每个 URDF mesh 的相对路径均能找到本目录中的真实资源 |
| 结构 | 15 links、14 hinges；唯一根 `trunk_base`；无重复 child，关节树连通 |
| 关节顺序 | 与原生 MJCF 的 14 个 position actuator 一致；顺序显式保存在 `joint_map.yaml` |
| 质量 | 15 个 link 的质量均为正，总质量 **0.73724318 kg**；这是模型参数合计，未进行硬件称重 |
| 惯量合法性 | 每个 link 的完整惯量矩阵均正定，主惯量满足三角关系 |
| 源文件到 URDF | 每个 link 的质量、局部重心、完整 3×3 惯量均与源数据一致；未从网格估算或填入虚构惯量 |
| 编译后的惯量对照 | MJCF 与 URDF 经 MuJoCo 分别加载后，逐 link 比较质量、局部重心和旋转回 link 坐标的完整惯量；本次三个误差指标的最大值均为 **0**，验收容差为 `1e-12` |
| 文件加载 | 原生 MJCF 与 URDF 均成功编译；结构差异符合导出预期，见下表 |
| 正运动学 FK | 零姿态、STAND、10 组固定随机种子 `20261206` 生成的限位内姿态，共 **12 组**；每组比较全部 **15 个 link** |
| FK 对照路径 | 原生 MJCF ↔ MuJoCo 导入 URDF；另以独立 URDF 变换链实现进行交叉检查；比较前显式对齐根坐标 |
| 最大位置误差 | **1.1102230246251565 × 10⁻¹⁶ m** |
| 最大旋转矩阵元素误差 | **1.3600232051658168 × 10⁻¹⁵**；位置、旋转矩阵及独立 FK 均低于 `1e-10` 验收容差 |

| 编译属性 | 原生 MJCF | 导出 URDF |
|---|---:|---:|
| body（包含 world） | 16 | 16 |
| qpos 维数 `nq` | 21 | 14 |
| 速度维数 `nv` | 20 | 14 |
| joint 数 | 15（1 free + 14 hinge） | 14（hinge） |
| actuator 数 | 14 | 0 |
| geom 数 | 76 | 78 |

URDF 增加的两个 geom 来自将原生躯干 capsule 拆成 cylinder 与两个 sphere。URDF 没有自动添加自由基座或执行器，因此 `nq`、`nv`、actuator 数不同是已知表达差异，不是遗漏未被发现。

## 限制与后续验收

1. **尚未验证动力学等价或稳定行走。** 本次没有运行训练、策略 rollout、奖励验收、接触传感器验收或 MotrixSim 回放；不能把 `PASS` 解释为课程环境已全部可用。
2. URDF 不承载原 MJCF 的自由根、执行器 kp/kv、armature、完整接触位掩码、IMU 和 foot site 等语义。课程训练和回放应继续使用固定版本的官方 MJCF / MotrixLab 任务。
3. capsule 的三 primitive 表达保持几何并集，但可能改变接触点数量和求解行为；URDF 的碰撞规则需要目标引擎单独配置。
4. 源模型没有速度上限。本次 `--export-velocity-rad-s 20` 仅填充 URDF 必需字段，**不是硬件额定值或安全限制**，且不参与原生课程训练。
5. 源 XML 和真实网格未经本次修改；网格继续受 **CC BY-NC-SA 4.0** 约束，具体归属与上游许可见 `licenses/`。模型整理与测试不会扩展其许可范围。
6. 供应商改版尚未选定。若换用改版，需重新核对模型授权、关节零位/方向/限位、质量惯量、脚底碰撞与执行器参数，并重跑本验收及训练回放验收。

临时验证虚拟环境 `.validation-venv/` 不属于发布资产，已从模型清单和版本跟踪中排除；镜像构建应按照依赖文件另行安装，现场无需安装。
