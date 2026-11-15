# MicroDuck MJCF → URDF 转换与验证说明

## 模型选择

固定 MotrixLab `1ddb9743c220c3a1e234ea0237d5ce9cc56d0499` 的 MicroDuck。源 XML 和 40 个网格不作几何修改。保留 MotrixLab 为其任务加入的 `trunk_collision` 胶囊与 `sole_left_collision.stl` / `sole_right_collision.stl` 脚底凸代理；不混用其他分支的高精度脚底作为碰撞体。

`tools/convert_mjcf_to_urdf.py` 只处理该模型具有的一组特征：单自由根、每个非根 body 一个零锚点 hinge、弧度关节范围、显式 `fullinertia`、mesh / capsule 几何。遇到不同结构会报错，不能视为通用 MJCF 转换器。

## 保留与表达方式

| 项目 | 本次处理 |
|---|---|
| 刚体与关节 | 15 个同名 link，14 个同名 revolute joint；父子拓扑与执行器次序一致 |
| 坐标变换 | MJCF body 的局部 pos / quat 转成 URDF joint origin 的 xyz / rpy；四元数归一化，单位不变 |
| 质量 / 重心 / 惯量 | 使用源数据；完整 3×3 惯量在 link 坐标中保留，不从网格体积反算、不编造参数 |
| 限位 | MJCF `range` 原样传递到 URDF `lower` / `upper` |
| 网格 | 保留原始真实 STL、尺度和逐 geom 的位姿；同一 link 的多个视觉/碰撞体分别保留 |
| 颜色 | 原 `material rgba` 转为 URDF material/color |
| 躯干 capsule | URDF 没有胶囊 primitive，用一段 cylinder 加两端 sphere 表示；几何并集相同，接触行为可能不同 |
| 关节阻尼与摩擦 | 将 chosen_actuator 的 joint damping / frictionloss 记入 URDF dynamics；不同引擎解释仍需检查 |
| effort | 来自 chosen_actuator 的 `forcerange = [-0.96, 0.96]`，并非新增硬件额定值 |
| velocity | 源没有速度限制；使用命令行显式传入的 20 rad/s 填充必需字段，单独在 joint_map 标注格式用途 |

## URDF 不能完整表达的内容

- 根 `trunk_base_freejoint` 在 MJCF 中自由运动；URDF 的无父根 link 不自动创建自由关节。导入 MuJoCo 做对比时是固定根，验证将两者根坐标对齐。课程原生 MJCF 训练保留自由根。
- URDF 未添加 position 执行器、kp/kv、armature、控制范围或力饱和控制逻辑。`joint_map.yaml` 记录源执行器参数；真正训练仍使用官方 MJCF / MotrixLab 控制器。
- `self_collision_only` 的 `contype=2, conaffinity=2` 等接触位掩码、接触组、排除规则没有标准 URDF 等价物。URDF 保留相应 collision 几何，导入其他环境后必须重新配置碰撞规则。
- IMU / framequat / gyro / velocimeter 等传感器、`left_foot` / `right_foot` sites 与相机等 MJCF 专有节点未伪造为 URDF 设备；原生 XML 完整保留。
- 胶囊拆成三个碰撞 primitive 可能改变接触点数量与接触求解，不能据此宣称两个文件动力学等价。
- 此导出没有引入第二仿真器，也没有更改“学生面前最终呈现 MotrixSim”的课程决定。

## 实测验证

完整机器结果：`tools/validation_report.json`。验证运行使用隔离 Python 环境中的 MuJoCo **3.3.7**；不是 MotrixSim 实机验证。

| 检查 | 结果 |
|---|---|
| LFS 真文件 | 40 / 40，与固定源 SHA-256、字节数匹配，二进制 STL 面数与长度匹配，顶点均有限 |
| 资源闭包 | 每个 URDF mesh 引用均可在本目录找到 |
| 拓扑 | 15 links / 14 hinges，唯一根 trunk_base，无重复 child 或断开的关节树 |
| 质量 / 惯量 | 全部质量正值、惯量正定并满足主惯量三角关系；源与导出逐 link 一致 |
| 编译后的惯量 | 原生 MJCF 与 URDF 导入后的逐 link mass、COM、旋转回 link 坐标的完整 tensor 一致（容差 1e-12） |
| 加载 | 原生 MJCF：自由根 + 14 hinge、14 actuator；URDF：14 hinge、0 actuator，均加载成功 |
| 正运动学 | 零位、STAND、10 组固定随机种子的范围内姿态，共 12 组；全部 15 link 比对原生 MJCF、URDF 导入、独立 URDF FK |
| 最大位置误差 | 约 1.11 × 10⁻¹⁶ m |
| 最大旋转矩阵元素误差 | 约 1.36 × 10⁻¹⁵ |

MuJoCo 3.3.7 在 Windows 直接打开中文绝对路径时失败，验证程序改由 Python 读取原始文件字节并通过 MuJoCo VFS 提供相同资源。此情况记录在机器报告中，不修改原始文件与网格。正式镜像宜使用 `/opt/microduck-course` 等固定 Linux 路径。

## 尚待联调的内容

以上验收范围仅为模型资产与静态运动学。仍需使用固定版本的 MotrixLab / MotrixSim 完成场景加载、碰撞/触地观测、任务复位、checkpoint 对齐和行走训练/回放；不能拿本报告替代训练收敛和行走录像验收。

如选择供应商改版小鸭子，必须先获取其明确的模型/网格授权、关节表、关节零位与方向、限位、质量/惯量、足底碰撞几何、执行器参数及版本号，然后另立模型目录进行一致性验收。不得只替换外壳网格后宣称策略可直接通用。
