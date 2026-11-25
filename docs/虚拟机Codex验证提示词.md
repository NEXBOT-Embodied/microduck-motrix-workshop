# 虚拟机 Codex 验证提示词

将下面整段发给虚拟机里的 Codex。先验证 Ubuntu x86_64 普通虚拟机；Windows x64 用同一提交另跑一次。同一台虚拟机的结果只覆盖其实际系统和硬件。

---

请在这台虚拟机上实际验证 MicroDuck × Motrix 课程工程的国内源安装和跨平台兼容性。直接执行并保存证据。

1. 记录系统及架构、虚拟机类型、内存、剩余磁盘、有无桌面会话、显卡和可用 CUDA。确认 Ubuntu x86_64 或 Windows x64。用新目录、空项目虚拟环境和独立 uv 缓存测试。已有 Python 3.12 和 uv 可作为基础工具，记录其版本和来源。勿为验证更改系统源、安装宿主机驱动或关闭 TLS 校验。
2. 从 `https://github.com/NEXBOT-Embodied/microduck-motrix-workshop.git` 干净检出 `install-cn-vm-check` 分支，并记录完整提交 SHA。Windows 克隆时加 `git -c core.longpaths=true clone`，选较短目录。下载仓库失败时可用同一提交的可信 Git bundle，记录实际来源。
3. 先读 README 的课前环境准备。在仓库根目录执行相应的课程安装脚本，保留完整安装日志和退出码。普通虚拟机用 `cpu`，只有虚拟机内能实际使用 NVIDIA CUDA 时才用 `cuda`。记录 uv、Python、通用 Python 包、PyTorch 与 MotrixSim 各自实际使用的来源，镜像不可达或缺包时保留原报错。若另外测试原厂源，将两次结果分开。不要更改版本锁定、屏蔽哈希校验或用别的包冒充缺失的版本。
4. 使用脚本创建的 Python。验证 `uv pip check`、Python 与 SDK 版本、`torch.cuda.is_available()`，然后运行课堂 `check --brief --allow-cpu` 和 `train --steps 256 --constraints configs/walking_constraints.yaml --dry-run`。检查两份教学权重的校验状态，并保存预览中实际读取的配置值。预览单列为配置校验。
5. 无图形设备时，运行 `tools/evaluate_walking.py` 做真实 CPU 物理步进。用 `tools/local_tutorial.py` 的 `CASES["straight"]` 找到原始 `run-dir`、`checkpoint` 和同一教师作业的 `course_config.json`，用 `--seconds 5 --seed 42 --output <新目录>`。保存 `report.json` 与 `steps.csv`，确认实际执行 250 个控制步。有图形环境时，另外运行课堂 `play --case straight` 看实际窗口，再运行 `play --case straight --record --seconds 5`，核对视频能解码且能看到机器人。离屏录像同样需要可用的图形后端。
6. 有 CUDA 且已验证上一步时，实际运行一次 `train --steps 256 --constraints configs/walking_constraints.yaml`。拿本次返回的新训练编号执行 `play-trained --run <编号> --record --seconds 5`。保存训练清单、权重哈希和回放证据。普通 CPU 虚拟机将课堂 CUDA 续训标为「未验证」，继续完成前面可跑的步骤。
7. 把命令、退出码、实际下载源、关键日志和结论写到 `outputs/vm-verification/REPORT.md`。为每项给出「通过 / 失败 / 未验证」及证据文件。区分网络或依赖、操作系统、虚拟机图形驱动、课程代码和 CUDA 硬件问题。发现问题时先记录首个可复现的失败；若用额外改动作诊断，注明改了什么，并保留原始结果。凭据和代理口令不得写进日志。
