# 可重复的离线验收

默认执行：

```bash
PYTHONPATH=src python -m unittest discover -s checks -p 'test_*.py' -v
```

PowerShell：

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONIOENCODING = 'utf-8'
python -m unittest discover -s checks -p 'test_*.py' -v
```

大部分检查只需标准库和 PyYAML。真实 task dataclass 集成检查还需要 upstream core 的轻依赖（gymnasium、OmegaConf、Numba、SciPy）；没有它们会明确跳过。本仓库不复制 `checks/.venv`，请在课前准备好的 Python 环境中运行。2026 年 11 月 15 日，发布目录使用已准备的 MotrixSim 环境完成 22 项检查，全部通过；`lesson.cmd check --brief` 和使用中文约束 YAML 的训练预览也通过。

覆盖配置边界、学员目录隔离、dry-run 不写输出、缺失权重阻塞、异常 run 不标成功、starter 身份在训练前落盘且失败保留、更新门防止只采样冒充学习、导出 checkpoint 身份绑定、实际官方任务 reward/固定评估命令注入，以及官方 typed FastSAC 配置。原任务 reference 到课程评估 context 的测试使用实际官方 context 类型和注册 factory，并 mock 训练器调用；它不是 SDK 回放验收。没有 Torch/MotrixSim SDK/GPU 的本机不运行训练，不生成假的 checkpoint 或回放结果。

`image/acceptance_plan.md` 另列真实云实例、行走、图形和 50 人容量门槛。
