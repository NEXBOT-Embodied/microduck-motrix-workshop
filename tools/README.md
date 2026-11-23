# 课堂入口

在工程根目录运行。Windows 使用 `.\tools\lesson.cmd`，macOS 和 Linux 使用 `sh tools/lesson.sh`，完整步骤见根目录 README。

- `local_tutorial.py`：检查固定案例、运行回放、读取 YAML、训练和选择新结果。
- `evaluate_walking.py`：对指定策略做行走评估。
- `record_training_stages.py`：录制多个阶段的结果。
- `style_training_media.py`：为已有仿真素材添加课程说明。

启动器先使用 `COURSE_PYTHON`，否则查找工程内两处预备环境，不自动安装软件。训练输出写到本工程 `outputs/`，不进入 Git。
