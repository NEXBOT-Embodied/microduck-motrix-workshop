"""Opt-in native observer of actual synchronous training, never policy replay.

The narrow wrapper hooks call the original step once, then only render/read it.
No extra physics, observation generation, policy call or RNG sampling is added.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import re
import subprocess
import time

from microduck_course.config import CourseError
from microduck_course.storage import sha256, write_json


LABEL = "真实训练视图 · 含探索动作 · 非 deterministic 评估"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingView:
    """Own the temporary wrapper hooks and both native renderer lifetimes."""

    def __init__(self, options: dict, run_path: Path, initial_step: int, source_checkpoint: Path | None):
        self.options = options
        self.enabled = options["enabled"]
        self.path = run_path / "training_visualization"
        self.initial_step = initial_step
        self.wrapper = None
        self.offscreen = None
        self.writer = None
        self.frame_stream = None
        self.local_steps = 0
        self.window_frames = 0
        self.total_frames = 0
        self.clip = None
        self._original_init = self._original_step = None
        self.started = time.monotonic()
        self._last_step_wall = self.started
        self.report = {"enabled": self.enabled}
        self.render_config = None
        if self.enabled:
            from motrix_env_core.renderer import RenderConfig
            self.render_config = RenderConfig(headless=False, camera_lookat=(0, 0, .12),
                                              camera_distance=.65, camera_elevation=-20, camera_azimuth=180)
            self.report.update(schema=1, status="starting", label=LABEL, created_utc=utc_now(),
                source_checkpoint=str(source_checkpoint.resolve()) if source_checkpoint else None,
                source_checkpoint_sha256=sha256(source_checkpoint) if source_checkpoint else None,
                initial_global_step=initial_step, options=options, clips=[],
                video_source="native_headless_renderer_of_same_live_training_env0",
                window_source="native_windowed_renderer_of_actual_training_batch",
                window_capture_supported=False, offscreen_render_instances=1,
                observation_or_policy_calls_added=0, physics_steps_added=0, rng_samples_added=0,
                frame_sampling="after_original_wrapper_step_and_possible_auto_reset_before_learner_update",
                global_index_definition="initial_global_step + local_step - 1; collector iteration before learner/global counter increment",
                playback_executed=False, deterministic_evaluation=False, pacing_changes_wall_time=True)

    def __enter__(self):
        if not self.enabled:
            return self
        from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap
        if getattr(FastSacNpEnvWrap, "_course_training_view_active", False):
            raise CourseError("不能同时安装两个 training visualization observer。")
        self.path.mkdir(parents=True, exist_ok=False)
        self.frame_stream = (self.path / "frames.jsonl").open("w", encoding="utf-8")
        self._original_init = FastSacNpEnvWrap.__init__
        self._original_step = FastSacNpEnvWrap.step
        observer = self

        def observed_init(wrapper, *args, **kwargs):
            observer._original_init(wrapper, *args, **kwargs)
            if observer.wrapper is not None:
                raise CourseError("真实训练视图只允许一个同步 NumPy 训练环境包装器。")
            observer.wrapper = wrapper
            if wrapper._renderer is None or wrapper._renderer._headless:
                raise CourseError("真实训练视图需要原生可见窗口 renderer。")
            observer.report["training_batch_instances"] = wrapper.num_envs
            observer.report["window_created_utc"] = utc_now()
            print(f"[course live training] {LABEL}; native window opened, batch={wrapper.num_envs}", flush=True)

        def observed_step(wrapper, actions):
            result = observer._original_step(wrapper, actions)
            observer.after_step(wrapper, result)
            return result

        FastSacNpEnvWrap.__init__ = observed_init
        FastSacNpEnvWrap.step = observed_step
        FastSacNpEnvWrap._course_training_view_active = True
        write_json(self.path / "manifest.json", self.report)
        return self

    def _follow_window(self):
        import numpy as np
        raw = self.wrapper.env
        renderer = self.wrapper._renderer
        link = raw.sim._model.get_link(raw.cfg.scene.objs.robot.resolved_base_link_name)
        xyz = np.asarray(link.get_position(renderer._data_source())).reshape(-1, 3)[0].copy()
        xyz[2] += .05
        renderer._render.system_camera.set_view(xyz.tolist(), .65, -20., 180.)
        renderer._sync_render_data = True
        renderer.render()
        self.window_frames += 1

    def _create_offscreen(self):
        import numpy as np
        from motrix_env_core.renderer import RenderConfig
        raw = self.wrapper.env
        cfg = RenderConfig(headless=True, path=self.path / "unused-native-pacing.mp4",
            fps=self.options["record_fps"], num_frames=self.options["max_recorded_frames"],
            width=self.options["width"], height=self.options["height"],
            camera_distance=.65, camera_elevation=-20., camera_azimuth=180.)
        # Launch ONE rendering instance. Select a native read-only SceneData view
        # of row zero; no new simulation/environment or policy is constructed.
        self.offscreen = raw.sim.create_renderer(cfg, num_envs=1, render_spacing=0.,
                                                 system_camera=raw.cfg.scene.system_camera)
        source = self.wrapper._renderer._data_source
        mask = np.zeros(raw.num_envs, dtype=bool)
        mask[0] = True
        self.offscreen._data_source = lambda: source()[mask]
        self.report["offscreen_created_utc"] = utc_now()

    def _start_clip(self, stage: int):
        self.clip = {"stage": stage, "path": str(self.path / f"live-stage-{stage:04d}.mp4"),
                     "frames": 0, "first_local_step": self.local_steps,
                     "fps": self.options["record_fps"], "label": LABEL}
        self.report["clips"].append(self.clip)

    def _capture(self, step_result):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        if self.offscreen is None:
            self._create_offscreen()
        frame = self.offscreen.capture()
        height, width = frame.shape[:2]
        if (width, height) != (self.options["width"], self.options["height"]):
            raise CourseError("native live training capture size differs from requested size")
        # Original renderer pixels are unscaled below a separate annotation bar.
        canvas = Image.new("RGB", (width, height + 64), "#20242a")
        canvas.paste(Image.fromarray(frame), (0, 64))
        draw = ImageDraw.Draw(canvas)
        font_size = 20
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        while font_size > 8 and draw.textbbox((0, 0), LABEL, font=font)[2] > width - 24:
            font_size -= 1
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        draw.text((12, 5), LABEL, font=font, fill="#e6ebf1")
        global_index = self.initial_step + self.local_steps - 1
        draw.text((12, 33), f"本次采样循环 {self.local_steps} | 可视化环境 0",
                  font=font, fill="#a4afbd")
        if self.writer is None:
            self.writer = imageio.get_writer(self.clip["path"], fps=self.options["record_fps"],
                codec="libx264", pixelformat="yuv420p", macro_block_size=None)
        import numpy as np
        self.writer.append_data(np.asarray(canvas))
        if self.total_frames == 0:
            poster = self.path / "live-first-frame.png"
            canvas.save(poster)
            self.report["first_frame_poster"] = {"path": str(poster), "sha256": sha256(poster),
                                                  "source": "actual_native_training_capture_with_external_header"}
        row = {"frame_index_in_clip": self.clip["frames"], "clip_stage": self.clip["stage"],
            "collector_iteration_global_index": global_index, "local_step": self.local_steps,
            "completed_collector_steps": self.initial_step + self.local_steps,
            "nominal_control_elapsed_seconds": self.local_steps * self.wrapper.env.cfg.ctrl_dt,
            "capture_utc": utc_now(), "wall_elapsed_seconds": time.monotonic() - self.started,
            "env0_terminated": bool(step_result.terminated[0].item()),
            "env0_truncated": bool(step_result.truncated[0].item()),
            "state_may_already_be_auto_reset": bool((step_result.terminated[0] | step_result.truncated[0]).item())}
        self.frame_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.frame_stream.flush()
        self.clip["frames"] += 1
        self.clip["last_local_step"] = self.local_steps
        self.total_frames += 1

    def _finish_clip(self):
        if self.clip is None:
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.clip["frames"]:
            import imageio_ffmpeg
            path = Path(self.clip["path"])
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-xerror",
                "-i", str(path), "-map", "0:v:0", "-an", "-vsync", "0", "-f", "null", "-",
                "-progress", "pipe:1", "-nostats"]
            decoded = subprocess.run(command, capture_output=True, text=True, timeout=120)
            counts = re.findall(r"^frame=(\d+)\s*$", decoded.stdout, re.MULTILINE)
            count = int(counts[-1]) if counts else 0
            if decoded.returncode or decoded.stderr.strip() or "progress=end" not in decoded.stdout or count != self.clip["frames"]:
                raise CourseError("真实训练录像未通过完整解码/帧数检查。")
            reader = imageio_ffmpeg.read_frames(str(path))
            try:
                metadata = next(reader)
            finally:
                reader.close()
            if list(metadata["size"]) != [self.options["width"], self.options["height"] + 64] or not math.isclose(metadata["fps"], self.options["record_fps"], abs_tol=.01):
                raise CourseError("真实训练录像分辨率/FPS不符。")
            self.clip.update(status="verified_complete_video", decoded_to_end=True,
                actual_frame_count=count, size=list(metadata["size"]), sha256=sha256(path),
                encoded_duration_seconds=count / self.options["record_fps"])
        else:
            self.clip["status"] = "no_frames"
        self.clip = None

    def after_step(self, wrapper, result):
        if wrapper is not self.wrapper:
            raise CourseError("真实训练观察器收到未绑定的环境。")
        self.local_steps += 1
        window = wrapper._renderer
        if window._render.is_closed:
            self.report["window_closed_during_training"] = True
        elif self.local_steps == 1 or self.local_steps % self.options["render_every_steps"] == 0:
            self._follow_window()
        if self.local_steps == 1:
            deadline = time.monotonic() + self.options["startup_hold_seconds"]
            while not window._render.is_closed and time.monotonic() < deadline:
                self._follow_window()
                time.sleep(.03)
        period = self.options["record_every_iterations"]
        offset = (self.local_steps - 1) % period if period else self.local_steps - 1
        stage = (self.local_steps - 1) // period if period else 0
        recording = (not period or offset < self.options["clip_iterations"])
        recording = recording and self.options["record"] and self.total_frames < self.options["max_recorded_frames"]
        if recording:
            if self.clip is None:
                self._start_clip(stage)
            target = int((offset + 1) * wrapper.env.cfg.ctrl_dt * self.options["record_fps"] + 1e-9)
            if target > self.clip["frames"]:
                self._capture(result)
        if self.clip is not None and (not recording or (period and offset + 1 >= self.options["clip_iterations"])):
            self._finish_clip()
        if self.options["realtime"]:
            remaining = wrapper.env.cfg.ctrl_dt - (time.monotonic() - self._last_step_wall)
            if remaining > 0:
                time.sleep(remaining)
        self._last_step_wall = time.monotonic()

    def __exit__(self, exc_type, exc, traceback):
        if not self.enabled:
            return False
        from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap
        FastSacNpEnvWrap.__init__ = self._original_init
        FastSacNpEnvWrap.step = self._original_step
        FastSacNpEnvWrap._course_training_view_active = False
        cleanup_errors = []
        try:
            self._finish_clip()
        except Exception as failure:
            cleanup_errors.append(f"video: {type(failure).__name__}: {failure}")
        for name, resource in (("offscreen", self.offscreen), ("window", self.wrapper._renderer if self.wrapper else None)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as failure:
                    cleanup_errors.append(f"{name}: {type(failure).__name__}: {failure}")
        if self.frame_stream is not None:
            self.frame_stream.close()
        self.report.update(status="failed" if exc is not None or cleanup_errors else "completed",
            original_wrapper_steps_observed=self.local_steps, window_render_calls=self.window_frames,
            total_recorded_frames=self.total_frames, finished_utc=utc_now(),
            frame_index_path=str(self.path / "frames.jsonl"), cleanup_errors=cleanup_errors,
            hooks_restored=True,
            renderers_closed=not any(error.startswith(("offscreen:", "window:")) for error in cleanup_errors))
        self.report["frame_index_sha256"] = sha256(self.path / "frames.jsonl")
        if exc is not None:
            self.report["error"] = f"{exc_type.__name__}: {exc}"
        write_json(self.path / "manifest.json", self.report)
        if cleanup_errors and exc is None:
            raise CourseError("真实训练视图清理/录像校验失败: " + "; ".join(cleanup_errors))
        return False
