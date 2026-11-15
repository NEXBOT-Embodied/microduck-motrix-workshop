"""Narrow compatibility fixes for the pinned, actually tested upstream/SDK pair."""

from __future__ import annotations

import inspect
import time

import numpy as np

APPLIED_ADAPTERS: set[str] = set()
RENDER_EVENTS: list[dict] = []
PRESENTATION = {"engine": "MotrixSim", "follow_base": True, "camera_distance_m": 0.65,
                "camera_height_offset_m": 0.05, "camera_elevation_degrees": -20.0,
                "camera_azimuth_degrees": 180.0, "warmup_wall_seconds": 1.0,
                "minimum_discarded_warmup_captures": 1, "physics_steps_during_warmup": 0,
                "first_frame_ready_timeout_seconds": 15.0,
                "first_frame_check": "visible_orange_MicroDuck_material_pixels_at_least_20"}


def install_render_compatibility(base_link_name: str = "trunk_base") -> None:
    """Keep official rendering while adapting the locked SDK's API differences.

    MotrixLab 1ddb9743 passes fps=None for headless RenderApp, but its locked
    SDK 0.10.2.dev132873 has no fps constructor parameter. VideoRecorder still
    controls the saved video's FPS and simulation pacing. Do not modify the
    vendored upstream or change the simulator/model/policy.
    """
    from motrix_env_motrixsim import renderer
    from motrixsim.render import RenderClosedError

    if not getattr(renderer, "_microduck_closed_window_compatibility", False):
        original_render = renderer.MotrixSimRenderer.render

        def course_render(self) -> bool | None:
            if self._headless:
                return original_render(self)
            try:
                return original_render(self)
            except RenderClosedError:
                # FastSAC play exits when env.render() returns False. Only a
                # closed interactive SDK window is a normal playback stop.
                return False

        renderer.MotrixSimRenderer.render = course_render
        renderer._microduck_closed_window_compatibility = True
        APPLIED_ADAPTERS.add("closed_native_MotrixSim_window_returns_False")

    if getattr(renderer, "_microduck_fps_compatibility", False):
        return
    original = renderer.RenderApp
    parameters = inspect.signature(original).parameters
    if "fps" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return

    def render_app_without_unsupported_fps(*args, **kwargs):
        kwargs.pop("fps", None)
        return original(*args, **kwargs)

    renderer.RenderApp = render_app_without_unsupported_fps
    renderer._microduck_fps_compatibility = True
    APPLIED_ADAPTERS.add("omit_RenderApp_fps_absent_from_locked_SDK")

    # The official docs' snapshot generator warms rendering and discards its
    # first capture. Do the same at first actual capture, after env.init_state.
    # Keep the physical state fixed and keep these frames out of VideoRecorder.
    original_init = renderer.MotrixSimRenderer.__init__
    original_capture = renderer.MotrixSimRenderer.capture

    def course_render_init(self, model, data_source, config, **kwargs):
        original_init(self, model, data_source, config, **kwargs)
        self._course_follow_link = model.get_link(base_link_name) if config.headless else None
        if config.headless and self._course_follow_link is None:
            raise ValueError(f"MotrixSim camera cannot find base link: {base_link_name}")
        self._course_render_warmed = False

    def course_capture(self):
        if self._course_follow_link is not None:
            xyz = np.asarray(self._course_follow_link.get_position(self._data_source())).reshape(-1, 3)[0].copy()
            xyz[2] += PRESENTATION["camera_height_offset_m"]
            self._render.system_camera.set_view(xyz.tolist(), PRESENTATION["camera_distance_m"],
                                                PRESENTATION["camera_elevation_degrees"], PRESENTATION["camera_azimuth_degrees"])
        if not self._course_render_warmed:
            started = time.monotonic()
            self.render()
            time.sleep(PRESENTATION["warmup_wall_seconds"])
            original_capture(self)
            discarded = 1
            # Asset uploads can outlast the official fixed 1s warmup on a busy
            # desktop. Hold this same physical state until the actual duck's
            # orange material is visible. Return that real frame unmodified.
            # A blank/sky-only initial frame must not count as frame zero.
            deadline = started + PRESENTATION["first_frame_ready_timeout_seconds"]
            while True:
                frame = original_capture(self)
                rgb = np.asarray(frame).astype(np.int16)
                orange = (rgb[..., 0] > 110) & (rgb[..., 0] > rgb[..., 1] + 12) & (rgb[..., 1] > rgb[..., 2] + 18)
                visible_pixels = int(orange.sum())
                if visible_pixels >= 20:
                    break
                discarded += 1
                if time.monotonic() >= deadline:
                    raise RuntimeError("MotrixSim assets not visibly ready within 15s; no blank first frame recorded")
                time.sleep(0.05)
            RENDER_EVENTS.append({"event": "first_actual_frame_ready", "discarded_real_captures": discarded,
                                  "visible_orange_material_pixels": visible_pixels,
                                  "warmup_wall_seconds": time.monotonic() - started, "physics_steps_during_wait": 0})
            self._course_render_warmed = True
            return frame
        return original_capture(self)

    renderer.MotrixSimRenderer.__init__ = course_render_init
    renderer.MotrixSimRenderer.capture = course_capture
    APPLIED_ADAPTERS.update(("official_snapshot_render_warmup_without_physics_steps", "headless_camera_follows_actual_base_xyz"))
