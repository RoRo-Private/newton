# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay PhysTwin cloth motion from final_data.pkl without physics simulation."""

from __future__ import annotations

import os
import pickle

import numpy as np
import warp as wp

import newton.examples


REPO_ROOT = os.path.dirname(os.path.dirname(newton.examples.get_source_directory()))
_CLOTH_DATA_CANDIDATES = (
    os.path.join(REPO_ROOT, "newton", "cloth_data"),
    os.path.join(REPO_ROOT, "cloth_data"),
)
CLOTH_DATA_ROOT = next((path for path in _CLOTH_DATA_CANDIDATES if os.path.isdir(path)), _CLOTH_DATA_CANDIDATES[1])
DEFAULT_DATASET_PATH = os.path.join(CLOTH_DATA_ROOT, "cloth_1_2", "final_data.pkl")


class Example:
    """Replay-only viewer for cloth_data final_data.pkl trajectories."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_cursor = 0.0

        with open(args.dataset, "rb") as f:
            data = pickle.load(f)

        self.object_points_np = np.asarray(data["object_points"], dtype=np.float32)
        self.object_vis_np = np.asarray(data.get("object_visibilities", np.ones(self.object_points_np.shape[:2], dtype=bool)))
        self.object_colors_np = np.asarray(data.get("object_colors", np.zeros_like(self.object_points_np)), dtype=np.float32)

        self.controller_points_np = np.asarray(data.get("controller_points", np.zeros((self.object_points_np.shape[0], 0, 3))), dtype=np.float32)
        self.controller_vis_np = np.asarray(
            data.get("controller_visibilities", np.ones(self.controller_points_np.shape[:2], dtype=bool)),
            dtype=bool,
        )

        self.frame_count = int(self.object_points_np.shape[0])
        self.point_count = int(self.object_points_np.shape[1])
        self.controller_count = int(self.controller_points_np.shape[1])

        self.object_points_wp = wp.array(self.object_points_np[0], dtype=wp.vec3, device=self.device)
        self.object_colors_wp = wp.zeros(self.point_count, dtype=wp.vec3, device=self.device)
        self.object_radii_wp = wp.full(self.point_count, args.object_radius, dtype=wp.float32, device=self.device)

        self.controller_points_wp = wp.array(
            self.controller_points_np[0] if self.controller_count > 0 else np.zeros((0, 3), dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        self.controller_colors_wp = wp.full(
            self.controller_count, wp.vec3(0.1, 0.6, 1.0), dtype=wp.vec3, device=self.device
        )
        self.controller_radii_wp = wp.full(
            self.controller_count, args.controller_radius, dtype=wp.float32, device=self.device
        )

        bounds_min = self.object_points_np[0].min(axis=0)
        bounds_max = self.object_points_np[0].max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        extent = float(np.max(bounds_max - bounds_min))
        camera_distance = max(0.35, extent * 1.6)
        camera_height = max(0.12, extent * 0.9)
        self.viewer.set_camera(
            wp.vec3(center[0], center[1] - camera_distance, center[2] + camera_height),
            -20.0,
            90.0,
        )

    def _wrapped_frame(self) -> int:
        frame_index = int(self.frame_cursor)
        if self.args.loop and self.frame_count > 0:
            return (frame_index * self.args.frame_stride) % self.frame_count
        idx = frame_index * self.args.frame_stride
        return min(idx, max(self.frame_count - 1, 0))

    def _update_frame_buffers(self, frame_id: int) -> None:
        self.object_points_wp.assign(wp.array(self.object_points_np[frame_id], dtype=wp.vec3, device=self.device))

        colors = np.zeros((self.point_count, 3), dtype=np.float32)
        visible = self.object_vis_np[frame_id]
        source_colors = self.object_colors_np[frame_id]
        if source_colors.shape[0] == self.point_count:
            colors[visible] = source_colors[visible]
        else:
            colors[visible] = np.array([0.95, 0.35, 0.20], dtype=np.float32)
        colors[~visible] = np.array([0.18, 0.18, 0.18], dtype=np.float32)
        self.object_colors_wp.assign(wp.array(colors, dtype=wp.vec3, device=self.device))

        if self.controller_count > 0:
            controller_points = self.controller_points_np[frame_id].copy()
            controller_vis = self.controller_vis_np[frame_id]
            if not np.all(controller_vis):
                controller_points[~controller_vis] = np.array([1.0e6, 1.0e6, 1.0e6], dtype=np.float32)
            self.controller_points_wp.assign(wp.array(controller_points, dtype=wp.vec3, device=self.device))

    def step(self):
        if not self.args.loop and int(self.frame_cursor) * self.args.frame_stride >= self.frame_count and self.args.auto_exit:
            raise SystemExit(0)

        frame_id = self._wrapped_frame()
        self._update_frame_buffers(frame_id)

        self.frame_cursor += max(self.args.playback_speed, 1.0e-3)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/cloth/object_points",
            self.object_points_wp,
            radii=self.object_radii_wp,
            colors=self.object_colors_wp,
        )
        if self.controller_count > 0 and self.args.show_controllers:
            self.viewer.log_points(
                "/cloth/controller_points",
                self.controller_points_wp,
                radii=self.controller_radii_wp,
                colors=self.controller_colors_wp,
            )
        self.viewer.end_frame()

    def test_final(self):
        points = self.object_points_wp.numpy()
        assert np.isfinite(points).all()
        assert points.shape[0] == self.point_count

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to final_data.pkl.")
        parser.add_argument("--frame-stride", type=int, default=1, help="Stride through recorded frames.")
        parser.add_argument(
            "--playback-speed",
            type=float,
            default=0.35,
            help="Playback speed multiplier. <1.0 is slower, >1.0 is faster.",
        )
        parser.add_argument("--loop", action="store_true", help="Loop playback.")
        parser.add_argument("--auto-exit", action="store_true", help="Exit when playback reaches the end in non-loop mode.")
        parser.add_argument("--object-radius", type=float, default=0.004, help="Rendered radius for cloth points [m].")
        parser.add_argument("--controller-radius", type=float, default=0.008, help="Rendered radius for controller points [m].")
        parser.add_argument("--show-controllers", action="store_true", help="Render controller points from the dataset.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
