# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preview a previously exported preview_episode.npz sequence."""

import json
import os

import numpy as np
import warp as wp

import newton.examples


REPO_ROOT = os.path.dirname(os.path.dirname(newton.examples.get_source_directory()))
_CLOTH_DATA_CANDIDATES = (
    os.path.join(REPO_ROOT, "newton", "cloth_data"),
    os.path.join(REPO_ROOT, "cloth_data"),
)
CLOTH_DATA_ROOT = next((path for path in _CLOTH_DATA_CANDIDATES if os.path.isdir(path)), _CLOTH_DATA_CANDIDATES[1])
DEFAULT_PREVIEW_PATH = os.path.join(CLOTH_DATA_ROOT, "sequence", "preview_episode.npz")


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index = 0

        preview = np.load(args.preview_episode, allow_pickle=True)
        try:
            self.global_segment_index = np.asarray(preview["global_segment_index"], dtype=np.int32)
            self.global_local_frame = np.asarray(preview["global_local_frame"], dtype=np.int32)
            self.segment_names = [str(x) for x in np.asarray(preview["segment_name"])]
            self.metadata = json.loads(str(preview["metadata_json"]))

            self.segment_entries = []
            for segment_index, segment_name in enumerate(self.segment_names):
                object_points = np.asarray(preview[f"segment_{segment_index}_object_points"], dtype=np.float32)
                faces = np.asarray(preview[f"segment_{segment_index}_faces"], dtype=np.int32)
                self.segment_entries.append(
                    {
                        "name": segment_name,
                        "object_points": object_points,
                        "face_indices_wp": wp.array(faces.reshape(-1), dtype=wp.int32, device=self.device),
                    }
                )
        finally:
            preview.close()

        self.total_frame_count = int(len(self.global_segment_index))

        bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        for entry in self.segment_entries:
            object_points = entry["object_points"]
            bounds_min = np.minimum(bounds_min, object_points.reshape(-1, 3).min(axis=0))
            bounds_max = np.maximum(bounds_max, object_points.reshape(-1, 3).max(axis=0))

        center = 0.5 * (bounds_min + bounds_max)
        extent = float(np.max(bounds_max - bounds_min))
        camera_distance = max(0.35, extent * 1.8)
        camera_height = max(0.12, extent * 1.0)
        self.viewer.set_camera(
            wp.vec3(center[0], center[1] - camera_distance, center[2] + camera_height),
            -20.0,
            90.0,
        )

    def _wrapped_frame(self) -> int:
        if self.args.loop and self.total_frame_count > 0:
            return self.frame_index % self.total_frame_count
        return min(self.frame_index, max(self.total_frame_count - 1, 0))

    def step(self):
        if not self.args.loop and self.frame_index >= self.total_frame_count and self.args.auto_exit:
            raise SystemExit(0)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        frame_id = self._wrapped_frame()
        active_segment = int(self.global_segment_index[frame_id])
        local_frame = int(self.global_local_frame[frame_id])

        for segment_index, entry in enumerate(self.segment_entries):
            mesh_name = f"/preview_episode_mesh_{segment_index}"
            if segment_index == active_segment:
                object_points = entry["object_points"][local_frame]
                self.viewer.log_mesh(
                    mesh_name,
                    wp.array(object_points, dtype=wp.vec3, device=self.device),
                    entry["face_indices_wp"],
                    hidden=False,
                    backface_culling=False,
                )
            else:
                self.viewer.log_mesh(
                    mesh_name,
                    wp.array(entry["object_points"][0], dtype=wp.vec3, device=self.device),
                    entry["face_indices_wp"],
                    hidden=True,
                    backface_culling=False,
                )
        self.viewer.end_frame()

    def test_final(self):
        assert self.total_frame_count > 0

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--preview-episode", type=str, default=DEFAULT_PREVIEW_PATH, help="Path to preview_episode.npz.")
        parser.add_argument("--loop", action="store_true", help="Loop the preview episode.")
        parser.add_argument("--auto-exit", action="store_true", help="Exit automatically after the last frame when not looping.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
