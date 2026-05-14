# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot v2.0 dataset collector for the bimanual OMX cloth pick-and-place task.

Usage:
    uv run python newton/examples/cloth/collect_lerobot_dataset.py \\
        --output-dir ./lerobot_dataset \\
        --num-episodes 50 \\
        --record-fps 30

Dataset layout produced:
    <output-dir>/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── episodes.jsonl
    │   └── tasks.jsonl
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.top_episode_000000.mp4
            └── ...

State / Action schema (10-dim, float32):
    [j1_l, j2_l, j3_l, j4_l, grip_l, j1_r, j2_r, j3_r, j4_r, grip_r]

Camera note:
    SensorTiledCamera renders rigid-body shapes (robot + table).
    Cloth particles are NOT visible in this renderer; the T-shirt appears
    only as deformation inferred from robot motion.  To include cloth
    visuals, use ViewerGL screenshots instead.

Dependencies (not in Newton core – install separately):
    pip install pyarrow imageio imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.sensors import SensorTiledCamera

from .example_cloth_omx_bimanual import EPISODE_DURATION, STATE_DIM, Example

# ---------------------------------------------------------------------------
# Camera configuration (top-down view)
# ---------------------------------------------------------------------------
_CAM_WIDTH = 224
_CAM_HEIGHT = 224
_CAM_FOV_DEG = 60.0

# Transform: positioned above the workspace, looking straight down.
# With Newton's convention (camera local -Z = forward), an identity rotation
# already points the camera downward when placed above the scene.
# Adjust _CAM_POS if the framing needs to change.
_CAM_POS = wp.vec3f(0.28, 0.0, 1.0)  # 1 m above table centre
_CAM_ROT = wp.quatf(0.0, 0.0, 0.0, 1.0)  # identity → looking down (-Z)

# ---------------------------------------------------------------------------
# LeRobot v2 format helpers
# ---------------------------------------------------------------------------
_TASK_DESCRIPTION = "Pick up the T-shirt with both arms and move it to the target location."
_ROBOT_TYPE = "open_manipulator_x_bimanual"
_STATE_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_gripper",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_gripper",
]
_CHUNKS_SIZE = 1000  # max episodes per chunk


def _episode_chunk(episode_index: int) -> int:
    return episode_index // _CHUNKS_SIZE


def _episode_parquet_path(output_dir: Path, episode_index: int) -> Path:
    chunk = _episode_chunk(episode_index)
    return output_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _episode_video_path(output_dir: Path, episode_index: int) -> Path:
    chunk = _episode_chunk(episode_index)
    return (
        output_dir
        / "videos"
        / f"chunk-{chunk:03d}"
        / f"observation.images.top_episode_{episode_index:06d}.mp4"
    )


# ---------------------------------------------------------------------------
# Camera sensor wrapper
# ---------------------------------------------------------------------------

class _TopCamera:
    """Thin wrapper around SensorTiledCamera for a single top-down view."""

    def __init__(self, model):
        self._sensor = SensorTiledCamera(
            model=model,
            options=SensorTiledCamera.Options(
                default_light=True,
                default_light_shadows=False,
                colors_per_shape=True,
                checkerboard_texture=False,
                backface_culling=True,
            ),
        )
        self._rays = self._sensor.compute_pinhole_camera_rays(
            _CAM_WIDTH, _CAM_HEIGHT, math.radians(_CAM_FOV_DEG)
        )
        self._color_image = self._sensor.create_color_image_output(
            _CAM_WIDTH, _CAM_HEIGHT, camera_count=1
        )
        # Camera transform array: shape [world_count=1, camera_count=1]
        self._cam_xform = wp.array(
            [[wp.transformf(_CAM_POS, _CAM_ROT)]],
            dtype=wp.transformf,
        )

    def capture(self, state) -> np.ndarray:
        """Render and return an (H, W, 3) uint8 RGB array."""
        self._sensor.update(
            state,
            self._cam_xform,
            self._rays,
            color_image=self._color_image,
        )
        # color_image shape: [world=1, camera=1, H, W]  dtype uint32 RGBA
        rgba = self._color_image.numpy()[0, 0]  # (H, W) uint32
        # Unpack RGBA uint32 → uint8 channels
        r = (rgba & 0x000000FF).astype(np.uint8)
        g = ((rgba >> 8) & 0xFF).astype(np.uint8)
        b = ((rgba >> 16) & 0xFF).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Episode collector
# ---------------------------------------------------------------------------

class _EpisodeBuffer:
    """Accumulates one episode's worth of frames."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self.timestamps: list[float] = []

    def append(self, state: np.ndarray, action: np.ndarray, image: np.ndarray, t: float):
        self.states.append(state)
        self.actions.append(action)
        self.images.append(image)
        self.timestamps.append(t)

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------

class LeRobotDatasetCollector:
    def __init__(self, output_dir: str | Path, num_episodes: int, record_fps: int = 30):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            self._pa = pa
            self._pq = pq
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for dataset collection.\n"
                "Install with: pip install pyarrow"
            ) from e

        try:
            import imageio
            self._imageio = imageio
        except ImportError as e:
            raise ImportError(
                "imageio is required for video encoding.\n"
                "Install with: pip install imageio imageio-ffmpeg"
            ) from e

        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.record_fps = record_fps
        # How many simulation frames to skip between recorded frames
        self._sim_fps = 60
        self._frame_skip = max(1, self._sim_fps // self.record_fps)

        # Create directory structure
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)

        self._global_frame_index = 0
        self._episode_stats: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def collect(self):
        """Collect all episodes and write the LeRobot dataset."""
        print(f"Collecting {self.num_episodes} episodes → {self.output_dir}")
        print(f"  record_fps={self.record_fps}, sim_fps={self._sim_fps}, "
              f"frame_skip={self._frame_skip}")

        null_viewer = newton.viewer.ViewerFile.__new__(newton.viewer.ViewerFile)
        # Use a minimal no-op viewer that satisfies the Example interface
        null_viewer = _NullViewer()

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []

        for ep_idx in range(self.num_episodes):
            t0 = time.time()
            print(f"  Episode {ep_idx:04d}/{self.num_episodes} ...", end="", flush=True)

            example = Example(null_viewer)
            camera = _TopCamera(example.model)

            buf = self._run_episode(example, camera)

            self._save_episode(buf, ep_idx)
            all_states.extend(buf.states)
            all_actions.extend(buf.actions)

            self._episode_stats.append(
                {
                    "episode_index": ep_idx,
                    "tasks": [_TASK_DESCRIPTION],
                    "length": len(buf),
                }
            )
            self._global_frame_index += len(buf)

            elapsed = time.time() - t0
            print(f" {len(buf)} frames  ({elapsed:.1f}s)")

        self._save_metadata(
            np.stack(all_states),
            np.stack(all_actions),
        )
        print(f"\nDataset saved to {self.output_dir}")

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def _run_episode(self, example: Example, camera: _TopCamera) -> _EpisodeBuffer:
        buf = _EpisodeBuffer()
        total_sim_frames = int(EPISODE_DURATION * self._sim_fps) + self._sim_fps  # +1 s margin
        episode_time = 0.0

        for frame_idx in range(total_sim_frames):
            example.step()
            episode_time += example.frame_dt

            if frame_idx % self._frame_skip == 0:
                state = example.get_obs_state()
                action = example.get_action()
                image = camera.capture(example.state_0)
                buf.append(state, action, image, episode_time)

        return buf

    # ------------------------------------------------------------------
    # Parquet + video writers
    # ------------------------------------------------------------------

    def _save_episode(self, buf: _EpisodeBuffer, episode_index: int):
        pa = self._pa
        pq = self._pq

        n = len(buf)
        ep_chunk = _episode_chunk(episode_index)

        # Ensure chunk directories exist
        (self.output_dir / "data" / f"chunk-{ep_chunk:03d}").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos" / f"chunk-{ep_chunk:03d}").mkdir(parents=True, exist_ok=True)

        # --- Write video ---
        video_path = _episode_video_path(self.output_dir, episode_index)
        self._write_video(buf.images, video_path)

        # --- Write parquet ---
        parquet_path = _episode_parquet_path(self.output_dir, episode_index)
        table = pa.table(
            {
                "observation.state": pa.array(
                    [s.tolist() for s in buf.states],
                    type=pa.list_(pa.float32(), list_size=STATE_DIM),
                ),
                "action": pa.array(
                    [a.tolist() for a in buf.actions],
                    type=pa.list_(pa.float32(), list_size=STATE_DIM),
                ),
                "timestamp": pa.array(buf.timestamps, type=pa.float32()),
                "frame_index": pa.array(list(range(n)), type=pa.int64()),
                "episode_index": pa.array([episode_index] * n, type=pa.int64()),
                "index": pa.array(
                    list(range(self._global_frame_index, self._global_frame_index + n)),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * n, type=pa.int64()),
                "next.done": pa.array([False] * (n - 1) + [True], type=pa.bool_()),
            }
        )
        pq.write_table(table, parquet_path)

    def _write_video(self, frames: list[np.ndarray], path: Path):
        imageio = self._imageio
        writer = imageio.get_writer(
            str(path),
            fps=self.record_fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

    # ------------------------------------------------------------------
    # Metadata writers
    # ------------------------------------------------------------------

    def _save_metadata(self, all_states: np.ndarray, all_actions: np.ndarray):
        total_episodes = self.num_episodes
        total_frames = self._global_frame_index
        record_fps = self.record_fps

        # info.json
        info = {
            "codebase_version": "v2.0",
            "robot_type": _ROBOT_TYPE,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes,
            "total_chunks": max(1, (total_episodes - 1) // _CHUNKS_SIZE + 1),
            "chunks_size": _CHUNKS_SIZE,
            "fps": record_fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/"
                "observation.images.top_episode_{episode_index:06d}.mp4"
            ),
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [STATE_DIM],
                    "names": _STATE_NAMES,
                },
                "observation.images.top": {
                    "dtype": "video",
                    "shape": [_CAM_HEIGHT, _CAM_WIDTH, 3],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video_backend": "pyav",
                        "video.fps": record_fps,
                        "video.height": _CAM_HEIGHT,
                        "video.width": _CAM_WIDTH,
                        "video.channels": 3,
                        "video.codec": "libx264",
                        "video.pix_fmt": "yuv420p",
                    },
                },
                "action": {
                    "dtype": "float32",
                    "shape": [STATE_DIM],
                    "names": _STATE_NAMES,
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
            },
        }
        (self.output_dir / "meta" / "info.json").write_text(
            json.dumps(info, indent=2)
        )

        # tasks.jsonl
        tasks_line = json.dumps({"task_index": 0, "task": _TASK_DESCRIPTION})
        (self.output_dir / "meta" / "tasks.jsonl").write_text(tasks_line + "\n")

        # episodes.jsonl
        lines = [json.dumps(ep) for ep in self._episode_stats]
        (self.output_dir / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n")

        # stats.json
        stats = _compute_stats(all_states, all_actions)
        (self.output_dir / "meta" / "stats.json").write_text(
            json.dumps(stats, indent=2)
        )


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _compute_stats(states: np.ndarray, actions: np.ndarray) -> dict:
    """Compute mean/std/min/max for state and action arrays."""

    def _stat(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }

    return {
        "observation.state": _stat(states),
        "action": _stat(actions),
    }


# ---------------------------------------------------------------------------
# Minimal no-op viewer (satisfies Example's viewer interface)
# ---------------------------------------------------------------------------

class _NullViewer:
    """No-op viewer for headless data collection."""

    def set_model(self, model):
        pass

    def set_camera(self, *args, **kwargs):
        pass

    def begin_frame(self, t):
        pass

    def log_state(self, state):
        pass

    def end_frame(self):
        pass

    def apply_forces(self, state):
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect a LeRobot v2 dataset from the bimanual OMX cloth task."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lerobot_cloth_omx",
        help="Root directory for the dataset (default: ./lerobot_cloth_omx)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to collect (default: 50)",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="Recording frame rate (default: 30; sim runs at 60 fps)",
    )
    args = parser.parse_args()

    collector = LeRobotDatasetCollector(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        record_fps=args.record_fps,
    )
    collector.collect()


if __name__ == "__main__":
    main()
