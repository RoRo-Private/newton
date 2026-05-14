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

###########################################################################
# Example Cloth SO-ARM 101 Bimanual Replay
#
# Replays PhysTwin cloth data in the bimanual SO-ARM 101 scene using a
# hybrid approach:
#   - Top-3 most dynamic driver particles (from driver_vertex_ids) are
#     driven directly from recorded driver_targets trajectories.
#   - The remaining cloth particles follow via VBD physics simulation.
#
# Command: uv run -m newton.examples cloth_so_arm_bimanual_replay
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone, SolverVBD

_SO_ARM_BIMANUAL_URDF = (
    Path(__file__).resolve().parents[3] / "so_arm_description" / "so101_bimanual_set" / "so101_bimanual_set2.urdf"
)

_CLOTH_DATA_DEFAULT = (
    Path(__file__).resolve().parents[3] / "cloth_data" / "cloth_1_2" / "newton_cloth" / "cloth_export.npz"
)

_N_COORDS_PER_ARM = 6
STATE_DIM = _N_COORDS_PER_ARM * 2

_HOME_POSE = np.zeros(12, dtype=np.float32)
_HOME_POSE[3] = 90.0 * np.pi / 180.0
_HOME_POSE[9] = 90.0 * np.pi / 180.0

# Number of dynamic driver vertices to use for cloth driving
_N_DRIVERS = 3

EPISODE_DURATION = 30.0  # [s]


@wp.kernel
def _expand_by_vmapping(
    src: wp.array(dtype=wp.vec3),
    vmapping: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3),
) -> None:
    """Expand per-original-vertex buffer to seam-split vertex buffer."""
    i = wp.tid()
    dst[i] = src[vmapping[i]]


@wp.kernel
def _apply_driver_spring(
    driver_vertex_ids: wp.array(dtype=wp.int32),
    driver_targets: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    ke: float,
) -> None:
    """Apply spring force pulling each driver particle toward its target."""
    tid = wp.tid()
    vid = driver_vertex_ids[tid]
    delta = driver_targets[tid] - particle_q[vid]
    particle_f[vid] = particle_f[vid] + delta * ke


class Example:
    def __init__(self, viewer, args=None):
        self.sim_substeps = 8
        self.iterations = 4
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self._step_count = 0
        self._replay_speed_divisor = 16  # advance replay frame every N physics steps
        self._settling_steps = 300       # physics steps to run before replay starts (~5 s at 60 fps)
        self._driver_ke = 500.0          # spring stiffness pulling drivers toward targets [N/m]

        self.viewer = viewer
        self._cloth_dataset_path = (
            getattr(args, "cloth_dataset", None) or str(_CLOTH_DATA_DEFAULT)
        )

        # ------------------------------------------------------------------
        # Build scene
        # ------------------------------------------------------------------
        scene = ModelBuilder()

        scene.add_urdf(
            str(_SO_ARM_BIMANUAL_URDF),
            xform=wp.transform_identity(),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        scene.joint_q[:STATE_DIM] = _HOME_POSE.tolist()

        vertices, indices, cloth_kwargs = self._load_cloth_mesh()
        scene.add_cloth_mesh(
            vertices=vertices,
            indices=indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-7,
            edge_ke=1e-3,
            edge_kd=1e-4,
            particle_radius=0.008,
            **cloth_kwargs,
        )

        scene.add_shape_plane(
            body=-1,
            xform=wp.transform(wp.vec3(0.3, 0.0, 0.018), wp.quat_identity()),
            width=1.0,
            length=3.0,
        )

        scene.color(include_bending=True)

        self.model = scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = 100.0
        self.model.soft_contact_kd = 2e-3
        self.model.soft_contact_mu = 0.5

        n = self.model.shape_count
        self.model.shape_material_ke = wp.array(
            np.full(n, 100.0, dtype=np.float32),
            dtype=self.model.shape_material_ke.dtype,
            device=self.model.shape_material_ke.device,
        )
        self.model.shape_material_kd = wp.array(
            np.full(n, 2e-3, dtype=np.float32),
            dtype=self.model.shape_material_kd.dtype,
            device=self.model.shape_material_kd.device,
        )
        self.model.shape_material_mu = wp.array(
            np.full(n, 1.0, dtype=np.float32),
            dtype=self.model.shape_material_mu.dtype,
            device=self.model.shape_material_mu.device,
        )


        # ------------------------------------------------------------------
        # Select driver vertices (all remain ACTIVE; driven via soft spring)
        # ------------------------------------------------------------------
        self._select_drivers()

        # ------------------------------------------------------------------
        # States, control, pipelines, solvers
        # ------------------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._robot_solver = SolverFeatherstone(self.model)
        self._rigid_pipeline = newton.CollisionPipeline(self.model)
        self._rigid_contacts = self._rigid_pipeline.contacts()

        self._cloth_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=0.004
        )
        self._cloth_contacts = self._cloth_pipeline.contacts()

        self._cloth_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            rigid_contact_k_start=100.0,

        )

        self._gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self._gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # GPU driver buffers
        dev = self.model.device
        self._driver_vertex_ids_wp = wp.array(
            self._driver_vertex_ids_np, dtype=wp.int32, device=dev
        )
        self._driver_targets_wp = wp.array(
            self._driver_traj_np[0], dtype=wp.vec3, device=dev
        )

        # ------------------------------------------------------------------
        # Viewer
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.3, 0.6, 0.8), -30.0, -50.0)
        self.viewer.show_triangles = True
        self.viewer.show_particles = False

    # ------------------------------------------------------------------
    # Cloth mesh loading
    # ------------------------------------------------------------------

    def _load_cloth_mesh(self) -> tuple[list, list, dict]:
        path = Path(self._cloth_dataset_path)
        if path.exists():
            dataset = np.load(str(path))
            verts = dataset["vertices"].astype(np.float32)
            faces = dataset["faces"].astype(np.int32)

            self._xy_mean = verts[:, :2].mean(axis=0)
            self._z_min   = verts[:, 2].min()

            verts[:, :2] -= self._xy_mean
            verts[:, 2]  -= self._z_min
            vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]

            tex_path = path.parent / "cloth_texture.png"
            dev = wp.get_device()
            has_uv = (
                "vmapping" in dataset.files
                and "faces_uv" in dataset.files
                and "uvs_uv" in dataset.files
                and tex_path.exists()
            )
            if has_uv:
                vmapping_np  = dataset["vmapping"].astype(np.int32)
                faces_uv_np  = dataset["faces_uv"].astype(np.int32)
                uvs_uv_np    = dataset["uvs_uv"].astype(np.float32)
                self._cloth_vmapping      = wp.array(vmapping_np, dtype=wp.int32, device=dev)
                self._cloth_face_indices  = wp.array(faces_uv_np.reshape(-1), dtype=wp.int32, device=dev)
                self._cloth_uvs           = wp.array(uvs_uv_np, dtype=wp.vec2, device=dev)
                self._cloth_render_q      = wp.zeros(len(vmapping_np), dtype=wp.vec3, device=dev)
                self._cloth_texture       = str(tex_path)
                print(f"[Replay] UV texture loaded ({tex_path.name})")
            else:
                self._cloth_vmapping     = None
                self._cloth_uvs          = None
                self._cloth_texture      = None
                self._cloth_face_indices = wp.array(faces.reshape(-1), dtype=wp.int32, device=dev)
                self._cloth_render_q     = None
                print("[Replay] No UV texture found.")

            kwargs = dict(
                scale=1.0,
                pos=wp.vec3(0.35, 0.0, 0.025),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0),
            )
            print(f"[Replay] Cloth: {len(verts)} vertices, {faces.shape[0]} faces")
            return vertices, faces.reshape(-1).tolist(), kwargs

        # USD fallback
        self._xy_mean = np.zeros(2, dtype=np.float32)
        self._z_min   = 0.0
        self._cloth_vmapping = self._cloth_uvs = self._cloth_texture = None
        self._cloth_face_indices = self._cloth_render_q = None
        print(f"[Replay] NPZ not found ({path}), falling back to USD t-shirt.")
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        kwargs = dict(
            scale=0.01,
            pos=wp.vec3(0.40, 0.0, 0.15),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
        )
        return [wp.vec3(v) for v in shirt_mesh.vertices], shirt_mesh.indices, kwargs

    # ------------------------------------------------------------------
    # Driver selection
    # ------------------------------------------------------------------

    def _select_drivers(self) -> None:
        """Load driver vertex ids and trajectories from cloth_export.npz.

        driver_vertex_ids are pre-selected during export (by displacement score).
        Invalid frames (visibility=False) are forward-filled from the last
        valid position.
        """
        path = Path(self._cloth_dataset_path)
        dataset = np.load(str(path))

        driver_vertex_ids = dataset["driver_vertex_ids"].astype(np.int32)  # (K,)
        obj_pts = dataset["object_points"].astype(np.float32)              # (T, N, 3)
        vis     = dataset["object_visibilities"]                           # (T, N)

        # Use at most _N_DRIVERS driving points
        driver_vertex_ids = driver_vertex_ids[:_N_DRIVERS]
        self._driver_vertex_ids_np = driver_vertex_ids                     # (K,)

        driver_traj = obj_pts[:, driver_vertex_ids, :]   # (T, K, 3)
        driver_vis  = vis[:, driver_vertex_ids]          # (T, K)

        vis_counts = driver_vis.sum(axis=0)
        print(f"[Replay] Using {len(driver_vertex_ids)} driver points: "
              f"vertex ids {driver_vertex_ids}, "
              f"visibility {vis_counts.tolist()}/{vis.shape[0]} frames")

        # Forward-fill invalid (invisible) frames with last valid position
        for k in range(_N_DRIVERS):
            last_valid = driver_traj[0, k].copy()
            for t in range(len(driver_traj)):
                if driver_vis[t, k]:
                    last_valid = driver_traj[t, k].copy()
                else:
                    driver_traj[t, k] = last_valid

        # Apply same coordinate transform as rest vertices
        R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
        T_frames, K = driver_traj.shape[:2]
        pts = driver_traj.reshape(-1, 3).copy()
        pts[:, 0] -= self._xy_mean[0]
        pts[:, 1] -= self._xy_mean[1]
        pts[:, 2] -= self._z_min
        pts = (R @ pts.T).T
        pts += np.array([0.35, 0.0, 0.06], dtype=np.float32)
        self._driver_traj_np = pts.reshape(T_frames, K, 3)                # (T, K, 3)

    # ------------------------------------------------------------------
    # Simulation / step / render
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        frame_id = self.frame_index % self._driver_traj_np.shape[0]
        dev = self.model.device

        # Update driver target positions for this frame
        self._driver_targets_wp.assign(
            wp.array(self._driver_traj_np[frame_id], dtype=wp.vec3, device=dev)
        )

        self._cloth_solver.rebuild_bvh(self.state_0)
        particle_count_saved = self.model.particle_count

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Robot step (home pose, no actuation)
            self.model.particle_count = 0
            self.model.gravity.assign(self._gravity_zero)
            self._rigid_pipeline.collide(self.state_0, self._rigid_contacts)
            self._robot_solver.step(self.state_0, self.state_1, self.control,
                                    self._rigid_contacts, self.sim_dt)

            # Cloth step — apply soft spring toward driver targets before solve
            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count_saved
            self.model.gravity.assign(self._gravity_earth)

            if self._step_count > self._settling_steps:
                wp.launch(
                    _apply_driver_spring,
                    dim=_N_DRIVERS,
                    inputs=[
                        self._driver_vertex_ids_wp,
                        self._driver_targets_wp,
                        self.state_0.particle_q,
                        self.state_0.particle_f,
                        self._driver_ke,
                    ],
                    device=dev,
                )

            self._cloth_pipeline.collide(self.state_0, self._cloth_contacts)
            self._cloth_solver.step(self.state_0, self.state_1, self.control,
                                    self._cloth_contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self._step_count += 1
        # Hold frame 0 during settling; advance only after settling is done
        if self._step_count > self._settling_steps:
            replay_step = self._step_count - self._settling_steps
            if replay_step % self._replay_speed_divisor == 0:
                self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)

        if self._cloth_uvs is not None and self._cloth_texture is not None:
            self.viewer.show_triangles = False
            wp.launch(
                _expand_by_vmapping,
                dim=self._cloth_render_q.shape[0],
                inputs=[self.state_0.particle_q, self._cloth_vmapping],
                outputs=[self._cloth_render_q],
                device=self.state_0.particle_q.device,
            )
            self.viewer.log_mesh(
                "/cloth_textured",
                self._cloth_render_q,
                self._cloth_face_indices,
                uvs=self._cloth_uvs,
                texture=self._cloth_texture,
                backface_culling=False,
            )
        else:
            self.viewer.show_triangles = True

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Replay particles contain non-finite values"
        assert particle_q.shape[0] == self.model.particle_count


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(EPISODE_DURATION * 60))
    parser.add_argument(
        "--cloth-dataset",
        type=str,
        default=str(_CLOTH_DATA_DEFAULT),
        help="Path to cloth_export.npz (must contain driver_vertex_ids, object_points).",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
