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
# Example Cloth SO-ARM 101 Bimanual
#
# Bimanual cloth manipulation using two SO-ARM 101 arms driven by IK +
# optional WebXR (Meta Quest 2) teleoperation.
#
# The simulation uses VBD for cloth and Featherstone for the robots.
# Joint state/action (12-dim, one per revolute joint):
#   [shoulder_pan_l, shoulder_lift_l, elbow_flex_l, wrist_flex_l,
#    wrist_roll_l, gripper_l,
#    shoulder_pan_r, shoulder_lift_r, elbow_flex_r, wrist_flex_r,
#    wrist_roll_r, gripper_r]
#
# Command: uv run -m newton.examples cloth_so_arm_bimanual
#          uv run -m newton.examples cloth_so_arm_bimanual --webxr
#
###########################################################################

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone, SolverVBD

_SO_ARM_BIMANUAL_URDF = (
    Path(__file__).resolve().parents[3] / "so_arm_description" / "so101_bimanual_set" / "so101_bimanual_set2.urdf"
)

_CLOTH_DATA_DEFAULT = (
    Path(__file__).resolve().parents[3] / "cloth_data" / "cloth_1_2" / "newton_cloth" / "cloth_export.npz"
)

# ---------------------------------------------------------------------------
# Joint layout after add_urdf (collapse_fixed_joints=True):
#   Left arm:
#   [0]  left_shoulder_pan    [1]  left_shoulder_lift
#   [2]  left_elbow_flex      [3]  left_wrist_flex
#   [4]  left_wrist_roll      [5]  left_gripper
#   Right arm:
#   [6]  right_shoulder_pan   [7]  right_shoulder_lift
#   [8]  right_elbow_flex     [9]  right_wrist_flex
#   [10] right_wrist_roll     [11] right_gripper
#
# Body indices (collapse_fixed_joints=True, base_link fixed to world):
#   Left:  [0] shoulder [1] upper_arm [2] lower_arm [3] wrist [4] gripper [5] moving_jaw
#   Right: [6] shoulder [7] upper_arm [8] lower_arm [9] wrist [10] gripper [11] moving_jaw
#
# NOTE: verify EE body indices by checking printed body_label at startup.
# ---------------------------------------------------------------------------
_N_COORDS_PER_ARM = 6
STATE_DIM = _N_COORDS_PER_ARM * 2  # 12

# Body indices of the EE links (gripper_link of each arm).
# NOTE: these are best-guess estimates — verify from startup print.
_LEFT_EE_BODY  = 4
_RIGHT_EE_BODY = 10

# Gripper joint range [rad]: open=0.0, close=1.5 (toward upper limit 1.745)
# Tune after visual inspection if the direction is reversed.
_GRIP_OPEN  = 0.0
_GRIP_CLOSE = 1.5

# Indices into 12-dim joint_q for state/action (all joints, no mimic)
_LEFT_STATE_IDX  = list(range(0, 6))
_RIGHT_STATE_IDX = list(range(6, 12))
_STATE_IDX = _LEFT_STATE_IDX + _RIGHT_STATE_IDX

# Home pose: all joints at 0 (neutral).
# Tune shoulder_pan / shoulder_lift for a good pre-grasp position.
_HOME_POSE = np.zeros(12, dtype=np.float32)
_HOME_POSE[1] = 0.0 * np.pi / 180.0
_HOME_POSE[3] = 90.0 * np.pi / 180.0

_HOME_POSE[7] = 0.0 * np.pi / 180.0
_HOME_POSE[9] = 90.0 * np.pi / 180.0

EPISODE_DURATION = 30.0  # [s] default teleoperation episode length


# ---------------------------------------------------------------------------
# WebXR → Newton coordinate transform
#
# Assumes the user stands facing the +X direction of the Newton world:
#   WebXR X (user right)  → Newton -Y
#   WebXR Y (user up)     → Newton +Z
#   WebXR Z (toward user) → Newton -X
# ---------------------------------------------------------------------------
_Q_AXIS = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)   # xyzw
_Q_AXIS_INV = np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float64)

_VR_STALE_TIMEOUT = 0.5


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in xyzw format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions) in xyzw format."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _xr_pos_to_newton(p: tuple) -> np.ndarray:
    """Map WebXR position (x, y, z) to Newton coordinate axes."""
    x, y, z = p
    return np.array([-z, -x, y], dtype=np.float64)


def _xr_rot_delta_to_newton(q_ref: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Map WebXR rotation delta (q_ref→q_cur) to Newton frame (xyzw)."""
    delta_xr = _quat_mul(_quat_conj(q_ref), q_cur)
    return _quat_mul(_quat_mul(_Q_AXIS, delta_xr), _Q_AXIS_INV)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64)
    return qn / np.linalg.norm(qn)


def _quat_slerp_np(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation for xyzw quaternions."""
    q0n = _quat_normalize(q0)
    q1n = _quat_normalize(q1)
    dot = float(np.dot(q0n, q1n))
    if dot < 0.0:
        q1n = -q1n
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0n + t * (q1n - q0n))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return _quat_normalize(s0 * q0n + s1 * q1n)


def _quat_to_world_yaw(q_xyzw: np.ndarray) -> float:
    """Extract world-Z yaw [rad] from a quaternion in xyzw format."""
    x, y, z, w = q_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@wp.kernel
def _expand_by_vmapping(
    src: wp.array(dtype=wp.vec3),
    vmapping: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=wp.vec3),
) -> None:
    """Expand a per-original-vertex buffer to the seam-split vertex buffer.

    dst[i] = src[vmapping[i]]  for each seam-split vertex i.
    """
    i = wp.tid()
    dst[i] = src[vmapping[i]]


@wp.kernel
def _p_controller_kernel(
    ik_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    kp: float,
    qd_max: float,
    n_arm_coords: int,
    target_joint_qd: wp.array(dtype=wp.float32),
) -> None:
    """Compute P-controller velocities: qd = clamp(kp*(q_target - q_current), ±qd_max)."""
    i = wp.tid()
    if i < n_arm_coords:
        target_joint_qd[i] = wp.clamp(kp * (ik_joint_q[i] - joint_q[i]), -qd_max, qd_max)
    else:
        target_joint_qd[i] = wp.float32(0.0)


class Example:
    def __init__(self, viewer, args=None):
        # ------------------------------------------------------------------
        # Simulation parameters
        # ------------------------------------------------------------------
        self.sim_substeps = 8
        self.iterations = 2
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Contact
        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.01
        self.particle_self_contact_radius = 0.002
        self.particle_self_contact_margin = 0.003
        self.soft_contact_ke = 100.0
        self.soft_contact_kd = 2e-3

        # Cloth elasticity
        self.tri_ke = 1e2
        self.tri_ka = 1e2
        self.tri_kd = 1.5e-6
        self.bending_ke = 1e-4
        self.bending_kd = 1e-3

        # Joint-space P-controller gain [1/s]
        self.kp = 3.0

        self.viewer = viewer

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
        # Set home pose
        scene.joint_q[:STATE_DIM] = _HOME_POSE.tolist()


        # --- Cloth mesh ---
        self._cloth_dataset_path = (
            getattr(args, "cloth_dataset", None) or str(_CLOTH_DATA_DEFAULT)
        )
        vertices, indices, cloth_mesh_kwargs = self._get_cloth_mesh()

        scene.add_cloth_mesh(
            vertices=vertices,
            indices=indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
            **cloth_mesh_kwargs,
        )

        scene.add_shape_plane(
            body=-1,
            xform=wp.transform(wp.vec3(0.3, 0.0, 0.018), wp.quat_identity()),
            width=1.0,
            length=3.0,
        )

        scene.color()

        # ------------------------------------------------------------------
        # Finalize model
        # ------------------------------------------------------------------
        self.model = scene.finalize(requires_grad=False)

        # Print body labels to help verify EE body indices
        print("[SO-ARM] body_label:", self.model.body_label)

        # Restrict cloth-particle collision to EE links only (gripper + moving jaw).
        # All other robot links (shoulder, upper_arm, lower_arm, wrist) have
        # COLLIDE_PARTICLES disabled to reduce cloth pipeline cost.
        _ee_bodies = {_LEFT_EE_BODY, _LEFT_EE_BODY + 1, _RIGHT_EE_BODY, _RIGHT_EE_BODY + 1}
        _flags = self.model.shape_flags.numpy().copy()
        for body_idx, shapes in self.model.body_shapes.items():
            # Skip static body (-1): ground plane and table must keep cloth collision.
            # Only disable COLLIDE_PARTICLES for non-EE robot links (body_idx >= 0).
            if body_idx >= 0 and body_idx not in _ee_bodies:
                for s in shapes:
                    _flags[s] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        self.model.shape_flags = wp.array(
            _flags, dtype=self.model.shape_flags.dtype, device=self.model.shape_flags.device
        )

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = 0.5

        n = self.model.shape_count
        self.model.shape_material_ke = wp.array(
            np.full(n, self.soft_contact_ke, dtype=np.float32),
            dtype=self.model.shape_material_ke.dtype,
            device=self.model.shape_material_ke.device,
        )
        self.model.shape_material_kd = wp.array(
            np.full(n, self.soft_contact_kd, dtype=np.float32),
            dtype=self.model.shape_material_kd.dtype,
            device=self.model.shape_material_kd.device,
        )
        self.model.shape_material_mu = wp.array(
            np.full(n, 1.0, dtype=np.float32),
            dtype=self.model.shape_material_mu.dtype,
            device=self.model.shape_material_mu.device,
        )

        # ------------------------------------------------------------------
        # States and control
        # ------------------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._target_joint_qd = wp.zeros(
            self.model.joint_dof_count, dtype=wp.float32, device=self.model.device
        )

        # ------------------------------------------------------------------
        # Collision pipelines
        # ------------------------------------------------------------------
        self._rigid_pipeline = newton.CollisionPipeline(self.model)
        self._rigid_contacts = self._rigid_pipeline.contacts()

        self._collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self._contacts = self._collision_pipeline.contacts()

        # ------------------------------------------------------------------
        # Solvers
        # ------------------------------------------------------------------
        self._robot_solver = SolverFeatherstone(
            self.model, update_mass_matrix_interval=self.sim_substeps
        )

        self.model.edge_rest_angle.zero_()
        self._cloth_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=self.soft_contact_ke,
        )

        self._gravity_zero  = wp.zeros(1, dtype=wp.vec3)
        self._gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ------------------------------------------------------------------
        # IK solver
        # ------------------------------------------------------------------
        self._setup_ik()

        # ------------------------------------------------------------------
        # WebXR teleoperation (optional)
        # ------------------------------------------------------------------
        if args is not None and getattr(args, "webxr", False):
            self._setup_vr(args.cert, args.key)
        else:
            self._vr = None

        # ------------------------------------------------------------------
        # Viewer + CUDA graph capture
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.3, 0.6, 0.8), -30.0, -50.0)
        self.viewer.show_triangles = True
        self.viewer.show_particles = False

        # Neutral lighting — slight cool sky tint to give mesh depth without orange cast.
        # sky/ground set the hemispherical ambient; _light_color drives the directional diffuse.
        # The shader multiplies diffuse by 3.0 internally, so keep light_color moderate.
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.sky_upper = (0.3, 0.3, 0.3)
            self.viewer.renderer.sky_lower = (0.18, 0.18, 0.20)
            self.viewer.renderer._light_color = (0.45, 0.45, 0.45)

        self._graph = None  # CUDA graph disabled: placo IK is a CPU op

    # ------------------------------------------------------------------
    # Cloth mesh loading
    # ------------------------------------------------------------------

    def _get_cloth_mesh(self) -> tuple[list, list, dict]:
        """Load cloth mesh. Returns (vertices, indices, add_cloth_mesh_kwargs).

        Loads from cloth_export.npz when available; falls back to USD t-shirt.
        """
        path = Path(self._cloth_dataset_path)
        if path.exists():
            dataset = np.load(str(path))
            verts = dataset["vertices"].astype(np.float32)   # (V, 3)
            faces = dataset["faces"].astype(np.int32)        # (F, 3)
            # Center XY at origin; lift bottom to z=0.
            verts[:, :2] -= verts[:, :2].mean(axis=0)
            verts[:, 2] -= verts[:, 2].min()
            vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts]

            # UV + texture (populated by bake_uv.py)
            # New format: vmapping/faces_uv/uvs_uv (seam-split topology)
            tex_path = path.parent / "cloth_texture.png"
            _dev = wp.get_device()
            has_seam_uv = (
                "vmapping" in dataset.files
                and "faces_uv" in dataset.files
                and "uvs_uv" in dataset.files
                and tex_path.exists()
            )
            if has_seam_uv:
                vmapping_np = dataset["vmapping"].astype(np.int32)     # (V_new,)
                faces_uv_np = dataset["faces_uv"].astype(np.int32)    # (F, 3)
                uvs_uv_np = dataset["uvs_uv"].astype(np.float32)      # (V_new, 2)
                self._cloth_vmapping = wp.array(vmapping_np, dtype=wp.int32, device=_dev)
                self._cloth_face_indices = wp.array(faces_uv_np.reshape(-1), dtype=wp.int32, device=_dev)
                self._cloth_uvs = wp.array(uvs_uv_np, dtype=wp.vec2, device=_dev)
                self._cloth_render_q = wp.zeros(len(vmapping_np), dtype=wp.vec3, device=_dev)
                self._cloth_texture = str(tex_path)
                print(f"[SO-ARM Cloth] UV texture loaded ({tex_path.name}), "
                      f"seam-split V={len(vmapping_np)}")
            else:
                self._cloth_vmapping = None
                self._cloth_uvs = None
                self._cloth_texture = None
                self._cloth_face_indices = wp.array(faces.reshape(-1), dtype=wp.int32, device=_dev)
                self._cloth_render_q = None
                print("[SO-ARM Cloth] No UV texture — run cloth_data/bake_uv.py to enable.")

            kwargs = dict(
                scale=1.0,
                pos=wp.vec3(0.35, 0.0, 0.025),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0),
            )
            print(f"[SO-ARM Cloth] Loaded cloth from NPZ: {path} ({len(verts)} particles)")
            return vertices, faces.reshape(-1).tolist(), kwargs

        # Fallback: USD t-shirt (no UV texture)
        self._cloth_vmapping = None
        self._cloth_uvs = None
        self._cloth_texture = None
        self._cloth_face_indices = None
        self._cloth_render_q = None
        print(f"[SO-ARM Cloth] NPZ not found ({path}), falling back to USD t-shirt.")
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
        kwargs = dict(
            scale=0.01,
            pos=wp.vec3(0.40, 0.0, 0.15),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
        )
        return vertices, shirt_mesh.indices, kwargs

    # ------------------------------------------------------------------
    # WebXR teleoperation
    # ------------------------------------------------------------------

    def _setup_vr(self, cert: str, key: str) -> None:
        import sys
        _REPO_ROOT = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(_REPO_ROOT))
        from webxr.vr_receiver import VRReceiver  # noqa: PLC0415

        self._vr = VRReceiver(cert=cert, key=key)
        self._vr.start()

        self._vr_calibrated = False
        self._vr_calib: dict | None = None
        self._vr_triggering = False

    def _calibrate_vr(self, left, right) -> None:
        """Capture current controller and EE poses as calibration reference."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        # Use placo FK to get current gripper_frame poses.
        q_np = self.state_0.joint_q.numpy()
        for i, name in enumerate(self._left_joint_names):
            self._placo_robot_left.set_joint(name, float(q_np[i]))
        self._placo_robot_left.update_kinematics()
        for i, name in enumerate(self._right_joint_names):
            self._placo_robot_right.set_joint(name, float(q_np[_N_COORDS_PER_ARM + i]))
        self._placo_robot_right.update_kinematics()

        left_T  = self._placo_robot_left.get_T_world_frame("left_gripper_frame_link")
        right_T = self._placo_robot_right.get_T_world_frame("right_gripper_frame_link")
        left_q_newton  = R.from_matrix(left_T[:3, :3]).as_quat()   # xyzw
        right_q_newton = R.from_matrix(right_T[:3, :3]).as_quat()  # xyzw

        self._vr_calib = {
            "left_xr_pos":    np.array(left.position,         dtype=np.float64),
            "right_xr_pos":   np.array(right.position,        dtype=np.float64),
            "left_xr_rot":    np.array(left.quaternion_xyzw,  dtype=np.float64),
            "right_xr_rot":   np.array(right.quaternion_xyzw, dtype=np.float64),
            "left_newton_pos":  left_T[:3, 3].copy(),
            "right_newton_pos": right_T[:3, 3].copy(),
            "left_newton_rot":  left_q_newton.copy(),
            "right_newton_rot": right_q_newton.copy(),
        }
        self._vr_calibrated = True
        print("[VR] Calibrated — release trigger to start controlling.")

    def _update_from_vr(self) -> None:
        """Read latest VR poses and update IK targets + gripper state."""
        import time
        left  = self._vr.get_left()
        right = self._vr.get_right()

        now = time.monotonic()
        if now - left.received > _VR_STALE_TIMEOUT or now - right.received > _VR_STALE_TIMEOUT:
            return

        trigger_pressed = left.trigger > 0.5 or right.trigger > 0.5
        if trigger_pressed and not self._vr_triggering:
            self._calibrate_vr(left, right)
        self._vr_triggering = trigger_pressed

        if not self._vr_calibrated:
            return

        c = self._vr_calib

        left_delta_newton  = _xr_pos_to_newton(np.array(left.position)  - c["left_xr_pos"])
        right_delta_newton = _xr_pos_to_newton(np.array(right.position) - c["right_xr_pos"])

        left_pos  = c["left_newton_pos"]  + left_delta_newton
        right_pos = c["right_newton_pos"] + right_delta_newton

        left_rot_delta  = _xr_rot_delta_to_newton(c["left_xr_rot"],  np.array(left.quaternion_xyzw))
        right_rot_delta = _xr_rot_delta_to_newton(c["right_xr_rot"], np.array(right.quaternion_xyzw))

        left_rot  = _quat_mul(c["left_newton_rot"],  left_rot_delta)
        right_rot = _quat_mul(c["right_newton_rot"], right_rot_delta)

        left_rot  /= np.linalg.norm(left_rot)
        right_rot /= np.linalg.norm(right_rot)

        self._left_ee_tf = wp.transform(
            wp.vec3(*left_pos.tolist()),
            wp.quat(*left_rot.tolist()),
        )
        self._right_ee_tf = wp.transform(
            wp.vec3(*right_pos.tolist()),
            wp.quat(*right_rot.tolist()),
        )

        self.grip_left  = _GRIP_OPEN + left.grip  * (_GRIP_CLOSE - _GRIP_OPEN)
        self.grip_right = _GRIP_OPEN + right.grip * (_GRIP_CLOSE - _GRIP_OPEN)

    # ------------------------------------------------------------------
    # IK setup (placo)
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        try:
            import placo  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "placo is required for IK. Install it with: pip install placo"
            ) from e

        urdf_path = str(_SO_ARM_BIMANUAL_URDF)
        _LEFT_JOINT_NAMES  = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                               "left_wrist_flex", "left_wrist_roll"]
        _RIGHT_JOINT_NAMES = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                               "right_wrist_flex", "right_wrist_roll"]

        # Left arm placo solver
        self._placo_robot_left = placo.RobotWrapper(urdf_path)
        self._placo_solver_left = placo.KinematicsSolver(self._placo_robot_left)
        self._placo_solver_left.mask_fbase(True)
        self._left_frame_task = self._placo_solver_left.add_frame_task(
            "left_gripper_frame_link", np.eye(4)
        )
        self._left_joint_names = _LEFT_JOINT_NAMES

        # Right arm placo solver
        self._placo_robot_right = placo.RobotWrapper(urdf_path)
        self._placo_solver_right = placo.KinematicsSolver(self._placo_robot_right)
        self._placo_solver_right.mask_fbase(True)
        self._right_frame_task = self._placo_solver_right.add_frame_task(
            "right_gripper_frame_link", np.eye(4)
        )
        self._right_joint_names = _RIGHT_JOINT_NAMES

        # Seed placo with home pose
        home_np = self.model.joint_q.numpy()
        for i, name in enumerate(_LEFT_JOINT_NAMES):
            self._placo_robot_left.set_joint(name, float(home_np[i]))
        for i, name in enumerate(_RIGHT_JOINT_NAMES):
            self._placo_robot_right.set_joint(name, float(home_np[_N_COORDS_PER_ARM + i]))
        self._placo_robot_left.update_kinematics()
        self._placo_robot_right.update_kinematics()

        # Compute initial gripper_frame poses via FK
        left_T  = self._placo_robot_left.get_T_world_frame("left_gripper_frame_link")
        right_T = self._placo_robot_right.get_T_world_frame("right_gripper_frame_link")

        def _T_to_wp_transform(T: np.ndarray) -> wp.transform:
            from scipy.spatial.transform import Rotation as R  # noqa: PLC0415
            q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
            return wp.transform(wp.vec3(*T[:3, 3].tolist()), wp.quat(*q.tolist()))

        self._T_to_wp_transform = _T_to_wp_transform

        # Raw command targets (VR or gizmo mutates these) — in gripper_frame space.
        self._left_ee_tf  = _T_to_wp_transform(left_T)
        self._right_ee_tf = _T_to_wp_transform(right_T)
        # Filtered targets used by IK.
        self._left_ee_tf_ik  = _T_to_wp_transform(left_T)
        self._right_ee_tf_ik = _T_to_wp_transform(right_T)

        # IK solution buffer on GPU (1D, length = joint_coord_count).
        self._ik_joint_q = wp.array(
            home_np.copy(),
            dtype=wp.float32,
            device=self.model.device,
        )

        # Gripper / wrist-roll (placo doesn't solve these; we set them directly).
        self.grip_left  = _GRIP_OPEN
        self.grip_right = _GRIP_OPEN
        self.roll_left  = float(home_np[4])
        self.roll_right = float(home_np[10])

        lq = wp.transform_get_rotation(self._left_ee_tf_ik)
        rq = wp.transform_get_rotation(self._right_ee_tf_ik)
        self._left_roll_ref_q  = np.array([lq[0], lq[1], lq[2], lq[3]], dtype=np.float64)
        self._right_roll_ref_q = np.array([rq[0], rq[1], rq[2], rq[3]], dtype=np.float64)
        self._left_roll_ref_value  = float(self.roll_left)
        self._right_roll_ref_value = float(self.roll_right)

        # Command target smoothing and speed limits.
        self._target_pos_lpf_alpha = 0.35
        self._target_rot_lpf_alpha = 0.30
        self._target_pos_max_speed = 1.2   # [m/s]
        self._target_rot_max_speed = 4.0   # [rad/s]

    def _solve_ik_and_push(self) -> None:
        """Run placo IK for both arms and push joint targets to GPU."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        def _wp_tf_to_T(tf: wp.transform) -> np.ndarray:
            p = np.array(wp.transform_get_translation(tf), dtype=np.float64)
            q = np.array(wp.transform_get_rotation(tf), dtype=np.float64)  # xyzw
            T = np.eye(4)
            T[:3, :3] = R.from_quat(q).as_matrix()
            T[:3, 3]  = p
            return T

        # Warm-start from previous IK solution (not physics state) to avoid
        # collision-induced corruption propagating into the IK seed.
        q_seed = self._ik_joint_q.numpy()

        # -- Left arm --
        for i, name in enumerate(self._left_joint_names):
            self._placo_robot_left.set_joint(name, float(q_seed[i]))
        self._placo_robot_left.update_kinematics()
        self._left_frame_task.T_world_frame = _wp_tf_to_T(self._left_ee_tf_ik)
        self._left_frame_task.configure("left_gripper_frame_link", "soft", 1.0, 0.1)
        self._placo_solver_left.solve(True)
        self._placo_robot_left.update_kinematics()

        # -- Right arm --
        for i, name in enumerate(self._right_joint_names):
            self._placo_robot_right.set_joint(name, float(q_seed[_N_COORDS_PER_ARM + i]))
        self._placo_robot_right.update_kinematics()
        self._right_frame_task.T_world_frame = _wp_tf_to_T(self._right_ee_tf_ik)
        self._right_frame_task.configure("right_gripper_frame_link", "soft", 1.0, 0.1)
        self._placo_solver_right.solve(True)
        self._placo_robot_right.update_kinematics()

        # Assemble and clamp to joint limits.
        q_lo = self.model.joint_limit_lower.numpy()
        q_hi = self.model.joint_limit_upper.numpy()
        q_target = q_seed.copy()
        for i, name in enumerate(self._left_joint_names):
            q_target[i] = self._placo_robot_left.get_joint(name)
        q_target[4]  = self.roll_left   # left wrist_roll
        q_target[5]  = self.grip_left   # left gripper
        for i, name in enumerate(self._right_joint_names):
            q_target[_N_COORDS_PER_ARM + i] = self._placo_robot_right.get_joint(name)
        q_target[10] = self.roll_right  # right wrist_roll
        q_target[11] = self.grip_right  # right gripper

        q_target[:STATE_DIM] = np.clip(q_target[:STATE_DIM], q_lo[:STATE_DIM], q_hi[:STATE_DIM])

        self._ik_joint_q.assign(q_target[:self.model.joint_coord_count].astype(np.float32))

    def _update_filtered_ee_targets(self) -> None:
        """Low-pass + rate-limit raw EE commands before IK."""

        def _filter_one(raw_tf: wp.transform, cur_tf: wp.transform) -> wp.transform:
            raw_p = np.array(wp.transform_get_translation(raw_tf), dtype=np.float64)
            cur_p = np.array(wp.transform_get_translation(cur_tf), dtype=np.float64)
            pos_des = cur_p + self._target_pos_lpf_alpha * (raw_p - cur_p)
            max_dp = self._target_pos_max_speed * self.frame_dt
            dp = pos_des - cur_p
            dp_norm = np.linalg.norm(dp)
            if dp_norm > max_dp and dp_norm > 1e-9:
                pos_des = cur_p + dp * (max_dp / dp_norm)

            raw_q = np.array(wp.transform_get_rotation(raw_tf), dtype=np.float64)
            cur_q = np.array(wp.transform_get_rotation(cur_tf), dtype=np.float64)
            q_des = _quat_slerp_np(cur_q, raw_q, self._target_rot_lpf_alpha)

            q_delta = _quat_mul(_quat_conj(cur_q), q_des)
            q_delta = _quat_normalize(q_delta)
            angle = 2.0 * np.arctan2(np.linalg.norm(q_delta[:3]), max(1e-9, q_delta[3]))
            max_da = self._target_rot_max_speed * self.frame_dt
            if angle > max_da:
                q_step = _quat_slerp_np(np.array([0.0, 0.0, 0.0, 1.0]), q_delta, max_da / angle)
                q_des = _quat_mul(cur_q, q_step)
                q_des = _quat_normalize(q_des)

            return wp.transform(
                wp.vec3(*pos_des.tolist()),
                wp.quat(float(q_des[0]), float(q_des[1]), float(q_des[2]), float(q_des[3])),
            )

        self._left_ee_tf_ik = _filter_one(self._left_ee_tf, self._left_ee_tf_ik)
        self._right_ee_tf_ik = _filter_one(self._right_ee_tf, self._right_ee_tf_ik)

    def _get_gripper_frame_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current gripper_frame world positions [m] via placo FK."""
        q_np = self.state_0.joint_q.numpy()
        for i, name in enumerate(self._left_joint_names):
            self._placo_robot_left.set_joint(name, float(q_np[i]))
        self._placo_robot_left.update_kinematics()
        for i, name in enumerate(self._right_joint_names):
            self._placo_robot_right.set_joint(name, float(q_np[_N_COORDS_PER_ARM + i]))
        self._placo_robot_right.update_kinematics()
        left_pos  = self._placo_robot_left.get_T_world_frame("left_gripper_frame_link")[:3, 3]
        right_pos = self._placo_robot_right.get_T_world_frame("right_gripper_frame_link")[:3, 3]
        return left_pos, right_pos

    def _update_roll_from_gizmos(self) -> None:
        """Update wrist-roll targets from interactive EE gizmo rotations."""
        lq = wp.transform_get_rotation(self._left_ee_tf)
        rq = wp.transform_get_rotation(self._right_ee_tf)

        lq_np = np.array([lq[0], lq[1], lq[2], lq[3]], dtype=np.float64)
        rq_np = np.array([rq[0], rq[1], rq[2], rq[3]], dtype=np.float64)

        ldelta = _quat_mul(_quat_conj(self._left_roll_ref_q), lq_np)
        rdelta = _quat_mul(_quat_conj(self._right_roll_ref_q), rq_np)
        ldelta /= np.linalg.norm(ldelta)
        rdelta /= np.linalg.norm(rdelta)

        self.roll_left = self._left_roll_ref_value + _quat_to_world_yaw(ldelta)
        self.roll_right = self._right_roll_ref_value + _quat_to_world_yaw(rdelta)

        q_lo = self.model.joint_limit_lower.numpy()
        q_hi = self.model.joint_limit_upper.numpy()
        self.roll_left = float(np.clip(self.roll_left, q_lo[4], q_hi[4]))
        self.roll_right = float(np.clip(self.roll_right, q_lo[10], q_hi[10]))

    # ------------------------------------------------------------------
    # Public helpers for data collection
    # ------------------------------------------------------------------

    def get_obs_state(self) -> np.ndarray:
        """Return 12-dim joint state [shoulder_pan..gripper x2]. [rad]"""
        wp.synchronize()
        return self.state_0.joint_q.numpy()[_STATE_IDX].astype(np.float32)

    def get_action(self) -> np.ndarray:
        """Return 12-dim absolute joint position action (current IK target). [rad]"""
        wp.synchronize()
        return self._ik_joint_q.numpy()[:STATE_DIM].astype(np.float32)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        # placo IK runs on CPU; result is pushed to GPU _ik_joint_q.
        self._solve_ik_and_push()

        # P-controller: convert IK joint targets → velocity targets.
        wp.launch(
            _p_controller_kernel,
            dim=self.model.joint_dof_count,
            inputs=[
                self._ik_joint_q,
                self.state_0.joint_q,
                self.kp,
                4.8,
                _N_COORDS_PER_ARM * 2,
                self._target_joint_qd,
            ],
            device=self.model.device,
        )

        self._cloth_solver.rebuild_bvh(self.state_0)
        particle_count_saved = self.model.particle_count
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.model.particle_count = 0
            self.model.gravity.assign(self._gravity_zero)

            self._rigid_pipeline.collide(self.state_0, self._rigid_contacts)
            self.state_0.joint_qd.assign(self._target_joint_qd)
            self._robot_solver.step(self.state_0, self.state_1, self.control, self._rigid_contacts, self.sim_dt)

            self.state_0.particle_f.zero_()

            self.model.particle_count = particle_count_saved
            self.model.gravity.assign(self._gravity_earth)

            self._collision_pipeline.collide(self.state_0, self._contacts)
            self._cloth_solver.step(
                self.state_0, self.state_1, self.control, self._contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self._vr is not None:
            self._update_from_vr()
        else:
            self._update_roll_from_gizmos()
        self._update_filtered_ee_targets()
        # placo IK is solved inside simulate() / _solve_ik_and_push()
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("left_ee", self._left_ee_tf)
        self.viewer.log_gizmo("right_ee", self._right_ee_tf)

        # Textured cloth mesh: override default triangle rendering when UV is available.
        if self._cloth_uvs is not None and self._cloth_texture is not None:
            self.viewer.show_triangles = False
            # Expand original particle_q → seam-split vertex buffer via vmapping
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
                color=(0.82, 0.82, 0.82),
                backface_culling=False,
            )
        else:
            self.viewer.show_triangles = True

        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-0.1, -0.5, -0.05)
        p_upper = wp.vec3(0.8, 0.5, 0.60)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 3.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 5.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(EPISODE_DURATION * 60))
    parser.add_argument(
        "--cloth-dataset",
        type=str,
        default=str(_CLOTH_DATA_DEFAULT),
        help="Path to cloth_export.npz. Defaults to cloth_data/cloth_1_2/newton_cloth/cloth_export.npz.",
    )
    parser.add_argument("--webxr", action="store_true", help="Enable WebXR teleoperation via Quest 2")
    parser.add_argument("--cert", default="webxr/cert.pem", help="TLS certificate for WebXR WSS server")
    parser.add_argument("--key", default="webxr/key.pem", help="TLS private key for WebXR WSS server")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
