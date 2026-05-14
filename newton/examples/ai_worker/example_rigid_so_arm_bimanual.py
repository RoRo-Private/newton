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
# Example Rigid SO-ARM 101 Bimanual VR
#
# Bimanual rigid-object manipulation using two SO-ARM 101 arms.
# A box (box_figjam) and a cylinder (cylinder_figjam) mesh object are placed
# on the table as manipulation targets.
# No cloth or soft-body physics — MuJoCo rigid contact pipeline only.
# Supports optional WebXR (Meta Quest 2) teleoperation.
#
# Command: uv run -m newton.examples rigid_so_arm_bimanual
#          uv run -m newton.examples rigid_so_arm_bimanual --webxr
###########################################################################

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import warp as wp

import trimesh

import newton
import newton.examples
from newton import ModelBuilder
from newton.solvers import SolverMuJoCo

# ---------------------------------------------------------------------------
# URDF path
# ---------------------------------------------------------------------------
_SO_ARM_BIMANUAL_URDF = (
    Path(__file__).resolve().parents[3] / "so_arm_description" / "so101_bimanual_set" / "so101_bimanual_set2.urdf"
)

# ---------------------------------------------------------------------------
# Joint layout (collapse_fixed_joints=True):
#   Left arm  [0..5]:  shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
#   Right arm [6..11]: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
#
# Body indices:
#   Left:  [0] shoulder [1] upper_arm [2] lower_arm [3] wrist [4] gripper [5] moving_jaw
#   Right: [6] shoulder [7] upper_arm [8] lower_arm [9] wrist [10] gripper [11] moving_jaw
# ---------------------------------------------------------------------------
_N_COORDS_PER_ARM = 6
STATE_DIM = _N_COORDS_PER_ARM * 2  # 12

_LEFT_EE_BODY  = 4
_RIGHT_EE_BODY = 10

_GRIP_OPEN  = 0.0
_GRIP_CLOSE = 1.5

_HOME_POSE = np.zeros(12, dtype=np.float32)
_HOME_POSE[3] = 90.0 * np.pi / 180.0
_HOME_POSE[9] = 90.0 * np.pi / 180.0

EPISODE_DURATION = 30.0  # [s]

# ---------------------------------------------------------------------------
# Floor geometry (matches URDF so101_bimanual floor collision box)
# Floor mesh: 1m x 3m x 0.01m, origin (0.3, 0., 0.0137943) in so101_bimanual link frame
# Top surface z = 0.0137943 + 0.005 = 0.0187943 m
# ---------------------------------------------------------------------------
_FLOOR_SURF_Z = 0.018  # [m] top surface of URDF floor collider

# ---------------------------------------------------------------------------
# Free rigid objects: box collision proxy + STL visual mesh
# ---------------------------------------------------------------------------
_MESH_DIR = Path(__file__).resolve().parents[3] / "so_arm_description" / "meshes"

# Collision box half-extents
_BOX_COLL_HZ = 0.015   # [m] box_figjam
_CYL_COLL_HZ = 0.010   # [m] cylinder_figjam

# Body z: collision box bottom sits on floor surface + 5 cm drop height
_BOX_Z_WORLD = _FLOOR_SURF_Z + _BOX_COLL_HZ + 0.05   # [m]
_CYL_Z_WORLD = _FLOOR_SURF_Z + _CYL_COLL_HZ + 0.05   # [m]

# Visual mesh local z offset (so STL bottom aligns with collision box bottom)
# box_figjam STL: z_min = -0.000375 in mesh space → offset = -(hz - 0.000375)
_BOX_VIS_Z = -(_BOX_COLL_HZ - 0.000375)   # ≈ -0.014625 [m]
# cylinder_figjam STL: z_min = -0.010 = -hz → no offset needed
_CYL_VIS_Z = 0.0

_OBJECT_CONFIGS: list[tuple[str, wp.transform]] = [
    ("box_figjam",      wp.transform(wp.vec3(0.15,  0.05, _BOX_Z_WORLD), wp.quat_identity())),
    ("cylinder_figjam", wp.transform(wp.vec3(0.15, -0.05, _CYL_Z_WORLD), wp.quat_identity())),
]


def _load_stl_mesh(name: str) -> newton.Mesh:
    """Load an STL file from the so_arm_description meshes directory."""
    m = trimesh.load(str(_MESH_DIR / f"{name}.stl"))
    return newton.Mesh(
        vertices=np.array(m.vertices, dtype=np.float32),
        indices=np.array(m.faces, dtype=np.int32).flatten(),
    )


# Contact material
_OBJ_DENSITY = 4000.0  # [kg/m³] ~heavy plastic / light metal; box≈182g, cylinder≈73g
_OBJ_KE = 2.0e3   # [N/m]
_OBJ_KD = 2.0e2   # [N·s/m]
_OBJ_MU = 0.9
_OBJ_MARGIN = 1e-3
_OBJ_MU_TORSIONAL = 0.02
_OBJ_MU_ROLLING = 0.01

# ---------------------------------------------------------------------------
# WebXR coordinate transform helpers
#
# Assumes user faces +X direction of the Newton world:
#   WebXR X (right)      → Newton -Y
#   WebXR Y (up)         → Newton +Z
#   WebXR Z (toward user)→ Newton -X
# ---------------------------------------------------------------------------
_Q_AXIS     = np.array([ 0.5, -0.5, -0.5, 0.5], dtype=np.float64)   # xyzw
_Q_AXIS_INV = np.array([-0.5,  0.5,  0.5, 0.5], dtype=np.float64)

_VR_STALE_TIMEOUT = 0.5   # [s]


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
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _xr_pos_to_newton(p: tuple) -> np.ndarray:
    x, y, z = p
    return np.array([-z, -x, y], dtype=np.float64)


def _xr_rot_delta_to_newton(q_ref: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    delta_xr = _quat_mul(_quat_conj(q_ref), q_cur)
    return _quat_mul(_quat_mul(_Q_AXIS, delta_xr), _Q_AXIS_INV)


def _quat_to_world_yaw(q_xyzw: np.ndarray) -> float:
    """Extract world-Z yaw [rad] from a quaternion in xyzw format."""
    x, y, z, w = q_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


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


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _set_arm_joint_q_kernel(
    ik_joint_q: wp.array(dtype=wp.float32),
    n_arm_coords: int,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
) -> None:
    """Directly overwrite arm DOF positions and zero their velocities."""
    i = wp.tid()
    if i < n_arm_coords:
        joint_q[i]  = ik_joint_q[i]
        joint_qd[i] = wp.float32(0.0)


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

class Example:
    """Bimanual SO-ARM 101 rigid-object manipulation with optional VR teleoperation."""

    def __init__(self, viewer, args=None):
        # ------------------------------------------------------------------
        # Simulation parameters
        # ------------------------------------------------------------------
        self.sim_substeps = 16
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self._profile = args is not None and getattr(args, "profile", False)
        self._prof_t_ik = 0.0
        self._prof_t_collide = 0.0
        self._prof_t_solver = 0.0
        self._prof_frames = 0

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


        # Free rigid objects: box collision proxy (fast, stable) + STL visual mesh.
        # Collision config: physics only, hidden from renderer.
        _obj_coll_cfg = ModelBuilder.ShapeConfig(
            density=_OBJ_DENSITY,
            ke=_OBJ_KE,
            kd=_OBJ_KD,
            mu=_OBJ_MU,
            margin=_OBJ_MARGIN,
            mu_torsional=_OBJ_MU_TORSIONAL,
            mu_rolling=_OBJ_MU_ROLLING,
            is_visible=False,
        )
        # Visual config: render only, no physics contribution.
        _obj_vis_cfg = ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        _obj_meshes = {name: _load_stl_mesh(name) for name, _ in _OBJECT_CONFIGS}
        self._object_body_indices = []

        # box_figjam: collision box hx=0.0195, hy=0.0195, hz=0.015
        _box_body = scene.add_body(xform=_OBJECT_CONFIGS[0][1], label=_OBJECT_CONFIGS[0][0])
        scene.add_shape_box(_box_body, hx=0.0195, hy=0.0195, hz=0.015, cfg=_obj_coll_cfg)
        scene.add_shape_mesh(
            _box_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, _BOX_VIS_Z), wp.quat_identity()),
            mesh=_obj_meshes["box_figjam"],
            cfg=_obj_vis_cfg,
        )
        self._object_body_indices.append(_box_body)

        # cylinder_figjam: collision box hx=0.017, hy=0.017, hz=0.010
        _cyl_body = scene.add_body(xform=_OBJECT_CONFIGS[1][1], label=_OBJECT_CONFIGS[1][0])
        scene.add_shape_box(_cyl_body, hx=0.017, hy=0.017, hz=0.010, cfg=_obj_coll_cfg)
        scene.add_shape_mesh(_cyl_body, mesh=_obj_meshes["cylinder_figjam"], cfg=_obj_vis_cfg)
        self._object_body_indices.append(_cyl_body)

        # Floor collision plane (body=-1: world-fixed, bypasses shape_contact_pairs filter).
        # Match URDF so101_bimanual_floor_floor.obj top surface:
        #   floor mesh is centered at z=0.0137943 with thickness 0.01 m
        #   -> top surface z = 0.0188 m.
        scene.add_shape_plane(
            body=-1,
            xform=wp.transform(wp.vec3(0.3, 0.0, _FLOOR_SURF_Z), wp.quat_identity()),
            width=1.0,
            length=3.0,
        )
        # Safety fallback: catch objects that fall off the floor edges.

        # ------------------------------------------------------------------
        # Finalize model
        # ------------------------------------------------------------------
        self.model = scene.finalize(requires_grad=False)
        print("[SO-ARM Rigid] body_label:", self.model.body_label)

        # Filter shape_contact_pairs to exclude non-EE robot links.
        # The explicit broad phase uses model.shape_contact_pairs directly, so
        # shape_flags alone cannot suppress mesh-plane triangle pair generation.
        # We build the set of shapes to exclude (non-EE robot links) and
        # drop any pair that involves one of those shapes.
        _ee_bodies = {_LEFT_EE_BODY, _LEFT_EE_BODY + 1, _RIGHT_EE_BODY, _RIGHT_EE_BODY + 1}
        # Only exclude non-EE robot links (body indices 0..STATE_DIM//6*2-1 = 0..11).
        # Free box bodies (indices >= 12) must keep their collision pairs.
        _robot_body_count = len(self.model.body_label) - len(_OBJECT_CONFIGS)
        _excluded_shapes: set[int] = set()
        for body_idx, shapes in self.model.body_shapes.items():
            if 0 <= body_idx < _robot_body_count and body_idx not in _ee_bodies:
                _excluded_shapes.update(shapes)

        if self.model.shape_contact_pairs is not None:
            _pairs_np = self.model.shape_contact_pairs.numpy()
            _mask = np.array(
                [p[0] not in _excluded_shapes and p[1] not in _excluded_shapes
                 for p in _pairs_np],
                dtype=bool,
            )
            _pairs_filtered = _pairs_np[_mask]
            self.model.shape_contact_pairs = wp.array(
                _pairs_filtered,
                dtype=self.model.shape_contact_pairs.dtype,
                device=self.model.shape_contact_pairs.device,
            )
            print(f"[SO-ARM Rigid] shape_contact_pairs: {len(_pairs_np)} → {len(_pairs_filtered)}")


        # ------------------------------------------------------------------
        # States and control
        # ------------------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # ------------------------------------------------------------------
        # Collision and solver
        # ------------------------------------------------------------------
        self._contacts = self.model.contacts()

        self._robot_solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=20,
            ls_iterations=50,
            nconmax=2048,
            njmax=4096,
            cone="elliptic",
            impratio=1000.0,
        )

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
        # Viewer
        # ------------------------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.3, 0.6, 0.8), -30.0, -50.0)

        for body_idx in self._object_body_indices:
            for s in self.model.body_shapes[body_idx]:
                self.viewer.update_shape_colors({s: wp.vec3(0.3, 0.6, 0.9)})
        
        # Settle boxes onto the table.
        # placo IK runs on CPU so CUDA graph capture is not used.
        _settle_frames = int(1.0 * self.fps)
        print(f"[SO-ARM Rigid] Settling {_settle_frames} frames...")
        for _ in range(_settle_frames):
            self.simulate()
        print("[SO-ARM Rigid] Settling done.")

        self._graph = None  # CUDA graph disabled: placo IK is a CPU op

        self._fps_count = 0
        self._fps_t0 = time.perf_counter()
        self._frame_saved = False

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

        left_delta  = _xr_pos_to_newton(np.array(left.position)  - c["left_xr_pos"])
        right_delta = _xr_pos_to_newton(np.array(right.position) - c["right_xr_pos"])

        left_pos  = c["left_newton_pos"]  + left_delta
        right_pos = c["right_newton_pos"] + right_delta

        left_rot_delta  = _xr_rot_delta_to_newton(c["left_xr_rot"],  np.array(left.quaternion_xyzw))
        right_rot_delta = _xr_rot_delta_to_newton(c["right_xr_rot"], np.array(right.quaternion_xyzw))

        left_rot  = _quat_mul(c["left_newton_rot"],  left_rot_delta)
        right_rot = _quat_mul(c["right_newton_rot"], right_rot_delta)
        left_rot  /= np.linalg.norm(left_rot)
        right_rot /= np.linalg.norm(right_rot)

        self._left_ee_tf  = wp.transform(wp.vec3(*left_pos.tolist()),  wp.quat(*left_rot.tolist()))
        self._right_ee_tf = wp.transform(wp.vec3(*right_pos.tolist()), wp.quat(*right_rot.tolist()))

        self.grip_left  = _GRIP_OPEN + left.grip  * (_GRIP_CLOSE - _GRIP_OPEN)
        self.grip_right = _GRIP_OPEN + right.grip * (_GRIP_CLOSE - _GRIP_OPEN)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        if self._profile:
            wp.synchronize()
            _t0 = time.perf_counter()

        # placo IK runs on CPU; result is pushed to GPU _ik_joint_q.
        self._solve_ik_and_push()

        if self._profile:
            wp.synchronize()
            self._prof_t_ik += time.perf_counter() - _t0

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self._profile:
                wp.synchronize()
                _tc0 = time.perf_counter()

            self.model.collide(self.state_0, self._contacts)

            if self._profile:
                wp.synchronize()
                self._prof_t_collide += time.perf_counter() - _tc0

            # Directly set arm joint positions and zero velocities.
            # Free-body DOFs are untouched and evolve under dynamics.
            wp.launch(
                _set_arm_joint_q_kernel,
                dim=STATE_DIM,
                inputs=[
                    self._ik_joint_q,
                    _N_COORDS_PER_ARM * 2,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                ],
                device=self.model.device,
            )

            if self._profile:
                wp.synchronize()
                _ts0 = time.perf_counter()

            self._robot_solver.step(
                self.state_0, self.state_1, self.control, self._contacts, self.sim_dt
            )

            if self._profile:
                wp.synchronize()
                self._prof_t_solver += time.perf_counter() - _ts0

            self.state_0, self.state_1 = self.state_1, self.state_0

        if self._profile:
            self._prof_frames += 1

    # ------------------------------------------------------------------
    # step / render
    # ------------------------------------------------------------------

    def step(self) -> None:
        if self._vr is not None:
            self._update_from_vr()
        else:
            self._update_roll_from_gizmos()
        self._update_filtered_ee_targets()
        # placo IK is solved inside simulate() / _solve_ik_and_push()
        if self._graph and not self._profile:
            wp.capture_launch(self._graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        self._fps_count += 1
        now = time.perf_counter()
        if now - self._fps_t0 >= 5.0:
            fps = self._fps_count / (now - self._fps_t0)
            if self._profile and self._prof_frames > 0:
                f = self._prof_frames
                print(
                    f"[Rigid SO-ARM] FPS: {fps:.1f} | "
                    f"ik+ctrl: {self._prof_t_ik/f*1e3:.2f} ms  "
                    f"collide: {self._prof_t_collide/f*1e3:.2f} ms  "
                    f"solver: {self._prof_t_solver/f*1e3:.2f} ms  "
                    f"(per frame, substeps={self.sim_substeps})"
                )
                self._prof_t_ik = self._prof_t_collide = self._prof_t_solver = 0.0
                self._prof_frames = 0
            self._fps_count = 0
            self._fps_t0 = now

    def render(self) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_gizmo("left_ee", self._left_ee_tf)
        self.viewer.log_gizmo("right_ee", self._right_ee_tf)
        self.viewer.log_state(self.state_0)

        # Visualize virtual camera poses as gizmos.
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        # Main camera (D455): fixed position above arm bases.
        self.viewer.log_gizmo("cam_main", wp.transform(
            wp.vec3(0.0, 0.0, 0.52),
            wp.quat_identity(),
        ))

        # Gripper cameras: use placo FK.
        q_np = self._ik_joint_q.numpy()
        for i, name in enumerate(self._left_joint_names):
            self._placo_robot_left.set_joint(name, float(q_np[i]))
        self._placo_robot_left.update_kinematics()
        for i, name in enumerate(self._right_joint_names):
            self._placo_robot_right.set_joint(name, float(q_np[_N_COORDS_PER_ARM + i]))
        self._placo_robot_right.update_kinematics()
        for robot, frame_name, cam_name in [
            (self._placo_robot_left,  "left_gripper_frame_link",  "cam_gripper_left"),
            (self._placo_robot_right, "right_gripper_frame_link", "cam_gripper_right"),
        ]:
            T = robot.get_T_world_frame(frame_name)
            q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
            self.viewer.log_gizmo(cam_name, wp.transform(
                wp.vec3(*T[:3, 3].tolist()),
                wp.quat(*q.tolist()),
            ))

        self.viewer.end_frame()

        if not self._frame_saved:
            self._frame_saved = True
            self._save_frames()

    def _render_and_save(self, cam, fname: str) -> None:
        """Render at cam's current resolution and save as PNG."""
        from PIL import Image  # noqa: PLC0415
        self.viewer.renderer.render(cam, self.viewer.objects, self.viewer.lines)
        img = self.viewer.get_frame().numpy()
        Image.fromarray(img).save(fname)
        print(f"[SO-ARM Rigid] Saved → {fname}")

    def _save_frames(self) -> None:
        """Save main camera (640×480, D455) + gripper cams (224×224, webcam) as PNG."""
        from pyglet.math import Vec3 as PyVec3  # noqa: PLC0415

        cam = self.viewer.camera

        # Save original state.
        orig_pos    = (cam.pos.x, cam.pos.y, cam.pos.z)
        orig_pitch  = cam.pitch
        orig_yaw    = cam.yaw
        orig_fov    = cam.fov
        orig_width  = cam.width
        orig_height = cam.height

        # -- Main camera: D455 640×480, FOV 58°, fixed at (0,0,0.52), pitch=-60° --
        cam.pos    = PyVec3(0.0, 0.0, 0.52)
        cam.pitch  = -60.0
        cam.yaw    = 0.0
        cam.fov    = 58.0
        cam.width  = 640
        cam.height = 480
        self._render_and_save(cam, "frame_main.png")

        # -- Gripper cam frames --
        # Get current gripper_frame world transforms via placo FK.
        q_np = self._ik_joint_q.numpy()
        for i, name in enumerate(self._left_joint_names):
            self._placo_robot_left.set_joint(name, float(q_np[i]))
        self._placo_robot_left.update_kinematics()
        for i, name in enumerate(self._right_joint_names):
            self._placo_robot_right.set_joint(name, float(q_np[_N_COORDS_PER_ARM + i]))
        self._placo_robot_right.update_kinematics()

        left_T  = self._placo_robot_left.get_T_world_frame("left_gripper_frame_link")
        right_T = self._placo_robot_right.get_T_world_frame("right_gripper_frame_link")

        def _T_to_pitch_yaw(T: np.ndarray) -> tuple[np.ndarray, float, float]:
            pos = T[:3, 3]
            forward = T[:3, 2]
            pitch = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
            yaw   = float(np.degrees(np.arctan2(forward[1], forward[0])))
            return pos, pitch, yaw

        # Wrist webcam: 224×224, FOV 140°.
        cam.fov    = 140.0
        cam.width  = 224
        cam.height = 224

        for T, fname in [(left_T, "frame_gripper_left.png"), (right_T, "frame_gripper_right.png")]:
            pos, pitch, yaw = _T_to_pitch_yaw(T)
            cam.pos   = PyVec3(*pos.tolist())
            cam.pitch = pitch
            cam.yaw   = yaw
            self._render_and_save(cam, fname)

        # Restore original state.
        cam.pos    = PyVec3(*orig_pos)
        cam.pitch  = orig_pitch
        cam.yaw    = orig_yaw
        cam.fov    = orig_fov
        cam.width  = orig_width
        cam.height = orig_height

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------

    def get_obs_state(self) -> np.ndarray:
        """Return 12-dim joint state [rad]."""
        wp.synchronize()
        return self.state_0.joint_q.numpy()[:STATE_DIM].astype(np.float32)

    def get_action(self) -> np.ndarray:
        """Return 12-dim absolute joint position action (current IK target). [rad]"""
        wp.synchronize()
        return self._ik_joint_q.numpy()[:STATE_DIM].astype(np.float32)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 10.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "free objects remain above ground",
            lambda q, qd: wp.transform_get_translation(q)[2] > -0.1,
            indices=self._object_body_indices,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(EPISODE_DURATION * 60))
    parser.add_argument("--profile", action="store_true", help="Print per-stage timing (disables CUDA graph)")
    parser.add_argument("--webxr", action="store_true", help="Enable WebXR teleoperation via Quest 2")
    parser.add_argument("--cert", default="webxr/cert.pem", help="TLS certificate for WebXR WSS server")
    parser.add_argument("--key", default="webxr/key.pem", help="TLS private key for WebXR WSS server")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
