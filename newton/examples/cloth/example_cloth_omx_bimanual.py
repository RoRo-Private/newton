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
# Example Cloth OMX Bimanual
#
# Bimanual pick-and-place simulation using two Open Manipulator X (OMX) arms
# and a cloth T-shirt, driven by a scripted joint-space controller.
#
# The simulation uses VBD for cloth and Featherstone for the robots.
# Joint state/action (10-dim):
#   [j1_l, j2_l, j3_l, j4_l, grip_l, j1_r, j2_r, j3_r, j4_r, grip_r]
#
# Command: python -m newton.examples cloth_omx_bimanual
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone, SolverVBD

# Bimanual URDF: both arms in a single file, single coordinate frame
# Left arm base:  xyz="0  0.15 0.10"
# Right arm base: xyz="0 -0.15 0.10"
_OMX_BIMANUAL_URDF = (
    Path(__file__).resolve().parents[3] / "omx_description" / "open_manipulator_x_bimanual.urdf"
)

# ---------------------------------------------------------------------------
# Joint layout after add_urdf (collapse_fixed_joints=True):
#   [0]  left_joint1            [1]  left_joint2
#   [2]  left_joint3            [3]  left_joint4
#   [4]  left_gripper_left_joint
#   [5]  left_gripper_right_joint  (mimic of [4])
#   [6]  right_joint1           [7]  right_joint2
#   [8]  right_joint3           [9]  right_joint4
#   [10] right_gripper_left_joint
#   [11] right_gripper_right_joint (mimic of [10])
# ---------------------------------------------------------------------------
_N_COORDS_PER_ARM = 6   # j1,j2,j3,j4,grip_left,grip_right
_N_STATE_PER_ARM  = 5   # j1,j2,j3,j4,grip_left  (grip_right excluded: mimic)
STATE_DIM = _N_STATE_PER_ARM * 2  # 10

# Body indices of the EE links (left_link5, right_link5).
# Gripper links are children of these and follow via FK.
_LEFT_EE_BODY  = 3
_RIGHT_EE_BODY = 9

# Gripper open/close joint position [m]
_GRIP_OPEN  =  0.019
_GRIP_CLOSE = -0.008

# Indices into 12-dim joint_q → 10-dim state/action
_LEFT_STATE_IDX  = [0, 1, 2, 3, 4]
_RIGHT_STATE_IDX = [6, 7, 8, 9, 10]
_STATE_IDX = _LEFT_STATE_IDX + _RIGHT_STATE_IDX

# ---------------------------------------------------------------------------
# Scripted key poses  [j1_l, j2_l, j3_l, j4_l, grip_l,
#                      j1_r, j2_r, j3_r, j4_r, grip_r]
# T-shirt centred at (0.40, 0, ~0.16) on a table.
# NOTE: Tune j2-j4 after inspecting in viewer.
# ---------------------------------------------------------------------------
_KEY_POSES = np.array(
    [
        # Phase 0 – Home: arms raised, gripper open
        [0.00, -0.80, 0.30, 0.80, 0.019,  0.00, -0.80, 0.30, 0.80, 0.019],
        # Phase 1 – Pre-grasp: hover above opposite shirt corners
        [-0.40, 0.25, 0.50, -0.20, 0.019, 0.40, 0.25, 0.50, -0.20, 0.019],
        # Phase 2 – Grasp: lower to cloth surface
        [-0.40, 0.55, 0.50, -0.50, 0.019, 0.40, 0.55, 0.50, -0.50, 0.019],
        # Phase 3 – Close grippers
        [-0.40, 0.55, 0.50, -0.50, -0.008, 0.40, 0.55, 0.50, -0.50, -0.008],
        # Phase 4 – Lift
        [-0.40, -0.30, 0.50, 0.30, -0.008, 0.40, -0.30, 0.50, 0.30, -0.008],
        # Phase 5 – Move to target (swing arms)
        [0.70, -0.30, 0.50, 0.30, -0.008, -0.70, -0.30, 0.50, 0.30, -0.008],
        # Phase 6 – Lower at target
        [0.70, 0.55, 0.50, -0.50, -0.008, -0.70, 0.55, 0.50, -0.50, -0.008],
        # Phase 7 – Release
        [0.70, 0.55, 0.50, -0.50, 0.019, -0.70, 0.55, 0.50, -0.50, 0.019],
        # Phase 8 – Return home
        [0.00, -0.80, 0.30, 0.80, 0.019,  0.00, -0.80, 0.30, 0.80, 0.019],
    ],
    dtype=np.float32,
)

_KEY_POSE_DURATIONS = np.array(
    [1.5, 2.0, 1.5, 0.8, 1.5, 2.0, 1.5, 0.8, 1.5], dtype=np.float32
)
_KEY_POSE_TIMES_CUM = np.cumsum(_KEY_POSE_DURATIONS)
EPISODE_DURATION = float(_KEY_POSE_TIMES_CUM[-1])


def _q10_to_q12(q10: np.ndarray) -> np.ndarray:
    """Expand 10-dim state to 12-dim joint_q, mirroring the mimic gripper joints."""
    q12 = np.empty(12, dtype=np.float32)
    q12[0:5]  = q10[0:5]   # left:  j1,j2,j3,j4,grip_left
    q12[5]    = q10[4]      # left:  grip_right = grip_left  (mimic)
    q12[6:11] = q10[5:10]  # right: j1,j2,j3,j4,grip_left
    q12[11]   = q10[9]      # right: grip_right = grip_left  (mimic)
    return q12


def _add_cross_arm_contact_pairs(model) -> None:
    """Add a single cross-arm contact pair: left gripper_right_link <-> right gripper_left_link.

    With enable_self_collisions=False, all intra-articulation shape pairs are
    excluded.  This function re-introduces only the inner-palm pair (the two
    gripper faces that can actually touch each other) to avoid the FPS cost of
    checking all 144 cross-arm shape combinations.
    """
    flags = model.shape_flags.numpy()

    def _collision_shapes(label_substr):
        bodies = [i for i, lbl in enumerate(model.body_label) if label_substr in lbl]
        # Keep only shapes with the COLLIDE_SHAPES flag (odd-indexed in practice)
        return [
            s
            for b in bodies
            for s in model.body_shapes.get(b, [])
            if flags[s] & newton.ShapeFlags.COLLIDE_SHAPES
        ]

    left_palm_shapes = _collision_shapes("/left_gripper_right_link")
    right_palm_shapes = _collision_shapes("/right_gripper_left_link")

    if not left_palm_shapes or not right_palm_shapes:
        return

    existing = model.shape_contact_pairs.numpy().tolist()
    existing_set = {(min(a, b), max(a, b)) for a, b in existing}

    new_pairs = list(existing)
    for ls in left_palm_shapes:
        for rs in right_palm_shapes:
            pair = (min(ls, rs), max(ls, rs))
            if pair not in existing_set:
                new_pairs.append([ls, rs])
                existing_set.add(pair)

    model.shape_contact_pairs = wp.array(
        np.array(new_pairs, dtype=np.int32), dtype=wp.vec2i, device=model.device
    )
    model.shape_contact_pair_count = len(new_pairs)


# ---------------------------------------------------------------------------
# WebXR → Newton coordinate transform
#
# Assumes the user stands facing the +X direction of the Newton world:
#   WebXR X (user right)  → Newton -Y
#   WebXR Y (user up)     → Newton +Z
#   WebXR Z (toward user) → Newton -X
#
# As a rotation quaternion (xyzw): [0.5, -0.5, -0.5, 0.5]
# (verified: det = +1, R = [[0,0,-1],[-1,0,0],[0,1,0]])
# ---------------------------------------------------------------------------
_Q_AXIS = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)   # xyzw
_Q_AXIS_INV = np.array([-0.5, 0.5, 0.5, 0.5], dtype=np.float64)  # conjugate

# Stale pose timeout [s]: ignore VR data older than this
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


@wp.kernel
def _warm_start_kernel(
    joint_q: wp.array(dtype=wp.float32),
    ik_joint_q: wp.array2d(dtype=wp.float32),
) -> None:
    """Copy joint_q → ik_joint_q[0, :] on GPU (avoids GPU→CPU→GPU round-trip)."""
    i = wp.tid()
    ik_joint_q[0, i] = joint_q[i]


@wp.kernel
def _set_gripper_kernel(
    grip_left: float,
    grip_right: float,
    ik_joint_q: wp.array2d(dtype=wp.float32),
) -> None:
    """Override gripper joints in ik_joint_q after IK solve (IK doesn't control them)."""
    ik_joint_q[0, 4] = grip_left
    ik_joint_q[0, 5] = grip_left    # mimic
    ik_joint_q[0, 10] = grip_right
    ik_joint_q[0, 11] = grip_right  # mimic


@wp.kernel
def _p_controller_kernel(
    ik_joint_q: wp.array2d(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    kp: float,
    qd_max: float,
    n_arm_coords: int,
    target_joint_qd: wp.array(dtype=wp.float32),
) -> None:
    """Compute P-controller velocities: qd = clamp(kp*(q_target - q_current), ±qd_max)."""
    i = wp.tid()
    if i < n_arm_coords:
        target_joint_qd[i] = wp.clamp(kp * (ik_joint_q[0, i] - joint_q[i]), -qd_max, qd_max)
    else:
        target_joint_qd[i] = wp.float32(0.0)


class Example:
    def __init__(self, viewer, args=None):
        # ------------------------------------------------------------------
        # Simulation parameters
        # ------------------------------------------------------------------
        self.sim_substeps = 15
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

        # --- Both arms from a single bimanual URDF ---
        # left_world_fixed  (xyz="0  0.15 0.10") and
        # right_world_fixed (xyz="0 -0.15 0.10") are collapsed automatically.
        scene.add_urdf(
            str(_OMX_BIMANUAL_URDF),
            xform=wp.transform_identity(),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            force_show_colliders=True,
        )
        # Set home pose for both arms
        q12_home = _q10_to_q12(_KEY_POSES[0])
        scene.joint_q[:_N_COORDS_PER_ARM * 2] = q12_home.tolist()

        # --- Table ---
        # Top at z=0.04.
        # x range: [-0.05, 0.75] covers arm bases (x=0) and shirt (x=[0.074, 0.725])
        # center x=0.35, hx=0.40
        scene.add_shape_box(
            -1,
            wp.transform(wp.vec3(0.35, 0.0, 0.02), wp.quat_identity()),
            hx=0.40,
            hy=0.35,
            hz=0.02,
        )

        # --- T-shirt cloth ---
        # Mesh bbox at scale=0.01:
        #   X: [-0.326, 0.325], Y: [0.880, 1.526] (center ~1.205), Z: [-0.201, 0.074]
        # After rot=π around Z: Y -> -Y
        # pos_x=0.40 → shirt X [0.074, 0.725] (clears arm bases at x=0)
        # pos_y=1.205 → world Y center = 0
        # pos_z=0.32  → shirt bottom at z=0.12 (2 cm above table top)
        _SHIRT_POS_Y = 1.205
        _SHIRT_POS_Z = 0.22

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        vertices = [wp.vec3(v) for v in shirt_mesh.vertices]

        scene.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.40, _SHIRT_POS_Y, _SHIRT_POS_Z),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            scale=0.01,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )
        scene.color()

        scene.add_ground_plane()

        # ------------------------------------------------------------------
        # Finalize model
        # ------------------------------------------------------------------
        self.model = scene.finalize(requires_grad=False)

        # Add cross-arm (left <-> right) shape contact pairs that were excluded
        # by enable_self_collisions=False on the bimanual URDF.
        # _add_cross_arm_contact_pairs(self.model)

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
        self._current_action = _KEY_POSES[0].copy()

        # ------------------------------------------------------------------
        # Collision pipelines
        # ------------------------------------------------------------------
        # Rigid pipeline: robot-table contacts (no soft margin needed)
        self._rigid_pipeline = newton.CollisionPipeline(self.model)
        self._rigid_contacts = self._rigid_pipeline.contacts()

        # Cloth pipeline: cloth-particle vs rigid shape contacts
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

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self._graph = capture.graph
        else:
            self._graph = None

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
        self._vr_triggering = False  # edge-detect: avoid repeated calibration on hold

    def _calibrate_vr(self, left, right) -> None:
        """Capture current controller and EE poses as calibration reference."""
        body_q = self.state_0.body_q.numpy()

        self._vr_calib = {
            # WebXR reference positions/rotations at calibration time
            "left_xr_pos":  np.array(left.position,        dtype=np.float64),
            "right_xr_pos": np.array(right.position,       dtype=np.float64),
            "left_xr_rot":  np.array(left.quaternion_xyzw, dtype=np.float64),
            "right_xr_rot": np.array(right.quaternion_xyzw,dtype=np.float64),
            # Newton EE home poses at calibration time
            "left_newton_pos":  body_q[_LEFT_EE_BODY, :3].copy(),
            "right_newton_pos": body_q[_RIGHT_EE_BODY, :3].copy(),
            "left_newton_rot":  body_q[_LEFT_EE_BODY, 3:].copy(),   # xyzw
            "right_newton_rot": body_q[_RIGHT_EE_BODY, 3:].copy(),
        }
        self._vr_calibrated = True
        print("[VR] Calibrated — release trigger to start controlling.")

    def _update_from_vr(self) -> None:
        """Read latest VR poses and update IK targets + gripper state."""
        import time
        left  = self._vr.get_left()
        right = self._vr.get_right()

        now = time.monotonic()
        # Ignore stale data (e.g. Quest asleep or disconnected)
        if now - left.received > _VR_STALE_TIMEOUT or now - right.received > _VR_STALE_TIMEOUT:
            return

        # ── calibration on trigger rising edge ───────────────────────────
        trigger_pressed = left.trigger > 0.5 or right.trigger > 0.5
        if trigger_pressed and not self._vr_triggering:
            self._calibrate_vr(left, right)
        self._vr_triggering = trigger_pressed

        if not self._vr_calibrated:
            return

        c = self._vr_calib

        # ── position: home + mapped delta from calibration origin ────────
        left_delta_newton  = _xr_pos_to_newton(np.array(left.position)  - c["left_xr_pos"])
        right_delta_newton = _xr_pos_to_newton(np.array(right.position) - c["right_xr_pos"])

        left_pos  = c["left_newton_pos"]  + left_delta_newton
        right_pos = c["right_newton_pos"] + right_delta_newton

        # ── rotation: home rotation + delta from calibration orientation ─
        left_rot_delta  = _xr_rot_delta_to_newton(c["left_xr_rot"],  np.array(left.quaternion_xyzw))
        right_rot_delta = _xr_rot_delta_to_newton(c["right_xr_rot"], np.array(right.quaternion_xyzw))

        left_rot  = _quat_mul(c["left_newton_rot"],  left_rot_delta)
        right_rot = _quat_mul(c["right_newton_rot"], right_rot_delta)

        # Normalize (accumulation may drift)
        left_rot  /= np.linalg.norm(left_rot)
        right_rot /= np.linalg.norm(right_rot)

        # ── update IK targets ─────────────────────────────────────────────
        self._left_ee_tf = wp.transform(
            wp.vec3(*left_pos.tolist()),
            wp.quat(*left_rot.tolist()),   # xyzw
        )
        self._right_ee_tf = wp.transform(
            wp.vec3(*right_pos.tolist()),
            wp.quat(*right_rot.tolist()),
        )

        # ── gripper: analog grip → joint position ─────────────────────────
        self.grip_left  = _GRIP_OPEN + left.grip  * (_GRIP_CLOSE - _GRIP_OPEN)
        self.grip_right = _GRIP_OPEN + right.grip * (_GRIP_CLOSE - _GRIP_OPEN)

    # ------------------------------------------------------------------
    # IK setup
    # ------------------------------------------------------------------

    def _setup_ik(self) -> None:
        body_q_np = self.state_0.body_q.numpy()

        # Gizmo transforms — viewer mutates these in-place on drag
        self._left_ee_tf  = wp.transform(*body_q_np[_LEFT_EE_BODY])
        self._right_ee_tf = wp.transform(*body_q_np[_RIGHT_EE_BODY])

        def _rot4(tf):
            q = wp.transform_get_rotation(tf)
            return wp.vec4(q[0], q[1], q[2], q[3])

        self._left_pos_obj = ik.IKObjectivePosition(
            link_index=_LEFT_EE_BODY,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self._left_ee_tf)], dtype=wp.vec3),
        )
        self._left_rot_obj = ik.IKObjectiveRotation(
            link_index=_LEFT_EE_BODY,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_rot4(self._left_ee_tf)], dtype=wp.vec4),
        )
        self._right_pos_obj = ik.IKObjectivePosition(
            link_index=_RIGHT_EE_BODY,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.transform_get_translation(self._right_ee_tf)], dtype=wp.vec3),
        )
        self._right_rot_obj = ik.IKObjectiveRotation(
            link_index=_RIGHT_EE_BODY,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_rot4(self._right_ee_tf)], dtype=wp.vec4),
        )
        self._joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self._ik_joint_q = wp.array(
            self.model.joint_q.numpy().reshape(1, -1),
            shape=(1, self.model.joint_coord_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        self._ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                self._left_pos_obj, self._left_rot_obj,
                self._right_pos_obj, self._right_rot_obj,
                self._joint_limit_obj,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._ik_iters = 8

        # Gripper open/close state per arm (toggled externally, e.g. WebXR)
        self.grip_left  = _GRIP_OPEN
        self.grip_right = _GRIP_OPEN

    def _run_ik(self) -> np.ndarray:
        """Solve IK from current gizmo targets. Returns 12-dim joint_q target."""
        # Push gizmo targets into objectives (small CPU→GPU writes, unavoidable)
        lp = wp.transform_get_translation(self._left_ee_tf)
        lq = wp.transform_get_rotation(self._left_ee_tf)
        self._left_pos_obj.set_target_position(0, lp)
        self._left_rot_obj.set_target_rotation(0, wp.vec4(lq[0], lq[1], lq[2], lq[3]))

        rp = wp.transform_get_translation(self._right_ee_tf)
        rq = wp.transform_get_rotation(self._right_ee_tf)
        self._right_pos_obj.set_target_position(0, rp)
        self._right_rot_obj.set_target_rotation(0, wp.vec4(rq[0], rq[1], rq[2], rq[3]))

        # Warm-start: GPU→GPU copy, no CPU round-trip
        wp.launch(
            _warm_start_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.state_0.joint_q, self._ik_joint_q],
            device=self.model.device,
        )

        self._ik_solver.step(self._ik_joint_q, self._ik_joint_q, iterations=self._ik_iters)

        # Override gripper joints on GPU before reading back
        wp.launch(
            _set_gripper_kernel,
            dim=1,
            inputs=[self.grip_left, self.grip_right, self._ik_joint_q],
            device=self.model.device,
        )
        wp.synchronize()

        return self._ik_joint_q.numpy()[0].copy()

    # ------------------------------------------------------------------
    # Public helpers for data collection
    # ------------------------------------------------------------------

    def get_obs_state(self) -> np.ndarray:
        """Return 10-dim joint state [j1_l..grip_l, j1_r..grip_r]. [rad or m]"""
        q = self.state_0.joint_q.numpy()
        return q[_STATE_IDX].astype(np.float32)

    def get_action(self) -> np.ndarray:
        """Return 10-dim absolute joint position action (current controller target)."""
        return self._current_action.copy()

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _generate_control(self) -> None:
        q12 = self._run_ik()

        # Update 10-dim action for data collection (exclude mimic joints)
        self._current_action = np.concatenate([q12[0:5], q12[6:11]]).astype(np.float32)

        # P-controller on GPU: reads q_target from _ik_joint_q, q_current from state_0.joint_q
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

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._cloth_solver.rebuild_bvh(self.state_0)
        particle_count_saved = self.model.particle_count
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Robot step: no particles, no gravity, but rigid-rigid contacts enabled
            self.model.particle_count = 0
            self.model.gravity.assign(self._gravity_zero)

            self._rigid_pipeline.collide(self.state_0, self._rigid_contacts)
            self.state_0.joint_qd.assign(self._target_joint_qd)
            self._robot_solver.step(self.state_0, self.state_1, self.control, self._rigid_contacts, self.sim_dt)

            self.state_0.particle_f.zero_()

            # Cloth step
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
        self._generate_control()
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
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-0.1, -0.5, -0.05)
        p_upper = wp.vec3(0.8, 0.5, 0.50)
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
    parser.add_argument("--webxr", action="store_true", help="Enable WebXR teleoperation via Quest 2")
    parser.add_argument("--cert", default="webxr/cert.pem", help="TLS certificate for WebXR WSS server")
    parser.add_argument("--key", default="webxr/key.pem", help="TLS private key for WebXR WSS server")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
