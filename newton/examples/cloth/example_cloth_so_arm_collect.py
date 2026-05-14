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
# Example Cloth SO-ARM 101 Collect
#
# Robot-only benchmark: runs bimanual arm control (IK + Featherstone)
# without any cloth (VBD solver and cloth collision pipeline removed).
# Supports two control modes via --control-mode:
#   ik       (default): IK solve + P-controller → joint velocity
#   jacobian          : EE spring-damper → body_f (Jacobian transpose via Featherstone)
#
# Command: uv run -m newton.examples cloth_so_arm_collect
#          uv run -m newton.examples cloth_so_arm_collect --webxr
#          uv run -m newton.examples cloth_so_arm_collect --control-mode jacobian
###########################################################################

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ModelBuilder
from newton.solvers import SolverFeatherstone
from newton.examples.cloth.example_cloth_so_arm_bimanual import (
    EPISODE_DURATION,
    Example as _BimanualExample,
    _SO_ARM_BIMANUAL_URDF,
    STATE_DIM,
    _HOME_POSE,
    _LEFT_EE_BODY,
    _RIGHT_EE_BODY,
    _N_COORDS_PER_ARM,
    _warm_start_kernel,
    _set_gripper_kernel,
    _p_controller_kernel,
)

# ---------------------------------------------------------------------------
# Jacobian transpose control kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _apply_ee_spring_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    ee_targets: wp.array(dtype=wp.vec3),
    ee_bodies: wp.array(dtype=wp.int32),
    kp: float,
    kd: float,
) -> None:
    """Apply Cartesian spring-damper force to each EE body toward its target."""
    i = wp.tid()  # 0 = left EE, 1 = right EE
    body_idx = ee_bodies[i]
    pos = wp.transform_get_translation(body_q[body_idx])
    lin_vel = wp.spatial_top(body_qd[body_idx])
    force = kp * (ee_targets[i] - pos) - kd * lin_vel
    wp.atomic_add(body_f, body_idx, wp.spatial_vector(force, wp.vec3(0.0, 0.0, 0.0)))


@wp.kernel
def _set_gripper_qd_kernel(
    grip_buf: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    kp: float,
    qd_max: float,
    joint_qd: wp.array(dtype=wp.float32),
) -> None:
    """P-control gripper joints only (indices 5 and 11); leave arm joints untouched."""
    joint_qd[5]  = wp.clamp(kp * (grip_buf[0] - joint_q[5]),  -qd_max, qd_max)
    joint_qd[11] = wp.clamp(kp * (grip_buf[1] - joint_q[11]), -qd_max, qd_max)


class Example(_BimanualExample):
    """Robot-only benchmark with selectable control mode (ik / jacobian)."""

    def __init__(self, viewer, args=None):
        self._control_mode = getattr(args, "control_mode", "ik") if args else "ik"

        # ------------------------------------------------------------------
        # Simulation parameters
        # ------------------------------------------------------------------
        self.sim_substeps = 8
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.kp = 3.0
        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build scene (robot + table + ground, no cloth)
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

        scene.add_shape_box(
            -1,
            wp.transform(wp.vec3(0.35, 0.0, 0.01), wp.quat_identity()),
            hx=0.40,
            hy=0.35,
            hz=0.01,
        )

        scene.add_ground_plane()

        # ------------------------------------------------------------------
        # Finalize
        # ------------------------------------------------------------------
        self.model = scene.finalize(requires_grad=False)
        print("[SO-ARM] body_label:", self.model.body_label)

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
        # Collision pipeline (rigid only)
        # ------------------------------------------------------------------
        self._rigid_pipeline = newton.CollisionPipeline(self.model)
        self._rigid_contacts = self._rigid_pipeline.contacts()

        # ------------------------------------------------------------------
        # Solver (Featherstone only)
        # ------------------------------------------------------------------
        self._robot_solver = SolverFeatherstone(
            self.model, update_mass_matrix_interval=self.sim_substeps
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ------------------------------------------------------------------
        # IK solver (used in both modes: jacobian mode still needs IK state
        # for EE pose tracking and gripper)
        # ------------------------------------------------------------------
        self._setup_ik()

        # ------------------------------------------------------------------
        # Jacobian mode: EE target position array (updated before graph launch)
        # ------------------------------------------------------------------
        self._ee_target_pos = wp.zeros(2, dtype=wp.vec3, device=self.model.device)
        self._ee_bodies = wp.array(
            [_LEFT_EE_BODY, _RIGHT_EE_BODY], dtype=wp.int32, device=self.model.device
        )
        self._jac_kp = getattr(args, "jac_kp", 500.0) if args else 500.0
        self._jac_kd = getattr(args, "jac_kd", 50.0) if args else 50.0

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

        # FPS counter
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # simulate() dispatches to the selected control mode
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        if self._control_mode == "jacobian":
            self._simulate_jacobian()
        else:
            self._simulate_ik()

    def _simulate_ik(self) -> None:
        """IK solve + P-controller → joint velocity (original bimanual logic)."""
        wp.launch(
            _warm_start_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.state_0.joint_q, self._ik_joint_q],
            device=self.model.device,
        )
        self._ik_solver.step(self._ik_joint_q, self._ik_joint_q, iterations=self._ik_iters)
        wp.launch(
            _set_gripper_kernel,
            dim=1,
            inputs=[self._grip_buf, self._ik_joint_q],
            device=self.model.device,
        )
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

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self._rigid_pipeline.collide(self.state_0, self._rigid_contacts)
            self.state_0.joint_qd.assign(self._target_joint_qd)
            self._robot_solver.step(
                self.state_0, self.state_1, self.control, self._rigid_contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _simulate_jacobian(self) -> None:
        """EE Cartesian spring-damper → body_f (Jacobian transpose via Featherstone).
        Arm joints evolve freely under dynamics; gripper joints are P-controlled.
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # EE spring-damper force → body_f (Featherstone converts via J^T internally)
            wp.launch(
                _apply_ee_spring_kernel,
                dim=2,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self._ee_target_pos,
                    self._ee_bodies,
                    self._jac_kp,
                    self._jac_kd,
                ],
                device=self.model.device,
            )

            # Gripper P-control (only joints 5 and 11; arm joints left free)
            wp.launch(
                _set_gripper_qd_kernel,
                dim=1,
                inputs=[
                    self._grip_buf,
                    self.state_0.joint_q,
                    self.kp,
                    4.8,
                    self.state_0.joint_qd,
                ],
                device=self.model.device,
            )

            self._rigid_pipeline.collide(self.state_0, self._rigid_contacts)
            self._robot_solver.step(
                self.state_0, self.state_1, self.control, self._rigid_contacts, self.sim_dt
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    # ------------------------------------------------------------------
    # _push_ik_targets: in jacobian mode, push EE positions to warp array
    # ------------------------------------------------------------------

    def _push_ik_targets(self) -> None:
        if self._control_mode == "jacobian":
            lp = wp.transform_get_translation(self._left_ee_tf)
            rp = wp.transform_get_translation(self._right_ee_tf)
            self._ee_target_pos.assign(
                np.array([[lp[0], lp[1], lp[2]], [rp[0], rp[1], rp[2]]], dtype=np.float32)
            )
            self._grip_buf.assign(
                np.array([self.grip_left, self.grip_right], dtype=np.float32)
            )
        else:
            super()._push_ik_targets()

    # ------------------------------------------------------------------
    # step / render
    # ------------------------------------------------------------------

    def step(self) -> None:
        if self._vr is not None:
            self._update_from_vr()
        self._push_ik_targets()
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
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(EPISODE_DURATION * 60))
    parser.add_argument("--webxr", action="store_true")
    parser.add_argument("--cert", default="webxr/cert.pem")
    parser.add_argument("--key", default="webxr/key.pem")
    parser.add_argument(
        "--control-mode",
        choices=["ik", "jacobian"],
        default="ik",
        help="Control mode: 'ik' (IK+P-controller) or 'jacobian' (EE spring-damper via body_f)",
    )
    parser.add_argument("--jac-kp", type=float, default=500.0, help="Jacobian mode: EE position stiffness [N/m]")
    parser.add_argument("--jac-kd", type=float, default=50.0, help="Jacobian mode: EE velocity damping [N·s/m]")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
