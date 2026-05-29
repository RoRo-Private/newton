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
# Example Basic Coupler Demo
#
# Lightweight side-by-side visualization of a Legacy-like coupler:
# - Left cluster: no coupler correction
# - Right cluster: coupler correction against a rigid proxy plane
#
# Command:
#   uv run python -m newton.examples basic_coupler_demo --viewer gl --device cuda:0
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import CouplerOptions, LegacyLikeCoupler, RigidCouplingMaterial, SolverXPBD


@wp.kernel
def _couple_particles_with_plane_kernel(
    particle_indices: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    plane_z: float,
    coup_friction: float,
    coup_softness: float,
    coup_restitution: float,
    eps: float,
):
    tid = wp.tid()
    i = particle_indices[tid]

    pos = particle_q[i]
    vel = particle_qd[i]

    signed_distance = pos[2] - plane_z
    if signed_distance > coup_softness * 3.0:
        return

    softness = wp.max(coup_softness, 1.0e-10)
    influence = wp.min(wp.exp(-signed_distance / softness), 1.0)
    if influence <= 0.1:
        return

    # Proxy rigid plane has zero velocity and world normal +Z.
    rvel_n_mag = vel[2]
    if rvel_n_mag >= 0.0:
        return

    rvel_tan = wp.vec3(vel[0], vel[1], 0.0)
    tan_norm = wp.max(wp.sqrt(rvel_tan[0] * rvel_tan[0] + rvel_tan[1] * rvel_tan[1]), eps)
    tan_after = wp.max(0.0, tan_norm + rvel_n_mag * coup_friction)
    rvel_tan = rvel_tan * (tan_after / tan_norm)

    rvel_normal = wp.vec3(0.0, 0.0, -rvel_n_mag * coup_restitution)
    rvel_new = rvel_tan + rvel_normal
    new_vel = rvel_new * influence + vel * (1.0 - influence)
    particle_qd[i] = new_vel

    # Positional correction to avoid visible tunneling.
    if pos[2] < plane_z:
        particle_q[i] = wp.vec3(pos[0], pos[1], plane_z)


@wp.kernel
def _couple_particles_with_box_kernel(
    particle_indices: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    box_center: wp.vec3,
    box_half_extents: wp.vec3,
    coup_friction: float,
    coup_softness: float,
    coup_restitution: float,
    eps: float,
):
    tid = wp.tid()
    i = particle_indices[tid]

    pos = particle_q[i]
    vel = particle_qd[i]
    rel = pos - box_center
    a = wp.abs(rel)
    he = box_half_extents

    clamped = wp.vec3(
        wp.clamp(rel[0], -he[0], he[0]),
        wp.clamp(rel[1], -he[1], he[1]),
        wp.clamp(rel[2], -he[2], he[2]),
    )
    closest = box_center + clamped
    delta = pos - closest
    delta_n = wp.length(delta)
    inside = (a[0] <= he[0]) and (a[1] <= he[1]) and (a[2] <= he[2])

    normal = wp.vec3(0.0, 0.0, 1.0)
    signed_distance = 1.0e6

    if inside:
        dx = he[0] - a[0]
        dy = he[1] - a[1]
        dz = he[2] - a[2]
        if dx <= dy and dx <= dz:
            normal = wp.vec3(wp.sign(rel[0]), 0.0, 0.0)
            signed_distance = -dx
        elif dy <= dz:
            normal = wp.vec3(0.0, wp.sign(rel[1]), 0.0)
            signed_distance = -dy
        else:
            normal = wp.vec3(0.0, 0.0, wp.sign(rel[2]))
            signed_distance = -dz
    else:
        if delta_n < eps:
            return
        normal = delta / delta_n
        signed_distance = delta_n

    if signed_distance > coup_softness * 3.0:
        return

    softness = wp.max(coup_softness, 1.0e-10)
    influence = wp.min(wp.exp(-signed_distance / softness), 1.0)
    if influence <= 0.1:
        return

    # Proxy rigid box has zero velocity.
    rvel_n_mag = wp.dot(vel, normal)
    if rvel_n_mag >= 0.0:
        return

    rvel_tan = vel - normal * rvel_n_mag
    tan_norm = wp.max(wp.length(rvel_tan), eps)
    tan_after = wp.max(0.0, tan_norm + rvel_n_mag * coup_friction)
    rvel_tan = rvel_tan * (tan_after / tan_norm)

    rvel_normal = normal * (-rvel_n_mag * coup_restitution)
    rvel_new = rvel_tan + rvel_normal
    new_vel = rvel_new * influence + vel * (1.0 - influence)
    particle_qd[i] = new_vel

    if inside:
        # Push to the box surface to avoid persistent interpenetration.
        particle_q[i] = pos - normal * signed_distance


class _ParticlePlaneCouplerAdapter:
    """Couples selected particles with a fixed proxy rigid plane."""

    def __init__(
        self,
        sim_dt: float,
        state_out: newton.State,
        particle_indices: np.ndarray,
        plane_z: float,
        material: RigidCouplingMaterial,
        proxy_type: str = "plane",
        box_center: tuple[float, float, float] = (0.38, -0.02, 0.15),
        box_half_extents: tuple[float, float, float] = (0.18, 0.18, 0.03),
        eps: float = 1.0e-12,
    ):
        self._sim_dt = sim_dt
        self._state_out = state_out
        self._particle_indices = wp.array(
            particle_indices,
            dtype=wp.int32,
            device=state_out.particle_q.device,
        )
        self._plane_z = plane_z
        self._proxy_type = proxy_type
        self._box_center = wp.vec3(*box_center)
        self._box_half_extents = wp.vec3(*box_half_extents)
        self._material = material
        self._eps = eps
        self._coupler: LegacyLikeCoupler | None = None

    def set_coupler(self, coupler: LegacyLikeCoupler):
        self._coupler = coupler

    def is_rigid_active(self) -> bool:
        return True

    def is_xpbd_active(self) -> bool:
        return True

    def substep_dt(self) -> float:
        return self._sim_dt

    def rigid_velocity_at_point(self, rigid_body_id: int, pos_world: np.ndarray) -> np.ndarray:
        _ = rigid_body_id, pos_world
        return np.zeros(3, dtype=np.float64)

    def apply_coupling_force(self, rigid_body_id: int, force_world: np.ndarray, at_pos_world: np.ndarray) -> None:
        _ = rigid_body_id, force_world, at_pos_world

    def couple_xpbd_with_rigid(self) -> None:
        if self._coupler is None:
            return

        if self._proxy_type == "box":
            wp.launch(
                _couple_particles_with_box_kernel,
                dim=len(self._particle_indices),
                inputs=[
                    self._particle_indices,
                    self._state_out.particle_q,
                    self._state_out.particle_qd,
                    self._box_center,
                    self._box_half_extents,
                    self._material.coup_friction,
                    self._material.coup_softness,
                    self._material.coup_restitution,
                    self._eps,
                ],
            )
        else:
            wp.launch(
                _couple_particles_with_plane_kernel,
                dim=len(self._particle_indices),
                inputs=[
                    self._particle_indices,
                    self._state_out.particle_q,
                    self._state_out.particle_qd,
                    self._plane_z,
                    self._material.coup_friction,
                    self._material.coup_softness,
                    self._material.coup_restitution,
                    self._eps,
                ],
            )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.reset_interval_frames = 240

        self.use_coupler = args.use_coupler
        self.compare_side_by_side = args.compare_side_by_side
        self.proxy_type = args.proxy_type

        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.04

        left_idx: list[int] = []
        right_idx: list[int] = []

        # Two lightweight 4x4 particle clusters.
        nx = 4
        ny = 4
        spacing = 0.09
        z0 = 0.9
        y0 = -0.2

        # Left: baseline (no coupler)
        for ix in range(nx):
            for iy in range(ny):
                p = wp.vec3(-0.45 + ix * spacing, y0 + iy * spacing, z0)
                i = builder.add_particle(
                    p,
                    wp.vec3(0.0, 0.0, -0.2),
                    mass=1.0,
                    flags=int(newton.ParticleFlags.ACTIVE),
                )
                left_idx.append(i)

        # Right: coupler candidate
        for ix in range(nx):
            for iy in range(ny):
                p = wp.vec3(0.20 + ix * spacing, y0 + iy * spacing, z0)
                i = builder.add_particle(
                    p,
                    wp.vec3(0.0, 0.0, -0.2),
                    mass=1.0,
                    flags=int(newton.ParticleFlags.ACTIVE),
                )
                right_idx.append(i)

        self.model = builder.finalize()
        self.model.particle_grid = None
        self.model.particle_max_radius = 0.0

        self.solver = SolverXPBD(self.model, iterations=2)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._left_idx = np.array(left_idx, dtype=np.int32)
        self._right_idx = np.array(right_idx, dtype=np.int32)

        self._coupler_enabled_indices = (
            self._right_idx if self.compare_side_by_side else np.concatenate([self._left_idx, self._right_idx])
        )

        self._plane_z = 0.15
        self._coupler = None
        if self.use_coupler:
            material = RigidCouplingMaterial(
                needs_coup=True,
                coup_friction=0.25,
                coup_softness=0.08,
                coup_restitution=0.2,
            )
            options = CouplerOptions(rigid_xpbd=True)
            adapter = _ParticlePlaneCouplerAdapter(
                sim_dt=self.sim_dt,
                state_out=self.state_1,
                particle_indices=self._coupler_enabled_indices,
                plane_z=self._plane_z,
                material=material,
                proxy_type=self.proxy_type,
                eps=options.eps,
            )
            self._coupler = LegacyLikeCoupler(adapter, options)
            adapter.set_coupler(self._coupler)
            self._coupler.build()

        self.viewer.set_model(self.model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.pos = type(self.viewer.camera.pos)(1.2, -1.6, 0.9)
            self.viewer.camera.pitch = 20.0
            self.viewer.camera.yaw = -35.0

        self._init_q = self.state_0.particle_q.numpy().copy()
        self._init_qd = self.state_0.particle_qd.numpy().copy()

        dev = wp.get_device()
        print(
            f"[coupler_demo] warp_device={dev} is_cuda={dev.is_cuda} "
            f"use_coupler={self.use_coupler} proxy_type={self.proxy_type}"
        )

    def _reset_scene(self) -> None:
        self.state_0.particle_q.assign(self._init_q)
        self.state_0.particle_qd.assign(self._init_qd)
        self.state_1.particle_q.assign(self._init_q)
        self.state_1.particle_qd.assign(self._init_qd)

    def step(self):
        for substep_idx in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if self._coupler is not None:
                self._coupler.preprocess(substep_idx)
                self._coupler.couple(substep_idx)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame += 1

        # Print a coarse metric so the interaction is numerically visible too.
        if self.frame % 30 == 0:
            q = self.state_0.particle_q.numpy()
            z_left = float(q[self._left_idx, 2].mean())
            z_right = float(q[self._right_idx, 2].mean())
            print(f"[coupler_demo] frame={self.frame:04d} z_left={z_left:+.3f} z_right={z_right:+.3f}")

        if self.frame % self.reset_interval_frames == 0:
            self._reset_scene()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Explicit particle visualization to avoid viewer-dependent particle rendering.
        q = self.state_0.particle_q.numpy()
        left_tf = wp.array(
            [wp.transform(wp.vec3(*q[i]), wp.quat_identity()) for i in self._left_idx],
            dtype=wp.transform,
        )
        right_tf = wp.array(
            [wp.transform(wp.vec3(*q[i]), wp.quat_identity()) for i in self._right_idx],
            dtype=wp.transform,
        )
        self.viewer.log_shapes(
            "/particles_left",
            newton.GeoType.SPHERE,
            0.03,
            left_tf,
            wp.array([wp.vec3(0.2, 0.7, 1.0)] * len(self._left_idx), dtype=wp.vec3),
        )
        self.viewer.log_shapes(
            "/particles_right",
            newton.GeoType.SPHERE,
            0.03,
            right_tf,
            wp.array([wp.vec3(0.2, 1.0, 0.3)] * len(self._right_idx), dtype=wp.vec3),
        )

        # Visual guides for the proxy plane under each cluster.
        if self.proxy_type == "box":
            xform = wp.array(
                [wp.transform(wp.vec3(0.38, -0.02, self._plane_z), wp.quat_identity())], dtype=wp.transform
            )
            self.viewer.log_shapes(
                "/proxy_rigid_box",
                newton.GeoType.BOX,
                (0.18, 0.18, 0.03),
                xform,
                wp.array([wp.vec3(0.9, 0.5, 0.2)], dtype=wp.vec3),
            )
        else:
            plane_x = [-0.28, 0.38] if self.compare_side_by_side else [0.05]
            for k, x in enumerate(plane_x):
                color = wp.array([wp.vec3(0.8, 0.2 + 0.5 * k, 0.2)], dtype=wp.vec3)
                xform = wp.array(
                    [wp.transform(wp.vec3(x, -0.02, self._plane_z), wp.quat_identity())],
                    dtype=wp.transform,
                )
                self.viewer.log_shapes(
                    f"/proxy_plane_{k}",
                    newton.GeoType.BOX,
                    (0.25, 0.25, 0.005),
                    xform,
                    color,
                )

        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        z_left = q[self._left_idx, 2].mean()
        z_right = q[self._right_idx, 2].mean()
        if self.use_coupler and self.compare_side_by_side:
            assert z_right > z_left + 0.05, "Coupler group should stay higher than baseline group"


if __name__ == "__main__":
    import argparse

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--use-coupler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable coupler correction.",
    )
    parser.add_argument(
        "--compare-side-by-side",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare baseline (left) vs coupler (right) in one scene.",
    )
    parser.add_argument(
        "--proxy-type",
        type=str,
        default="plane",
        choices=["plane", "box"],
        help="Coupler rigid proxy shape: plane or box.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
