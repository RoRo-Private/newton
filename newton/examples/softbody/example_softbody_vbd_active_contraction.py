# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bend a soft-body plate using directional active tetrahedral contraction."""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD


class Example:
    """Cantilevered plate bent by contracting tetrahedra in its upper layer."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.target_activation = float(np.clip(args.activation, 0.0, 1.0))
        self.ramp_time = args.ramp_time
        self.log_interval = args.log_interval
        self.log_metrics_enabled = not args.test
        self.frame = 0

        plate_width = args.dim_y * args.cell_size
        plate_thickness = args.dim_z * args.thickness_cell_size
        if not 0.0 < args.active_layer_fraction < 1.0:
            raise ValueError("active-layer-fraction must be between 0 and 1.")
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, -0.5 * plate_width, -0.5 * plate_thickness),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            dim_z=args.dim_z,
            cell_x=args.cell_size,
            cell_y=args.cell_size,
            cell_z=args.thickness_cell_size,
            density=args.density,
            k_mu=args.k_mu,
            k_lambda=args.k_lambda,
            k_damp=args.k_damp,
            fix_left=args.fix_left,
            particle_radius=0.0,
            add_surface_mesh_edges=False,
        )
        builder.color()

        self.model = builder.finalize()
        self.model.gravity.zero_()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        rest_positions = self.model.particle_q.numpy()
        self.tet_indices = self.model.tet_indices.numpy()
        tet_centers = np.mean(rest_positions[self.tet_indices], axis=1)
        active_layer_start = np.max(rest_positions[:, 2]) - args.active_layer_fraction * plate_thickness
        self.active_tet_mask = (tet_centers[:, 2] > active_layer_start).astype(np.float32)
        if not np.any(self.active_tet_mask):
            raise ValueError("The active-layer selection contains no tetrahedra.")
        if np.all(self.active_tet_mask):
            raise ValueError("The active layer must not contain every tetrahedron.")
        self.activation_values = np.zeros(self.model.tet_count, dtype=np.float32)

        self.solver = SolverVBD(
            self.model,
            iterations=args.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        director = np.array([args.director_x, args.director_y, args.director_z], dtype=np.float32)
        self.solver.set_tet_active_contraction(
            directors=np.tile(director, (self.model.tet_count, 1)),
            activations=np.zeros(self.model.tet_count, dtype=np.float32),
            stiffness=np.full(self.model.tet_count, args.active_stiffness, dtype=np.float32),
        )

        tip_threshold = np.max(rest_positions[:, 0]) - 0.5 * args.cell_size
        self.tip_indices = np.flatnonzero(rest_positions[:, 0] > tip_threshold)
        self.initial_tip_center = np.mean(rest_positions[self.tip_indices], axis=0)
        self.initial_metrics = self.measure(rest_positions)
        self.current_metrics = self.initial_metrics
        self.current_activation = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.2, -1.5, 0.8), pitch=-20.0, yaw=118.0)

        if self.log_metrics_enabled:
            active_fraction = np.mean(self.active_tet_mask)
            print(f"Active upper-layer tetrahedra: {np.count_nonzero(self.active_tet_mask)}/{self.model.tet_count}")
            print(f"Active tet fraction: {active_fraction:.3f}")
            print("time activation L/L0 W/W0 T/T0 V/V0 tip_dz/L0 COM_displacement")
            self.log_metrics()

        self.graph = None
        self.capture()

    def measure(self, positions):
        extents = np.ptp(positions, axis=0)
        tet_positions = positions[self.tet_indices]
        ds = np.stack(
            (
                tet_positions[:, 1] - tet_positions[:, 0],
                tet_positions[:, 2] - tet_positions[:, 0],
                tet_positions[:, 3] - tet_positions[:, 0],
            ),
            axis=2,
        )
        volume = float(np.sum(np.abs(np.linalg.det(ds))) / 6.0)
        center_of_mass = np.mean(positions, axis=0)
        return extents, volume, center_of_mass

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.ramp_time > 0.0:
            ramp = min(self.sim_time / self.ramp_time, 1.0)
        else:
            ramp = 1.0
        self.current_activation = self.target_activation * ramp
        np.multiply(self.active_tet_mask, self.current_activation, out=self.activation_values)
        self.solver.tet_active_activations.assign(self.activation_values)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame += 1
        if self.log_metrics_enabled and self.frame % self.log_interval == 0:
            self.log_metrics()

    def log_metrics(self):
        positions = self.state_0.particle_q.numpy()
        self.current_metrics = self.measure(positions)
        extents, volume, center_of_mass = self.current_metrics
        initial_extents, initial_volume, initial_center_of_mass = self.initial_metrics
        ratios = extents / initial_extents
        volume_ratio = volume / initial_volume
        com_displacement = np.linalg.norm(center_of_mass - initial_center_of_mass)
        tip_center = np.mean(positions[self.tip_indices], axis=0)
        normalized_tip_deflection = (tip_center[2] - self.initial_tip_center[2]) / initial_extents[0]
        print(
            f"{self.sim_time:.3f} {self.current_activation:.3f} "
            f"{ratios[0]:.6f} {ratios[1]:.6f} {ratios[2]:.6f} "
            f"{volume_ratio:.6f} {normalized_tip_deflection:.6f} {com_displacement:.6e}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify stable differential contraction bends the plate."""
        newton.examples.test_particle_state(
            self.state_0,
            "active soft-body particles remain finite",
            lambda q, qd: wp.length(q) < 100.0 and wp.length(qd) < 100.0,
        )
        final_positions = self.state_0.particle_q.numpy()
        self.current_metrics = self.measure(final_positions)
        initial_length = self.initial_metrics[0][0]
        tip_center = np.mean(final_positions[self.tip_indices], axis=0)
        normalized_tip_deflection = abs(tip_center[2] - self.initial_tip_center[2]) / initial_length
        if self.target_activation > 0.0:
            if normalized_tip_deflection < 0.02:
                raise ValueError(
                    f"Upper-layer contraction did not bend the plate: |tip dz|/L0={normalized_tip_deflection:.6f}"
                )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--activation", type=float, default=1.0)
        parser.add_argument("--active-stiffness", type=float, default=1.0e5)
        parser.add_argument("--ramp-time", type=float, default=1.0)
        parser.add_argument("--director-x", type=float, default=1.0)
        parser.add_argument("--director-y", type=float, default=0.0)
        parser.add_argument("--director-z", type=float, default=0.0)
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--substeps", type=int, default=15)
        parser.add_argument("--fix-left", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--dim-x", type=int, default=16)
        parser.add_argument("--dim-y", type=int, default=8)
        parser.add_argument("--dim-z", type=int, default=2)
        parser.add_argument("--cell-size", type=float, default=0.05)
        parser.add_argument("--thickness-cell-size", type=float, default=0.05)
        parser.add_argument("--active-layer-fraction", type=float, default=0.5)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--k-mu", type=float, default=1.0e5)
        parser.add_argument("--k-lambda", type=float, default=1.0e5)
        parser.add_argument("--k-damp", type=float, default=1.0e3)
        parser.add_argument("--log-interval", type=int, default=30)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
