# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bend a rectangular soft-body plate using tetrahedral activation."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverXPBD


class Example:
    """Cantilevered plate bent by contracting tetrahedra in its upper layer."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.activation = args.activation
        self.activation_duration = args.activation_duration

        builder = newton.ModelBuilder()
        plate_width = args.width_cells * args.cell_size
        builder.add_soft_grid(
            pos=wp.vec3(0.0, -0.5 * plate_width, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=args.length_cells,
            dim_y=args.width_cells,
            dim_z=args.thickness_cells,
            cell_x=args.cell_size,
            cell_y=args.cell_size,
            cell_z=args.thickness_cell_size,
            density=args.density,
            k_mu=args.stiffness,
            k_lambda=args.stiffness,
            k_damp=args.damping,
            fix_left=True,
            particle_radius=0.0,
            add_surface_mesh_edges=False,
        )
        builder.color()

        self.model = builder.finalize()
        self.model.gravity.zero_()

        rest_positions = self.model.particle_q.numpy()
        tet_indices = self.model.tet_indices.numpy()
        tet_centers = np.mean(rest_positions[tet_indices], axis=1)
        plate_midplane = 0.5 * args.thickness_cells * args.thickness_cell_size
        self.activation_profile = (tet_centers[:, 2] > plate_midplane).astype(np.float32)

        plate_length = args.length_cells * args.cell_size
        self.tip_indices = np.flatnonzero(rest_positions[:, 0] > plate_length - 0.5 * args.cell_size).tolist()
        self.initial_tip_height = float(np.mean(rest_positions[self.tip_indices, 2]))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = SolverXPBD(
            self.model,
            iterations=args.iterations,
            soft_body_relaxation=args.relaxation,
        )

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.2, -2.8, 1.5), pitch=-18.0, yaw=90.0)

        self.graph = None
        self.capture()

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
        ramp = min(self.sim_time / self.activation_duration, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        self.control.tet_activations.assign(self.activation_profile * (self.activation * ramp))

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that activation bends the plate without destabilizing it."""
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions and velocities",
            lambda q, qd: wp.length(q) < 100.0 and wp.length(qd) < 100.0,
        )

        tip_height = float(np.mean(self.state_0.particle_q.numpy()[self.tip_indices, 2]))
        if abs(tip_height - self.initial_tip_height) < 0.02:
            raise ValueError("Tet activation did not bend the plate tip")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=5)
        parser.add_argument("--iterations", type=int, default=10)
        parser.add_argument("--relaxation", type=float, default=0.7)
        parser.add_argument("--activation", type=float, default=0.4)
        parser.add_argument("--activation-duration", type=float, default=1.5)
        parser.add_argument("--length-cells", type=int, default=20)
        parser.add_argument("--width-cells", type=int, default=6)
        parser.add_argument("--thickness-cells", type=int, default=2)
        parser.add_argument("--cell-size", type=float, default=0.1)
        parser.add_argument("--thickness-cell-size", type=float, default=0.05)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--stiffness", type=float, default=5.0e4)
        parser.add_argument("--damping", type=float, default=1.0e-2)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
