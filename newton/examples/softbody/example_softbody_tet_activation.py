# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare active bending of three XPBD plates with different materials."""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverXPBD


class Example:
    """Bend soft, medium, and stiff plates using identical tet activation."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / args.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.activation = args.activation
        self.activation_duration = args.activation_duration
        self.log_interval = args.log_interval
        self.log_metrics_enabled = not args.test
        self.current_activation = 0.0
        self.frame = 0

        self.plate_names = ("soft", "medium", "stiff")
        self.material_scales = np.asarray(args.material_scales, dtype=np.float32)
        if not np.all(np.isfinite(self.material_scales)) or np.any(self.material_scales <= 0.0):
            raise ValueError("material-scales must contain three positive finite values.")
        if not 0.0 < args.active_layer_fraction < 1.0:
            raise ValueError("active-layer-fraction must be between 0 and 1.")

        plate_length = args.length_cells * args.cell_size
        plate_width = args.width_cells * args.cell_size
        plate_thickness = args.thickness_cells * args.thickness_cell_size
        plate_pitch = plate_length + args.plate_gap

        builder = newton.ModelBuilder()
        self.plate_particle_ranges = []
        self.plate_tri_ranges = []
        for plate_index, material_scale in enumerate(self.material_scales):
            particle_start = len(builder.particle_q)
            tri_start = len(builder.tri_indices)
            builder.add_soft_grid(
                pos=wp.vec3(plate_index * plate_pitch, -0.5 * plate_width, -0.5 * plate_thickness),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=args.length_cells,
                dim_y=args.width_cells,
                dim_z=args.thickness_cells,
                cell_x=args.cell_size,
                cell_y=args.cell_size,
                cell_z=args.thickness_cell_size,
                density=args.density,
                k_mu=args.stiffness * float(material_scale),
                k_lambda=args.stiffness * float(material_scale),
                k_damp=args.damping,
                fix_left=args.fix_left,
                particle_radius=0.0,
                add_surface_mesh_edges=False,
            )
            self.plate_particle_ranges.append((particle_start, len(builder.particle_q)))
            self.plate_tri_ranges.append((tri_start, len(builder.tri_indices)))
        builder.color()

        self.model = builder.finalize()
        self.model.gravity.zero_()

        rest_positions = self.model.particle_q.numpy()
        tet_indices = self.model.tet_indices.numpy()
        tet_centers = np.mean(rest_positions[tet_indices], axis=1)
        active_layer_start = np.max(rest_positions[:, 2]) - args.active_layer_fraction * plate_thickness
        self.activation_profile = (tet_centers[:, 2] > active_layer_start).astype(np.float32)
        if not np.any(self.activation_profile) or np.all(self.activation_profile):
            raise ValueError("The active layer must contain some, but not all, tetrahedra.")
        self.activation_values = np.zeros(self.model.tet_count, dtype=np.float32)

        self.tip_indices = []
        self.initial_tip_centers = []
        self.initial_lengths = []
        for particle_start, particle_end in self.plate_particle_ranges:
            particle_indices = np.arange(particle_start, particle_end)
            plate_positions = rest_positions[particle_start:particle_end]
            tip_threshold = np.max(plate_positions[:, 0]) - 0.5 * args.cell_size
            tip_indices = particle_indices[plate_positions[:, 0] > tip_threshold]
            self.tip_indices.append(tip_indices)
            self.initial_tip_centers.append(np.mean(rest_positions[tip_indices], axis=0))
            self.initial_lengths.append(np.ptp(plate_positions[:, 0]))
        self.initial_tip_centers = np.asarray(self.initial_tip_centers)
        self.initial_lengths = np.asarray(self.initial_lengths)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = SolverXPBD(
            self.model,
            iterations=args.iterations,
            soft_body_relaxation=args.relaxation,
        )

        tri_indices = self.model.tri_indices.numpy()
        self.plate_mesh_indices = [
            wp.array(
                tri_indices[tri_start:tri_end].reshape(-1),
                dtype=wp.int32,
                device=self.model.device,
            )
            for tri_start, tri_end in self.plate_tri_ranges
        ]
        self.plate_colors = ((0.28, 0.58, 0.95), (0.95, 0.72, 0.25), (0.9, 0.3, 0.3))

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        if hasattr(self.viewer, "set_camera"):
            scene_center_x = plate_pitch + 0.5 * plate_length
            scene_span = 3.0 * plate_length + 2.0 * args.plate_gap
            camera_distance = max(4.0, 1.35 * scene_span)
            camera_height = 0.25 * camera_distance
            self.viewer.set_camera(
                pos=wp.vec3(scene_center_x, -camera_distance, camera_height),
                pitch=-14.0,
                yaw=90.0,
            )

        if self.log_metrics_enabled:
            for name, scale in zip(self.plate_names, self.material_scales, strict=True):
                print(f"{name}: stiffness={args.stiffness * scale:.3g} Pa")
            print(f"Active upper-layer tetrahedra: {np.count_nonzero(self.activation_profile)}/{self.model.tet_count}")
            print("time activation soft_tip_dz/L0 medium_tip_dz/L0 stiff_tip_dz/L0")
            self.log_metrics()

        self.graph = None
        self.capture()

    def normalized_tip_deflections(self, positions):
        return np.asarray(
            [
                (np.mean(positions[tip_indices], axis=0)[2] - initial_tip_center[2]) / initial_length
                for tip_indices, initial_tip_center, initial_length in zip(
                    self.tip_indices,
                    self.initial_tip_centers,
                    self.initial_lengths,
                    strict=True,
                )
            ]
        )

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
        if self.activation_duration > 0.0:
            ramp = min(self.sim_time / self.activation_duration, 1.0)
        else:
            ramp = 1.0
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        self.current_activation = self.activation * ramp
        np.multiply(self.activation_profile, self.current_activation, out=self.activation_values)
        self.control.tet_activations.assign(self.activation_values)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame += 1
        if self.log_metrics_enabled and self.frame % self.log_interval == 0:
            self.log_metrics()

    def log_metrics(self):
        tip_deflections = self.normalized_tip_deflections(self.state_0.particle_q.numpy())
        print(
            f"{self.sim_time:.3f} {self.current_activation:.3f} "
            f"{tip_deflections[0]:.6f} {tip_deflections[1]:.6f} {tip_deflections[2]:.6f}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, indices, color in zip(
            self.plate_names,
            self.plate_mesh_indices,
            self.plate_colors,
            strict=True,
        ):
            self.viewer.log_mesh(
                f"activation_plate_{name}",
                self.state_0.particle_q,
                indices,
                color=color,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify that activation bends the three plates without instability."""
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions and velocities",
            lambda q, qd: wp.length(q) < 100.0 and wp.length(qd) < 100.0,
        )

        tip_deflections = np.abs(self.normalized_tip_deflections(self.state_0.particle_q.numpy()))
        if self.activation > 0.0 and np.max(tip_deflections) < 0.02:
            raise ValueError(f"Tet activation did not bend the plate tips: {tip_deflections}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=10)
        parser.add_argument("--relaxation", type=float, default=0.7)
        parser.add_argument("--activation", type=float, default=0.4)
        parser.add_argument("--activation-duration", type=float, default=1.5)
        parser.add_argument("--length-cells", type=int, default=20)
        parser.add_argument("--width-cells", type=int, default=6)
        parser.add_argument("--thickness-cells", type=int, default=2)
        parser.add_argument("--cell-size", type=float, default=0.1)
        parser.add_argument("--thickness-cell-size", type=float, default=0.05)
        parser.add_argument("--active-layer-fraction", type=float, default=0.5)
        parser.add_argument("--plate-gap", type=float, default=0.2)
        parser.add_argument("--material-scales", type=float, nargs=3, default=(0.5, 1.0, 2.0))
        parser.add_argument("--fix-left", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--stiffness", type=float, default=5.0e4)
        parser.add_argument("--damping", type=float, default=1.0e-2)
        parser.add_argument("--log-interval", type=int, default=30)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
