# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare active bending of three VBD plates with different materials."""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD


class Example:
    """Bend soft, medium, and stiff plates using identical active contraction."""

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

        self.plate_names = ("soft", "medium", "stiff")
        self.material_scales = np.asarray(args.material_scales, dtype=np.float32)
        if not np.all(np.isfinite(self.material_scales)) or np.any(self.material_scales <= 0.0):
            raise ValueError("material-scales must contain three positive finite values.")

        plate_length = args.dim_x * args.cell_size
        plate_width = args.dim_y * args.cell_size
        plate_thickness = args.dim_z * args.thickness_cell_size
        if not 0.0 < args.active_layer_fraction < 1.0:
            raise ValueError("active-layer-fraction must be between 0 and 1.")

        builder = newton.ModelBuilder()
        self.plate_particle_ranges = []
        self.plate_tet_ranges = []
        self.plate_tri_ranges = []
        plate_pitch = plate_length + args.plate_gap
        for plate_index, material_scale in enumerate(self.material_scales):
            particle_start = len(builder.particle_q)
            tet_start = len(builder.tet_indices)
            tri_start = len(builder.tri_indices)
            builder.add_soft_grid(
                pos=wp.vec3(plate_index * plate_pitch, -0.5 * plate_width, -0.5 * plate_thickness),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=args.dim_x,
                dim_y=args.dim_y,
                dim_z=args.dim_z,
                cell_x=args.cell_size,
                cell_y=args.cell_size,
                cell_z=args.thickness_cell_size,
                density=args.density,
                k_mu=args.k_mu * float(material_scale),
                k_lambda=args.k_lambda * float(material_scale),
                k_damp=args.k_damp,
                fix_left=args.fix_left,
                particle_radius=0.0,
                add_surface_mesh_edges=False,
            )
            self.plate_particle_ranges.append((particle_start, len(builder.particle_q)))
            self.plate_tet_ranges.append((tet_start, len(builder.tet_indices)))
            self.plate_tri_ranges.append((tri_start, len(builder.tri_indices)))
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
        self.current_activation = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        if hasattr(self.viewer, "set_camera"):
            scene_center_x = plate_pitch + 0.5 * plate_length
            self.viewer.set_camera(pos=wp.vec3(scene_center_x, -4.0, 1.25), pitch=-14.0, yaw=90.0)

        if self.log_metrics_enabled:
            active_fraction = np.mean(self.active_tet_mask)
            for name, scale in zip(self.plate_names, self.material_scales, strict=True):
                print(
                    f"{name}: material scale={scale:.3g}, "
                    f"k_mu={args.k_mu * scale:.3g} Pa, k_lambda={args.k_lambda * scale:.3g} Pa"
                )
            print(f"Active upper-layer tetrahedra: {np.count_nonzero(self.active_tet_mask)}/{self.model.tet_count}")
            print(f"Active tet fraction: {active_fraction:.3f}")
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
        tip_deflections = self.normalized_tip_deflections(positions)
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
                f"active_plate_{name}",
                self.state_0.particle_q,
                indices,
                color=color,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify stable differential contraction bends the plate."""
        newton.examples.test_particle_state(
            self.state_0,
            "active soft-body particles remain finite",
            lambda q, qd: wp.length(q) < 100.0 and wp.length(qd) < 100.0,
        )
        final_positions = self.state_0.particle_q.numpy()
        tip_deflections = np.abs(self.normalized_tip_deflections(final_positions))
        if self.target_activation > 0.0:
            if np.max(tip_deflections) < 0.02:
                raise ValueError(f"Upper-layer contraction did not bend the plates: |tip dz|/L0={tip_deflections}")
            if tip_deflections[0] <= tip_deflections[2]:
                raise ValueError(f"The soft plate did not bend more than the stiff plate: {tip_deflections}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--activation", type=float, default=0.3)
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
        parser.add_argument("--plate-gap", type=float, default=0.2)
        parser.add_argument("--material-scales", type=float, nargs=3, default=(0.5, 1.0, 2.0))
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--k-mu", type=float, default=1.0e5)
        parser.add_argument("--k-lambda", type=float, default=1.0e5)
        parser.add_argument("--k-damp", type=float, default=1.0e4)
        parser.add_argument("--log-interval", type=int, default=30)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
