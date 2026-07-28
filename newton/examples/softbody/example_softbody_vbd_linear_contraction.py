# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract fixed-end VBD actuators with Poisson-style transverse stretch."""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD


def _append_oriented_tet(vertices, indices, tet):
    a, b, c, d = tet
    pa = np.asarray(vertices[a], dtype=np.float64)
    pb = np.asarray(vertices[b], dtype=np.float64)
    pc = np.asarray(vertices[c], dtype=np.float64)
    pd = np.asarray(vertices[d], dtype=np.float64)
    if np.linalg.det(np.stack((pb - pa, pc - pa, pd - pa), axis=1)) < 0.0:
        tet = (a, b, d, c)
    indices.extend(tet)


def _make_cylinder_tet_mesh(length, radius, length_segments, radial_segments, angular_segments):
    vertices = []
    slice_stride = 1 + radial_segments * angular_segments

    def vertex_index(xi, ri, ti=0):
        base = xi * slice_stride
        if ri == 0:
            return base
        return base + 1 + (ri - 1) * angular_segments + (ti % angular_segments)

    for xi in range(length_segments + 1):
        x = length * xi / length_segments
        vertices.append((x, 0.0, 0.0))
        for ri in range(1, radial_segments + 1):
            r = radius * ri / radial_segments
            for ti in range(angular_segments):
                theta = 2.0 * np.pi * ti / angular_segments
                vertices.append((x, r * np.cos(theta), r * np.sin(theta)))

    cross_section_tris = []
    for ti in range(angular_segments):
        cross_section_tris.append(((0, 0), (1, ti), (1, ti + 1)))
    for ri in range(1, radial_segments):
        for ti in range(angular_segments):
            cross_section_tris.append(((ri, ti), (ri + 1, ti), (ri + 1, ti + 1)))
            cross_section_tris.append(((ri, ti), (ri + 1, ti + 1), (ri, ti + 1)))

    indices = []
    for xi in range(length_segments):
        for (ari, ati), (bri, bti), (cri, cti) in cross_section_tris:
            a = vertex_index(xi, ari, ati)
            b = vertex_index(xi, bri, bti)
            c = vertex_index(xi, cri, cti)
            ap = vertex_index(xi + 1, ari, ati)
            bp = vertex_index(xi + 1, bri, bti)
            cp = vertex_index(xi + 1, cri, cti)
            _append_oriented_tet(vertices, indices, (a, b, c, ap))
            _append_oriented_tet(vertices, indices, (b, bp, cp, ap))
            _append_oriented_tet(vertices, indices, (b, cp, c, ap))

    return vertices, indices


def _make_rest_wireframe_lines(positions, tri_indices, device):
    edges = set()
    for tri in tri_indices.reshape(-1, 3):
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    starts = np.asarray([positions[a] for a, _ in edges], dtype=np.float32)
    ends = np.asarray([positions[b] for _, b in edges], dtype=np.float32)
    return (
        wp.array(starts, dtype=wp.vec3, device=device),
        wp.array(ends, dtype=wp.vec3, device=device),
    )


class Example:
    """Linearly contract soft actuator shapes while transverse dimensions expand."""

    def __init__(self, viewer, args):
        if not 0.0 <= args.contraction < 1.0:
            raise ValueError("contraction must be in the range [0, 1).")
        if not -1.0 < args.poissons_ratio < 0.5:
            raise ValueError("poissons-ratio must be between -1 and 0.5.")
        if min(args.dim_x, args.dim_y, args.dim_z) < 1:
            raise ValueError("dim-x, dim-y, and dim-z must be positive.")
        if args.shape == "cylinder":
            if args.cylinder_radius <= 0.0:
                raise ValueError("cylinder-radius must be positive.")
            if args.radial_segments < 1:
                raise ValueError("radial-segments must be positive.")
            if args.angular_segments < 3:
                raise ValueError("angular-segments must be at least 3.")
        if args.log_interval < 1:
            raise ValueError("log-interval must be positive.")

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

        self.parallel_stretch = 1.0 - args.contraction
        self.perpendicular_stretch = 1.0 + args.poissons_ratio * args.contraction
        if self.perpendicular_stretch <= 0.0:
            raise ValueError("poissons-ratio and contraction produce a non-positive transverse stretch.")

        builder = newton.ModelBuilder()
        if args.shape == "strip":
            length = args.dim_x * args.cell_size
            width = args.dim_y * args.cell_size
            thickness = args.dim_z * args.thickness_cell_size
            builder.add_soft_grid(
                pos=wp.vec3(0.0, -0.5 * width, -0.5 * thickness),
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
                fix_left=True,
                particle_radius=0.0,
                add_surface_mesh_edges=False,
            )
        else:
            length = args.dim_x * args.cell_size
            width = 2.0 * args.cylinder_radius
            thickness = 2.0 * args.cylinder_radius
            vertices, indices = _make_cylinder_tet_mesh(
                length,
                args.cylinder_radius,
                args.dim_x,
                args.radial_segments,
                args.angular_segments,
            )
            particle_start = len(builder.particle_q)
            builder.add_soft_mesh(
                pos=(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
                vertices=vertices,
                indices=indices,
                density=args.density,
                k_mu=args.k_mu,
                k_lambda=args.k_lambda,
                k_damp=args.k_damp,
                particle_radius=0.0,
                add_surface_mesh_edges=False,
            )
            for vi, vertex in enumerate(vertices):
                if vertex[0] <= 1.0e-6:
                    builder.particle_mass[particle_start + vi] = 0.0
        builder.color()

        self.model = builder.finalize()
        self.model.gravity.zero_()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.rest_positions = self.model.particle_q.numpy()
        self.rest_min = np.min(self.rest_positions, axis=0)
        self.rest_max = np.max(self.rest_positions, axis=0)
        self.rest_extents = self.rest_max - self.rest_min
        self.rest_extents = np.maximum(self.rest_extents, 1.0e-8)
        self.activation_values = np.zeros(self.model.tet_count, dtype=np.float32)

        self.solver = SolverVBD(
            self.model,
            iterations=args.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )
        self.solver.set_tet_active_strain(
            directors=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (self.model.tet_count, 1)),
            activations=np.zeros(self.model.tet_count, dtype=np.float32),
            parallel_stretch=np.full(self.model.tet_count, self.parallel_stretch, dtype=np.float32),
            perpendicular_stretch=np.full(self.model.tet_count, self.perpendicular_stretch, dtype=np.float32),
        )

        self.tri_indices = wp.array(self.model.tri_indices.numpy().reshape(-1), dtype=wp.int32, device=self.model.device)
        self.rest_line_starts, self.rest_line_ends = _make_rest_wireframe_lines(
            self.rest_positions,
            self.model.tri_indices.numpy(),
            self.model.device,
        )
        self.current_activation = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        if hasattr(self.viewer, "set_camera"):
            camera_distance = max(1.4, 2.7 * length)
            camera_height = max(0.25, 3.0 * thickness)
            self.viewer.set_camera(
                pos=wp.vec3(0.5 * length, -camera_distance, camera_height),
                pitch=-8.0,
                yaw=90.0,
            )

        if self.log_metrics_enabled:
            print(
                "target stretches "
                f"parallel={self.parallel_stretch:.4f} perpendicular={self.perpendicular_stretch:.4f}"
            )
            print("time activation length_ratio width_ratio height_ratio volume_ratio")
            self.log_metrics()

        self.graph = None
        self.capture()

    def extent_ratios(self, positions):
        extents = np.max(positions, axis=0) - np.min(positions, axis=0)
        return extents / self.rest_extents

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
        ramp = min(self.sim_time / self.ramp_time, 1.0) if self.ramp_time > 0.0 else 1.0
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        self.current_activation = self.target_activation * ramp
        self.activation_values.fill(self.current_activation)
        self.solver.tet_active_strain_activations.assign(self.activation_values)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.frame += 1
        if self.log_metrics_enabled and self.frame % self.log_interval == 0:
            self.log_metrics()

    def log_metrics(self):
        ratios = self.extent_ratios(self.state_0.particle_q.numpy())
        volume_ratio = ratios[0] * ratios[1] * ratios[2]
        print(
            f"{self.sim_time:.3f} {self.current_activation:.3f} "
            f"{ratios[0]:.6f} {ratios[1]:.6f} {ratios[2]:.6f} {volume_ratio:.6f}"
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "active_strain_linear_contraction",
            self.state_0.particle_q,
            self.tri_indices,
            color=(0.25, 0.58, 0.95),
            backface_culling=False,
        )
        self.viewer.log_lines(
            "rest_pose_wireframe",
            self.rest_line_starts,
            self.rest_line_ends,
            (0.92, 0.92, 0.92),
            width=0.003,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify linear contraction stays finite and shortens the active axis."""
        newton.examples.test_particle_state(
            self.state_0,
            "linear contraction particles remain finite",
            lambda q, qd: wp.length(q) < 100.0 and wp.length(qd) < 100.0,
        )
        ratios = self.extent_ratios(self.state_0.particle_q.numpy())
        if self.target_activation > 0.0:
            if ratios[0] > 0.98:
                raise ValueError(f"block did not contract along x: extent ratios={ratios}")
            if self.perpendicular_stretch > 1.0 and (ratios[1] <= 1.0 or ratios[2] <= 1.0):
                raise ValueError(f"block did not expand transversely: extent ratios={ratios}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--shape", choices=("strip", "cylinder"), default="strip")
        parser.add_argument("--activation", type=float, default=1.0)
        parser.add_argument("--contraction", type=float, default=0.2)
        parser.add_argument("--poissons-ratio", type=float, default=0.3)
        parser.add_argument("--ramp-time", type=float, default=1.0)
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--substeps", type=int, default=15)
        parser.add_argument("--dim-x", type=int, default=14)
        parser.add_argument("--dim-y", type=int, default=3)
        parser.add_argument("--dim-z", type=int, default=1)
        parser.add_argument("--cell-size", type=float, default=0.06)
        parser.add_argument("--thickness-cell-size", type=float, default=0.018)
        parser.add_argument("--cylinder-radius", type=float, default=0.045)
        parser.add_argument("--radial-segments", type=int, default=2)
        parser.add_argument("--angular-segments", type=int, default=12)
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
