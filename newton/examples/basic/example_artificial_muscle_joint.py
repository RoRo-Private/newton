# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualize a passive hinge joint made from CFRP BCC lattice links."""

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
from newton.solvers import SolverVBD


def _grid_point(i, j, k, cell_size):
    return np.array((i * cell_size, j * cell_size, k * cell_size), dtype=np.float32)


def _build_bcc_lattice_lines(cells_x, cells_y, cells_z, cell_size):
    starts = []
    ends = []

    for i in range(cells_x):
        for j in range(cells_y + 1):
            for k in range(cells_z + 1):
                starts.append(_grid_point(i, j, k, cell_size))
                ends.append(_grid_point(i + 1, j, k, cell_size))

    for i in range(cells_x + 1):
        for j in range(cells_y):
            for k in range(cells_z + 1):
                starts.append(_grid_point(i, j, k, cell_size))
                ends.append(_grid_point(i, j + 1, k, cell_size))

    for i in range(cells_x + 1):
        for j in range(cells_y + 1):
            for k in range(cells_z):
                starts.append(_grid_point(i, j, k, cell_size))
                ends.append(_grid_point(i, j, k + 1, cell_size))

    diagonal_pairs = (
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0)),
        ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    diagonal_pairs = np.asarray(diagonal_pairs, dtype=np.float32)

    for i in range(cells_x):
        for j in range(cells_y):
            for k in range(cells_z):
                cell_origin = _grid_point(i, j, k, cell_size)
                for start_offset, end_offset in diagonal_pairs:
                    starts.append(cell_origin + start_offset * cell_size)
                    ends.append(cell_origin + end_offset * cell_size)

    for side in (-1.0, 1.0):
        z_index = 0 if side < 0.0 else cells_z
        z_base = 0.0 if side < 0.0 else cells_z * cell_size
        z_ridge = z_base + side * 0.5 * cell_size
        y_ridge = 0.5 * cells_y * cell_size
        for i in (0, cells_x):
            x = i * cell_size
            ridge = np.array((x, y_ridge, z_ridge), dtype=np.float32)
            starts.append(_grid_point(i, 0, z_index, cell_size))
            ends.append(ridge)
            starts.append(_grid_point(i, cells_y, z_index, cell_size))
            ends.append(ridge)

        starts.append(np.array((0.0, y_ridge, z_ridge), dtype=np.float32))
        ends.append(np.array((cells_x * cell_size, y_ridge, z_ridge), dtype=np.float32))

    return np.asarray(starts, dtype=np.float32), np.asarray(ends, dtype=np.float32)


def _count_bcc_lattice_lines(cells_x, cells_y, cells_z):
    return (
        cells_x * (cells_y + 1) * (cells_z + 1)
        + (cells_x + 1) * cells_y * (cells_z + 1)
        + (cells_x + 1) * (cells_y + 1) * cells_z
        + 4 * cells_x * cells_y * cells_z
        + _count_gable_lines()
    )


def _count_bcc_diagonal_lines(cells_x, cells_y, cells_z):
    return 4 * cells_x * cells_y * cells_z


def _count_gable_lines():
    return 10


def _basis_from_z_axis(direction):
    z_axis = np.asarray(direction, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)
    ref = np.array((0.0, 1.0, 0.0), dtype=np.float64) if abs(z_axis[2]) > 0.9 else np.array((0.0, 0.0, 1.0))
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _build_lattice_capsule_mesh(starts, ends, radius):
    vertices = []
    indices = []
    for start, end in zip(starts, ends, strict=True):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 0.0:
            continue

        half_height = max(0.0, 0.5 * length - radius)
        midpoint = 0.5 * (start + end)
        direction = segment / length

        capsule = newton.Mesh.create_capsule(
            radius,
            half_height,
            up_axis=newton.Axis.Z,
            segments=12,
            compute_inertia=False,
        )
        local_vertices = np.asarray(capsule.vertices, dtype=np.float32)
        rotation = _basis_from_z_axis(direction).astype(np.float32)
        start_index = len(vertices)
        vertices.extend((local_vertices @ rotation.T + midpoint).tolist())
        indices.extend((np.asarray(capsule.indices, dtype=np.int32) + start_index).tolist())

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def _box_inertia(mass, size_x, size_y, size_z):
    i_xx = mass * (size_y * size_y + size_z * size_z) / 12.0
    i_yy = mass * (size_x * size_x + size_z * size_z) / 12.0
    i_zz = mass * (size_x * size_x + size_y * size_y) / 12.0
    return wp.mat33(i_xx, 0.0, 0.0, 0.0, i_yy, 0.0, 0.0, 0.0, i_zz)


def _transform_vertices(vertices, transform):
    position = transform[:3]
    rotation = np.array(wp.quat_to_matrix(wp.quat(*transform[3:7])), dtype=np.float32).reshape(3, 3)
    return vertices @ rotation.T + position


def _transform_point(point, transform):
    return _transform_vertices(np.asarray([point], dtype=np.float32), transform)[0]


def _soft_grid_particle_index(start, dim_x, dim_y, x, y, z):
    return start + z * (dim_y + 1) * (dim_x + 1) + y * (dim_x + 1) + x


def _smooth_triangular_activation(time, half_period):
    if half_period <= 0.0:
        return 1.0

    phase = time % (2.0 * half_period)
    ramp = phase / half_period if phase < half_period else 2.0 - phase / half_period
    return ramp * ramp * (3.0 - 2.0 * ramp)


class Example:
    """Simulate two CFRP lattice links connected by a passive revolute joint."""

    def __init__(self, viewer, args):
        if min(args.cells_x, args.cells_y, args.cells_z) < 1:
            raise ValueError("cells-x, cells-y, and cells-z must be positive.")
        if args.cell_size <= 0.0:
            raise ValueError("cell-size must be positive.")
        if args.wire_diameter <= 0.0:
            raise ValueError("wire-diameter must be positive.")
        if args.link_mass <= 0.0:
            raise ValueError("link-mass must be positive.")
        if args.joint_stiffness < 0.0:
            raise ValueError("joint-stiffness must be non-negative.")
        if args.joint_damping < 0.0:
            raise ValueError("joint-damping must be non-negative.")
        if args.joint_friction < 0.0:
            raise ValueError("joint-friction must be non-negative.")
        if args.actuator_width <= 0.0:
            raise ValueError("actuator-width must be positive.")
        if args.actuator_thickness <= 0.0:
            raise ValueError("actuator-thickness must be positive.")
        if args.actuator_attachment_length <= 0.0:
            raise ValueError("actuator-attachment-length must be positive.")
        if args.actuator_ramp_time < 0.0:
            raise ValueError("actuator-ramp-time must be non-negative.")
        if min(args.actuator_dim_x, args.actuator_dim_y, args.actuator_dim_z) < 1:
            raise ValueError("actuator dimensions must be positive.")
        if not np.isfinite(args.actuator_active_stress):
            raise ValueError("actuator-active-stress must be finite.")
        if args.attachment_stiffness < 0.0:
            raise ValueError("attachment-stiffness must be non-negative.")
        if args.attachment_damping < 0.0:
            raise ValueError("attachment-damping must be non-negative.")
        if args.camera_speed < 0.0:
            raise ValueError("camera-speed must be non-negative.")

        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        starts_np, ends_np = _build_bcc_lattice_lines(args.cells_x, args.cells_y, args.cells_z, args.cell_size)
        offset = 0.5 * np.array(
            (args.cells_x * args.cell_size, args.cells_y * args.cell_size, args.cells_z * args.cell_size),
            dtype=np.float32,
        )
        starts_np -= offset
        ends_np -= offset

        self.line_count = starts_np.shape[0]
        self.diagonal_line_count = _count_bcc_diagonal_lines(args.cells_x, args.cells_y, args.cells_z)
        self.gable_line_count = _count_gable_lines()
        self.wire_radius = 0.5 * args.wire_diameter
        vertices_np, indices_np = _build_lattice_capsule_mesh(starts_np, ends_np, self.wire_radius)
        self.link_vertices_local = vertices_np
        self.link_indices = indices_np
        self.expected_line_count = _count_bcc_lattice_lines(args.cells_x, args.cells_y, args.cells_z)
        self.lattice_indices = wp.array(self.link_indices, dtype=wp.int32)

        link_size_x = args.cells_x * args.cell_size
        link_size_y = args.cells_y * args.cell_size
        link_size_z = (args.cells_z + 1) * args.cell_size
        link_inertia = _box_inertia(args.link_mass, link_size_x, link_size_y, link_size_z)
        self.parent_hinge_local = wp.vec3(0.0, 0.0, 0.5 * link_size_z)
        self.child_hinge_local = wp.vec3(0.0, 0.0, -0.5 * link_size_z)
        hinge_world_z = float(self.parent_hinge_local[2])
        self.actuator_inside_y = -0.5 * link_size_y + args.actuator_clearance + 0.5 * args.actuator_thickness
        self.actuator_attachment_length = min(args.actuator_attachment_length, 0.45 * link_size_z)
        self.actuator_width = args.actuator_width
        self.actuator_thickness = args.actuator_thickness
        self.actuator_activation = float(np.clip(args.actuator_activation, 0.0, 1.0))
        self.actuator_ramp_time = args.actuator_ramp_time
        self.actuator_mode = args.actuator_mode
        self.actuator_dim_x = args.actuator_dim_x
        self.actuator_dim_y = args.actuator_dim_y
        self.actuator_dim_z = args.actuator_dim_z
        self.actuator_length = 2.0 * self.actuator_attachment_length
        self.actuator_z_min = hinge_world_z - self.actuator_attachment_length
        self.actuator_z_max = hinge_world_z + self.actuator_attachment_length
        self.current_activation = 0.0

        builder = newton.ModelBuilder()
        self.link_1 = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=args.link_mass,
            inertia=link_inertia,
            label="cfrp_lattice_link_1",
        )
        self.link_2 = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, link_size_z), q=wp.quat_identity()),
            mass=args.link_mass,
            inertia=link_inertia,
            label="cfrp_lattice_link_2",
        )

        root_joint = builder.add_joint_fixed(
            parent=-1,
            child=self.link_1,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="world_to_link_1",
        )
        self.hinge_joint = builder.add_joint_revolute(
            parent=self.link_1,
            child=self.link_2,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=self.parent_hinge_local, q=wp.quat_identity()),
            child_xform=wp.transform(p=self.child_hinge_local, q=wp.quat_identity()),
            target_pos=args.rest_angle,
            target_vel=0.0,
            target_ke=args.joint_stiffness,
            target_kd=args.joint_damping,
            damping=args.joint_damping,
            friction=args.joint_friction,
            label="passive_lattice_hinge",
        )
        builder.add_articulation([root_joint, self.hinge_joint], label="artificial_muscle_joint")
        builder.joint_q[-1] = args.initial_angle

        actuator_particle_start = builder.particle_count
        actuator_tri_start = len(builder.tri_indices)
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * self.actuator_width, self.actuator_inside_y - 0.5 * self.actuator_thickness, self.actuator_z_min),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=self.actuator_dim_x,
            dim_y=self.actuator_dim_y,
            dim_z=self.actuator_dim_z,
            cell_x=self.actuator_width / self.actuator_dim_x,
            cell_y=self.actuator_thickness / self.actuator_dim_y,
            cell_z=self.actuator_length / self.actuator_dim_z,
            density=args.actuator_density,
            k_mu=args.actuator_k_mu,
            k_lambda=args.actuator_k_lambda,
            k_damp=args.actuator_k_damp,
            particle_radius=0.0,
            add_surface_mesh_edges=False,
            label="ipcnt_bending_actuator",
        )
        actuator_particle_end = builder.particle_count
        actuator_tri_end = len(builder.tri_indices)

        link_2_origin = np.array((0.0, 0.0, link_size_z), dtype=np.float32)
        actuator_positions = np.asarray(builder.particle_q[actuator_particle_start:actuator_particle_end], dtype=np.float32)
        self.attachment_count = 0
        attachment_corners = (
            (0, 0),
            (self.actuator_dim_x, 0),
            (0, self.actuator_dim_y),
            (self.actuator_dim_x, self.actuator_dim_y),
        )
        for z, body in ((0, self.link_1), (self.actuator_dim_z, self.link_2)):
            for x, y in attachment_corners:
                particle = _soft_grid_particle_index(
                    actuator_particle_start,
                    self.actuator_dim_x,
                    self.actuator_dim_y,
                    x,
                    y,
                    z,
                )
                point_world = actuator_positions[particle - actuator_particle_start]
                point_body = point_world if body == self.link_1 else point_world - link_2_origin
                SolverCoupledADMM.add_body_particle_attachment(
                    builder,
                    body,
                    particle,
                    body_point=wp.vec3(*point_body),
                    stiffness=args.attachment_stiffness,
                    damping=args.attachment_damping,
                )
                self.attachment_count += 1

        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 0.0
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_mu = 0.0
        self.hinge_q_start = int(self.model.joint_q_start.numpy()[self.hinge_joint])
        self.actuator_particle_range = (actuator_particle_start, actuator_particle_end)
        self.actuator_particles = list(range(actuator_particle_start, actuator_particle_end))
        self.actuator_tri_indices = wp.array(
            self.model.tri_indices.numpy()[actuator_tri_start:actuator_tri_end].reshape(-1),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=lambda v: SolverVBD(model=v, iterations=args.rigid_iterations),
                    bodies=[self.link_1, self.link_2],
                    joints=[root_joint, self.hinge_joint],
                ),
                SolverCoupled.Entry(
                    name="actuator",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=args.soft_iterations,
                        particle_enable_self_contact=False,
                        particle_enable_tile_solve=False,
                    ),
                    particles=self.actuator_particles,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=args.coupling_iterations,
                rho=args.coupling_rho,
                gamma=args.coupling_gamma,
                baumgarte=args.coupling_baumgarte,
            ),
        )
        self.soft_solver = self.solver.solver("actuator")
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = None
        self.control = self.model.control()

        rest_positions = self.model.particle_q.numpy()
        tet_indices = self.model.tet_indices.numpy()
        tet_centers = np.mean(rest_positions[tet_indices], axis=1)
        if self.actuator_mode == "bending":
            thickness_half = 0.5 * self.actuator_thickness
            signed_layer_coordinate = np.clip((tet_centers[:, 1] - self.actuator_inside_y) / thickness_half, -1.0, 1.0)
            self.active_stress_values = (args.actuator_active_stress * signed_layer_coordinate).astype(np.float32)
            if not (np.any(self.active_stress_values > 0.0) and np.any(self.active_stress_values < 0.0)):
                raise ValueError("The bending actuator needs tetrahedra on both sides of the thickness centerline.")
        else:
            self.active_stress_values = np.full(self.model.tet_count, args.actuator_active_stress, dtype=np.float32)
        self.active_tet_mask = np.ones(self.model.tet_count, dtype=np.float32)
        self.activation_values = np.zeros(self.model.tet_count, dtype=np.float32)
        self.soft_solver.set_tet_active_stress(
            directors=np.tile(np.array((0.0, 0.0, 1.0), dtype=np.float32), (self.model.tet_count, 1)),
            activations=self.activation_values,
            stress=self.active_stress_values,
        )

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        if hasattr(self.viewer, "set_camera"):
            length = 2.0 * link_size_z
            self.viewer.set_camera(
                pos=wp.vec3(0.7 * length, -2.3 * length, 0.55 * length),
                pitch=-16.0,
                yaw=118.0,
            )
        if hasattr(self.viewer, "camera_speed"):
            self.viewer.camera_speed = args.camera_speed

        if not args.test:
            print(
                f"bcc lattice cells={args.cells_x}x{args.cells_y}x{args.cells_z} "
                f"cell_size={args.cell_size:.4f}m wire_diameter={args.wire_diameter:.4f}m "
                f"link_mass={args.link_mass:.4f}kg joint_stiffness={args.joint_stiffness:.4f}N*m/rad "
                f"joint_damping={args.joint_damping:.4f}N*m*s/rad "
                f"actuator_mode={self.actuator_mode} actuator_active_stress={args.actuator_active_stress:.1f}Pa "
                f"stress_range=({np.min(self.active_stress_values):.1f}, {np.max(self.active_stress_values):.1f})Pa "
                f"attachments={self.attachment_count} "
                f"lines_per_link={self.line_count} diagonals_per_link={self.diagonal_line_count} "
                f"gables_per_link={self.gable_line_count} hinge_axis=x"
            )

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_actuator_activation(self):
        ramp = _smooth_triangular_activation(self.sim_time, self.actuator_ramp_time)
        self.current_activation = self.actuator_activation * ramp
        np.multiply(self.active_tet_mask, self.current_activation, out=self.activation_values)
        self.soft_solver.tet_active_stress_activations.assign(self.activation_values)

    def step(self):
        self.update_actuator_activation()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        body_q = self.state_0.body_q.numpy()
        link_1_vertices = wp.array(_transform_vertices(self.link_vertices_local, body_q[self.link_1]), dtype=wp.vec3)
        link_2_vertices = wp.array(_transform_vertices(self.link_vertices_local, body_q[self.link_2]), dtype=wp.vec3)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh(
            "/cfrp_bcc_lattice/link_1",
            link_1_vertices,
            self.lattice_indices,
            color=(0.0, 0.0, 0.0),
            backface_culling=False,
        )
        self.viewer.log_mesh(
            "/cfrp_bcc_lattice/link_2",
            link_2_vertices,
            self.lattice_indices,
            color=(0.0, 0.0, 0.0),
            backface_culling=False,
        )
        self.viewer.log_mesh(
            f"/{self.actuator_mode}_active_stress_actuator",
            self.state_0.particle_q,
            self.actuator_tri_indices,
            color=(0.15, 0.45, 0.9),
            backface_culling=False,
            roughness=0.55,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the passive lattice joint remains finite and connected."""
        if self.line_count != self.expected_line_count:
            raise ValueError(f"expected {self.expected_line_count} BCC frame lines, got {self.line_count}")
        if len(self.link_vertices_local) == 0 or len(self.lattice_indices) == 0:
            raise ValueError("BCC lattice capsule mesh is empty.")
        body_q = self.state_0.body_q.numpy()
        link_1_hinge = _transform_point(np.array(self.parent_hinge_local, dtype=np.float32), body_q[self.link_1])
        link_2_hinge = _transform_point(np.array(self.child_hinge_local, dtype=np.float32), body_q[self.link_2])
        if not np.isfinite(body_q).all():
            raise ValueError("artificial muscle joint body state contains non-finite values.")
        if np.linalg.norm(link_1_hinge - link_2_hinge) > 2.0e-3:
            raise ValueError("passive revolute joint anchors separated too far.")
        particle_q = self.state_0.particle_q.numpy()
        if not np.isfinite(particle_q).all():
            raise ValueError("deformable actuator particle state contains non-finite values.")
        if self.attachment_count == 0:
            raise ValueError("deformable actuator has no rigid-link attachments.")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720)
        parser.add_argument("--cells-x", type=int, default=1)
        parser.add_argument("--cells-y", type=int, default=1)
        parser.add_argument("--cells-z", type=int, default=2)
        parser.add_argument("--cell-size", type=float, default=0.02)
        parser.add_argument("--wire-diameter", type=float, default=0.001)
        parser.add_argument("--link-mass", type=float, default=0.01)
        parser.add_argument("--joint-stiffness", type=float, default=2.0e-3)
        parser.add_argument("--joint-damping", type=float, default=4.0e-4)
        parser.add_argument("--joint-friction", type=float, default=5.0e-5)
        parser.add_argument("--rest-angle", type=float, default=0.0)
        parser.add_argument("--initial-angle", type=float, default=0.0)
        parser.add_argument("--actuator-activation", type=float, default=1.0)
        parser.add_argument("--actuator-mode", choices=("bending", "linear"), default="bending")
        parser.add_argument("--actuator-ramp-time", type=float, default=3.0)
        parser.add_argument("--actuator-width", type=float, default=0.014)
        parser.add_argument("--actuator-thickness", type=float, default=0.008)
        parser.add_argument("--actuator-clearance", type=float, default=0.0015)
        parser.add_argument("--actuator-attachment-length", type=float, default=0.026)
        parser.add_argument("--actuator-dim-x", type=int, default=4)
        parser.add_argument("--actuator-dim-y", type=int, default=6)
        parser.add_argument("--actuator-dim-z", type=int, default=12)
        parser.add_argument("--actuator-density", type=float, default=1000.0)
        parser.add_argument("--actuator-k-mu", type=float, default=5.0e4)
        parser.add_argument("--actuator-k-lambda", type=float, default=5.0e4)
        parser.add_argument("--actuator-k-damp", type=float, default=1.0e3)
        parser.add_argument("--actuator-active-stress", type=float, default=2.0e4)
        parser.add_argument("--attachment-stiffness", type=float, default=3.0e3)
        parser.add_argument("--attachment-damping", type=float, default=1.0)
        parser.add_argument("--rigid-iterations", type=int, default=8)
        parser.add_argument("--soft-iterations", type=int, default=12)
        parser.add_argument("--coupling-iterations", type=int, default=4)
        parser.add_argument("--coupling-rho", type=float, default=80.0)
        parser.add_argument("--coupling-gamma", type=float, default=0.05)
        parser.add_argument("--coupling-baumgarte", type=float, default=0.02)
        parser.add_argument("--camera-speed", type=float, default=0.08)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
