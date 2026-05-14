# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim Cloth Phystwin
#
# Loads a cloth trajectory exported from PhysTwin and replays it inside
# Newton. In `full` mode the exported cloth mesh is rendered directly.
# In `anchors` mode a Newton cloth mesh is driven by exported anchor
# trajectories.
#
# Command: WARP_CACHE_PATH=/tmp/warp-cache uv run --module newton.examples diffsim_cloth_phystwin --show-target
#
###########################################################################

# cloth_export.npz를 읽고, 두 가지 모드로 시뮬레이션을 실행하는 예제

import json
import os

import numpy as np
import warp as wp

import newton
import newton.examples


REPO_ROOT = os.path.dirname(os.path.dirname(newton.examples.get_source_directory()))
_CLOTH_DATA_CANDIDATES = (
    os.path.join(REPO_ROOT, "newton", "cloth_data"),
    os.path.join(REPO_ROOT, "cloth_data"),
)
CLOTH_DATA_ROOT = next((path for path in _CLOTH_DATA_CANDIDATES if os.path.isdir(path)), _CLOTH_DATA_CANDIDATES[1])
DEFAULT_DATASET_PATH = os.path.join(CLOTH_DATA_ROOT, "cloth_1_2", "newton_cloth", "cloth_export.npz")
DEFAULT_META_PATH = os.path.join(CLOTH_DATA_ROOT, "cloth_1_2", "newton_cloth", "meta.json")
FULL_MODE_POINT_RADIUS = 0.006


@wp.kernel
def set_anchor_targets(
    anchor_vertex_ids: wp.array[int],
    anchor_targets: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    particle_id = anchor_vertex_ids[tid]
    particle_q[particle_id] = anchor_targets[tid]
    particle_qd[particle_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_particle_targets(
    particle_targets: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    particle_q[tid] = particle_targets[tid]
    particle_qd[tid] = wp.vec3(0.0, 0.0, 0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0

        dataset = np.load(args.dataset)
        self.rest_vertices = dataset["vertices"].astype(np.float32)
        self.faces = dataset["faces"].astype(np.int32)
        self.anchor_vertex_ids_np = dataset["anchor_vertex_ids"].astype(np.int32)
        self.anchor_targets_np = dataset["anchor_targets"].astype(np.float32)
        self.object_points_np = dataset["object_points"].astype(np.float32)
        self.object_visibilities_np = dataset["object_visibilities"].astype(bool)
        self.frame_count = int(self.object_points_np.shape[0])

        print(f"Loading PhysTwin cloth data from: {args.dataset}")

        self.plane_height = 0.0
        if os.path.exists(args.meta):
            with open(args.meta, encoding="utf-8") as f:
                meta = json.load(f)
            self.plane_height = float(meta["plane_center"][2])

        self.target_points = wp.array(self.object_points_np[0], dtype=wp.vec3, device=self.device)
        self.target_indices = wp.array(self.faces.reshape(-1), dtype=wp.int32, device=self.device)
        self.target_colors = wp.zeros(self.rest_vertices.shape[0], dtype=wp.vec3, device=self.device)
        self.target_radii = wp.full(
            self.rest_vertices.shape[0],
            args.target_radius,
            dtype=wp.float32,
            device=self.device,
        )
        self._update_target_colors(0)
        self.full_mode_colors = wp.full(
            self.rest_vertices.shape[0],
            wp.vec3(1.0, 0.55, 0.2),
            dtype=wp.vec3,
            device=self.device,
        )
        self.full_mode_radii = wp.full(
            self.rest_vertices.shape[0],
            FULL_MODE_POINT_RADIUS,
            dtype=wp.float32,
            device=self.device,
        )

        if self.args.drive_mode == "anchors":
            builder = newton.ModelBuilder(gravity=args.gravity)
            builder.default_particle_radius = args.particle_radius
            builder.default_shape_cfg.ke = args.contact_ke
            builder.default_shape_cfg.kd = args.contact_kd
            builder.default_shape_cfg.mu = args.contact_mu
            if args.with_ground:
                builder.add_ground_plane()
            # Ensure the initial cloth mesh is above the ground plane to avoid
            # large initial penetration that can destabilize VBD.
            min_rest_z = float(np.min(self.rest_vertices[:, 2]))
            auto_lift = max(0.0, -min_rest_z + args.cloth_z_clearance)
            self.cloth_z_offset = -self.plane_height + auto_lift
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, self.cloth_z_offset),
                rot=wp.quat_identity(),
                scale=1.0,
                vertices=[wp.vec3(v) for v in self.rest_vertices],
                indices=self.faces.reshape(-1).tolist(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=args.density,
                tri_ke=args.tri_ke,
                tri_ka=args.tri_ka,
                tri_kd=args.tri_kd,
                edge_ke=args.edge_ke,
                edge_kd=args.edge_kd,
                particle_radius=args.particle_radius,
            )
            builder.color(include_bending=True)

            self.model = builder.finalize()
            self.model.soft_contact_ke = args.contact_ke
            self.model.soft_contact_kd = args.contact_kd
            self.model.soft_contact_mu = args.contact_mu

            flags = self.model.particle_flags.numpy()
            for vertex_id in self.anchor_vertex_ids_np:
                flags[vertex_id] = flags[vertex_id] & ~newton.ParticleFlags.ACTIVE
            self.model.particle_flags = wp.array(flags, dtype=int, device=self.model.device)

            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=args.iterations,
                particle_enable_self_contact=args.self_contact,
                particle_self_contact_radius=args.self_contact_radius,
                particle_self_contact_margin=args.self_contact_margin,
            )

            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            self.contacts = self.model.contacts()
            self.anchor_vertex_ids = wp.array(self.anchor_vertex_ids_np, dtype=int, device=self.model.device)
            self.anchor_targets_np = self.anchor_targets_np + np.array(
                [0.0, 0.0, self.cloth_z_offset], dtype=np.float32
            )
            self.object_points_np = self.object_points_np + np.array(
                [0.0, 0.0, self.cloth_z_offset], dtype=np.float32
            )
            self.current_anchor_targets = wp.array(self.anchor_targets_np[0], dtype=wp.vec3, device=self.model.device)
            self.viewer.set_model(self.model)

        bounds_min = self.object_points_np[0].min(axis=0)
        bounds_max = self.object_points_np[0].max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        extent = float(np.max(bounds_max - bounds_min))
        camera_distance = max(0.35, extent * 1.6)
        camera_height = max(0.12, extent * 0.8)
        self.viewer.set_camera(
            wp.vec3(center[0], center[1] - camera_distance, center[2] + camera_height),
            -20.0,
            90.0,
        )

        self.capture()

    def _wrap_frame(self, frame_index: int) -> int:
        return (frame_index * self.args.frame_stride) % self.frame_count

    def _update_target_colors(self, frame_index: int) -> None:
        colors = np.zeros((self.rest_vertices.shape[0], 3), dtype=np.float32)
        visible = self.object_visibilities_np[frame_index]
        colors[visible] = np.array([0.95, 0.25, 0.15], dtype=np.float32)
        colors[~visible] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.target_colors = wp.array(colors, dtype=wp.vec3, device=self.device)

    def _set_frame_targets(self, frame_index: int) -> None:
        self.target_points.assign(wp.array(self.object_points_np[frame_index], dtype=wp.vec3, device=self.device))
        self._update_target_colors(frame_index)
        if self.args.drive_mode == "anchors":
            self.current_anchor_targets.assign(
                wp.array(self.anchor_targets_np[frame_index], dtype=wp.vec3, device=self.model.device)
            )

    def capture(self):
        self.graph = None

    def simulate(self):
        frame_id = self._wrap_frame(self.frame_index)
        self._set_frame_targets(frame_id)

        if self.args.drive_mode == "full":
            return

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                set_anchor_targets,
                dim=self.anchor_vertex_ids.shape[0],
                inputs=[
                    self.anchor_vertex_ids,
                    self.current_anchor_targets,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                ],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            set_anchor_targets,
            dim=self.anchor_vertex_ids.shape[0],
            inputs=[
                self.anchor_vertex_ids,
                self.current_anchor_targets,
                self.state_0.particle_q,
                self.state_0.particle_qd,
            ],
            device=self.model.device,
        )

    def step(self):
        self.simulate()
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.args.drive_mode == "full":
            self.viewer.log_mesh(
                "/cloth_export_mesh",
                self.target_points,
                self.target_indices,
                hidden=False,
                backface_culling=False,
            )
            self.viewer.log_points(
                "/cloth_export_points",
                self.target_points,
                radii=self.full_mode_radii,
                colors=self.full_mode_colors,
            )
        else:
            self.viewer.log_state(self.state_0)
        if self.args.show_target:
            self.viewer.log_points(
                "/target_points",
                self.target_points,
                radii=self.target_radii,
                colors=self.target_colors,
            )
        self.viewer.end_frame()

    def test_final(self):
        if self.args.drive_mode == "full":
            particle_q = self.target_points.numpy()
        else:
            particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all()
        assert particle_q.shape[0] == self.rest_vertices.shape[0]

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to cloth_export.npz.")
        parser.add_argument("--meta", type=str, default=DEFAULT_META_PATH, help="Path to export meta.json.")
        parser.add_argument(
            "--drive-mode",
            type=str,
            choices=["full", "anchors"],
            default="full",
            help="Use 'full' to replay exported cloth vertices exactly, or 'anchors' to simulate from anchor motion.",
        )
        parser.add_argument("--frame-stride", type=int, default=1, help="Stride through exported frames.")
        parser.add_argument("--sim-substeps", type=int, default=8, help="Number of Newton substeps per frame.")
        parser.add_argument("--iterations", type=int, default=10, help="VBD iterations per substep.")
        parser.add_argument("--density", type=float, default=0.15, help="Cloth areal density.")
        parser.add_argument("--particle-radius", type=float, default=0.0035, help="Cloth particle radius.")
        parser.add_argument("--gravity", type=float, default=0.0, help="Gravity acceleration along +Z axis [m/s^2].")
        parser.add_argument(
            "--with-ground",
            action="store_true",
            help="Enable ground contact. Disabled by default.",
        )
        parser.add_argument("--tri-ke", type=float, default=4.0e3, help="Triangle stretching stiffness.")
        parser.add_argument("--tri-ka", type=float, default=4.0e3, help="Triangle area stiffness.")
        parser.add_argument("--tri-kd", type=float, default=5.0e-2, help="Triangle damping.")
        parser.add_argument("--edge-ke", type=float, default=5.0e1, help="Bending stiffness.")
        parser.add_argument("--edge-kd", type=float, default=1.0, help="Bending damping.")
        parser.add_argument("--contact-ke", type=float, default=1.0e3, help="Ground contact stiffness.")
        parser.add_argument("--contact-kd", type=float, default=1.0e1, help="Ground contact damping.")
        parser.add_argument("--contact-mu", type=float, default=0.8, help="Ground contact friction.")
        parser.add_argument("--self-contact", action="store_true", help="Enable cloth self-contact in VBD mode.")
        parser.add_argument("--self-contact-radius", type=float, default=0.004, help="Self-contact radius.")
        parser.add_argument("--self-contact-margin", type=float, default=0.006, help="Self-contact margin.")
        parser.add_argument(
            "--cloth-z-clearance",
            type=float,
            default=0.01,
            help="Minimum initial clearance from the ground plane [m] in anchors mode.",
        )
        parser.add_argument("--show-target", action="store_true", help="Overlay exported cloth points.")
        parser.add_argument("--target-radius", type=float, default=0.004, help="Rendered radius for target points.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
