# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim Cloth Phystwin Interact
#
# Loads a PhysTwin cloth mesh and interacts with it inside Newton by moving
# a subset of anchor vertices directly. This is useful for prototyping
# cloth pulling / dragging behaviors without replaying recorded anchors.
#
# Command:
#   WARP_CACHE_PATH=/tmp/warp-cache uv run --module newton.examples \
#       diffsim_cloth_phystwin_interact --keyboard-speed 0.02
#
###########################################################################

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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.manual_anchor_offset_np = np.zeros(3, dtype=np.float32)
        self._reset_key_prev = False

        cloth = np.load(args.dataset)
        self.vertices_np = cloth["vertices"].astype(np.float32)
        self.faces_np = cloth["faces"].astype(np.int32)
        self.face_indices_wp = wp.array(self.faces_np.reshape(-1), dtype=wp.int32, device=wp.get_device())
        self.plane_height = 0.0
        if os.path.exists(args.meta):
            with open(args.meta, encoding="utf-8") as f:
                meta = json.load(f)
            self.plane_height = float(meta["plane_center"][2])

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.particle_radius
        builder.default_shape_cfg.ke = args.contact_ke
        builder.default_shape_cfg.kd = args.contact_kd
        builder.default_shape_cfg.mu = args.contact_mu
        if args.with_ground:
            builder.add_ground_plane()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, args.cloth_z_offset - self.plane_height),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=[wp.vec3(v) for v in self.vertices_np],
            indices=self.faces_np.reshape(-1).tolist(),
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

        selected_ids = self._select_pull_anchors(args.anchor_count, args.anchor_mode)
        if len(selected_ids) == 0:
            raise ValueError("No anchors selected. Adjust --anchor-start/--anchor-count.")

        self.selected_anchor_ids_np = selected_ids
        self.selected_anchor_rest_np = self.vertices_np[self.selected_anchor_ids_np].copy()
        self.selected_anchor_targets_np = self.selected_anchor_rest_np.copy()

        # Freeze only the selected anchors so they can pull the cloth.
        flags = self.model.particle_flags.numpy()
        for vertex_id in self.selected_anchor_ids_np:
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
        if args.with_ground:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="nxn",
                soft_contact_margin=max(args.particle_radius * 1.5, 0.004),
            )
            self.contacts = self.collision_pipeline.contacts()
        else:
            self.collision_pipeline = None
            self.contacts = self.model.contacts()

        self.selected_anchor_ids = wp.array(self.selected_anchor_ids_np, dtype=int, device=self.model.device)
        self.selected_anchor_targets = wp.array(
            self.selected_anchor_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.selected_anchor_points = wp.array(
            self.selected_anchor_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.selected_anchor_colors = wp.full(
            len(self.selected_anchor_ids_np),
            wp.vec3(1.0, 0.2, 0.1),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.selected_anchor_radii = wp.full(
            len(self.selected_anchor_ids_np),
            args.anchor_radius,
            dtype=wp.float32,
            device=self.model.device,
        )

        bounds_min = self.vertices_np.min(axis=0)
        bounds_max = self.vertices_np.max(axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        extent = float(np.max(bounds_max - bounds_min))
        camera_distance = max(0.35, extent * 1.8)
        camera_height = max(0.12, extent * 1.0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            wp.vec3(center[0], center[1] - camera_distance, center[2] + camera_height),
            -20.0,
            90.0,
        )

    def _select_pull_anchors(self, anchor_count: int, anchor_mode: str) -> np.ndarray:
        if anchor_mode == "dataset":
            dataset_anchor_ids = np.load(self.args.dataset)["anchor_vertex_ids"].astype(np.int32)
            count = min(anchor_count, len(dataset_anchor_ids))
            return dataset_anchor_ids[:count]

        verts = self.vertices_np
        top_mask = verts[:, 1] >= np.quantile(verts[:, 1], 0.98)
        candidate_ids = np.where(top_mask)[0]
        if len(candidate_ids) == 0:
            candidate_ids = np.arange(len(verts))

        if anchor_mode == "top_left":
            ordered = candidate_ids[np.argsort(verts[candidate_ids, 0])]
        elif anchor_mode == "top_right":
            ordered = candidate_ids[np.argsort(-verts[candidate_ids, 0])]
        else:  # spread_top
            ordered = candidate_ids[np.argsort(verts[candidate_ids, 0])]
            if anchor_count > 1:
                pick_positions = np.linspace(0, len(ordered) - 1, num=min(anchor_count, len(ordered))).astype(int)
                return ordered[pick_positions].astype(np.int32)

        return ordered[: min(anchor_count, len(ordered))].astype(np.int32)

    def _compute_anchor_targets(self) -> np.ndarray:
        targets = self.selected_anchor_rest_np.copy()

        pull_time = max(0.0, self.sim_time - self.args.pull_delay)
        ramp = min(pull_time / max(self.args.pull_ramp, 1e-6), 1.0)
        smooth_ramp = ramp * ramp * (3.0 - 2.0 * ramp)

        displacement = np.array(
            [
                self.args.pull_dx,
                self.args.pull_dy,
                self.args.pull_dz,
            ],
            dtype=np.float32,
        )

        targets += displacement * smooth_ramp
        targets += self.manual_anchor_offset_np
        return targets

    def _update_keyboard_controls(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        speed = self.args.keyboard_speed * self.frame_dt
        delta = np.zeros(3, dtype=np.float32)

        if self.viewer.is_key_down("j"):
            delta[0] -= speed
        if self.viewer.is_key_down("l"):
            delta[0] += speed
        if self.viewer.is_key_down("k"):
            delta[1] -= speed
        if self.viewer.is_key_down("i"):
            delta[1] += speed
        if self.viewer.is_key_down("o"):
            delta[2] -= speed
        if self.viewer.is_key_down("u"):
            delta[2] += speed

        self.manual_anchor_offset_np += delta

        reset_down = bool(self.viewer.is_key_down("p"))
        if reset_down and not self._reset_key_prev:
            self.manual_anchor_offset_np.fill(0.0)
        self._reset_key_prev = reset_down

    def _apply_anchor_targets(self):
        self.selected_anchor_targets_np = self._compute_anchor_targets()
        self.selected_anchor_targets.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )
        self.selected_anchor_points.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )

        wp.launch(
            set_anchor_targets,
            dim=len(self.selected_anchor_ids_np),
            inputs=[
                self.selected_anchor_ids,
                self.selected_anchor_targets,
                self.state_0.particle_q,
                self.state_0.particle_qd,
            ],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_anchor_targets()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Re-apply after stepping so the visualized anchor positions stay fixed.
        self._apply_anchor_targets()

    def step(self):
        self._update_keyboard_controls()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh(
            "/cloth_interact_mesh",
            self.state_0.particle_q,
            self.face_indices_wp,
            hidden=False,
            backface_culling=False,
        )
        self.viewer.log_points(
            "/pull_anchors",
            self.selected_anchor_points,
            radii=self.selected_anchor_radii,
            colors=self.selected_anchor_colors,
        )
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to cloth_export.npz.")
        parser.add_argument("--meta", type=str, default=DEFAULT_META_PATH, help="Path to export meta.json.")
        parser.add_argument("--sim-substeps", type=int, default=18, help="Number of substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4, help="VBD iterations per substep.")
        parser.add_argument("--density", type=float, default=0.08, help="Cloth areal density.")
        parser.add_argument("--particle-radius", type=float, default=0.0025, help="Cloth particle radius.")
        parser.add_argument("--gravity", type=float, default=0.0, help="Gravity acceleration along the up axis [m/s^2].")
        parser.add_argument("--cloth-z-offset", type=float, default=0.02, help="Lift cloth above the fitted plane [m].")
        parser.add_argument(
            "--with-ground",
            action="store_true",
            help="Enable ground plane contact. Disabled by default for stability.",
        )
        parser.add_argument("--tri-ke", type=float, default=3.0e2, help="Triangle stretching stiffness.")
        parser.add_argument("--tri-ka", type=float, default=3.0e2, help="Triangle area stiffness.")
        parser.add_argument("--tri-kd", type=float, default=3.0e-1, help="Triangle damping.")
        parser.add_argument("--edge-ke", type=float, default=3.0, help="Bending stiffness.")
        parser.add_argument("--edge-kd", type=float, default=4.0, help="Bending damping.")
        parser.add_argument("--contact-ke", type=float, default=1.0e2, help="Ground contact stiffness.")
        parser.add_argument("--contact-kd", type=float, default=3.0e1, help="Ground contact damping.")
        parser.add_argument("--contact-mu", type=float, default=0.2, help="Ground contact friction.")
        parser.add_argument("--self-contact", action="store_true", help="Enable cloth self-contact.")
        parser.add_argument("--self-contact-radius", type=float, default=0.004, help="Self-contact radius.")
        parser.add_argument("--self-contact-margin", type=float, default=0.006, help="Self-contact margin.")
        parser.add_argument(
            "--anchor-mode",
            type=str,
            choices=["top_left", "top_right", "spread_top", "dataset"],
            default="top_left",
            help="How to choose pull anchors.",
        )
        parser.add_argument("--anchor-count", type=int, default=1, help="Number of anchors to pull.")
        parser.add_argument("--anchor-radius", type=float, default=0.01, help="Rendered radius for pull anchors.")
        parser.add_argument("--pull-delay", type=float, default=1.0, help="Seconds to wait before pulling.")
        parser.add_argument("--pull-ramp", type=float, default=4.0, help="Seconds to ramp pull strength.")
        parser.add_argument("--pull-dx", type=float, default=0.0, help="Anchor pull amplitude along x [m].")
        parser.add_argument("--pull-dy", type=float, default=0.0, help="Anchor pull amplitude along y [m].")
        parser.add_argument("--pull-dz", type=float, default=0.003, help="Anchor pull amplitude along z [m].")
        parser.add_argument(
            "--keyboard-speed",
            type=float,
            default=0.03,
            help="Anchor target speed for keyboard control [m/s]. Keys: J/L=x, K/I=y, O/U=z, P=reset.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
