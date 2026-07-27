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
# Example Diffsim Box XPBD
#
# Simulates a cardboard box with hinged flaps using XPBD.  Particle and
# spring data are loaded from box_data/apriltag_template/box_hinge_xpbd.npz,
# which was reconstructed from AprilTag motion capture.  The bottom panel
# and four walls are fixed; the four flaps are dynamic and swing open
# under gravity via hinge spring constraints.
#
# Panel layout (panels_built order in template_summary.json):
#   0: bottom (static)    1: front_wall (static)  2: left_wall (static)
#   3: back_wall (static) 4: right_wall (static)
#   5: front_flap  6: left_flap  7: back_flap  8: right_flap
#
# Compliance mapping (from box_hinge_xpbd.npz):
#   Type 0  compliance=1e-8  (rigid in-panel)
#   Type 1  compliance=1e-5  (wall edge)
#   Type 2  compliance=3e-5  (hinge)
#   Type 3  compliance=3e-5  (shear/bend)
#
# Newton's XPBD solve_springs uses ke = 1/compliance (N/m), where
# alpha = 1/(ke * dt^2).  Very large ke (>1e5) requires many Gauss-Seidel
# iterations to converge within a single substep.  We therefore cap ke at
# KE_MAX = 1e5, which keeps all 165k springs stable with 4 iterations while
# preserving the relative stiffness hierarchy between constraint types.
#
# Command: uv run python newton/examples/diffsim/example_diffsim_box_xpbd.py
#
###########################################################################

import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import CollisionSample, CouplerOptions, LegacyLikeCoupler, RigidCouplingMaterial, SolverXPBD

STATIC_PANELS = {0, 1, 2, 3, 4}  # bottom + 4 walls

# Per-type spring stiffness (ke = 1/compliance in XPBD) and damping (kd).
# Type 0: in-panel structural  → rigid panels
# Type 1: wall-edge connections → rigid panels
# Type 2: hinge                → replaced by dihedral bending constraints (see below)
# Type 3: shear/diagonal       → rigid panels (no in-plane shear)
KE_BY_TYPE = {0: 1e5, 1: 1e5, 3: 1e5}
KD_BY_TYPE = {0: 0.0, 1: 0.0, 3: 0.0}

# Dihedral bending constraints at hinge fold lines.
# gamma = HINGE_BEND_KD / (HINGE_BEND_KE * dt) ≈ 1.2 at dt=1/1200 → slightly overdamped.
HINGE_BEND_KD = 10.0  # damping; gamma = kd/(ke*dt) ≈ 1.2 at dt=1/1200


def _require(data: np.lib.npyio.NpzFile, key: str, path: str) -> np.ndarray:
    if key not in data:
        raise KeyError(
            f"'{key}' not found in {path}. "
            "Rebuild the npz with build_xpbd_model() to include dihedral constraints."
        )
    return data[key]


def _load_npz(npz_path: str) -> dict:
    data = np.load(npz_path)
    positions = data["init_vertices"].astype(np.float32)
    positions[:, 2] *= -1  # captured upside-down; flip Z to right-side up
    return dict(
        positions=positions,
        constraints=data["constraints"].astype(np.int32),
        constraint_type=data["constraint_type"].astype(np.int32),
        colors=data["colors"],
        substeps=int(data["substeps"]),
        panel_id=data["panel_id"].astype(np.int32),
        uv=data["uv"].astype(np.float32),
        material_id=data["material_id"].astype(np.int32),
        dihedral_quads=_require(data, "dihedral_quads", npz_path).astype(np.int32),
        dihedral_ke=float(1.0 / _require(data, "dihedral_compliance", npz_path)),
    )


def _build_per_panel_tris(
    panel_id: np.ndarray,
    uv: np.ndarray,
    positions: np.ndarray,
    max_uv_edge: float = 0.08,
) -> dict[int, np.ndarray]:
    """Delaunay-triangulate each panel in its own UV [0,1]² space.

    Returns {panel_id: (T, 3) array of global particle indices}.
    Triangles with any UV edge > *max_uv_edge* are discarded to avoid slivers.
    Winding order is fixed so every panel's normals point outward from the box.
    """
    from scipy.spatial import Delaunay

    box_center = positions[panel_id == 0].mean(axis=0)

    result: dict[int, np.ndarray] = {}
    for pid in np.unique(panel_id):
        global_idx = np.where(panel_id == pid)[0]
        local_uv = uv[global_idx]
        try:
            tri = Delaunay(local_uv).simplices.astype(np.int32)
        except Exception:
            result[int(pid)] = np.zeros((0, 3), dtype=np.int32)
            continue
        p = local_uv[tri]
        e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
        e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        valid = (e0 < max_uv_edge) & (e1 < max_uv_edge) & (e2 < max_uv_edge)
        tris = global_idx[tri[valid]]

        # Fix winding: ensure normals point outward (away from box center).
        # All Delaunay triangles in one panel have the same UV winding, so a
        # single panel-wide flip is sufficient.
        if len(tris) > 0:
            v0 = positions[tris[:, 0]]
            v1 = positions[tris[:, 1]]
            v2 = positions[tris[:, 2]]
            mean_normal = np.cross(v1 - v0, v2 - v0).sum(axis=0)
            outward = positions[global_idx].mean(axis=0) - box_center
            if np.dot(mean_normal, outward) < 0:
                tris = tris[:, [0, 2, 1]]  # flip winding

        result[int(pid)] = tris
    return result


def _build_tri_indices(
    panel_id: np.ndarray, uv: np.ndarray, positions: np.ndarray, max_uv_edge: float = 0.08
) -> np.ndarray:
    """Combined triangle index array across all panels (for rendering)."""
    per_panel = _build_per_panel_tris(panel_id, uv, positions, max_uv_edge)
    parts = [v for v in per_panel.values() if len(v) > 0]
    return np.vstack(parts).astype(np.int32) if parts else np.zeros((0, 3), dtype=np.int32)



def _build_hinge_stitch_tris(
    positions: np.ndarray,
    panel_id: np.ndarray,
    material_id: np.ndarray,
) -> np.ndarray:
    """Triangle strip bridging the wall-flap gap at each hinge.

    Uses material_id==1 (hinge-strip particles) to locate the fold edge on
    both the wall and flap sides.  For each flap, the nearest wall hinge
    particle is found per flap hinge particle via KD-tree, then both sets are
    sorted along the fold axis and zipped into a triangle strip.
    """
    from scipy.spatial import cKDTree

    static_mask = np.isin(panel_id, list(STATIC_PANELS))
    box_center = positions[panel_id == 0].mean(axis=0)

    # All static hinge-strip particles (fold-edge candidates on the wall side)
    static_hinge_idx = np.where(static_mask & (material_id == 1))[0]
    static_hinge_tree = cKDTree(positions[static_hinge_idx])

    all_tris: list[np.ndarray] = []

    for flap_pid in sorted(set(range(9)) - STATIC_PANELS):
        # Flap hinge-strip particles
        flap_hinge_idx = np.where((panel_id == flap_pid) & (material_id == 1))[0]
        if len(flap_hinge_idx) < 2:
            continue

        # Nearest static hinge particle per flap hinge particle
        _, nn = static_hinge_tree.query(positions[flap_hinge_idx])
        wall_matches = static_hinge_idx[nn]

        # Keep only matches from the single most-adjacent wall panel
        adj_pid = panel_id[wall_matches]
        wall_pid = np.bincount(adj_pid).argmax()
        valid = adj_pid == wall_pid
        flap_pts = flap_hinge_idx[valid]
        wall_pts = wall_matches[valid]
        if len(flap_pts) < 2:
            continue

        # Deduplicate: keep one flap particle per wall particle (closest).
        # The wall hinge is sparser than the flap hinge, so naive KD-tree
        # matching maps 2+ flap particles to each wall particle, causing
        # crossed triangles in the strip.
        flap_for_wall: dict[int, tuple[int, float]] = {}
        for fp_idx, wp_idx in zip(flap_pts.tolist(), wall_pts.tolist()):
            d = float(np.linalg.norm(positions[fp_idx] - positions[wp_idx]))
            if wp_idx not in flap_for_wall or d < flap_for_wall[wp_idx][1]:
                flap_for_wall[wp_idx] = (fp_idx, d)
        wall_uniq = np.array(list(flap_for_wall.keys()))
        flap_uniq = np.array([flap_for_wall[w][0] for w in wall_uniq], dtype=np.int32)

        # Sort both along the fold axis (PCA of wall hinge positions)
        wp = positions[wall_uniq]
        center = wp.mean(axis=0)
        _, _, Vt = np.linalg.svd(wp - center, full_matrices=False)
        order = np.argsort((wp - center) @ Vt[0])
        sorted_wall = wall_uniq[order]
        sorted_flap = flap_uniq[order]

        # Triangle strip: adjacent pair → quad → 2 triangles
        tris: list[list[int]] = []
        for k in range(len(sorted_flap) - 1):
            w0, w1 = int(sorted_wall[k]), int(sorted_wall[k + 1])
            f0, f1 = int(sorted_flap[k]), int(sorted_flap[k + 1])
            tris += [[w0, f0, w1], [w1, f0, f1]]

        if not tris:
            continue
        tris_arr = np.array(tris, dtype=np.int32)

        # Fix winding: normals should point outward from box center
        v0 = positions[tris_arr[:, 0]]
        v1 = positions[tris_arr[:, 1]]
        v2 = positions[tris_arr[:, 2]]
        mean_normal = np.cross(v1 - v0, v2 - v0).sum(axis=0)
        outward = positions[sorted_flap].mean(axis=0) - box_center
        if np.dot(mean_normal, outward) < 0:
            tris_arr = tris_arr[:, [0, 2, 1]]

        all_tris.append(tris_arr)

    return np.vstack(all_tris).astype(np.int32) if all_tris else np.zeros((0, 3), dtype=np.int32)


def _make_checker_texture(size: int = 512, tiles: int = 8) -> np.ndarray:
    """Generate a checker-pattern texture as a (H, W, 3) uint8 array."""
    t = (np.arange(size) * tiles // size) % 2
    checker = (t[:, None] ^ t[None, :]).astype(np.uint8)
    light = np.array([220, 210, 190], dtype=np.uint8)
    dark = np.array([55, 50, 45], dtype=np.uint8)
    return np.where(checker[:, :, None], light, dark).astype(np.uint8)


def _rotate_flaps_to_angle(
    positions: np.ndarray,
    panel_id: np.ndarray,
    constraints: np.ndarray,
    ctype: np.ndarray,
    target_angle_deg: float,
) -> np.ndarray:
    """Rotate each flap panel so its preferred hinge angle is *target_angle_deg*.

    Angle convention (per-flap, measured from the hinge line):
      0°   – perpendicular to the ground (+Z, flap standing straight up)
      +    – toward the box interior
      –    – toward the outside

    Returns a copy of *positions* with flap particles moved accordingly.
    The dihedral rest angles for ``add_edge`` are auto-computed from these
    positions, so the bending constraints will resist deviations from it.
    """
    from scipy.spatial.transform import Rotation

    pos = positions.copy()
    static_mask = np.isin(panel_id, list(STATIC_PANELS))
    hinge_pairs = constraints[ctype == 2]
    h_i, h_j = hinge_pairs[:, 0], hinge_pairs[:, 1]
    box_center = pos[panel_id == 0].mean(axis=0)

    for flap_pid in sorted(set(range(9)) - STATIC_PANELS):
        flap_idx = np.where(panel_id == flap_pid)[0]

        # Flap-side particles of springs crossing the fold line
        i_cross = np.isin(h_i, flap_idx) & static_mask[h_j]
        j_cross = np.isin(h_j, flap_idx) & static_mask[h_i]
        hinge_flap_idx = np.unique(np.concatenate([h_i[i_cross], h_j[j_cross]]))
        if len(hinge_flap_idx) == 0:
            continue

        hinge_center = pos[hinge_flap_idx].mean(axis=0)

        # Fold axis: principal direction of the hinge-edge particles
        _, _, Vt = np.linalg.svd(pos[hinge_flap_idx] - hinge_center, full_matrices=False)
        fold_axis = Vt[0] / np.linalg.norm(Vt[0])

        # Current flap direction from the hinge, projected ⊥ to fold axis
        flap_dir = pos[flap_idx].mean(axis=0) - hinge_center
        flap_dir -= np.dot(flap_dir, fold_axis) * fold_axis
        if np.linalg.norm(flap_dir) < 1e-6:
            continue
        current_dir = flap_dir / np.linalg.norm(flap_dir)

        # 0° base direction: +Z projected onto the plane ⊥ to fold axis
        z_perp = np.array([0.0, 0.0, 1.0])
        z_perp -= np.dot(z_perp, fold_axis) * fold_axis
        if np.linalg.norm(z_perp) < 1e-6:
            continue
        z_perp /= np.linalg.norm(z_perp)

        # Inward direction: toward box center, projected ⊥ to fold axis
        inward = box_center - hinge_center
        inward -= np.dot(inward, fold_axis) * fold_axis
        if np.linalg.norm(inward) < 1e-6:
            inward = np.cross(fold_axis, z_perp)
        else:
            inward /= np.linalg.norm(inward)

        # Target direction in the plane ⊥ to fold axis
        a = np.radians(target_angle_deg)
        target_dir = np.cos(a) * z_perp + np.sin(a) * inward
        target_dir /= np.linalg.norm(target_dir)

        # Signed rotation angle from current_dir to target_dir around fold_axis
        cross = np.cross(current_dir, target_dir)
        rotation_angle = np.arctan2(np.dot(cross, fold_axis), np.dot(current_dir, target_dir))

        rot = Rotation.from_rotvec(rotation_angle * fold_axis)
        pos[flap_idx] = hinge_center + rot.apply(pos[flap_idx] - hinge_center)

    return pos


class _BoxXpbdCouplerAdapter:
    """LegacyLikeCoupler adapter with a fixed ground-plane rigid proxy."""

    def __init__(self, sim_dt: float, state_out, static_mask: np.ndarray):
        self._sim_dt = sim_dt
        self._state_out = state_out
        self._dynamic_idx = np.where(~static_mask)[0]
        self._plane_z = 0.0
        self._material = RigidCouplingMaterial(
            needs_coup=True,
            coup_friction=0.15,
            coup_softness=0.003,
            coup_restitution=0.0,
        )
        self._coupler: LegacyLikeCoupler | None = None
        self.reaction_force_accum = np.zeros(3, dtype=np.float64)

    def set_coupler(self, coupler: LegacyLikeCoupler) -> None:
        self._coupler = coupler

    def is_rigid_active(self) -> bool:
        return True

    def is_xpbd_active(self) -> bool:
        return True

    def substep_dt(self) -> float:
        return self._sim_dt

    def query_rigid_collision(self, pos_world: np.ndarray) -> CollisionSample:
        signed_distance = float(pos_world[2] - self._plane_z)
        return CollisionSample(
            valid=(signed_distance <= self._material.coup_softness * 3.0),
            signed_distance=signed_distance,
            normal_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            rigid_body_id=0,
            material=self._material,
        )

    def rigid_velocity_at_point(self, rigid_body_id: int, pos_world: np.ndarray) -> np.ndarray:
        _ = rigid_body_id, pos_world
        return np.zeros(3, dtype=np.float64)

    def apply_coupling_force(self, rigid_body_id: int, force_world: np.ndarray, at_pos_world: np.ndarray) -> None:
        _ = rigid_body_id, at_pos_world
        self.reaction_force_accum += force_world

    def couple_xpbd_with_rigid(self) -> None:
        if self._coupler is None:
            return
        q = self._state_out.particle_q.numpy()
        qd = self._state_out.particle_qd.numpy()
        self.reaction_force_accum[:] = 0.0
        for i in self._dynamic_idx:
            pos = q[i].astype(np.float64)
            vel = qd[i].astype(np.float64)
            sample = self.query_rigid_collision(pos)
            if not sample.valid:
                continue
            qd[i] = self._coupler.resolve_rigid_collision(pos, vel, mass=1.0, sample=sample).astype(np.float32)
        self._state_out.particle_qd.assign(qd)


class Example:
    def __init__(self, viewer, npz_path: str | None = None, preferred_angle_deg: float = 0.0):
        if npz_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(here, "../../../box_data/apriltag_template/box_hinge_xpbd.npz")

        self._npz_path = npz_path
        box = _load_npz(npz_path)

        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = box["substeps"]  # 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        panel_id = box["panel_id"]
        static_mask = np.isin(panel_id, list(STATIC_PANELS))

        # --- Build Newton model ---
        # Particle radius 2mm for visibility in the GL renderer.
        # Contact detection is disabled via particle_grid=None below.
        builder = newton.ModelBuilder()
        builder.default_particle_radius = 0.002

        pos = box["positions"]
        n = len(pos)

        # Rotate flaps to the preferred angle so hinge spring rest-lengths encode it.
        # Simulation starts at this angle; springs resist deviations from it.
        pref_pos = _rotate_flaps_to_angle(
            pos, panel_id, box["constraints"], box["constraint_type"], preferred_angle_deg
        )

        # Add particles individually to assign per-particle flags correctly.
        # Static panels: flags=0 (INACTIVE, position frozen), mass=0 (no
        #   spring force contribution, inv_mass=0).
        # Dynamic flaps: flags=ACTIVE, mass=1.
        for i in range(n):
            flag = 0 if static_mask[i] else int(newton.ParticleFlags.ACTIVE)
            mass = 0.0 if static_mask[i] else 1.0
            builder.add_particle(wp.vec3(*pref_pos[i]), wp.vec3(0.0, 0.0, 0.0), mass=mass, flags=flag)

        # Add XPBD spring constraints (type 2 hinge springs are replaced by
        # dihedral bending constraints below).
        cons = box["constraints"]
        ctype = box["constraint_type"]
        for idx in range(len(cons)):
            t = int(ctype[idx])
            if t == 2:
                continue  # replaced by add_edge bending constraints
            i, j = int(cons[idx, 0]), int(cons[idx, 1])
            builder.add_spring(i, j, ke=KE_BY_TYPE[t], kd=KD_BY_TYPE[t], control=0.0)

        # Dihedral bending constraints loaded from the pre-built npz.
        # rest_angle is auto-computed from pref_pos so it encodes preferred_angle_deg.
        ke_bend = box["dihedral_ke"]
        for quad in box["dihedral_quads"]:
            builder.add_edge(
                int(quad[0]), int(quad[1]), int(quad[2]), int(quad[3]),
                edge_ke=ke_bend, edge_kd=HINGE_BEND_KD,
            )

        self.model = builder.finalize()
        self.model.particle_max_radius = 0.0  # self-collision disabled (perf)
        self.model.particle_grid = None
        # 8 iterations: enough for ke=1e5 panel springs to converge
        self.solver = SolverXPBD(self.model, iterations=8)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._static_mask = static_mask
        self._init_pos = pref_pos.copy()

        # --- Cloth mesh for textured rendering ---
        device = self.model.particle_q.device
        panel_tris = _build_tri_indices(box["panel_id"], box["uv"], pref_pos)
        stitch_tris = _build_hinge_stitch_tris(pref_pos, box["panel_id"], box["material_id"])
        tri_idx = np.vstack([panel_tris, stitch_tris]) if len(stitch_tris) else panel_tris
        self._cloth_tri_indices = wp.array(tri_idx.flatten(), dtype=wp.int32, device=device)
        self._cloth_uvs = wp.array(box["uv"], dtype=wp.vec2, device=device)
        self._cloth_texture = _make_checker_texture()
        self._cloth_mesh_ready = False

        self._coupler_adapter = _BoxXpbdCouplerAdapter(self.sim_dt, self.state_1, static_mask)
        self._coupler = LegacyLikeCoupler(self._coupler_adapter, CouplerOptions(rigid_xpbd=True))
        self._coupler_adapter.set_coupler(self._coupler)
        self._coupler.build()

        self.viewer.set_model(self.model)
        self._set_camera()

    def _set_camera(self):
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            # Box center ≈ (0.05, 0.11, -0.07).  Camera 0.8m away looking inward.
            self.viewer.camera.pos = type(self.viewer.camera.pos)(0.7, -0.4, 0.5)
            self.viewer.camera.pitch = 30.5
            self.viewer.camera.yaw = -138.5

    def step(self):
        for substep_idx in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self._coupler.preprocess(substep_idx)
            self._coupler.couple(substep_idx)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _log_cloth_mesh(self):
        # Pass texture only on the first call; the GL viewer re-uploads on every
        # non-None texture argument, so skip it once the mesh object is created.
        texture = self._cloth_texture if not self._cloth_mesh_ready else None
        self.viewer.log_mesh(
            "/model/cloth",
            self.state_0.particle_q,
            self._cloth_tri_indices,
            uvs=self._cloth_uvs,
            texture=texture,
            backface_culling=False,
        )
        self._cloth_mesh_ready = True

    def render(self):
        self.viewer.begin_frame(self.frame * self.frame_dt)
        self.viewer.log_state(self.state_0)
        self._log_cloth_mesh()
        self.viewer.end_frame()
        self.frame += 1

    def test_final(self):
        pos = self.state_0.particle_q.numpy()
        dynamic_mask = ~self._static_mask
        displacement = np.linalg.norm(pos[dynamic_mask] - self._init_pos[dynamic_mask], axis=1)
        assert not np.any(np.isnan(displacement)), "Simulation produced NaN"
        assert displacement.max() > 0.001, "Flap particles did not move under gravity"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--npz", default=None, help="Path to box_hinge_xpbd.npz")
    parser.add_argument(
        "--preferred-angle",
        type=float,
        default=0.0,
        help="Hinge preferred angle [deg]. 0=vertical, +=inward, -=outward.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, npz_path=args.npz, preferred_angle_deg=args.preferred_angle)
    newton.examples.run(example, args)
