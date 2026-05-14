# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Diffsim Cloth PhysTwin Interact Spring
#
# Loads a PhysTwin cloth mesh and optional spring graph from an NPZ file, then
# builds a spring-aware Newton cloth model. A subset of anchor vertices can be
# moved directly for interactive pulling / dragging experiments.
#
# Expected NPZ keys:
#   - vertices [N, 3]
#   - faces [F, 3]
# Optional spring keys:
#   - spring_indices or spring_pairs [S, 2]
#   - spring_ke or spring_stiffness [S]
#   - spring_kd or spring_damping [S]
#   - spring_rest_length or spring_rest_lengths [S]
#
# Command:
#   WARP_CACHE_PATH=/tmp/warp-cache uv run --module newton.examples \
#       diffsim_cloth_phystwin_interact_spring --keyboard-speed 0.02
#
"""
WARP_CACHE_PATH=/tmp/warp-cache uv run --module newton.examples diffsim_cloth_phystwin_interact_spring \
    --anchor-mode left_arm \
    --anchor-count 8 \
    --script-pattern fold_left_arm \
    --script-amp-x 0.06 \
    --script-amp-z 0.05 \
    --script-period 3.0
"""
###########################################################################

import argparse
import json
import os
import pickle

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

SPRING_INDEX_KEYS = ("spring_indices", "spring_pairs")
SPRING_KE_KEYS = ("spring_ke", "spring_stiffness")
SPRING_KD_KEYS = ("spring_kd", "spring_damping")
SPRING_REST_KEYS = ("spring_rest_length", "spring_rest_lengths")


def _compact_cloth_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_vertex_ids: np.ndarray | None = None,
    anchor_targets: np.ndarray | None = None,
    object_points: np.ndarray | None = None,
    object_visibilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    used_vertex_ids = np.unique(faces.reshape(-1)).astype(np.int32)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids), dtype=np.int32)

    compacted = {
        "vertices": vertices[used_vertex_ids].astype(np.float32),
        "faces": remap[faces].astype(np.int32),
        "used_vertex_ids": used_vertex_ids,
    }

    if object_points is not None:
        compacted["object_points"] = object_points[:, used_vertex_ids].astype(np.float32)
    if object_visibilities is not None:
        compacted["object_visibilities"] = object_visibilities[:, used_vertex_ids].astype(bool)

    if anchor_vertex_ids is not None:
        valid_anchor_mask = remap[anchor_vertex_ids] >= 0
        compacted["anchor_vertex_ids"] = remap[anchor_vertex_ids[valid_anchor_mask]].astype(np.int32)
        compacted["anchor_valid_mask"] = valid_anchor_mask
        if anchor_targets is not None:
            compacted["anchor_targets"] = anchor_targets[:, valid_anchor_mask].astype(np.float32)

    return compacted


def _build_voxel_proxy_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    voxel_size: float,
    object_points: np.ndarray | None = None,
    object_visibilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if voxel_size <= 0.0:
        return {
            "vertices": vertices.astype(np.float32),
            "faces": faces.astype(np.int32),
            "source_to_proxy": np.arange(len(vertices), dtype=np.int32),
            "object_points": object_points.astype(np.float32) if object_points is not None else None,
            "object_visibilities": object_visibilities.astype(bool) if object_visibilities is not None else None,
        }

    grid = np.floor((vertices - np.min(vertices, axis=0, keepdims=True)) / voxel_size).astype(np.int32)
    _, source_to_proxy = np.unique(grid, axis=0, return_inverse=True)
    proxy_vertex_count = int(source_to_proxy.max()) + 1

    proxy_vertices = np.zeros((proxy_vertex_count, 3), dtype=np.float64)
    counts = np.bincount(source_to_proxy, minlength=proxy_vertex_count).astype(np.float64)
    np.add.at(proxy_vertices, source_to_proxy, vertices)
    proxy_vertices /= counts[:, None]

    remapped_faces = source_to_proxy[faces]
    face_keep = np.logical_and.reduce(
        [
            remapped_faces[:, 0] != remapped_faces[:, 1],
            remapped_faces[:, 1] != remapped_faces[:, 2],
            remapped_faces[:, 0] != remapped_faces[:, 2],
        ]
    )
    remapped_faces = remapped_faces[face_keep]
    if len(remapped_faces) > 0:
        remapped_faces = np.unique(np.sort(remapped_faces, axis=1), axis=0).astype(np.int32)
        remapped_faces = _filter_nonmanifold_faces(remapped_faces)
    else:
        remapped_faces = np.empty((0, 3), dtype=np.int32)

    proxy_object_points = None
    if object_points is not None:
        frame_count = object_points.shape[0]
        proxy_object_points = np.zeros((frame_count, proxy_vertex_count, 3), dtype=np.float64)
        for frame_id in range(frame_count):
            np.add.at(proxy_object_points[frame_id], source_to_proxy, object_points[frame_id])
        proxy_object_points /= counts[None, :, None]
        proxy_object_points = proxy_object_points.astype(np.float32)

    proxy_object_visibilities = None
    if object_visibilities is not None:
        frame_count = object_visibilities.shape[0]
        vis_counts = np.zeros((frame_count, proxy_vertex_count), dtype=np.float64)
        for frame_id in range(frame_count):
            np.add.at(vis_counts[frame_id], source_to_proxy, object_visibilities[frame_id].astype(np.float64))
        proxy_object_visibilities = vis_counts >= 0.5

    return {
        "vertices": proxy_vertices.astype(np.float32),
        "faces": remapped_faces.astype(np.int32),
        "source_to_proxy": source_to_proxy.astype(np.int32),
        "object_points": proxy_object_points,
        "object_visibilities": proxy_object_visibilities,
    }


def _filter_nonmanifold_faces(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces.astype(np.int32)

    edge_counts: dict[tuple[int, int], int] = {}
    face_edges: list[list[tuple[int, int]]] = []
    for tri in faces.tolist():
        tri_edges = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (min(a, b), max(a, b))
            tri_edges.append(edge)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        face_edges.append(tri_edges)

    keep = []
    for tri, tri_edges in zip(faces.tolist(), face_edges, strict=False):
        if all(edge_counts[edge] <= 2 for edge in tri_edges):
            keep.append(tri)
    if not keep:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(keep, dtype=np.int32)


def _resolve_reference_pkl_path(dataset_path: str, explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None

    dataset_dir = os.path.dirname(dataset_path)
    candidate = os.path.join(os.path.dirname(dataset_dir), "final_data.pkl")
    if os.path.exists(candidate):
        return candidate
    return None


def _load_reference_policy_template(
    dataset_path: str,
    reference_pkl_path: str | None,
    anchor_count: int,
) -> dict[str, object]:
    dataset = np.load(dataset_path)
    try:
        compacted = _compact_cloth_topology(
            np.asarray(dataset["vertices"], dtype=np.float32),
            np.asarray(dataset["faces"], dtype=np.int32),
            np.asarray(dataset["anchor_vertex_ids"], dtype=np.int32),
            np.asarray(dataset["anchor_targets"], dtype=np.float32),
        )
        vertices = compacted["vertices"]
        dataset_anchor_ids = compacted["anchor_vertex_ids"]
        anchor_targets = compacted["anchor_targets"]
        anchor_valid_mask = compacted["anchor_valid_mask"]
    finally:
        dataset.close()

    controller_points = anchor_targets
    pkl_path = _resolve_reference_pkl_path(dataset_path, reference_pkl_path)
    if pkl_path is not None:
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        if "controller_points" in pkl_data:
            controller_points = np.asarray(pkl_data["controller_points"], dtype=np.float32)
            controller_points = controller_points[:, anchor_valid_mask]

    controller_delta = controller_points - controller_points[0:1]
    frame_mean_delta = controller_delta.mean(axis=1)
    frame_motion = np.linalg.norm(frame_mean_delta, axis=1)
    peak_frame = int(np.argmax(frame_motion))
    peak_delta = frame_mean_delta[peak_frame]

    per_anchor_motion = np.linalg.norm(controller_delta[peak_frame], axis=1)
    sorted_anchor_idx = np.argsort(-per_anchor_motion)
    top_anchor_idx = sorted_anchor_idx[: max(anchor_count, min(12, len(sorted_anchor_idx)))]
    # Spatially bias towards a coherent left-side patch if the motion has a strong x component.
    if abs(float(peak_delta[0])) >= abs(float(peak_delta[1])):
        top_anchor_idx = top_anchor_idx[np.argsort(vertices[dataset_anchor_ids[top_anchor_idx], 0])]
    selected_anchor_idx = top_anchor_idx[: min(anchor_count, len(top_anchor_idx))]
    selected_anchor_vertex_ids = dataset_anchor_ids[selected_anchor_idx].astype(np.int32)

    selected_delta = controller_delta[:, selected_anchor_idx]
    selected_mean_delta = selected_delta.mean(axis=1)
    z_profile = np.maximum(selected_mean_delta[:, 2], 0.0)
    x_profile = np.abs(selected_mean_delta[:, 0])

    lift_peak_frame = int(np.argmax(z_profile))
    fold_peak_frame = int(np.argmax(x_profile))
    lift_phase_ratio = float(np.clip(max(lift_peak_frame, 1) / max(peak_frame, 1), 0.18, 0.55))

    z_peak = float(np.max(z_profile))
    hold_phase_ratio = 0.12
    if z_peak > 1.0e-6 and fold_peak_frame > lift_peak_frame:
        hold_mask = z_profile[lift_peak_frame : fold_peak_frame + 1] >= 0.88 * z_peak
        hold_phase_ratio = float(
            np.clip(np.mean(hold_mask) * max(fold_peak_frame - lift_peak_frame, 1) / max(peak_frame, 1), 0.05, 0.25)
        )

    vertex_extent = np.maximum(np.ptp(vertices, axis=0), 1.0e-6)
    anchor_points = vertices[selected_anchor_vertex_ids]
    patch_scale = float(
        np.clip(
            1.25
            + 1.5 * max(np.ptp(anchor_points[:, 1]) / vertex_extent[1], np.ptp(anchor_points[:, 0]) / vertex_extent[0]),
            1.0,
            2.0,
        )
    )

    return {
        "template_source": "pkl" if pkl_path is not None else "npz",
        "anchor_vertex_ids": selected_anchor_vertex_ids.tolist(),
        "candidate_anchor_vertex_ids": dataset_anchor_ids[top_anchor_idx].astype(np.int32).tolist(),
        "peak_frame": peak_frame,
        "lift_peak_frame": lift_peak_frame,
        "fold_peak_frame": fold_peak_frame,
        "base_period": float(np.clip(peak_frame / 45.0, 2.5, 6.0)),
        "base_amp_x": float(np.clip(abs(selected_mean_delta[peak_frame, 0]) * 0.18, 0.012, 0.08)),
        "base_amp_y": float(np.clip(abs(selected_mean_delta[peak_frame, 1]) * 0.10, 0.0, 0.03)),
        "base_amp_z": float(np.clip(max(z_peak * 0.22, 0.008), 0.008, 0.05)),
        "lift_phase_ratio": lift_phase_ratio,
        "hold_phase_ratio": hold_phase_ratio,
        "base_patch_scale": patch_scale,
        "direction_x": -1.0 if float(selected_mean_delta[peak_frame, 0]) >= 0.0 else 1.0,
    }


def _load_unified_template(unified_template_path: str) -> dict[str, object]:
    unified = np.load(unified_template_path, allow_pickle=True)
    try:
        if "metadata_json" not in unified:
            raise ValueError("Unified template file must contain metadata_json.")
        metadata = json.loads(str(unified["metadata_json"]))
        if "global_template" not in metadata:
            raise ValueError("Unified template metadata must contain global_template.")
        return metadata["global_template"]
    finally:
        unified.close()


def _load_phase_machine_template(phase_machine_template_path: str) -> dict[str, object]:
    with open(phase_machine_template_path, encoding="utf-8") as f:
        payload = json.load(f)
    if "phases" not in payload or not payload["phases"]:
        raise ValueError("Phase machine template must contain a non-empty phases list.")
    return payload


def _scaled_template(template: dict[str, object], args) -> dict[str, object]:
    scaled = dict(template)
    scaled["base_amp_x"] = float(template["base_amp_x"]) * args.unified_amp_x_scale
    scaled["base_amp_y"] = float(template.get("base_amp_y", 0.0)) * args.unified_amp_y_scale
    scaled["base_amp_z"] = float(template["base_amp_z"]) * args.unified_amp_z_scale
    scaled["base_period"] = float(template["base_period"]) * args.unified_period_scale
    scaled["base_patch_scale"] = max(
        float(template["base_patch_scale"]) * args.unified_patch_scale_factor,
        args.unified_patch_scale,
    )
    return scaled


def _range_value(range_values: list[float], mode: str, minimize_magnitude: bool = False) -> float:
    lo = float(range_values[0])
    hi = float(range_values[1])
    if mode == "random":
        raise ValueError("Random selection must be handled by the caller.")
    if mode == "fixed":
        if minimize_magnitude:
            return lo if abs(lo) <= abs(hi) else hi
        return lo
    return 0.5 * (lo + hi)


def _find_first_key(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def _load_optional_springs(
    data: np.lib.npyio.NpzFile,
    default_ke: float,
    default_kd: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    spring_index_key = _find_first_key(data, SPRING_INDEX_KEYS)
    if spring_index_key is None:
        return None, None, None, None

    spring_pairs = np.asarray(data[spring_index_key], dtype=np.int32)
    if spring_pairs.ndim == 1:
        if spring_pairs.size % 2 != 0:
            raise ValueError(f"{spring_index_key} must contain an even number of entries.")
        spring_pairs = spring_pairs.reshape(-1, 2)
    if spring_pairs.ndim != 2 or spring_pairs.shape[1] != 2:
        raise ValueError(f"{spring_index_key} must have shape [spring_count, 2].")

    spring_count = spring_pairs.shape[0]

    spring_ke_key = _find_first_key(data, SPRING_KE_KEYS)
    if spring_ke_key is None:
        spring_ke = np.full(spring_count, default_ke, dtype=np.float32)
    else:
        spring_ke = np.asarray(data[spring_ke_key], dtype=np.float32).reshape(-1)
        if spring_ke.shape[0] != spring_count:
            raise ValueError(f"{spring_ke_key} length must match {spring_index_key}.")

    spring_kd_key = _find_first_key(data, SPRING_KD_KEYS)
    if spring_kd_key is None:
        spring_kd = np.full(spring_count, default_kd, dtype=np.float32)
    else:
        spring_kd = np.asarray(data[spring_kd_key], dtype=np.float32).reshape(-1)
        if spring_kd.shape[0] != spring_count:
            raise ValueError(f"{spring_kd_key} length must match {spring_index_key}.")

    spring_rest_key = _find_first_key(data, SPRING_REST_KEYS)
    spring_rest = None
    if spring_rest_key is not None:
        spring_rest = np.asarray(data[spring_rest_key], dtype=np.float32).reshape(-1)
        if spring_rest.shape[0] != spring_count:
            raise ValueError(f"{spring_rest_key} length must match {spring_index_key}.")

    return spring_pairs, spring_ke, spring_kd, spring_rest


def _compute_bending_edges(faces: np.ndarray) -> np.ndarray:
    """Build edge rows [o0, o1, v0, v1] from a triangle mesh."""
    edge_map: dict[tuple[int, int], list[int]] = {}

    for a, b, c in faces.tolist():
        for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            key = (min(u, v), max(u, v))
            if key not in edge_map:
                edge_map[key] = [opposite, -1, u, v]
            else:
                if edge_map[key][1] != -1:
                    raise ValueError("Cloth mesh must be two-manifold to build bending edges.")
                edge_map[key][1] = opposite

    if not edge_map:
        return np.empty((0, 4), dtype=np.int32)

    return np.asarray(list(edge_map.values()), dtype=np.int32)


def _generate_mesh_springs(
    vertices: np.ndarray,
    faces: np.ndarray,
    include_shear: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate spring pairs and rest lengths from mesh topology.

    Structural springs are created on every unique triangle edge.
    If ``include_shear`` is enabled, an additional spring is created between
    the opposite vertices of each adjacent triangle pair.
    """
    spring_pairs: set[tuple[int, int]] = set()

    for a, b, c in faces.tolist():
        spring_pairs.add((min(a, b), max(a, b)))
        spring_pairs.add((min(b, c), max(b, c)))
        spring_pairs.add((min(c, a), max(c, a)))

    if include_shear:
        for o0, o1, _, _ in _compute_bending_edges(faces).tolist():
            if o0 != -1 and o1 != -1:
                spring_pairs.add((min(o0, o1), max(o0, o1)))

    spring_pairs_np = np.asarray(sorted(spring_pairs), dtype=np.int32)
    if len(spring_pairs_np) == 0:
        return spring_pairs_np, np.empty((0,), dtype=np.float32)

    rest = np.linalg.norm(
        vertices[spring_pairs_np[:, 0]] - vertices[spring_pairs_np[:, 1]],
        axis=1,
    ).astype(np.float32)
    return spring_pairs_np, rest


def _split_shear_springs(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_pairs, _ = _generate_mesh_springs(vertices, faces, include_shear=False)
    all_pairs, all_rest = _generate_mesh_springs(vertices, faces, include_shear=True)
    edge_pair_set = {tuple(pair) for pair in edge_pairs.tolist()}
    shear_pairs = []
    shear_rest = []
    for pair, rest in zip(all_pairs.tolist(), all_rest.tolist(), strict=False):
        if tuple(pair) not in edge_pair_set:
            shear_pairs.append(pair)
            shear_rest.append(rest)
    if not shear_pairs:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    return np.asarray(shear_pairs, dtype=np.int32), np.asarray(shear_rest, dtype=np.float32)


def _select_left_arm_end_anchors(vertices: np.ndarray, anchor_count: int) -> np.ndarray:
    x_cut = np.quantile(vertices[:, 0], 0.035)
    y_low = np.quantile(vertices[:, 1], 0.18)
    y_high = np.quantile(vertices[:, 1], 0.82)
    candidate_ids = np.where((vertices[:, 0] <= x_cut) & (vertices[:, 1] >= y_low) & (vertices[:, 1] <= y_high))[0]
    if len(candidate_ids) == 0:
        candidate_ids = np.where(vertices[:, 0] <= x_cut)[0]
    if len(candidate_ids) == 0:
        candidate_ids = np.arange(len(vertices))
    ordered = candidate_ids[np.argsort(vertices[candidate_ids, 1])]
    if anchor_count > 1:
        pick_positions = np.linspace(0, len(ordered) - 1, num=min(anchor_count, len(ordered))).astype(int)
        return ordered[pick_positions].astype(np.int32)
    return ordered[:1].astype(np.int32)


def _select_right_arm_end_anchors(vertices: np.ndarray, anchor_count: int) -> np.ndarray:
    x_cut = np.quantile(vertices[:, 0], 0.965)
    y_low = np.quantile(vertices[:, 1], 0.18)
    y_high = np.quantile(vertices[:, 1], 0.82)
    candidate_ids = np.where((vertices[:, 0] >= x_cut) & (vertices[:, 1] >= y_low) & (vertices[:, 1] <= y_high))[0]
    if len(candidate_ids) == 0:
        candidate_ids = np.where(vertices[:, 0] >= x_cut)[0]
    if len(candidate_ids) == 0:
        candidate_ids = np.arange(len(vertices))
    ordered = candidate_ids[np.argsort(vertices[candidate_ids, 1])]
    if anchor_count > 1:
        pick_positions = np.linspace(0, len(ordered) - 1, num=min(anchor_count, len(ordered))).astype(int)
        return ordered[pick_positions].astype(np.int32)
    return ordered[:1].astype(np.int32)


def _select_torso_center_anchors(vertices: np.ndarray, anchor_count: int) -> np.ndarray:
    x_center = float(np.median(vertices[:, 0]))
    x_half = max(0.10 * float(np.ptp(vertices[:, 0])), 0.03)
    y_low = np.quantile(vertices[:, 1], 0.30)
    y_high = np.quantile(vertices[:, 1], 0.72)
    z_low = np.quantile(vertices[:, 2], 0.20)
    z_high = np.quantile(vertices[:, 2], 0.85)
    candidate_ids = np.where(
        (vertices[:, 0] >= x_center - x_half)
        & (vertices[:, 0] <= x_center + x_half)
        & (vertices[:, 1] >= y_low)
        & (vertices[:, 1] <= y_high)
        & (vertices[:, 2] >= z_low)
        & (vertices[:, 2] <= z_high)
    )[0]
    if len(candidate_ids) == 0:
        candidate_ids = np.argsort(np.abs(vertices[:, 0] - x_center))[: max(anchor_count, 8)]
    center = np.mean(vertices[candidate_ids], axis=0)
    distances = np.linalg.norm(vertices[candidate_ids] - center[None, :], axis=1)
    ordered = candidate_ids[np.argsort(distances)]
    return ordered[: min(anchor_count, len(ordered))].astype(np.int32)


def _select_left_arm_grip_region(vertices: np.ndarray, visible_anchor_ids: np.ndarray) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_pad = max(0.10 * float(np.ptp(vertices[:, 0])), 0.03)
    y_pad = max(0.05 * float(np.ptp(vertices[:, 1])), 0.02)
    z_pad = max(0.08 * float(np.ptp(vertices[:, 2])), 0.02)
    x_max = float(np.max(anchor_points[:, 0])) + x_pad
    y_min = float(np.min(anchor_points[:, 1])) - y_pad
    y_max = float(np.max(anchor_points[:, 1])) + y_pad
    z_min = float(np.min(anchor_points[:, 2])) - z_pad
    z_max = float(np.max(anchor_points[:, 2])) + z_pad
    region_ids = np.where(
        (vertices[:, 0] <= x_max)
        & (vertices[:, 1] >= y_min)
        & (vertices[:, 1] <= y_max)
        & (vertices[:, 2] >= z_min)
        & (vertices[:, 2] <= z_max)
    )[0]
    if len(region_ids) == 0:
        return visible_anchor_ids
    return region_ids.astype(np.int32)


def _select_right_arm_grip_region(vertices: np.ndarray, visible_anchor_ids: np.ndarray) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_pad = max(0.10 * float(np.ptp(vertices[:, 0])), 0.03)
    y_pad = max(0.05 * float(np.ptp(vertices[:, 1])), 0.02)
    z_pad = max(0.08 * float(np.ptp(vertices[:, 2])), 0.02)
    x_min = float(np.min(anchor_points[:, 0])) - x_pad
    y_min = float(np.min(anchor_points[:, 1])) - y_pad
    y_max = float(np.max(anchor_points[:, 1])) + y_pad
    z_min = float(np.min(anchor_points[:, 2])) - z_pad
    z_max = float(np.max(anchor_points[:, 2])) + z_pad
    region_ids = np.where(
        (vertices[:, 0] >= x_min)
        & (vertices[:, 1] >= y_min)
        & (vertices[:, 1] <= y_max)
        & (vertices[:, 2] >= z_min)
        & (vertices[:, 2] <= z_max)
    )[0]
    if len(region_ids) == 0:
        return visible_anchor_ids
    return region_ids.astype(np.int32)


def _select_torso_center_patch(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    patch_scale: float,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_center = float(np.mean(anchor_points[:, 0]))
    y_center = float(np.mean(anchor_points[:, 1]))
    z_center = float(np.mean(anchor_points[:, 2]))
    x_half = max(0.14 * patch_scale * float(np.ptp(vertices[:, 0])), 0.05)
    y_half = max(0.16 * patch_scale * float(np.ptp(vertices[:, 1])), 0.06)
    z_half = max(0.14 * patch_scale * float(np.ptp(vertices[:, 2])), 0.04)
    patch_ids = np.where(
        (vertices[:, 0] >= x_center - x_half)
        & (vertices[:, 0] <= x_center + x_half)
        & (vertices[:, 1] >= y_center - y_half)
        & (vertices[:, 1] <= y_center + y_half)
        & (vertices[:, 2] >= z_center - z_half)
        & (vertices[:, 2] <= z_center + z_half)
    )[0]
    if len(patch_ids) == 0:
        patch_ids = visible_anchor_ids
    merged = np.unique(np.concatenate([visible_anchor_ids.astype(np.int32), patch_ids.astype(np.int32)]))
    return merged.astype(np.int32)


def _select_left_arm_cuff_patch(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_tip = float(np.min(anchor_points[:, 0]))
    x_pad = max(0.08 * float(np.ptp(vertices[:, 0])), 0.025)
    y_center = float(np.mean(anchor_points[:, 1]))
    y_pad = max(0.12 * float(np.ptp(vertices[:, 1])), 0.03)
    z_center = float(np.mean(anchor_points[:, 2]))
    z_pad = max(0.10 * float(np.ptp(vertices[:, 2])), 0.025)

    cuff_candidates = np.where(
        (vertices[:, 0] <= x_tip + x_pad)
        & (vertices[:, 1] >= y_center - y_pad)
        & (vertices[:, 1] <= y_center + y_pad)
        & (vertices[:, 2] >= z_center - z_pad)
        & (vertices[:, 2] <= z_center + z_pad)
    )[0]
    if len(cuff_candidates) == 0:
        cuff_candidates = _select_left_arm_grip_region(vertices, visible_anchor_ids)

    center = np.mean(anchor_points, axis=0)
    distances = np.linalg.norm(vertices[cuff_candidates] - center[None, :], axis=1)
    ordered = cuff_candidates[np.argsort(distances)]
    count = min(max(patch_count, len(visible_anchor_ids)), len(ordered))
    patch_ids = ordered[:count]
    merged = np.unique(np.concatenate([visible_anchor_ids.astype(np.int32), patch_ids.astype(np.int32)]))
    return merged.astype(np.int32)


def _select_right_arm_cuff_patch(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    patch_count: int,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_tip = float(np.max(anchor_points[:, 0]))
    x_pad = max(0.08 * float(np.ptp(vertices[:, 0])), 0.025)
    y_center = float(np.mean(anchor_points[:, 1]))
    y_pad = max(0.12 * float(np.ptp(vertices[:, 1])), 0.03)
    z_center = float(np.mean(anchor_points[:, 2]))
    z_pad = max(0.10 * float(np.ptp(vertices[:, 2])), 0.025)

    cuff_candidates = np.where(
        (vertices[:, 0] >= x_tip - x_pad)
        & (vertices[:, 1] >= y_center - y_pad)
        & (vertices[:, 1] <= y_center + y_pad)
        & (vertices[:, 2] >= z_center - z_pad)
        & (vertices[:, 2] <= z_center + z_pad)
    )[0]
    if len(cuff_candidates) == 0:
        cuff_candidates = _select_right_arm_grip_region(vertices, visible_anchor_ids)

    center = np.mean(anchor_points, axis=0)
    distances = np.linalg.norm(vertices[cuff_candidates] - center[None, :], axis=1)
    ordered = cuff_candidates[np.argsort(distances)]
    count = min(max(patch_count, len(visible_anchor_ids)), len(ordered))
    patch_ids = ordered[:count]
    merged = np.unique(np.concatenate([visible_anchor_ids.astype(np.int32), patch_ids.astype(np.int32)]))
    return merged.astype(np.int32)


def _select_left_arm_full_patch(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    patch_scale: float,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_tip = float(np.min(anchor_points[:, 0]))
    x_extent = float(np.ptp(vertices[:, 0]))
    y_extent = float(np.ptp(vertices[:, 1]))
    z_extent = float(np.ptp(vertices[:, 2]))

    x_max = x_tip + max(0.22 * patch_scale * x_extent, 0.08)
    y_center = float(np.mean(anchor_points[:, 1]))
    y_half = max(0.22 * patch_scale * y_extent, 0.08)
    z_center = float(np.mean(anchor_points[:, 2]))
    z_half = max(0.18 * patch_scale * z_extent, 0.06)

    patch_ids = np.where(
        (vertices[:, 0] <= x_max)
        & (vertices[:, 1] >= y_center - y_half)
        & (vertices[:, 1] <= y_center + y_half)
        & (vertices[:, 2] >= z_center - z_half)
        & (vertices[:, 2] <= z_center + z_half)
    )[0]
    if len(patch_ids) == 0:
        return _select_left_arm_cuff_patch(vertices, visible_anchor_ids, max(1, int(16 * patch_scale)))
    merged = np.unique(np.concatenate([visible_anchor_ids.astype(np.int32), patch_ids.astype(np.int32)]))
    return merged.astype(np.int32)


def _select_right_arm_full_patch(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    patch_scale: float,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    x_tip = float(np.max(anchor_points[:, 0]))
    x_extent = float(np.ptp(vertices[:, 0]))
    y_extent = float(np.ptp(vertices[:, 1]))
    z_extent = float(np.ptp(vertices[:, 2]))

    x_min = x_tip - max(0.22 * patch_scale * x_extent, 0.08)
    y_center = float(np.mean(anchor_points[:, 1]))
    y_half = max(0.22 * patch_scale * y_extent, 0.08)
    z_center = float(np.mean(anchor_points[:, 2]))
    z_half = max(0.18 * patch_scale * z_extent, 0.06)

    patch_ids = np.where(
        (vertices[:, 0] >= x_min)
        & (vertices[:, 1] >= y_center - y_half)
        & (vertices[:, 1] <= y_center + y_half)
        & (vertices[:, 2] >= z_center - z_half)
        & (vertices[:, 2] <= z_center + z_half)
    )[0]
    if len(patch_ids) == 0:
        return _select_right_arm_cuff_patch(vertices, visible_anchor_ids, max(1, int(16 * patch_scale)))
    merged = np.unique(np.concatenate([visible_anchor_ids.astype(np.int32), patch_ids.astype(np.int32)]))
    return merged.astype(np.int32)


def _compute_left_arm_grip_weights(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    control_vertex_ids: np.ndarray,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    control_points = vertices[control_vertex_ids]

    # Weight by proximity to the cuff tip so the patch behaves like a small gripper.
    dists = np.linalg.norm(control_points[:, None, :] - anchor_points[None, :, :], axis=2)
    min_dist = np.min(dists, axis=1)
    radius = max(0.16 * float(np.ptp(vertices[:, 0])), 0.06)
    weights = np.clip(1.0 - min_dist / radius, 0.0, 1.0)

    # Keep the whole cuff patch moving cohesively.
    weights = 0.55 + 0.45 * weights
    weights[np.isin(control_vertex_ids, visible_anchor_ids)] = 1.0
    return weights.astype(np.float32)


def _compute_region_grip_weights(
    vertices: np.ndarray,
    visible_anchor_ids: np.ndarray,
    control_vertex_ids: np.ndarray,
) -> np.ndarray:
    anchor_points = vertices[visible_anchor_ids]
    control_points = vertices[control_vertex_ids]
    dists = np.linalg.norm(control_points[:, None, :] - anchor_points[None, :, :], axis=2)
    min_dist = np.min(dists, axis=1)
    radius = max(0.18 * float(np.ptp(vertices[:, 0])), 0.05)
    weights = np.clip(1.0 - min_dist / radius, 0.0, 1.0)
    weights = 0.55 + 0.45 * weights
    weights[np.isin(control_vertex_ids, visible_anchor_ids)] = 1.0
    return weights.astype(np.float32)


def _limit_point_target_speed(
    previous_targets: np.ndarray,
    desired_targets: np.ndarray,
    max_step: float,
) -> np.ndarray:
    if max_step <= 0.0:
        return desired_targets.astype(np.float32, copy=True)

    delta = desired_targets - previous_targets
    step_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    safe_norm = np.maximum(step_norm, 1.0e-8)
    scale = np.minimum(1.0, max_step / safe_norm)
    return (previous_targets + delta * scale).astype(np.float32)


@wp.kernel
def set_anchor_targets(
    vertex_ids: wp.array[int],
    vertex_targets: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    particle_id = vertex_ids[tid]
    particle_q[particle_id] = vertex_targets[tid]
    particle_qd[particle_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_soft_pull_forces(
    vertex_ids: wp.array[int],
    vertex_targets: wp.array[wp.vec3],
    vertex_weights: wp.array[float],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    pull_ke: float,
    pull_kd: float,
    max_pull_force: float,
):
    tid = wp.tid()
    particle_id = vertex_ids[tid]
    if (particle_flags[particle_id] & newton.ParticleFlags.ACTIVE) == 0:
        return

    weight = vertex_weights[tid]
    displacement = vertex_targets[tid] - particle_q[particle_id]
    damping = -particle_qd[particle_id]
    force = weight * (pull_ke * displacement + pull_kd * damping)
    force_norm = wp.length(force)
    if max_pull_force > 0.0 and force_norm > max_pull_force:
        force = force * (max_pull_force / force_norm)
    wp.atomic_add(particle_f, particle_id, force)


@wp.kernel
def apply_global_velocity_damping(
    particle_qd: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    damping_kd: float,
):
    tid = wp.tid()
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        return
    wp.atomic_add(particle_f, tid, -damping_kd * particle_qd[tid])


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
        self.sequence_preview_mode = False

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.rollout_index = 0
        self.manual_anchor_offset_np = np.zeros(3, dtype=np.float32)
        self._reset_key_prev = False
        self.scripted_rollout_done = False
        self.rng = np.random.default_rng(args.policy_seed)
        self.phase_machine_spec = None
        self.phase_machine_active = False
        self.phase_machine_phase_index = 0
        self.phase_machine_schedule = []
        self.phase_machine_total_frames = 0
        self.phase_machine_phase_start_frame = 0
        self.phase_machine_phase_motion_start = 0.0

        if args.control_mode == "template_preview" and args.sequence_datasets:
            self._init_sequence_preview()
            return

        if args.phase_machine_template:
            self.phase_machine_spec = _load_phase_machine_template(args.phase_machine_template)
            self.phase_machine_active = args.control_mode == "scripted"

        if args.unified_template:
            self.reference_template = _scaled_template(_load_unified_template(args.unified_template), args)
        else:
            self.reference_template = _load_reference_policy_template(
                args.reference_dataset,
                args.reference_pkl,
                args.anchor_count,
            )
        if self.phase_machine_active:
            self.reference_template = self._phase_reference_template(self.phase_machine_spec["phases"][0])
        self.current_policy = {}

        cloth = np.load(args.dataset)
        compacted = _compact_cloth_topology(
            np.asarray(cloth["vertices"], dtype=np.float32),
            np.asarray(cloth["faces"], dtype=np.int32),
            np.asarray(cloth["anchor_vertex_ids"], dtype=np.int32) if "anchor_vertex_ids" in cloth else None,
            np.asarray(cloth["anchor_targets"], dtype=np.float32) if "anchor_targets" in cloth else None,
            np.asarray(cloth["object_points"], dtype=np.float32) if "object_points" in cloth else None,
            np.asarray(cloth["object_visibilities"], dtype=bool) if "object_visibilities" in cloth else None,
        )
        if args.control_mode == "template_preview":
            proxy_voxel_size = 0.0
        elif self.phase_machine_active:
            proxy_voxel_size = (
                args.phase_machine_proxy_voxel_size
                if args.phase_machine_proxy_voxel_size > 0.0
                else args.proxy_voxel_size
            )
        else:
            proxy_voxel_size = args.proxy_voxel_size
        proxy = _build_voxel_proxy_mesh(
            compacted["vertices"],
            compacted["faces"],
            proxy_voxel_size,
            object_points=compacted.get("object_points"),
            object_visibilities=compacted.get("object_visibilities"),
        )
        self.vertices_np = proxy["vertices"]
        self.faces_np = proxy["faces"]
        self.source_to_proxy_np = proxy["source_to_proxy"]
        self.dataset_anchor_ids_np = None
        self.dataset_anchor_targets_np = None
        if compacted.get("anchor_vertex_ids") is not None:
            proxy_anchor_ids = self.source_to_proxy_np[compacted["anchor_vertex_ids"]]
            unique_anchor_ids, unique_anchor_pos = np.unique(proxy_anchor_ids, return_index=True)
            self.dataset_anchor_ids_np = unique_anchor_ids.astype(np.int32)
            if compacted.get("anchor_targets") is not None:
                self.dataset_anchor_targets_np = compacted["anchor_targets"][:, unique_anchor_pos].astype(np.float32)
        self.reference_object_points_np = None
        self.reference_object_visibilities_np = None
        self.reference_frame_count = 0
        if proxy.get("object_points") is not None:
            self.reference_object_points_np = proxy["object_points"]
            self.reference_frame_count = int(self.reference_object_points_np.shape[0])
        if proxy.get("object_visibilities") is not None:
            self.reference_object_visibilities_np = proxy["object_visibilities"]
        self.face_indices_wp = wp.array(self.faces_np.reshape(-1), dtype=wp.int32, device=wp.get_device())
        self.spring_pairs_np, self.spring_ke_np, self.spring_kd_np, self.spring_rest_np = _load_optional_springs(
            cloth,
            default_ke=args.spring_ke,
            default_kd=args.spring_kd,
        )
        cloth.close()

        if args.require_springs and self.spring_pairs_np is None:
            raise ValueError("Spring-aware example requires spring_indices/spring_pairs in the dataset.")
        if self._is_reference_control() and self.reference_object_points_np is None:
            raise ValueError("control-mode=reference requires object_points in cloth_export.npz.")

        self.plane_height = 0.0
        if os.path.exists(args.meta):
            with open(args.meta, encoding="utf-8") as f:
                meta = json.load(f)
            self.plane_height = float(meta["plane_center"][2])

        if args.collision_minimal:
            args.with_ground = False
            args.self_contact = False
            args.spring_from_faces = "none"
            args.gravity = 0.0
            args.contact_ke = 0.0
            args.contact_kd = 0.0
            args.contact_mu = 0.0

        if args.material_preset == "paper_like":
            args.density = 0.02
            args.tri_ke = 4.0e3
            args.tri_ka = 4.0e3
            args.tri_kd = 6.0
            args.edge_ke = 60.0
            args.edge_kd = 10.0
            args.particle_radius = min(args.particle_radius, 0.002)

        self.cloth_offset_np = np.array([0.0, 0.0, args.cloth_z_offset - self.plane_height], dtype=np.float32)
        self.vertices_world_np = self.vertices_np + self.cloth_offset_np
        if self.reference_object_points_np is not None:
            self.reference_object_points_np = self.reference_object_points_np + self.cloth_offset_np[None, None, :]
        if hasattr(self, "source_to_proxy_np") and self.reference_template.get("anchor_vertex_ids") is not None:
            remapped_template_anchor_ids = self.source_to_proxy_np[
                np.asarray(self.reference_template["anchor_vertex_ids"], dtype=np.int32)
            ]
            remapped_template_anchor_ids = np.unique(remapped_template_anchor_ids).astype(np.int32)
            self.reference_template["anchor_vertex_ids"] = remapped_template_anchor_ids.tolist()
        if self.phase_machine_active:
            for phase_spec in self.phase_machine_spec["phases"]:
                phase_anchor_ids = np.asarray(phase_spec.get("anchor_vertex_ids", []), dtype=np.int32)
                if len(phase_anchor_ids) == 0:
                    continue
                remapped_phase_anchor_ids = np.unique(self.source_to_proxy_np[phase_anchor_ids]).astype(np.int32)
                phase_spec["anchor_vertex_ids"] = remapped_phase_anchor_ids.tolist()

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.default_particle_radius = args.particle_radius
        builder.default_shape_cfg.ke = args.contact_ke
        builder.default_shape_cfg.kd = args.contact_kd
        builder.default_shape_cfg.mu = args.contact_mu
        if args.with_ground:
            builder.add_ground_plane()

        start_vertex = len(builder.particle_q)
        builder.add_cloth_mesh(
            pos=wp.vec3(*self.cloth_offset_np.tolist()),
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
            add_springs=False,
            spring_ke=args.spring_ke,
            spring_kd=args.spring_kd,
            particle_radius=args.particle_radius,
        )

        spring_source = "dataset"
        if self.spring_pairs_np is not None:
            for spring_idx, (i, j) in enumerate(self.spring_pairs_np.tolist()):
                builder.add_spring(
                    start_vertex + int(i),
                    start_vertex + int(j),
                    float(self.spring_ke_np[spring_idx]),
                    float(self.spring_kd_np[spring_idx]),
                    control=0.0,
                )
                if self.spring_rest_np is not None:
                    builder.spring_rest_length[-1] = float(self.spring_rest_np[spring_idx])
        elif args.spring_from_faces == "edges":
            self.spring_pairs_np, self.spring_rest_np = _generate_mesh_springs(
                self.vertices_world_np,
                self.faces_np,
                include_shear=False,
            )
            for spring_idx, (i, j) in enumerate(self.spring_pairs_np.tolist()):
                builder.add_spring(start_vertex + int(i), start_vertex + int(j), args.spring_ke, args.spring_kd, control=0.0)
                builder.spring_rest_length[-1] = float(self.spring_rest_np[spring_idx])
            self.spring_ke_np = np.full(len(self.spring_pairs_np), args.spring_ke, dtype=np.float32)
            self.spring_kd_np = np.full(len(self.spring_pairs_np), args.spring_kd, dtype=np.float32)
            spring_source = "helper+custom:edges"
        elif args.spring_from_faces == "edges_shear":
            self.spring_pairs_np, self.spring_rest_np = _generate_mesh_springs(
                self.vertices_world_np,
                self.faces_np,
                include_shear=True,
            )
            for spring_idx, (i, j) in enumerate(self.spring_pairs_np.tolist()):
                builder.add_spring(
                    start_vertex + int(i),
                    start_vertex + int(j),
                    args.spring_ke,
                    args.spring_kd,
                    control=0.0,
                )
                builder.spring_rest_length[-1] = float(self.spring_rest_np[spring_idx])
            self.spring_ke_np = np.full(len(self.spring_pairs_np), args.spring_ke, dtype=np.float32)
            self.spring_kd_np = np.full(len(self.spring_pairs_np), args.spring_kd, dtype=np.float32)
            spring_source = "helper+custom:edges+shear"
        else:
            self.spring_pairs_np = np.empty((0, 2), dtype=np.int32)
            self.spring_rest_np = np.empty((0,), dtype=np.float32)
            self.spring_ke_np = np.empty((0,), dtype=np.float32)
            self.spring_kd_np = np.empty((0,), dtype=np.float32)
            spring_source = "disabled"

        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = args.contact_ke
        self.model.soft_contact_kd = args.contact_kd
        self.model.soft_contact_mu = args.contact_mu

        selected_ids = self._select_pull_anchors(args.anchor_count, args.anchor_mode)
        if len(selected_ids) == 0:
            raise ValueError("No anchors selected. Adjust --anchor-count or anchor mode.")

        self.selected_anchor_ids_np = selected_ids
        self.selected_anchor_rest_np = self.vertices_world_np[self.selected_anchor_ids_np].copy()
        self.selected_anchor_targets_np = self.selected_anchor_rest_np.copy()
        self.anchor_action_np = np.zeros((len(self.selected_anchor_ids_np), 3), dtype=np.float32)
        self.reference_anchor_rest_np = self.selected_anchor_rest_np.copy()
        self.reference_control_rest_np = None
        if self.phase_machine_active:
            self._initialize_phase_machine(sample_new=True)
        else:
            self._configure_policy(sample_new=True)

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
        self.control_vertex_ids = wp.array(self.control_vertex_ids_np, dtype=int, device=self.model.device)
        self.selected_anchor_targets = wp.array(
            self.selected_anchor_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.control_vertex_targets = wp.array(
            self.control_vertex_targets_np,
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

        self.rollout_particle_q = []
        self.rollout_particle_qd = []
        self.rollout_anchor_targets = []
        self.rollout_anchor_actions = []
        self.rollout_metadata = {
            "dataset": args.dataset,
            "reference_dataset": args.reference_dataset,
            "reference_pkl": _resolve_reference_pkl_path(args.reference_dataset, args.reference_pkl),
            "phase_machine_template": args.phase_machine_template,
            "proxy_voxel_size": proxy_voxel_size,
            "spring_from_faces": args.spring_from_faces,
            "vertex_count": int(len(self.vertices_np)),
            "spring_count": 0 if self.spring_pairs_np is None else int(len(self.spring_pairs_np)),
            "anchor_ids": self.selected_anchor_ids_np.astype(int).tolist(),
            "control_ids": self.control_vertex_ids_np.astype(int).tolist(),
            "reference_template": self.reference_template,
        }

        bounds_min = self.vertices_world_np.min(axis=0)
        bounds_max = self.vertices_world_np.max(axis=0)
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

        if self.spring_pairs_np is None or len(self.spring_pairs_np) == 0:
            print("No springs loaded; running with triangle/bending cloth only.")
        else:
            print(f"Loaded {len(self.spring_pairs_np)} springs ({spring_source}).")

    def _in_settle_phase(self) -> bool:
        return self.args.control_mode in ("scripted", "reference") and self.frame_index < self.args.settle_frames

    def _motion_time(self) -> float:
        settle_time = self.args.settle_frames * self.frame_dt
        return max(0.0, self.sim_time - settle_time)

    def _phase_motion_time(self) -> float:
        motion_time = self._motion_time()
        if self.phase_machine_active:
            return max(0.0, motion_time - self.phase_machine_phase_motion_start)
        return motion_time

    def _is_reference_control(self) -> bool:
        return self.args.control_mode == "reference"

    def _rollout_frame_limit(self) -> int:
        if self.phase_machine_active:
            return self.args.settle_frames + self.phase_machine_total_frames
        return self.args.rollout_frames

    def _wrap_reference_frame(self, motion_frame: int) -> int:
        if self.reference_frame_count <= 0:
            return 0
        return (motion_frame * self.args.reference_frame_stride) % self.reference_frame_count

    def _resolve_sequence_meta_paths(self) -> list[str | None]:
        meta_paths = list(self.args.sequence_metas or [])
        dataset_paths = list(self.args.sequence_datasets or [])
        while len(meta_paths) < len(dataset_paths):
            dataset_dir = os.path.dirname(dataset_paths[len(meta_paths)])
            default_meta = os.path.join(dataset_dir, "meta.json")
            meta_paths.append(default_meta if os.path.exists(default_meta) else None)
        return meta_paths[: len(dataset_paths)]

    def _init_sequence_preview(self) -> None:
        self.sequence_preview_mode = True
        self.sequence_entries = []
        dataset_paths = list(self.args.sequence_datasets)
        meta_paths = self._resolve_sequence_meta_paths()
        global_bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        global_bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        previous_end_centroid = None

        for dataset_path, meta_path in zip(dataset_paths, meta_paths, strict=False):
            dataset = np.load(dataset_path)
            try:
                compacted = _compact_cloth_topology(
                    np.asarray(dataset["vertices"], dtype=np.float32),
                    np.asarray(dataset["faces"], dtype=np.int32),
                    object_points=np.asarray(dataset["object_points"], dtype=np.float32),
                    object_visibilities=np.asarray(dataset["object_visibilities"], dtype=bool)
                    if "object_visibilities" in dataset
                    else None,
                )
            finally:
                dataset.close()

            plane_height = 0.0
            if meta_path is not None and os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                plane_height = float(meta["plane_center"][2])

            cloth_offset_np = np.array([0.0, 0.0, self.args.cloth_z_offset - plane_height], dtype=np.float32)
            object_points = compacted["object_points"] + cloth_offset_np[None, None, :]
            if self.args.sequence_align_endpoints and previous_end_centroid is not None:
                current_start_centroid = np.mean(object_points[0], axis=0)
                translation = previous_end_centroid - current_start_centroid
                object_points = object_points + translation[None, None, :]
            faces = compacted["faces"].astype(np.int32)
            face_indices = wp.array(faces.reshape(-1), dtype=wp.int32, device=wp.get_device())
            bounds_min = object_points.reshape(-1, 3).min(axis=0)
            bounds_max = object_points.reshape(-1, 3).max(axis=0)
            global_bounds_min = np.minimum(global_bounds_min, bounds_min)
            global_bounds_max = np.maximum(global_bounds_max, bounds_max)
            previous_end_centroid = np.mean(object_points[-1], axis=0)
            self.sequence_entries.append(
                {
                    "dataset": dataset_path,
                    "object_points": object_points.astype(np.float32),
                    "faces": faces,
                    "face_indices_wp": face_indices,
                    "frame_count": int(object_points.shape[0]),
                }
            )

        self.sequence_total_frames = sum(entry["frame_count"] for entry in self.sequence_entries)
        center = 0.5 * (global_bounds_min + global_bounds_max)
        extent = float(np.max(global_bounds_max - global_bounds_min))
        camera_distance = max(0.35, extent * 1.8)
        camera_height = max(0.12, extent * 1.0)
        self.viewer.set_camera(
            wp.vec3(center[0], center[1] - camera_distance, center[2] + camera_height),
            -20.0,
            90.0,
        )

    def _sequence_frame_entry(self) -> tuple[int, dict[str, object], int]:
        frame_id = self.frame_index
        if self.args.sequence_loop and self.sequence_total_frames > 0:
            frame_id = frame_id % self.sequence_total_frames

        for entry_index, entry in enumerate(self.sequence_entries):
            frame_count = int(entry["frame_count"])
            if frame_id < frame_count:
                return entry_index, entry, frame_id
            frame_id -= frame_count

        return len(self.sequence_entries) - 1, self.sequence_entries[-1], int(self.sequence_entries[-1]["frame_count"]) - 1

    def _pull_force_ramp(self) -> float:
        if self._in_settle_phase():
            return 0.0
        motion_time = self._motion_time()
        if self.phase_machine_active:
            motion_time = self._phase_motion_time()
        ramp_time = float(self.current_policy.get("force_ramp_time", self.args.force_ramp_time))
        ramp = min(motion_time / max(ramp_time, 1.0e-6), 1.0)
        return float(ramp * ramp * (3.0 - 2.0 * ramp))

    def _capture_settled_pose(self) -> None:
        particle_q_np = self.state_0.particle_q.numpy()
        self.selected_anchor_rest_np = particle_q_np[self.selected_anchor_ids_np].astype(np.float32).copy()
        self.selected_anchor_targets_np = self.selected_anchor_rest_np.copy()
        self.reference_anchor_rest_np = self.selected_anchor_rest_np.copy()
        self._filter_grounded_control_vertices(particle_q_np)
        self.anchor_action_np.fill(0.0)

        self.selected_anchor_targets.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )
        self.control_vertex_targets.assign(
            wp.array(self.control_vertex_targets_np, dtype=wp.vec3, device=self.model.device)
        )
        self.selected_anchor_points.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )

    def _select_pull_anchors(self, anchor_count: int, anchor_mode: str) -> np.ndarray:
        target_region = str(self.current_policy.get("target_region", ""))
        if anchor_mode == "reference_template" and target_region == "left_sleeve":
            return _select_left_arm_end_anchors(self.vertices_world_np, anchor_count)
        if anchor_mode == "reference_template" and target_region == "right_sleeve":
            return _select_right_arm_end_anchors(self.vertices_world_np, anchor_count)
        if anchor_mode == "reference_template" and target_region == "torso_center_fold":
            return _select_torso_center_anchors(self.vertices_world_np, anchor_count)
        if anchor_mode == "reference_template":
            template_anchor_ids = np.asarray(self.reference_template["anchor_vertex_ids"], dtype=np.int32)
            count = min(anchor_count, len(template_anchor_ids))
            return template_anchor_ids[:count]
        if anchor_mode == "dataset":
            if self.dataset_anchor_ids_np is None:
                raise ValueError("Dataset does not provide anchor_vertex_ids.")
            dataset_anchor_ids = self.dataset_anchor_ids_np
            count = min(anchor_count, len(dataset_anchor_ids))
            return dataset_anchor_ids[:count]

        verts = self.vertices_world_np
        if anchor_mode == "left_arm":
            return _select_left_arm_end_anchors(verts, anchor_count)

        top_mask = verts[:, 1] >= np.quantile(verts[:, 1], 0.98)
        candidate_ids = np.where(top_mask)[0]
        if len(candidate_ids) == 0:
            candidate_ids = np.arange(len(verts))

        if anchor_mode == "top_left":
            ordered = candidate_ids[np.argsort(verts[candidate_ids, 0])]
        elif anchor_mode == "top_right":
            ordered = candidate_ids[np.argsort(-verts[candidate_ids, 0])]
        else:
            ordered = candidate_ids[np.argsort(verts[candidate_ids, 0])]
            if anchor_count > 1:
                pick_positions = np.linspace(0, len(ordered) - 1, num=min(anchor_count, len(ordered))).astype(int)
                return ordered[pick_positions].astype(np.int32)

        return ordered[: min(anchor_count, len(ordered))].astype(np.int32)

    def _phase_reference_template(self, phase_spec: dict[str, object]) -> dict[str, object]:
        ref = dict(phase_spec.get("reference_template", {}))
        ref["anchor_vertex_ids"] = list(phase_spec.get("anchor_vertex_ids", []))
        ref["base_period"] = float(ref.get("base_period", 3.5))
        ref["base_amp_x"] = float(max(abs(x) for x in phase_spec["amp_x_range"]))
        ref["base_amp_y"] = float(max(abs(y) for y in phase_spec["amp_y_range"]))
        ref["base_amp_z"] = float(max(abs(z) for z in phase_spec["amp_z_range"]))
        ref["base_patch_scale"] = float(max(phase_spec["patch_scale_range"]))
        ref["direction_x"] = float(ref.get("direction_x", 1.0))
        return ref

    def _select_control_vertices(self, anchor_mode: str, visible_anchor_ids: np.ndarray) -> np.ndarray:
        if self._is_reference_control():
            return np.arange(len(self.vertices_world_np), dtype=np.int32)
        target_region = str(self.current_policy.get("target_region", ""))
        if anchor_mode == "reference_template" and target_region == "left_sleeve":
            return _select_left_arm_cuff_patch(
                self.vertices_world_np,
                visible_anchor_ids,
                patch_count=max(self.args.phase_machine_min_patch_count, self.args.hidden_grip_count + len(visible_anchor_ids)),
            )
        if anchor_mode == "reference_template" and target_region == "right_sleeve":
            return _select_right_arm_cuff_patch(
                self.vertices_world_np,
                visible_anchor_ids,
                patch_count=max(self.args.phase_machine_min_patch_count, self.args.hidden_grip_count + len(visible_anchor_ids)),
            )
        if anchor_mode == "reference_template" and target_region == "torso_center_fold":
            return _select_torso_center_patch(
                self.vertices_world_np,
                visible_anchor_ids,
                patch_scale=self.args.left_arm_patch_scale,
            )
        if anchor_mode in ("left_arm", "reference_template"):
            if anchor_mode == "reference_template" and self.args.unified_template:
                return _select_left_arm_full_patch(
                    self.vertices_world_np,
                    visible_anchor_ids,
                    patch_scale=max(self.args.left_arm_patch_scale, self.args.unified_patch_scale),
                )
            if self.args.left_arm_grip_mode == "full_patch":
                return _select_left_arm_full_patch(
                    self.vertices_world_np,
                    visible_anchor_ids,
                    patch_scale=self.args.left_arm_patch_scale,
                )
            return _select_left_arm_cuff_patch(
                self.vertices_world_np,
                visible_anchor_ids,
                patch_count=self.args.hidden_grip_count + len(visible_anchor_ids),
            )
        return visible_anchor_ids

    def _compute_control_vertex_weights(self, anchor_mode: str) -> np.ndarray:
        if self._is_reference_control():
            return np.full(len(self.control_vertex_ids_np), self.args.reference_vertex_weight, dtype=np.float32)
        target_region = str(self.current_policy.get("target_region", ""))
        if anchor_mode == "reference_template" and target_region in ("right_sleeve", "torso_center_fold"):
            return _compute_region_grip_weights(
                self.vertices_world_np,
                self.selected_anchor_ids_np,
                self.control_vertex_ids_np,
            )
        if anchor_mode in ("left_arm", "reference_template"):
            weights = _compute_left_arm_grip_weights(
                self.vertices_world_np,
                self.selected_anchor_ids_np,
                self.control_vertex_ids_np,
            )
            if anchor_mode == "reference_template" and self.args.unified_template:
                # Spread pull more uniformly for unified-template rollouts.
                weights = self.args.unified_weight_floor + (1.0 - self.args.unified_weight_floor) * weights
            return weights.astype(np.float32)
        return np.ones(len(self.control_vertex_ids_np), dtype=np.float32)

    def _limit_control_patch_size(self, control_vertex_ids: np.ndarray) -> np.ndarray:
        if self._is_reference_control():
            return control_vertex_ids.astype(np.int32)
        max_count = max(self.args.left_arm_max_control_count, len(self.selected_anchor_ids_np))
        if self.args.anchor_mode not in ("left_arm", "reference_template") or len(control_vertex_ids) <= max_count:
            return control_vertex_ids.astype(np.int32)

        anchor_center = np.mean(self.vertices_world_np[self.selected_anchor_ids_np], axis=0)
        control_points = self.vertices_world_np[control_vertex_ids]
        distances = np.linalg.norm(control_points - anchor_center[None, :], axis=1)
        ordered = control_vertex_ids[np.argsort(distances)]
        limited = np.unique(
            np.concatenate(
                [
                    self.selected_anchor_ids_np.astype(np.int32),
                    ordered[:max_count].astype(np.int32),
                ]
            )
        )
        return limited.astype(np.int32)

    def _filter_grounded_control_vertices(self, particle_q_np: np.ndarray) -> None:
        if self._is_reference_control():
            self.control_vertex_rest_np = particle_q_np[self.control_vertex_ids_np].astype(np.float32).copy()
            self.control_vertex_targets_np = self.control_vertex_rest_np.copy()
            self.reference_control_rest_np = self.control_vertex_rest_np.copy()
            self.control_vertex_weights_np = self._compute_control_vertex_weights(self.args.anchor_mode)
            self.control_vertex_ids = wp.array(self.control_vertex_ids_np, dtype=int, device=self.model.device)
            self.control_vertex_targets = wp.array(
                self.control_vertex_targets_np,
                dtype=wp.vec3,
                device=self.model.device,
            )
            self.control_vertex_weights = wp.array(
                self.control_vertex_weights_np,
                dtype=wp.float32,
                device=self.model.device,
            )
            return
        if self.args.anchor_mode not in ("left_arm", "reference_template") or len(self.control_vertex_ids_np) == 0:
            return

        ground_z = self.plane_height + self.args.settled_ground_margin
        control_ids = self.control_vertex_ids_np.astype(np.int32)
        keep_mask = particle_q_np[control_ids, 2] > ground_z
        keep_mask |= np.isin(control_ids, self.selected_anchor_ids_np)
        filtered_ids = control_ids[keep_mask]

        min_count = max(len(self.selected_anchor_ids_np), self.args.min_control_count_after_settle)
        if len(filtered_ids) < min_count:
            anchor_center = np.mean(particle_q_np[self.selected_anchor_ids_np], axis=0)
            distances = np.linalg.norm(particle_q_np[control_ids] - anchor_center[None, :], axis=1)
            ordered = control_ids[np.argsort(distances)]
            filtered_ids = np.unique(
                np.concatenate(
                    [
                        self.selected_anchor_ids_np.astype(np.int32),
                        ordered[:min_count].astype(np.int32),
                    ]
                )
            )

        self.control_vertex_ids_np = filtered_ids.astype(np.int32)
        self.control_vertex_rest_np = particle_q_np[self.control_vertex_ids_np].astype(np.float32).copy()
        self.control_vertex_targets_np = self.control_vertex_rest_np.copy()
        self.control_vertex_weights_np = self._compute_control_vertex_weights(self.args.anchor_mode)
        self.control_vertex_ids = wp.array(self.control_vertex_ids_np, dtype=int, device=self.model.device)
        self.control_vertex_targets = wp.array(
            self.control_vertex_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.control_vertex_weights = wp.array(
            self.control_vertex_weights_np,
            dtype=wp.float32,
            device=self.model.device,
        )

    def _sample_policy(self) -> dict[str, float | str]:
        if self._is_reference_control():
            return {
                "script_pattern": "reference_replay",
                "reference_displacement_scale": self.args.reference_displacement_scale,
            }
        if self.args.policy_mode == "fixed" and self.args.unified_template:
            base = self.reference_template
            return {
                "script_pattern": "lift_hold_fold_left_arm",
                "script_amp_x": float(base["base_amp_x"]),
                "script_amp_y": float(base["base_amp_y"]),
                "script_amp_z": float(base["base_amp_z"]),
                "script_period": float(base["base_period"]),
                "lift_phase_ratio": float(base["lift_phase_ratio"]),
                "hold_phase_ratio": float(base["hold_phase_ratio"]),
                "left_arm_patch_scale": float(max(float(base["base_patch_scale"]), self.args.unified_patch_scale)),
            }
        if self.args.policy_mode == "template_randomized":
            base = self.reference_template
            amp_scale = self.rng.uniform(self.args.template_amp_scale_min, self.args.template_amp_scale_max)
            z_scale = self.rng.uniform(self.args.template_z_scale_min, self.args.template_z_scale_max)
            return {
                "script_pattern": "lift_hold_fold_left_arm",
                "script_amp_x": float(base["base_amp_x"] * amp_scale),
                "script_amp_y": float(base["base_amp_y"]),
                "script_amp_z": float(base["base_amp_z"] * z_scale),
                "script_period": float(
                    base["base_period"] * self.rng.uniform(self.args.template_period_scale_min, self.args.template_period_scale_max)
                ),
                "lift_phase_ratio": float(
                    np.clip(
                        float(base["lift_phase_ratio"])
                        + self.rng.uniform(-self.args.template_phase_jitter, self.args.template_phase_jitter),
                        0.15,
                        0.65,
                    )
                ),
                "hold_phase_ratio": float(
                    np.clip(
                        float(base["hold_phase_ratio"])
                        + self.rng.uniform(-0.5 * self.args.template_phase_jitter, 0.5 * self.args.template_phase_jitter),
                        0.04,
                        0.30,
                    )
                ),
                "left_arm_patch_scale": float(
                    float(base["base_patch_scale"])
                    * self.rng.uniform(self.args.template_patch_scale_min, self.args.template_patch_scale_max)
                ),
            }
        if self.args.policy_mode == "fixed":
            return {
                "script_pattern": self.args.script_pattern,
                "script_amp_x": self.args.script_amp_x,
                "script_amp_y": self.args.script_amp_y,
                "script_amp_z": self.args.script_amp_z,
                "script_period": self.args.script_period,
                "lift_phase_ratio": self.args.lift_phase_ratio,
                "hold_phase_ratio": self.args.hold_phase_ratio,
                "left_arm_patch_scale": self.args.left_arm_patch_scale,
            }

        base = self.reference_template
        amp_scale = self.rng.uniform(self.args.rand_amp_scale_min, self.args.rand_amp_scale_max)
        z_scale = self.rng.uniform(self.args.rand_z_scale_min, self.args.rand_z_scale_max)
        return {
            "script_pattern": "lift_hold_fold_left_arm",
            "script_amp_x": float(base["base_amp_x"] * amp_scale),
            "script_amp_y": 0.0,
            "script_amp_z": float(base["base_amp_z"] * z_scale),
            "script_period": float(self.rng.uniform(self.args.rand_period_min, self.args.rand_period_max)),
            "lift_phase_ratio": float(
                self.rng.uniform(self.args.rand_lift_phase_ratio_min, self.args.rand_lift_phase_ratio_max)
            ),
            "hold_phase_ratio": float(
                self.rng.uniform(self.args.rand_hold_phase_ratio_min, self.args.rand_hold_phase_ratio_max)
            ),
            "left_arm_patch_scale": float(
                base["base_patch_scale"] * self.rng.uniform(self.args.rand_patch_scale_min, self.args.rand_patch_scale_max)
            ),
        }

    def _sample_phase_policy(self, phase_spec: dict[str, object]) -> dict[str, float | str | int]:
        if self.args.policy_mode == "fixed":
            duration_scale = float(phase_spec["duration_scale_range"][1])
            amp_x = _range_value(phase_spec["amp_x_range"], "fixed", minimize_magnitude=True)
            amp_y = _range_value(phase_spec["amp_y_range"], "fixed", minimize_magnitude=True)
            amp_z = _range_value(phase_spec["amp_z_range"], "fixed", minimize_magnitude=True)
            patch_scale = float(phase_spec["patch_scale_range"][0])
            pull_ke = float(phase_spec["pull_ke_range"][0])
            max_pull_force = float(phase_spec["max_pull_force_range"][0])
        else:
            duration_scale = self.rng.uniform(*phase_spec["duration_scale_range"])
            amp_x = float(self.rng.uniform(*phase_spec["amp_x_range"]))
            amp_y = float(self.rng.uniform(*phase_spec["amp_y_range"]))
            amp_z = float(self.rng.uniform(*phase_spec["amp_z_range"]))
            patch_scale = float(self.rng.uniform(*phase_spec["patch_scale_range"]))
            pull_ke = float(self.rng.uniform(*phase_spec["pull_ke_range"]))
            max_pull_force = float(self.rng.uniform(*phase_spec["max_pull_force_range"]))

        base_frames = max(1, int(phase_spec["global_end_frame"]) - int(phase_spec["global_start_frame"]))
        duration_frames = max(1, int(round(base_frames * duration_scale)))
        reference_template = self._phase_reference_template(phase_spec)
        return {
            "phase_name": str(phase_spec["phase_name"]),
            "script_pattern": "lift_hold_fold_left_arm",
            "script_amp_x": amp_x,
            "script_amp_y": amp_y,
            "script_amp_z": amp_z,
            "script_period": float(max(duration_frames / self.fps, 1.0e-3)),
            "duration_frames": duration_frames,
            "lift_phase_ratio": float(reference_template["lift_phase_ratio"]),
            "hold_phase_ratio": float(reference_template["hold_phase_ratio"]),
            "left_arm_patch_scale": patch_scale,
            "pull_ke": pull_ke,
            "max_pull_force": max_pull_force,
            "script_delay": max(self.args.script_delay, 1.5),
            "script_ramp": max(self.args.script_ramp, 3.0),
            "force_ramp_time": max(self.args.force_ramp_time, 3.0),
            "anchor_mode": str(phase_spec.get("anchor_mode", "reference_template")),
            "target_region": str(phase_spec.get("target_region", "")),
            "reference_template": reference_template,
        }

    def _sync_anchor_control_wp_arrays(self) -> None:
        self.selected_anchor_ids = wp.array(self.selected_anchor_ids_np, dtype=int, device=self.model.device)
        self.control_vertex_ids = wp.array(self.control_vertex_ids_np, dtype=int, device=self.model.device)
        self.selected_anchor_targets = wp.array(
            self.selected_anchor_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.selected_anchor_weights = wp.full(
            len(self.selected_anchor_ids_np),
            1.0,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.control_vertex_targets = wp.array(
            self.control_vertex_targets_np,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.control_vertex_weights = wp.array(
            self.control_vertex_weights_np,
            dtype=wp.float32,
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
            self.args.anchor_radius,
            dtype=wp.float32,
            device=self.model.device,
        )

    def _activate_phase_policy(self, phase_index: int, preserve_pose: bool) -> None:
        phase_entry = self.phase_machine_schedule[phase_index]
        self.phase_machine_phase_index = phase_index
        self.reference_template = dict(phase_entry["policy"]["reference_template"])
        self.current_policy = dict(phase_entry["policy"])
        self.current_policy["phase_index"] = phase_index

        selected_ids = self._select_pull_anchors(self.args.anchor_count, "reference_template")
        self.selected_anchor_ids_np = selected_ids.astype(np.int32)

        if preserve_pose and hasattr(self, "state_0"):
            particle_q_np = self.state_0.particle_q.numpy()
            self.selected_anchor_rest_np = particle_q_np[self.selected_anchor_ids_np].astype(np.float32).copy()
        else:
            self.selected_anchor_rest_np = self.vertices_world_np[self.selected_anchor_ids_np].astype(np.float32).copy()
        self.selected_anchor_targets_np = self.selected_anchor_rest_np.copy()
        self.reference_anchor_rest_np = self.selected_anchor_rest_np.copy()
        self.anchor_action_np = np.zeros((len(self.selected_anchor_ids_np), 3), dtype=np.float32)

        original_patch_scale = self.args.left_arm_patch_scale
        self.args.left_arm_patch_scale = float(self.current_policy["left_arm_patch_scale"])
        self.control_vertex_ids_np = self._select_control_vertices("reference_template", self.selected_anchor_ids_np)
        self.args.left_arm_patch_scale = original_patch_scale
        self.control_vertex_ids_np = self._limit_control_patch_size(self.control_vertex_ids_np)

        if preserve_pose and hasattr(self, "state_0"):
            particle_q_np = self.state_0.particle_q.numpy()
            self.control_vertex_rest_np = particle_q_np[self.control_vertex_ids_np].astype(np.float32).copy()
        else:
            self.control_vertex_rest_np = self.vertices_world_np[self.control_vertex_ids_np].astype(np.float32).copy()
        self.control_vertex_targets_np = self.control_vertex_rest_np.copy()
        self.reference_control_rest_np = self.control_vertex_rest_np.copy()
        self.control_vertex_weights_np = self._compute_control_vertex_weights("reference_template")

        if hasattr(self, "model"):
            self._sync_anchor_control_wp_arrays()

        self.phase_machine_phase_start_frame = int(phase_entry["start_frame"])
        self.phase_machine_phase_motion_start = self.phase_machine_phase_start_frame * self.frame_dt

    def _initialize_phase_machine(self, sample_new: bool) -> None:
        if not sample_new and self.phase_machine_schedule:
            self._activate_phase_policy(0, preserve_pose=False)
            return

        self.phase_machine_schedule = []
        frame_cursor = 0
        for phase_spec in self.phase_machine_spec["phases"]:
            policy = self._sample_phase_policy(phase_spec)
            duration_frames = int(policy["duration_frames"])
            self.phase_machine_schedule.append(
                {
                    "spec": phase_spec,
                    "policy": policy,
                    "start_frame": frame_cursor,
                    "end_frame": frame_cursor + duration_frames,
                }
            )
            frame_cursor += duration_frames
        self.phase_machine_total_frames = frame_cursor
        self._activate_phase_policy(0, preserve_pose=False)

    def _update_phase_machine_phase(self) -> None:
        if not self.phase_machine_active:
            return
        motion_frame = max(0, self.frame_index - self.args.settle_frames)
        for idx, phase_entry in enumerate(self.phase_machine_schedule):
            if motion_frame < int(phase_entry["end_frame"]):
                if idx != self.phase_machine_phase_index:
                    self._activate_phase_policy(idx, preserve_pose=True)
                return
        final_index = len(self.phase_machine_schedule) - 1
        if final_index >= 0 and self.phase_machine_phase_index != final_index:
            self._activate_phase_policy(final_index, preserve_pose=True)

    def _configure_policy(self, sample_new: bool):
        if sample_new:
            self.current_policy = self._sample_policy()
        if self.args.anchor_mode == "left_arm":
            original_patch_scale = self.args.left_arm_patch_scale
            self.args.left_arm_patch_scale = float(self.current_policy["left_arm_patch_scale"])
            self.control_vertex_ids_np = self._select_control_vertices(self.args.anchor_mode, self.selected_anchor_ids_np)
            self.args.left_arm_patch_scale = original_patch_scale
        else:
            self.control_vertex_ids_np = self._select_control_vertices(self.args.anchor_mode, self.selected_anchor_ids_np)
        self.control_vertex_ids_np = self._limit_control_patch_size(self.control_vertex_ids_np)
        self.control_vertex_rest_np = self.vertices_world_np[self.control_vertex_ids_np].copy()
        self.control_vertex_targets_np = self.control_vertex_rest_np.copy()
        self.reference_control_rest_np = self.control_vertex_rest_np.copy()
        self.control_vertex_weights_np = self._compute_control_vertex_weights(self.args.anchor_mode)
        if hasattr(self, "model"):
            self.control_vertex_ids = wp.array(self.control_vertex_ids_np, dtype=int, device=self.model.device)
            self.control_vertex_targets = wp.array(
                self.control_vertex_targets_np,
                dtype=wp.vec3,
                device=self.model.device,
            )
            self.control_vertex_weights = wp.array(
                self.control_vertex_weights_np,
                dtype=wp.float32,
                device=self.model.device,
            )

    def _reset_rollout_state(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_time = 0.0
        self.frame_index = 0
        self.selected_anchor_targets_np = self.selected_anchor_rest_np.copy()
        self.control_vertex_targets_np = self.control_vertex_rest_np.copy()
        self.anchor_action_np = np.zeros_like(self.anchor_action_np)
        self.rollout_particle_q = []
        self.rollout_particle_qd = []
        self.rollout_anchor_targets = []
        self.rollout_anchor_actions = []

    def _compute_anchor_targets(self) -> np.ndarray:
        if self._is_reference_control():
            anchor_targets, _, _ = self._compute_reference_targets()
            self.anchor_action_np = anchor_targets - self.reference_anchor_rest_np
            return anchor_targets
        if self.args.control_mode == "scripted":
            return self._compute_scripted_anchor_targets()

        targets = self.selected_anchor_rest_np.copy()

        pull_time = max(0.0, self._motion_time() - self.args.pull_delay)
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
        self.anchor_action_np = targets - self.selected_anchor_rest_np
        return targets

    def _compute_reference_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.reference_object_points_np is None or self.reference_frame_count <= 0:
            raise ValueError("Reference control requires object_points in the dataset.")

        motion_frame = max(0, self.frame_index - self.args.settle_frames)
        ref_frame = self._wrap_reference_frame(motion_frame)
        displacement = self.reference_object_points_np[ref_frame] - self.reference_object_points_np[0]
        scale = float(self.current_policy.get("reference_displacement_scale", self.args.reference_displacement_scale))

        anchor_targets = self.reference_anchor_rest_np + scale * displacement[self.selected_anchor_ids_np]
        control_targets = self.reference_control_rest_np + scale * displacement[self.control_vertex_ids_np]
        control_weights = self.control_vertex_weights_np.copy()
        if self.reference_object_visibilities_np is not None:
            vis = self.reference_object_visibilities_np[ref_frame, self.control_vertex_ids_np].astype(np.float32)
            control_weights *= vis
        return anchor_targets.astype(np.float32), control_targets.astype(np.float32), control_weights.astype(np.float32)

    def _compute_preview_particle_targets(self) -> np.ndarray:
        if self.reference_object_points_np is not None and self.reference_frame_count > 0:
            motion_frame = self.frame_index
            ref_frame = self._wrap_reference_frame(motion_frame)
            displacement = self.reference_object_points_np[ref_frame] - self.reference_object_points_np[0]
            scale = float(self.args.preview_displacement_scale)
            return (self.vertices_world_np + scale * displacement).astype(np.float32)

        anchor_targets = self._compute_anchor_targets()
        control_targets = self._compute_control_targets(anchor_targets)
        particle_targets = self.vertices_world_np.copy()
        particle_targets[self.control_vertex_ids_np] = control_targets
        return particle_targets.astype(np.float32)

    def _compute_scripted_anchor_targets(self) -> np.ndarray:
        targets = self.selected_anchor_rest_np.copy()

        script_delay = float(self.current_policy.get("script_delay", self.args.script_delay))
        script_ramp = float(self.current_policy.get("script_ramp", self.args.script_ramp))
        time_after_delay = max(0.0, self._phase_motion_time() - script_delay)
        script_period = float(self.current_policy["script_period"])
        if script_period <= 1e-6:
            phase = 0.0
        else:
            phase = 2.0 * np.pi * time_after_delay / script_period
        envelope = min(time_after_delay / max(script_ramp, 1e-6), 1.0)
        envelope = envelope * envelope * (3.0 - 2.0 * envelope)

        center_x = float(np.mean(self.selected_anchor_rest_np[:, 0]))
        left_mask = self.selected_anchor_rest_np[:, 0] < center_x
        right_mask = ~left_mask

        offset = np.zeros_like(targets)
        mode = str(self.current_policy["script_pattern"])
        amp_x = float(self.current_policy["script_amp_x"]) * envelope
        amp_y = float(self.current_policy["script_amp_y"]) * envelope
        amp_z = float(self.current_policy["script_amp_z"]) * envelope

        if mode == "lift":
            offset[:, 2] = amp_z * np.sin(phase)
        elif mode == "sway":
            offset[:, 0] = amp_x * np.sin(phase)
        elif mode == "figure8":
            offset[:, 0] = amp_x * np.sin(phase)
            offset[:, 2] = amp_z * np.sin(2.0 * phase)
        elif mode == "twist":
            offset[left_mask, 2] = amp_z * np.sin(phase)
            offset[right_mask, 2] = -amp_z * np.sin(phase)
            offset[left_mask, 0] = -0.5 * amp_x * np.sin(phase)
            offset[right_mask, 0] = 0.5 * amp_x * np.sin(phase)
        elif mode == "fold_left_arm":
            fold_wave = 0.5 * (1.0 - np.cos(phase))
            lateral_sign = np.sign(np.mean(self.selected_anchor_rest_np[:, 0]))
            if lateral_sign == 0.0:
                lateral_sign = -1.0
            offset[:, 0] = -lateral_sign * amp_x * fold_wave
            offset[:, 2] = amp_z * fold_wave
            offset[:, 1] = -0.5 * (amp_x + amp_y) * fold_wave
        elif mode == "lift_fold_left_arm":
            cycle = np.mod(time_after_delay / max(script_period, 1e-6), 1.0)
            lateral_sign = np.sign(np.mean(self.selected_anchor_rest_np[:, 0]))
            if lateral_sign == 0.0:
                lateral_sign = -1.0
            lift_phase_ratio = float(self.current_policy["lift_phase_ratio"])
            if cycle < lift_phase_ratio:
                lift_phase = cycle / max(lift_phase_ratio, 1.0e-6)
                smooth = lift_phase * lift_phase * (3.0 - 2.0 * lift_phase)
                offset[:, 2] = amp_z * smooth
            else:
                fold_phase = (cycle - lift_phase_ratio) / max(1.0 - lift_phase_ratio, 1.0e-6)
                smooth = fold_phase * fold_phase * (3.0 - 2.0 * fold_phase)
                offset[:, 2] = amp_z
                offset[:, 0] = -lateral_sign * amp_x * smooth
                offset[:, 1] = -0.5 * (amp_x + amp_y) * smooth
        elif mode == "lift_hold_fold_left_arm":
            cycle = np.mod(time_after_delay / max(script_period, 1.0e-6), 1.0)
            lateral_sign = np.sign(np.mean(self.selected_anchor_rest_np[:, 0]))
            if lateral_sign == 0.0:
                lateral_sign = -1.0
            lift_phase_ratio = float(self.current_policy["lift_phase_ratio"])
            hold_phase_ratio = float(self.current_policy["hold_phase_ratio"])
            fold_start = lift_phase_ratio + hold_phase_ratio
            if cycle < lift_phase_ratio:
                lift_phase = cycle / max(lift_phase_ratio, 1.0e-6)
                smooth = lift_phase * lift_phase * (3.0 - 2.0 * lift_phase)
                offset[:, 2] = amp_z * smooth
            elif cycle < fold_start:
                offset[:, 2] = amp_z
            else:
                fold_phase = (cycle - fold_start) / max(1.0 - fold_start, 1.0e-6)
                smooth = fold_phase * fold_phase * (3.0 - 2.0 * fold_phase)
                offset[:, 2] = amp_z
                offset[:, 0] = -lateral_sign * amp_x * smooth
                offset[:, 1] = -0.35 * (amp_x + amp_y) * smooth
        elif mode == "pull_release":
            wave = np.maximum(0.0, np.sin(phase))
            offset[:, 2] = amp_z * wave
            offset[:, 0] = amp_x * wave

        if amp_y != 0.0:
            offset[:, 1] += amp_y * np.sin(phase)

        targets += offset
        self.anchor_action_np = offset
        return targets

    def _compute_control_targets(self, visible_targets: np.ndarray) -> np.ndarray:
        if self.args.anchor_mode not in ("left_arm", "reference_template"):
            return visible_targets

        delta = np.mean(visible_targets - self.selected_anchor_rest_np, axis=0)
        if np.linalg.norm(delta) <= 1.0e-8:
            return self.control_vertex_rest_np.copy()

        # Use the inner edge of the cuff patch as a hinge axis so the sleeve lifts as
        # a patch instead of stretching from a single point.
        pivot_x = float(np.quantile(self.control_vertex_rest_np[:, 0], 0.85))
        pivot_y = float(np.mean(self.control_vertex_rest_np[:, 1]))
        pivot_z = float(np.mean(self.control_vertex_rest_np[:, 2]))
        pivot = np.array([pivot_x, pivot_y, pivot_z], dtype=np.float32)

        base_dx = float(delta[0])
        base_dz = float(delta[2])
        base_dy = float(delta[1])
        angle = np.clip(7.0 * abs(base_dx) + 5.0 * abs(base_dz), 0.0, 0.75)
        angle *= -1.0 if base_dx >= 0.0 else 1.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array(
            [
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a],
            ],
            dtype=np.float32,
        )

        local = self.control_vertex_rest_np - pivot
        rotated = local @ rot.T
        rotated_targets = rotated + pivot

        # Apply a coherent rigid-like lift to the whole cuff patch.
        lift = np.array([0.35 * base_dx, base_dy, max(base_dz, 0.0)], dtype=np.float32)
        weights = self.control_vertex_weights_np[:, None]
        rigid_targets = rotated_targets + lift
        return self.control_vertex_rest_np + weights * (rigid_targets - self.control_vertex_rest_np)

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
        if self._in_settle_phase():
            return

        if self._is_reference_control():
            desired_anchor_targets, desired_control_targets, desired_control_weights = self._compute_reference_targets()
            self.anchor_action_np = desired_anchor_targets - self.reference_anchor_rest_np
            self.control_vertex_weights_np = desired_control_weights.astype(np.float32)
            self.control_vertex_weights.assign(
                wp.array(self.control_vertex_weights_np, dtype=wp.float32, device=self.model.device)
            )
        else:
            desired_anchor_targets = self._compute_anchor_targets()
            desired_control_targets = self._compute_control_targets(desired_anchor_targets)

        max_anchor_step = self.args.max_anchor_speed * self.sim_dt
        self.selected_anchor_targets_np = _limit_point_target_speed(
            self.selected_anchor_targets_np,
            desired_anchor_targets,
            max_anchor_step,
        )
        max_control_step = self.args.max_control_speed * self.sim_dt
        self.control_vertex_targets_np = _limit_point_target_speed(
            self.control_vertex_targets_np,
            desired_control_targets,
            max_control_step,
        )
        self.selected_anchor_targets.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )
        self.control_vertex_targets.assign(
            wp.array(self.control_vertex_targets_np, dtype=wp.vec3, device=self.model.device)
        )
        self.selected_anchor_points.assign(
            wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
        )

        dynamic_weights = self.control_vertex_weights_np.copy()
        if self.args.disable_grounded_pull:
            particle_q_np = self.state_0.particle_q.numpy()
            grounded_mask = particle_q_np[self.control_vertex_ids_np, 2] <= (self.plane_height + self.args.grounded_pull_margin)
            grounded_mask &= ~np.isin(self.control_vertex_ids_np, self.selected_anchor_ids_np)
            dynamic_weights[grounded_mask] = 0.0

        ramp = self._pull_force_ramp()
        anchor_weights = np.full(len(self.selected_anchor_ids_np), ramp, dtype=np.float32)
        self.selected_anchor_weights.assign(
            wp.array(anchor_weights, dtype=wp.float32, device=self.model.device)
        )
        dynamic_weights *= ramp
        self.control_vertex_weights.assign(
            wp.array(dynamic_weights, dtype=wp.float32, device=self.model.device)
        )

        wp.launch(
            apply_soft_pull_forces,
            dim=len(self.selected_anchor_ids_np),
            inputs=[
                self.selected_anchor_ids,
                self.selected_anchor_targets,
                self.selected_anchor_weights,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_0.particle_f,
                self.model.particle_flags,
                float(self.current_policy.get("anchor_pull_ke", self.args.anchor_pull_ke)),
                self.args.pull_kd,
                float(self.current_policy.get("anchor_max_pull_force", self.args.anchor_max_pull_force)),
            ],
            device=self.model.device,
        )

        wp.launch(
            apply_soft_pull_forces,
            dim=len(self.control_vertex_ids_np),
            inputs=[
                self.control_vertex_ids,
                self.control_vertex_targets,
                self.control_vertex_weights,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_0.particle_f,
                self.model.particle_flags,
                float(self.current_policy.get("pull_ke", self.args.pull_ke)),
                self.args.pull_kd,
                float(self.current_policy.get("max_pull_force", self.args.max_pull_force)),
            ],
            device=self.model.device,
        )

    def simulate(self):
        if self.sequence_preview_mode:
            return
        if self.args.control_mode == "template_preview":
            preview_targets = self._compute_preview_particle_targets()
            self.selected_anchor_targets_np = preview_targets[self.selected_anchor_ids_np].astype(np.float32).copy()
            self.selected_anchor_points.assign(
                wp.array(self.selected_anchor_targets_np, dtype=wp.vec3, device=self.model.device)
            )
            wp.launch(
                set_particle_targets,
                dim=len(preview_targets),
                inputs=[
                    wp.array(preview_targets, dtype=wp.vec3, device=self.model.device),
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                ],
                device=self.model.device,
            )
            return

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_anchor_targets()
            if self.args.global_drag_kd > 0.0:
                wp.launch(
                    apply_global_velocity_damping,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_0.particle_qd,
                        self.state_0.particle_f,
                        self.model.particle_flags,
                        self.args.global_drag_kd,
                    ],
                    device=self.model.device,
                )
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self._apply_anchor_targets()

    def step(self):
        if self.sequence_preview_mode:
            self.sim_time += self.frame_dt
            self.frame_index += 1
            if not self.args.sequence_loop and self.frame_index >= self.sequence_total_frames and self.args.auto_exit_after_rollouts:
                raise SystemExit(0)
            return

        if self.scripted_rollout_done and self.args.control_mode in ("scripted", "template_preview"):
            return
        if self.args.control_mode == "interactive":
            self._update_keyboard_controls()

        settle_done_this_frame = self.args.control_mode in ("scripted", "reference") and self.frame_index == self.args.settle_frames
        if settle_done_this_frame:
            self._capture_settled_pose()

        if self.phase_machine_active:
            self._update_phase_machine_phase()

        self.simulate()
        self._record_rollout_frame()
        self.sim_time += self.frame_dt
        self.frame_index += 1

        if self.args.control_mode in ("scripted", "template_preview") and not self.scripted_rollout_done and self.frame_index >= self._rollout_frame_limit():
            self._save_rollout()
            if self.rollout_index + 1 >= self.args.num_rollouts:
                self.scripted_rollout_done = True
                if self.args.auto_exit_after_rollouts:
                    raise SystemExit(0)
            else:
                self.rollout_index += 1
                if self.phase_machine_active:
                    self._initialize_phase_machine(sample_new=self.args.policy_mode != "fixed")
                    self._reset_rollout_state()
                else:
                    self._configure_policy(sample_new=self.args.policy_mode != "fixed")
                    self._reset_rollout_state()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.sequence_preview_mode:
            entry_index, entry, local_frame = self._sequence_frame_entry()
            for idx, sequence_entry in enumerate(self.sequence_entries):
                mesh_name = f"/cloth_sequence_mesh_{idx}"
                if idx == entry_index:
                    object_points = sequence_entry["object_points"][local_frame]
                    self.viewer.log_mesh(
                        mesh_name,
                        wp.array(object_points, dtype=wp.vec3, device=wp.get_device()),
                        sequence_entry["face_indices_wp"],
                        hidden=False,
                        backface_culling=False,
                    )
                else:
                    self.viewer.log_mesh(
                        mesh_name,
                        wp.array(sequence_entry["object_points"][0], dtype=wp.vec3, device=wp.get_device()),
                        sequence_entry["face_indices_wp"],
                        hidden=True,
                        backface_culling=False,
                    )
            self.viewer.end_frame()
            return

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
        if self.sequence_preview_mode:
            _, entry, local_frame = self._sequence_frame_entry()
            particle_q = np.asarray(entry["object_points"][local_frame], dtype=np.float32)
            assert np.isfinite(particle_q).all()
            return

        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all()

    def _record_rollout_frame(self):
        if self.args.control_mode not in ("scripted", "template_preview"):
            return
        self.rollout_particle_q.append(self.state_0.particle_q.numpy().astype(np.float32))
        self.rollout_particle_qd.append(self.state_0.particle_qd.numpy().astype(np.float32))
        self.rollout_anchor_targets.append(self.selected_anchor_targets_np.astype(np.float32).copy())
        self.rollout_anchor_actions.append(self.anchor_action_np.astype(np.float32).copy())

    def _save_rollout(self):
        if not self.args.rollout_output:
            return

        output_dir = os.path.dirname(self.args.rollout_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_path = self._rollout_output_path()
        np.savez(
            output_path,
            particle_q=np.asarray(self.rollout_particle_q, dtype=np.float32),
            particle_qd=np.asarray(self.rollout_particle_qd, dtype=np.float32),
            anchor_targets=np.asarray(self.rollout_anchor_targets, dtype=np.float32),
            anchor_actions=np.asarray(self.rollout_anchor_actions, dtype=np.float32),
            anchor_vertex_ids=self.selected_anchor_ids_np.astype(np.int32),
            faces=self.faces_np.astype(np.int32),
            rest_vertices=self.vertices_world_np.astype(np.float32),
            spring_pairs=(
                np.asarray(self.spring_pairs_np, dtype=np.int32)
                if self.spring_pairs_np is not None
                else np.empty((0, 2), dtype=np.int32)
            ),
            spring_rest_length=(
                np.asarray(self.spring_rest_np, dtype=np.float32)
                if self.spring_rest_np is not None
                else np.empty((0,), dtype=np.float32)
            ),
            spring_ke=np.asarray(self.spring_ke_np, dtype=np.float32),
            spring_kd=np.asarray(self.spring_kd_np, dtype=np.float32),
            metadata_json=json.dumps(
                {
                    **self.rollout_metadata,
                    "rollout_index": self.rollout_index,
                    "policy": self.current_policy,
                    "phase_machine_schedule": self.phase_machine_schedule,
                }
            ),
        )
        print(f"Saved scripted rollout to {output_path}")

    def _rollout_output_path(self) -> str:
        if self.args.num_rollouts <= 1:
            return self.args.rollout_output
        root, ext = os.path.splitext(self.args.rollout_output)
        return f"{root}_{self.rollout_index:04d}{ext}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to cloth NPZ.")
        parser.add_argument("--meta", type=str, default=DEFAULT_META_PATH, help="Path to export meta.json.")
        parser.add_argument(
            "--sequence-datasets",
            nargs="*",
            default=[],
            help="Optional list of cloth_export.npz files to preview consecutively in template_preview mode.",
        )
        parser.add_argument(
            "--sequence-metas",
            nargs="*",
            default=[],
            help="Optional list of meta.json files aligned with --sequence-datasets.",
        )
        parser.add_argument(
            "--sequence-loop",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Loop the sequence preview instead of stopping at the last dataset.",
        )
        parser.add_argument(
            "--sequence-align-endpoints",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Translate each next preview clip so its first frame starts where the previous clip ended.",
        )
        parser.add_argument(
            "--collision-minimal",
            action="store_true",
            help="Disable ground, self-contact, fallback springs, and gravity so only the fold controller is visualized.",
        )
        parser.add_argument(
            "--material-preset",
            type=str,
            choices=["default", "paper_like"],
            default="default",
            help="Optional material preset for the simulated cloth.",
        )
        parser.add_argument(
            "--proxy-voxel-size",
            type=float,
            default=0.0,
            help="Voxel size [m] for building a lower-resolution proxy cloth mesh from cloth_export vertices/faces.",
        )
        parser.add_argument("--sim-substeps", type=int, default=24, help="Number of substeps per frame.")
        parser.add_argument("--iterations", type=int, default=6, help="VBD iterations per substep.")
        parser.add_argument("--density", type=float, default=0.08, help="Cloth areal density.")
        parser.add_argument("--particle-radius", type=float, default=0.0025, help="Cloth particle radius.")
        parser.add_argument("--gravity", type=float, default=0.0, help="Gravity acceleration along the up axis [m/s^2].")
        parser.add_argument("--cloth-z-offset", type=float, default=0.02, help="Lift cloth above the fitted plane [m].")
        parser.add_argument(
            "--settle-frames",
            type=int,
            default=0,
            help="Number of initial frames to let the cloth settle before any anchor motion starts.",
        )
        parser.add_argument(
            "--with-ground",
            action="store_true",
            help="Enable ground plane contact. Disabled by default for stability.",
        )
        parser.add_argument("--tri-ke", type=float, default=2.0e3, help="Triangle stretching stiffness.")
        parser.add_argument("--tri-ka", type=float, default=2.0e3, help="Triangle area stiffness.")
        parser.add_argument("--tri-kd", type=float, default=1.0, help="Triangle damping.")
        parser.add_argument("--edge-ke", type=float, default=20.0, help="Bending stiffness.")
        parser.add_argument("--edge-kd", type=float, default=8.0, help="Bending damping.")
        parser.add_argument("--spring-ke", type=float, default=5.0e3, help="Fallback spring stiffness if NPZ omits it.")
        parser.add_argument("--spring-kd", type=float, default=10.0, help="Fallback spring damping if NPZ omits it.")
        parser.add_argument(
            "--spring-from-faces",
            type=str,
            choices=["none", "edges", "edges_shear"],
            default="edges",
            help="How to generate springs from faces when the dataset has no spring graph.",
        )
        parser.add_argument(
            "--require-springs",
            action="store_true",
            help="Fail if the dataset does not include spring_indices/spring_pairs.",
        )
        parser.add_argument("--contact-ke", type=float, default=1.0e2, help="Ground contact stiffness.")
        parser.add_argument("--contact-kd", type=float, default=3.0e1, help="Ground contact damping.")
        parser.add_argument("--contact-mu", type=float, default=0.2, help="Ground contact friction.")
        parser.add_argument("--self-contact", action="store_true", help="Enable cloth self-contact.")
        parser.add_argument("--self-contact-radius", type=float, default=0.004, help="Self-contact radius.")
        parser.add_argument("--self-contact-margin", type=float, default=0.006, help="Self-contact margin.")
        parser.add_argument(
            "--control-mode",
            type=str,
            choices=["interactive", "scripted", "reference", "template_preview"],
            default="scripted",
            help="Interactive control, scripted helper rollout generation, reference-driven replay, or direct template preview.",
        )
        parser.add_argument(
            "--policy-mode",
            type=str,
            choices=["fixed", "reference_randomized", "template_randomized"],
            default="template_randomized",
            help="Use fixed scripted parameters or randomized fold templates extracted from a reference trajectory.",
        )
        parser.add_argument(
            "--reference-dataset",
            type=str,
            default=DEFAULT_DATASET_PATH,
            help="Reference NPZ used to estimate left-arm fold statistics for randomized policies.",
        )
        parser.add_argument(
            "--reference-pkl",
            type=str,
            default=None,
            help="Optional PKL trajectory source. Defaults to the final_data.pkl next to the reference dataset.",
        )
        parser.add_argument(
            "--unified-template",
            type=str,
            default=None,
            help="Optional unified_template.npz produced by the sequence exporter. Overrides single-reference template extraction.",
        )
        parser.add_argument(
            "--phase-machine-template",
            type=str,
            default=None,
            help="Optional phase_machine_template.json. When provided with scripted control, drives staged phase-machine rollout.",
        )
        parser.add_argument(
            "--phase-machine-proxy-voxel-size",
            type=float,
            default=0.03,
            help="Voxel size [m] used to build the Newton cloth proxy mesh for phase-machine scripted rollouts.",
        )
        parser.add_argument(
            "--unified-patch-scale",
            type=float,
            default=1.25,
            help="Minimum helper patch scale used for scripted rollouts driven by --unified-template.",
        )
        parser.add_argument(
            "--unified-weight-floor",
            type=float,
            default=0.75,
            help="Minimum control weight used across the helper patch for --unified-template rollouts.",
        )
        parser.add_argument(
            "--unified-amp-x-scale",
            type=float,
            default=1.0,
            help="Scale factor applied to unified-template x displacement amplitude.",
        )
        parser.add_argument(
            "--unified-amp-y-scale",
            type=float,
            default=1.0,
            help="Scale factor applied to unified-template y displacement amplitude.",
        )
        parser.add_argument(
            "--unified-amp-z-scale",
            type=float,
            default=1.0,
            help="Scale factor applied to unified-template z displacement amplitude.",
        )
        parser.add_argument(
            "--unified-period-scale",
            type=float,
            default=1.0,
            help="Scale factor applied to unified-template period.",
        )
        parser.add_argument(
            "--unified-patch-scale-factor",
            type=float,
            default=1.0,
            help="Scale factor applied to unified-template helper patch size before clamping to --unified-patch-scale.",
        )
        parser.add_argument("--policy-seed", type=int, default=0, help="Random seed for reference_randomized policy sampling.")
        parser.add_argument(
            "--reference-frame-stride",
            type=int,
            default=1,
            help="Stride through cloth_export reference frames when control-mode=reference.",
        )
        parser.add_argument(
            "--reference-displacement-scale",
            type=float,
            default=1.0,
            help="Scale applied to exported per-vertex displacement when control-mode=reference.",
        )
        parser.add_argument(
            "--preview-displacement-scale",
            type=float,
            default=1.0,
            help="Scale applied to exported per-vertex displacement when control-mode=template_preview.",
        )
        parser.add_argument(
            "--reference-vertex-weight",
            type=float,
            default=1.0,
            help="Base pull weight per vertex when control-mode=reference.",
        )
        parser.add_argument("--num-rollouts", type=int, default=1, help="Number of scripted rollouts to generate in one run.")
        parser.add_argument(
            "--auto-exit-after-rollouts",
            action="store_true",
            help="Exit the example automatically after the requested scripted rollouts are saved.",
        )
        parser.add_argument(
            "--anchor-mode",
            type=str,
            choices=["top_left", "top_right", "spread_top", "left_arm", "dataset", "reference_template"],
            default="reference_template",
            help="How to choose visible pull anchors. reference_template uses the anchor subset extracted from the reference trajectory.",
        )
        parser.add_argument(
            "--hidden-grip-count",
            type=int,
            default=5,
            help="Additional cuff-patch vertices to move with the visible anchor(s) in left_arm mode.",
        )
        parser.add_argument(
            "--left-arm-grip-mode",
            type=str,
            choices=["cuff_patch", "full_patch"],
            default="full_patch",
            help="Use a small cuff patch or a much larger left-sleeve patch in left_arm mode.",
        )
        parser.add_argument(
            "--left-arm-patch-scale",
            type=float,
            default=1.0,
            help="Scale factor for the left sleeve patch size when left-arm-grip-mode=full_patch.",
        )
        parser.add_argument(
            "--left-arm-max-control-count",
            type=int,
            default=384,
            help="Maximum number of soft-control vertices used by the left-arm helper patch.",
        )
        parser.add_argument(
            "--min-control-count-after-settle",
            type=int,
            default=12,
            help="Minimum number of control vertices kept after pruning grounded vertices at the end of settling.",
        )
        parser.add_argument(
            "--phase-machine-min-patch-count",
            type=int,
            default=12,
            help="Minimum control patch size used for target_region sleeve phases in phase-machine rollouts.",
        )
        parser.add_argument("--anchor-count", type=int, default=6, help="Number of anchors to pull.")
        parser.add_argument("--anchor-radius", type=float, default=0.01, help="Rendered radius for pull anchors.")
        parser.add_argument("--anchor-pull-ke", type=float, default=8.0, help="Soft pull stiffness applied to visible anchor vertices.")
        parser.add_argument(
            "--anchor-max-pull-force",
            type=float,
            default=0.05,
            help="Maximum pull force magnitude per visible anchor vertex [N]. Set to 0 to disable clamping.",
        )
        parser.add_argument("--pull-ke", type=float, default=150.0, help="Soft pull stiffness applied to control vertices.")
        parser.add_argument("--pull-kd", type=float, default=12.0, help="Soft pull damping applied to control vertices.")
        parser.add_argument(
            "--max-pull-force",
            type=float,
            default=2.5,
            help="Maximum pull force magnitude per control vertex [N]. Set to 0 to disable clamping.",
        )
        parser.add_argument(
            "--force-ramp-time",
            type=float,
            default=0.5,
            help="Seconds to ramp soft pull forces from zero after settling completes.",
        )
        parser.add_argument(
            "--global-drag-kd",
            type=float,
            default=0.0,
            help="Global velocity damping applied to all active cloth particles during simulation.",
        )
        parser.add_argument("--pull-delay", type=float, default=1.0, help="Seconds to wait before pulling.")
        parser.add_argument("--pull-ramp", type=float, default=4.0, help="Seconds to ramp pull strength.")
        parser.add_argument("--pull-dx", type=float, default=0.0, help="Anchor pull amplitude along x [m].")
        parser.add_argument("--pull-dy", type=float, default=0.0, help="Anchor pull amplitude along y [m].")
        parser.add_argument("--pull-dz", type=float, default=0.003, help="Anchor pull amplitude along z [m].")
        parser.add_argument(
            "--settled-ground-margin",
            type=float,
            default=0.004,
            help="Settled-pose ground clearance used to prune grounded helper vertices [m].",
        )
        parser.add_argument(
            "--disable-grounded-pull",
            action="store_true",
            help="Disable soft pull on control vertices that are still touching the ground at the current frame.",
        )
        parser.add_argument(
            "--grounded-pull-margin",
            type=float,
            default=0.006,
            help="Current-frame ground clearance threshold used by --disable-grounded-pull [m].",
        )
        parser.add_argument(
            "--keyboard-speed",
            type=float,
            default=0.03,
            help="Anchor target speed for keyboard control [m/s]. Keys: J/L=x, K/I=y, O/U=z, P=reset.",
        )
        parser.add_argument("--rollout-frames", type=int, default=240, help="Number of frames to save in scripted mode.")
        parser.add_argument(
            "--rollout-output",
            type=str,
            default=os.path.join(CLOTH_DATA_ROOT, "generated", "newton_cloth_rollout.npz"),
            help="NPZ path for scripted rollout output.",
        )
        parser.add_argument(
            "--script-pattern",
            type=str,
            choices=[
                "lift",
                "sway",
                "figure8",
                "twist",
                "fold_left_arm",
                "lift_fold_left_arm",
                "lift_hold_fold_left_arm",
                "pull_release",
            ],
            default="twist",
            help="Anchor motion pattern for scripted rollouts.",
        )
        parser.add_argument("--script-delay", type=float, default=0.5, help="Seconds before scripted motion starts.")
        parser.add_argument("--script-ramp", type=float, default=1.0, help="Seconds to ramp scripted motion amplitude.")
        parser.add_argument("--script-period", type=float, default=4.0, help="Seconds per scripted motion cycle.")
        parser.add_argument(
            "--template-amp-scale-min",
            type=float,
            default=0.85,
            help="Minimum scale applied to the extracted template fold amplitude.",
        )
        parser.add_argument(
            "--template-amp-scale-max",
            type=float,
            default=1.10,
            help="Maximum scale applied to the extracted template fold amplitude.",
        )
        parser.add_argument(
            "--template-z-scale-min",
            type=float,
            default=0.90,
            help="Minimum scale applied to the extracted template lift amplitude.",
        )
        parser.add_argument(
            "--template-z-scale-max",
            type=float,
            default=1.15,
            help="Maximum scale applied to the extracted template lift amplitude.",
        )
        parser.add_argument(
            "--template-period-scale-min",
            type=float,
            default=0.90,
            help="Minimum scale applied to the extracted template period.",
        )
        parser.add_argument(
            "--template-period-scale-max",
            type=float,
            default=1.10,
            help="Maximum scale applied to the extracted template period.",
        )
        parser.add_argument(
            "--template-patch-scale-min",
            type=float,
            default=0.90,
            help="Minimum scale applied to the extracted helper patch size.",
        )
        parser.add_argument(
            "--template-patch-scale-max",
            type=float,
            default=1.15,
            help="Maximum scale applied to the extracted helper patch size.",
        )
        parser.add_argument(
            "--template-phase-jitter",
            type=float,
            default=0.05,
            help="Maximum additive jitter applied to extracted phase ratios.",
        )
        parser.add_argument("--rand-amp-scale-min", type=float, default=0.8, help="Minimum scale on reference fold amplitude.")
        parser.add_argument("--rand-amp-scale-max", type=float, default=1.05, help="Maximum scale on reference fold amplitude.")
        parser.add_argument("--rand-z-scale-min", type=float, default=0.9, help="Minimum scale on reference lift amplitude.")
        parser.add_argument("--rand-z-scale-max", type=float, default=1.2, help="Maximum scale on reference lift amplitude.")
        parser.add_argument("--rand-period-min", type=float, default=4.0, help="Minimum randomized policy period [s].")
        parser.add_argument("--rand-period-max", type=float, default=7.0, help="Maximum randomized policy period [s].")
        parser.add_argument("--rand-patch-scale-min", type=float, default=0.9, help="Minimum scale on reference sleeve patch size.")
        parser.add_argument("--rand-patch-scale-max", type=float, default=1.2, help="Maximum scale on reference sleeve patch size.")
        parser.add_argument(
            "--rand-lift-phase-ratio-min",
            type=float,
            default=0.35,
            help="Minimum lift phase ratio for randomized lift-fold policies.",
        )
        parser.add_argument(
            "--rand-lift-phase-ratio-max",
            type=float,
            default=0.60,
            help="Maximum lift phase ratio for randomized lift-fold policies.",
        )
        parser.add_argument(
            "--rand-hold-phase-ratio-min",
            type=float,
            default=0.10,
            help="Minimum hold phase ratio for randomized lift-hold-fold policies.",
        )
        parser.add_argument(
            "--rand-hold-phase-ratio-max",
            type=float,
            default=0.25,
            help="Maximum hold phase ratio for randomized lift-hold-fold policies.",
        )
        parser.add_argument(
            "--lift-phase-ratio",
            type=float,
            default=0.45,
            help="Fraction of one scripted cycle spent lifting before folding in lift_fold_left_arm mode.",
        )
        parser.add_argument(
            "--hold-phase-ratio",
            type=float,
            default=0.15,
            help="Fraction of one scripted cycle spent holding after lift and before fold.",
        )
        parser.add_argument("--script-amp-x", type=float, default=0.02, help="Scripted anchor motion amplitude along x [m].")
        parser.add_argument("--script-amp-y", type=float, default=0.0, help="Scripted anchor motion amplitude along y [m].")
        parser.add_argument("--script-amp-z", type=float, default=0.03, help="Scripted anchor motion amplitude along z [m].")
        parser.add_argument(
            "--max-anchor-speed",
            type=float,
            default=0.20,
            help="Maximum visible-anchor target speed [m/s]. Set to 0 to disable target speed limiting.",
        )
        parser.add_argument(
            "--max-control-speed",
            type=float,
            default=0.12,
            help="Maximum hidden control-patch target speed [m/s]. Set to 0 to disable target speed limiting.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)
