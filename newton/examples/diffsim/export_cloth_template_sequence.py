# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export a unified manifest/template from multiple cloth_export clips."""

import argparse
import json
import os

import numpy as np

from newton.examples.diffsim.example_diffsim_cloth_phystwin_interact_spring import (
    _load_reference_policy_template,
    _resolve_reference_pkl_path,
)


def _default_meta_path(dataset_path: str) -> str | None:
    candidate = os.path.join(os.path.dirname(dataset_path), "meta.json")
    return candidate if os.path.exists(candidate) else None


def _phase_specs(frame_start: int, frame_count: int, template: dict[str, object], segment_name: str) -> list[dict[str, object]]:
    if frame_count <= 0:
        return []

    lift_ratio = float(template["lift_phase_ratio"])
    hold_ratio = float(template["hold_phase_ratio"])
    lift_count = max(1, int(round(frame_count * lift_ratio)))
    hold_count = max(1, int(round(frame_count * hold_ratio)))
    fold_count = max(1, frame_count - lift_count - hold_count)

    phase_bounds = [
        ("lift", frame_start, frame_start + lift_count),
        ("hold", frame_start + lift_count, frame_start + lift_count + hold_count),
        ("fold", frame_start + lift_count + hold_count, frame_start + lift_count + hold_count + fold_count),
    ]

    phases = []
    for phase_name, start, end in phase_bounds:
        phases.append(
            {
                "segment_name": segment_name,
                "phase_name": phase_name,
                "frame_start": int(start),
                "frame_end": int(min(end, frame_start + frame_count)),
            }
        )
    return phases


def _aggregate_templates(segment_templates: list[dict[str, object]]) -> dict[str, object]:
    if not segment_templates:
        raise ValueError("At least one segment template is required.")

    def avg(key: str) -> float:
        return float(np.mean([float(t[key]) for t in segment_templates]))

    anchor_ids = []
    for template in segment_templates:
        anchor_ids.extend(int(x) for x in template["anchor_vertex_ids"])

    return {
        "template_source": "unified_sequence",
        "anchor_vertex_ids": sorted(set(anchor_ids)),
        "base_period": avg("base_period"),
        "base_amp_x": avg("base_amp_x"),
        "base_amp_y": avg("base_amp_y"),
        "base_amp_z": avg("base_amp_z"),
        "lift_phase_ratio": avg("lift_phase_ratio"),
        "hold_phase_ratio": avg("hold_phase_ratio"),
        "base_patch_scale": avg("base_patch_scale"),
        "direction_x": float(np.sign(np.mean([float(t["direction_x"]) for t in segment_templates])) or 1.0),
    }


def export_sequence(
    sequence_datasets: list[str],
    sequence_metas: list[str | None],
    sequence_pkls: list[str | None],
    manifest_output: str,
    template_output: str,
    preview_output: str | None,
    anchor_count: int,
) -> None:
    segments = []
    segment_templates = []
    phases = []
    global_segment_index = []
    global_local_frame = []
    global_phase_index = []
    frame_cursor = 0
    previous_end_centroid = None
    preview_payload: dict[str, object] = {}

    for segment_index, dataset_path in enumerate(sequence_datasets):
        dataset = np.load(dataset_path)
        try:
            vertices = np.asarray(dataset["vertices"], dtype=np.float32)
            faces = np.asarray(dataset["faces"], dtype=np.int32)
            object_points = np.asarray(dataset["object_points"], dtype=np.float32)
            object_visibilities = (
                np.asarray(dataset["object_visibilities"], dtype=bool) if "object_visibilities" in dataset else None
            )
            anchor_vertex_ids = (
                np.asarray(dataset["anchor_vertex_ids"], dtype=np.int32) if "anchor_vertex_ids" in dataset else None
            )
            anchor_targets = np.asarray(dataset["anchor_targets"], dtype=np.float32) if "anchor_targets" in dataset else None
            frame_count = int(object_points.shape[0])
        finally:
            dataset.close()

        meta_path = sequence_metas[segment_index]
        pkl_path = _resolve_reference_pkl_path(dataset_path, sequence_pkls[segment_index])
        template = _load_reference_policy_template(dataset_path, pkl_path, anchor_count)

        start_centroid = object_points[0].mean(axis=0)
        end_centroid = object_points[-1].mean(axis=0)
        translation = np.zeros(3, dtype=np.float32)
        if previous_end_centroid is not None:
            translation = previous_end_centroid - start_centroid
            object_points = object_points + translation[None, None, :]
            if anchor_targets is not None:
                anchor_targets = anchor_targets + translation[None, None, :]
            end_centroid = end_centroid + translation
        previous_end_centroid = end_centroid

        segment_name = os.path.basename(os.path.dirname(os.path.dirname(dataset_path)))
        segment_manifest = {
            "segment_index": segment_index,
            "name": segment_name,
            "dataset": dataset_path,
            "meta": meta_path,
            "reference_pkl": pkl_path,
            "frame_start": frame_cursor,
            "frame_count": frame_count,
            "frame_end": frame_cursor + frame_count,
            "translation": translation.astype(float).tolist(),
            "template": template,
        }
        segments.append(segment_manifest)
        segment_templates.append(template)
        preview_payload[f"segment_{segment_index}_vertices"] = vertices.astype(np.float32)
        preview_payload[f"segment_{segment_index}_faces"] = faces.astype(np.int32)
        preview_payload[f"segment_{segment_index}_object_points"] = object_points.astype(np.float32)
        if object_visibilities is not None:
            preview_payload[f"segment_{segment_index}_object_visibilities"] = object_visibilities.astype(bool)
        if anchor_vertex_ids is not None:
            preview_payload[f"segment_{segment_index}_anchor_vertex_ids"] = anchor_vertex_ids.astype(np.int32)
        if anchor_targets is not None:
            preview_payload[f"segment_{segment_index}_anchor_targets"] = anchor_targets.astype(np.float32)

        segment_phases = _phase_specs(frame_cursor, frame_count, template, segment_name)
        phase_offset = len(phases)
        phases.extend(segment_phases)

        for local_frame in range(frame_count):
            global_segment_index.append(segment_index)
            global_local_frame.append(local_frame)
            if local_frame < int(round(frame_count * float(template["lift_phase_ratio"]))):
                phase_idx = phase_offset
            elif local_frame < int(round(frame_count * (float(template["lift_phase_ratio"]) + float(template["hold_phase_ratio"])))):
                phase_idx = phase_offset + 1
            else:
                phase_idx = phase_offset + 2
            global_phase_index.append(phase_idx)

        frame_cursor += frame_count

    global_template = _aggregate_templates(segment_templates)
    manifest = {
        "episode_name": os.path.splitext(os.path.basename(template_output))[0],
        "segment_count": len(segments),
        "total_frame_count": frame_cursor,
        "segments": segments,
        "phases": phases,
        "global_template": global_template,
    }

    manifest_dir = os.path.dirname(manifest_output)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    template_dir = os.path.dirname(template_output)
    if template_dir:
        os.makedirs(template_dir, exist_ok=True)

    with open(manifest_output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    np.savez(
        template_output,
        global_segment_index=np.asarray(global_segment_index, dtype=np.int32),
        global_local_frame=np.asarray(global_local_frame, dtype=np.int32),
        global_phase_index=np.asarray(global_phase_index, dtype=np.int32),
        phase_start_frame=np.asarray([phase["frame_start"] for phase in phases], dtype=np.int32),
        phase_end_frame=np.asarray([phase["frame_end"] for phase in phases], dtype=np.int32),
        phase_name=np.asarray([phase["phase_name"] for phase in phases]),
        segment_name=np.asarray([segment["name"] for segment in segments]),
        segment_frame_start=np.asarray([segment["frame_start"] for segment in segments], dtype=np.int32),
        segment_frame_count=np.asarray([segment["frame_count"] for segment in segments], dtype=np.int32),
        segment_translation=np.asarray([segment["translation"] for segment in segments], dtype=np.float32),
        metadata_json=json.dumps(manifest),
    )

    if preview_output:
        preview_dir = os.path.dirname(preview_output)
        if preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
        np.savez(
            preview_output,
            global_segment_index=np.asarray(global_segment_index, dtype=np.int32),
            global_local_frame=np.asarray(global_local_frame, dtype=np.int32),
            global_phase_index=np.asarray(global_phase_index, dtype=np.int32),
            phase_start_frame=np.asarray([phase["frame_start"] for phase in phases], dtype=np.int32),
            phase_end_frame=np.asarray([phase["frame_end"] for phase in phases], dtype=np.int32),
            phase_name=np.asarray([phase["phase_name"] for phase in phases]),
            segment_name=np.asarray([segment["name"] for segment in segments]),
            segment_frame_start=np.asarray([segment["frame_start"] for segment in segments], dtype=np.int32),
            segment_frame_count=np.asarray([segment["frame_count"] for segment in segments], dtype=np.int32),
            segment_translation=np.asarray([segment["translation"] for segment in segments], dtype=np.float32),
            metadata_json=json.dumps(manifest),
            **preview_payload,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-datasets", nargs="+", required=True, help="Ordered cloth_export.npz sequence.")
    parser.add_argument("--sequence-metas", nargs="*", default=[], help="Optional ordered meta.json sequence.")
    parser.add_argument("--sequence-pkls", nargs="*", default=[], help="Optional ordered final_data.pkl sequence.")
    parser.add_argument("--manifest-output", required=True, help="Path to write sequence_manifest.json.")
    parser.add_argument("--template-output", required=True, help="Path to write unified_template.npz.")
    parser.add_argument(
        "--preview-output",
        default=None,
        help="Optional path to write preview_episode.npz with aligned per-segment geometry and trajectories.",
    )
    parser.add_argument("--anchor-count", type=int, default=6, help="Anchor count used when extracting per-segment templates.")
    args = parser.parse_args()

    sequence_metas = list(args.sequence_metas)
    while len(sequence_metas) < len(args.sequence_datasets):
        sequence_metas.append(_default_meta_path(args.sequence_datasets[len(sequence_metas)]))

    sequence_pkls = list(args.sequence_pkls)
    while len(sequence_pkls) < len(args.sequence_datasets):
        sequence_pkls.append(None)

    export_sequence(
        sequence_datasets=list(args.sequence_datasets),
        sequence_metas=sequence_metas[: len(args.sequence_datasets)],
        sequence_pkls=sequence_pkls[: len(args.sequence_datasets)],
        manifest_output=args.manifest_output,
        template_output=args.template_output,
        preview_output=args.preview_output,
        anchor_count=args.anchor_count,
    )


if __name__ == "__main__":
    main()
