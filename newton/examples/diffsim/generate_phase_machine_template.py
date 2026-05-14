# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a phase_machine_template.json from phase_annotation.json."""

import argparse
import json
import os


def _signed_x_range(base_amp_x: float, direction_x: float, min_scale: float, max_scale: float) -> list[float]:
    signed = base_amp_x * (direction_x if direction_x != 0.0 else 1.0)
    lo = signed * min_scale
    hi = signed * max_scale
    return [float(min(lo, hi)), float(max(lo, hi))]


def _scaled_range(value: float, min_scale: float, max_scale: float, minimum: float = 0.0) -> list[float]:
    lo = max(minimum, value * min_scale)
    hi = max(lo, value * max_scale)
    return [float(lo), float(hi)]


def _phase_defaults(phase_name: str) -> dict[str, object]:
    if phase_name == "fold_left_sleeve":
        return {
            "duration_scale_range": [0.9, 1.15],
            "x_scales": (0.75, 1.05),
            "y_scales": (0.0, 4.0),
            "z_scales": (0.75, 1.35),
            "patch_scales": (0.9, 1.2),
            "pull_ke_range": [0.8, 1.8],
            "max_pull_force_range": [0.002, 0.010],
            "anchor_mode": "reference_template",
            "target_region": "left_sleeve",
        }
    if phase_name == "fold_right_sleeve":
        return {
            "duration_scale_range": [0.9, 1.15],
            "x_scales": (0.75, 1.05),
            "y_scales": (0.5, 1.5),
            "z_scales": (0.75, 1.35),
            "patch_scales": (0.9, 1.2),
            "pull_ke_range": [0.8, 1.8],
            "max_pull_force_range": [0.002, 0.010],
            "anchor_mode": "reference_template",
            "target_region": "right_sleeve",
        }
    return {
        "duration_scale_range": [0.95, 1.25],
        "x_scales": (0.7, 1.3),
        "y_scales": (0.6, 1.4),
        "z_scales": (0.8, 1.4),
        "patch_scales": (0.95, 1.25),
        "pull_ke_range": [1.0, 2.2],
        "max_pull_force_range": [0.003, 0.012],
        "anchor_mode": "reference_template",
        "target_region": "torso_center_fold",
    }


def generate_phase_machine_template(
    phase_annotation_path: str,
    sequence_manifest_path: str,
    output_path: str,
) -> None:
    with open(phase_annotation_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    with open(sequence_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    segment_templates = {segment["name"]: segment["template"] for segment in manifest["segments"]}

    phase_entries = []
    for phase in annotation["phases"]:
        phase_name = phase["phase_name"]
        segment_name = phase["segment_name"]
        segment_template = segment_templates[segment_name]
        defaults = _phase_defaults(phase_name)

        phase_entries.append(
            {
                "phase_index": int(phase["phase_index"]),
                "phase_name": phase_name,
                "global_start_frame": int(phase["global_start_frame"]),
                "global_end_frame": int(phase["global_end_frame"]),
                "segment_index": int(phase["segment_index"]),
                "segment_name": segment_name,
                "duration_scale_range": defaults["duration_scale_range"],
                "amp_x_range": _signed_x_range(
                    float(segment_template["base_amp_x"]),
                    float(segment_template["direction_x"]),
                    *defaults["x_scales"],
                ),
                "amp_y_range": _scaled_range(
                    float(segment_template["base_amp_y"]),
                    *defaults["y_scales"],
                    minimum=0.0,
                ),
                "amp_z_range": _scaled_range(
                    float(segment_template["base_amp_z"]),
                    *defaults["z_scales"],
                    minimum=0.002,
                ),
                "patch_scale_range": _scaled_range(
                    float(segment_template["base_patch_scale"]),
                    *defaults["patch_scales"],
                    minimum=0.8,
                ),
                "pull_ke_range": defaults["pull_ke_range"],
                "max_pull_force_range": defaults["max_pull_force_range"],
                "anchor_mode": defaults["anchor_mode"],
                "target_region": defaults["target_region"],
                "anchor_vertex_ids": list(segment_template["anchor_vertex_ids"]),
                "reference_template": {
                    "base_period": float(segment_template["base_period"]),
                    "lift_phase_ratio": float(segment_template["lift_phase_ratio"]),
                    "hold_phase_ratio": float(segment_template["hold_phase_ratio"]),
                    "direction_x": float(segment_template["direction_x"]),
                },
                "notes": phase.get("notes", ""),
            }
        )

    payload = {
        "episode_name": annotation["episode_name"],
        "total_frame_count": int(annotation["total_frame_count"]),
        "segment_count": int(annotation["segment_count"]),
        "phases": phase_entries,
        "global_template": annotation.get("global_template", {}),
        "source_phase_annotation": phase_annotation_path,
        "source_sequence_manifest": sequence_manifest_path,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-annotation", required=True, help="Path to phase_annotation.json.")
    parser.add_argument("--sequence-manifest", required=True, help="Path to sequence_manifest.json.")
    parser.add_argument("--output", required=True, help="Path to write phase_machine_template.json.")
    args = parser.parse_args()
    generate_phase_machine_template(args.phase_annotation, args.sequence_manifest, args.output)


if __name__ == "__main__":
    main()
