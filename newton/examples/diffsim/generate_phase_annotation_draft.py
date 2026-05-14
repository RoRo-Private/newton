# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a draft phase_annotation.json from preview_episode.npz."""

import argparse
import json
import os

import numpy as np


def generate_phase_annotation(preview_episode_path: str, output_path: str) -> None:
    preview = np.load(preview_episode_path, allow_pickle=True)
    try:
        metadata = json.loads(str(preview["metadata_json"]))
        phase_start = np.asarray(preview["phase_start_frame"], dtype=np.int32)
        phase_end = np.asarray(preview["phase_end_frame"], dtype=np.int32)
        phase_name = [str(x) for x in np.asarray(preview["phase_name"])]
        global_segment_index = np.asarray(preview["global_segment_index"], dtype=np.int32)
        global_local_frame = np.asarray(preview["global_local_frame"], dtype=np.int32)
        segment_names = [str(x) for x in np.asarray(preview["segment_name"])]
        segment_frame_start = np.asarray(preview["segment_frame_start"], dtype=np.int32)
        segment_frame_count = np.asarray(preview["segment_frame_count"], dtype=np.int32)
    finally:
        preview.close()

    phases = []
    for phase_index, (start, end, name) in enumerate(zip(phase_start.tolist(), phase_end.tolist(), phase_name, strict=False)):
        frame_slice = slice(start, max(start, end))
        if frame_slice.start >= len(global_segment_index):
            continue
        segment_index = int(global_segment_index[frame_slice.start])
        local_start = int(global_local_frame[frame_slice.start])
        local_end = int(global_local_frame[min(max(end - 1, start), len(global_local_frame) - 1)])
        phases.append(
            {
                "phase_index": phase_index,
                "phase_name": name,
                "global_start_frame": start,
                "global_end_frame": end,
                "segment_index": segment_index,
                "segment_name": segment_names[segment_index],
                "segment_local_start": local_start,
                "segment_local_end": local_end,
                "notes": "",
            }
        )

    annotation = {
        "episode_name": metadata.get("episode_name", os.path.splitext(os.path.basename(preview_episode_path))[0]),
        "total_frame_count": int(metadata.get("total_frame_count", len(global_segment_index))),
        "segment_count": int(metadata.get("segment_count", len(segment_names))),
        "segments": [
            {
                "segment_index": idx,
                "segment_name": segment_names[idx],
                "frame_start": int(segment_frame_start[idx]),
                "frame_count": int(segment_frame_count[idx]),
            }
            for idx in range(len(segment_names))
        ],
        "phases": phases,
        "global_template": metadata.get("global_template", {}),
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-episode", required=True, help="Path to preview_episode.npz.")
    parser.add_argument("--output", required=True, help="Path to write phase_annotation.json.")
    args = parser.parse_args()
    generate_phase_annotation(args.preview_episode, args.output)


if __name__ == "__main__":
    main()
