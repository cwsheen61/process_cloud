#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
from pathlib import Path
from itertools import combinations

def load_trajectory_file(file_path):
    try:
        data = np.loadtxt(file_path, skiprows=1)
        return data
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)

def compute_bbox(segment):
    """Returns the bounding box [xmin, xmax, ymin, ymax, zmin, zmax]"""
    xyz = segment[:, 1:4]
    return [
        float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0])),
        float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1])),
        float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2])),
    ]

def bbox_overlap_ratio(bbox1, bbox2):
    """Returns overlap volume / volume of smaller box"""
    x_overlap = max(0, min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0]))
    y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2]))
    z_overlap = max(0, min(bbox1[5], bbox2[5]) - max(bbox1[4], bbox2[4]))
    intersection_vol = x_overlap * y_overlap * z_overlap
    if intersection_vol == 0:
        return 0.0

    vol1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2]) * (bbox1[5] - bbox1[4])
    vol2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2]) * (bbox2[5] - bbox2[4])
    min_vol = min(vol1, vol2)
    return intersection_vol / min_vol

def build_overlap_graph(segments, threshold=0.5):
    graph = {seg["id"]: set() for seg in segments}
    for seg1, seg2 in combinations(segments, 2):
        ratio = bbox_overlap_ratio(
            [seg1["bbox"]["xmin"], seg1["bbox"]["xmax"],
             seg1["bbox"]["ymin"], seg1["bbox"]["ymax"],
             seg1["bbox"]["zmin"], seg1["bbox"]["zmax"]],
            [seg2["bbox"]["xmin"], seg2["bbox"]["xmax"],
             seg2["bbox"]["ymin"], seg2["bbox"]["ymax"],
             seg2["bbox"]["zmin"], seg2["bbox"]["zmax"]],
        )
        if ratio >= threshold:
            graph[seg1["id"]].add(seg2["id"])
            graph[seg2["id"]].add(seg1["id"])
    return graph

def connected_components(graph):
    visited = set()
    groups = []

    for node in graph:
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.add(current)
                    stack.extend(graph[current] - visited)
            groups.append(sorted(group))
    return groups

def split_and_group_trajectory(file_path, num_segments=200):
    data = load_trajectory_file(file_path)
    total_points = data.shape[0]
    segment_size = total_points // num_segments
    segments = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else total_points
        segment = data[start_idx:end_idx]
        if segment.size == 0:
            continue

        bbox = compute_bbox(segment)
        segments.append({
            "id": f"segment_{i:03}",
            "range": [int(start_idx), int(end_idx)],
            "bbox": {
                "xmin": bbox[0], "xmax": bbox[1],
                "ymin": bbox[2], "ymax": bbox[3],
                "zmin": bbox[4], "zmax": bbox[5]
            }
        })

    # Save individual segments
    out_dir = Path(file_path).parent
    split_path = out_dir / "trajectory_split.json"
    with open(split_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"✅ Wrote trajectory_split.json with {len(segments)} segments")

    # Build fuzzy overlap groups
    graph = build_overlap_graph(segments, threshold=0.5)
    groups = connected_components(graph)

    fuzzy_groups = [{"group_id": f"group_{i:03}", "segments": group} for i, group in enumerate(groups)]
    group_path = out_dir / "trajectory_fuzzy_groups.json"
    with open(group_path, "w") as f:
        json.dump(fuzzy_groups, f, indent=2)
    print(f"✅ Wrote trajectory_fuzzy_groups.json with {len(fuzzy_groups)} fuzzy groups")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_trajectory.py /path/to/trajectory.txt")
        sys.exit(1)

    split_and_group_trajectory(sys.argv[1])
