"""
Stage 2 -- Self-duplicate detection (within one member's own work).
Stage 3 -- Cross-member instance matching (which polygons, from
           different members, represent the same physical calamansi).

Uses simple IoU-threshold + connected components rather than optimal
(Hungarian) matching, because the fruits in these photos are physically
separated and do not touch -- so there is no risk of one polygon
legitimately overlapping two different neighboring fruits.
"""

from itertools import combinations

import networkx as nx

from .config import IOU_MATCH_THRESHOLD, IOU_SELFDUP_THRESHOLD


def compute_iou(poly_a, poly_b):
    if not poly_a.intersects(poly_b):
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union == 0:
        return 0.0
    return inter / union


def find_self_duplicates(anns_for_image):
    """Flag polygons from the SAME member that overlap each other."""
    flags = []
    members = {a["member"] for a in anns_for_image}
    for member in members:
        member_anns = [a for a in anns_for_image if a["member"] == member]
        for a, b in combinations(member_anns, 2):
            iou = compute_iou(a["polygon"], b["polygon"])
            if iou > IOU_SELFDUP_THRESHOLD:
                reason = ("SELF_DUPLICATE_SAME_CLASS"
                          if a["class_name"] == b["class_name"]
                          else "SELF_DUPLICATE_DIFF_CLASS")
                flags.append({
                    "queue": "REDO",
                    "reason": reason,
                    "image": a["file_name"],
                    "member": member,
                    "ann_ids": [a["raw_ann_id"], b["raw_ann_id"]],
                    "iou": round(iou, 3),
                })
    return flags


def build_instance_groups(anns_for_image):
    """Cluster polygons from DIFFERENT members that represent the same
    physical calamansi. Same-member pairs are skipped here -- they are
    handled separately by find_self_duplicates.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(anns_for_image)))

    for i, j in combinations(range(len(anns_for_image)), 2):
        a, b = anns_for_image[i], anns_for_image[j]
        if a["member"] == b["member"]:
            continue
        iou = compute_iou(a["polygon"], b["polygon"])
        if iou > IOU_MATCH_THRESHOLD:
            graph.add_edge(i, j, weight=iou)

    groups = []
    matched_indices = set()
    for component in nx.connected_components(graph):
        if len(component) > 1:
            groups.append([anns_for_image[idx] for idx in component])
            matched_indices.update(component)

    # Annotations that matched nobody are their own (singleton) group --
    # this is what later becomes an EXTRA_ANNOTATION flag.
    for idx, ann in enumerate(anns_for_image):
        if idx not in matched_indices:
            groups.append([ann])

    return groups


def average_pairwise_iou(group):
    if len(group) < 2:
        return 1.0  # nothing to compare against
    ious = [compute_iou(a["polygon"], b["polygon"]) for a, b in combinations(group, 2)]
    return sum(ious) / len(ious)
