"""
Stage 5 -- Consensus polygon selection.

When >=3 members agree on a class, we need ONE polygon to represent
that calamansi in the final dataset. We pick whichever agreeing
member's polygon has the highest average IoU against the other
agreeing members' polygons -- i.e. the most "typical" trace, rather
than an arbitrary pick or a synthetic averaged shape.
"""

from .matching import compute_iou


def pick_consensus_polygon(group, agreed_class):
    agreeing = [a for a in group if a["class_name"] == agreed_class]

    if len(agreeing) == 1:
        return agreeing[0]

    best_ann = None
    best_avg_iou = -1
    for candidate in agreeing:
        others = [a for a in agreeing if a is not candidate]
        ious = [compute_iou(candidate["polygon"], o["polygon"]) for o in others]
        avg_iou = sum(ious) / len(ious)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_ann = candidate

    return best_ann


def pick_representative_polygon(group, final_class):
    """Used when applying a DELIBERATED decision, where the team's final
    class may not match any single member's vote. Prefers a polygon from
    a member who voted that class; otherwise falls back to the most
    "typical" polygon among the whole group (highest average IoU to
    the rest), regardless of what class they originally voted.
    """
    agreeing = [a for a in group if a["class_name"] == final_class]
    if agreeing:
        return pick_consensus_polygon(group, final_class)

    if len(group) == 1:
        return group[0]

    best_ann, best_avg_iou = None, -1
    for candidate in group:
        others = [a for a in group if a is not candidate]
        ious = [compute_iou(candidate["polygon"], o["polygon"]) for o in others]
        avg_iou = sum(ious) / len(ious)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_ann = candidate
    return best_ann
