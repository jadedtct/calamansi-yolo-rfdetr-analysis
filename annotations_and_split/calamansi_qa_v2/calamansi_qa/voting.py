"""
Stage 4 -- Voting and merge/deliberation decision.

Rule: a class needs >= 3 of 4 votes to auto-merge. This check always
happens FIRST -- so e.g. 3 votes Extra Class / 1 vote Class 1 merges
immediately as Extra Class, regardless of adjacency.

Only when nobody has 3+ votes do we classify WHY the team disagreed:
  FULL_DISAGREEMENT        -- 1-1-1-1, every member picked differently
  TIE                      -- 2-2
  ADJACENT_CLASS_CONFUSION -- 2-1-1, and the minority classes sit next
                               to the majority class in the grading
                               hierarchy (a genuinely blurry boundary)
  SPLIT_MINORITY           -- 2-1-1, but the split crosses non-adjacent
                               classes (more likely a real error)
"""

from collections import Counter

from .config import ADJACENT_PAIRS, ALL_MEMBERS, IOU_LOWCONF_THRESHOLD
from .matching import average_pairwise_iou
from .consensus import pick_consensus_polygon


def _is_full_disagreement(class_votes):
    return len(class_votes) == 4 and all(v == 1 for v in class_votes.values())


def _is_even_tie(class_votes):
    counts = sorted(class_votes.values(), reverse=True)
    return len(counts) > 1 and counts[0] == counts[1]


def _classes_adjacent_to_majority(class_votes, top_class):
    others = [c for c in class_votes if c != top_class]
    return any(frozenset([top_class, other]) in ADJACENT_PAIRS for other in others)


def decide_group(group, image_name, group_id):
    """Returns (merged_record | None, conflict_flag | None, extra_flags list)"""
    extra_flags = []
    members_present = {a["member"] for a in group}
    class_votes = Counter(a["class_name"] for a in group)
    top_class, top_count = class_votes.most_common(1)[0]

    if len(group) == 1:
        # Nobody else on the team matched this fruit -- it might be a real
        # miss by the other 3, or a false positive by this member. We don't
        # know which yet, so we flag ONLY as EXTRA_ANNOTATION (ask this
        # member to verify) rather than also accusing the other 3 of
        # missing something that may not even be a real calamansi.
        extra_flags.append({
            "queue": "REDO",
            "reason": "EXTRA_ANNOTATION",
            "image": image_name,
            "group_id": group_id,
            "member": group[0]["member"],
            "ann_id": group[0]["raw_ann_id"],
        })
        return None, None, extra_flags

    if len(members_present) < 4:
        missing = ALL_MEMBERS - members_present
        extra_flags.append({
            "queue": "REDO",
            "reason": "MISSING_ANNOTATION",
            "image": image_name,
            "group_id": group_id,
            "missing_members": sorted(missing),
        })

    avg_iou = average_pairwise_iou(group)
    if avg_iou < IOU_LOWCONF_THRESHOLD:
        extra_flags.append({
            "queue": "REDO",
            "reason": "LOW_CONFIDENCE_MATCH",
            "image": image_name,
            "group_id": group_id,
            "avg_iou": round(avg_iou, 3),
        })

    if top_count >= 3:
        rep_ann = pick_consensus_polygon(group, top_class)
        merged = {
            "image": image_name,
            "group_id": group_id,
            "class_name": top_class,
            "polygon": rep_ann["polygon"],
            "image_w": rep_ann["image_w"],
            "image_h": rep_ann["image_h"],
            "contributing_members": sorted(members_present),
            "status": "MERGED",
        }
        return merged, None, extra_flags

    if _is_full_disagreement(class_votes):
        reason = "FULL_DISAGREEMENT"
    elif _is_even_tie(class_votes):
        reason = "TIE"
    elif _classes_adjacent_to_majority(class_votes, top_class):
        reason = "ADJACENT_CLASS_CONFUSION"
    else:
        reason = "SPLIT_MINORITY"

    conflict = {
        "queue": "DELIBERATION",
        "reason": reason,
        "image": image_name,
        "group_id": group_id,
        "votes": dict(class_votes),
        "members_present": sorted(members_present),
        "final_class": "",  # <-- your team fills this in after discussion
    }
    return None, conflict, extra_flags
