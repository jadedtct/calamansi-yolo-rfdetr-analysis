"""
Stage 6 -- Count mismatch report (per image, per member).
Stage 7 -- Write everything out: redo queue, deliberation queue,
           count-mismatch report, and the final merged COCO dataset.
"""

import csv
import json
from pathlib import Path

from .config import CLASSES


def check_count_mismatches(anns_for_image, groups, image_name):
    """Compare each member's raw annotation count on this image against
    how many instance groups they actually ended up matched into.
    A mismatch usually means a missed self-duplicate, or an unmatched
    extra annotation (both already flagged elsewhere, but this is a
    quick per-image sanity total).
    """
    flags = []
    members = {a["member"] for a in anns_for_image}
    consensus_count = len(groups)
    for member in members:
        raw_count = sum(1 for a in anns_for_image if a["member"] == member)
        matched_count = sum(1 for g in groups if any(a["member"] == member for a in g))
        if raw_count != matched_count:
            flags.append({
                "image": image_name,
                "member": member,
                "raw_annotation_count": raw_count,
                "matched_group_count": matched_count,
                "image_consensus_fruit_count": consensus_count,
            })
    return flags


def _write_csv(rows, filepath, fieldnames):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # flatten list/dict values so they display cleanly in a spreadsheet
            clean_row = {
                k: (json.dumps(v) if isinstance(v, (list, dict, set)) else v)
                for k, v in row.items()
            }
            writer.writerow(clean_row)


def write_redo_queue(redo_flags, filepath):
    fieldnames = ["queue", "reason", "image", "member", "group_id",
                  "ann_id", "ann_ids", "missing_members", "avg_iou", "iou"]
    _write_csv(redo_flags, filepath, fieldnames)
    print(f"[INFO] Wrote {len(redo_flags)} redo items -> {filepath}")


def write_deliberation_queue(conflict_flags, filepath):
    fieldnames = ["queue", "reason", "image", "group_id", "votes",
                  "members_present", "final_class"]
    _write_csv(conflict_flags, filepath, fieldnames)
    print(f"[INFO] Wrote {len(conflict_flags)} deliberation items -> {filepath}")


def write_count_mismatch_report(mismatch_flags, filepath):
    fieldnames = ["image", "member", "raw_annotation_count",
                  "matched_group_count", "image_consensus_fruit_count"]
    _write_csv(mismatch_flags, filepath, fieldnames)
    print(f"[INFO] Wrote {len(mismatch_flags)} count-mismatch rows -> {filepath}")


def combine_coco_files(filepaths, output_path):
    """Merge multiple COCO-segmentation JSON files (e.g. the auto-merged
    dataset and the deliberated dataset) into one final training-ready file.
    Assumes all inputs use the same category list/order.
    """
    all_images = {}   # file_name -> image record (re-ids as we go)
    all_annotations = []
    categories = None

    for fp in filepaths:
        data = json.loads(Path(fp).read_text())
        if categories is None:
            categories = data["categories"]

        old_id_to_new_image = {}
        for img in data["images"]:
            if img["file_name"] not in all_images:
                new_id = len(all_images)
                all_images[img["file_name"]] = {**img, "id": new_id}
            old_id_to_new_image[img["id"]] = all_images[img["file_name"]]["id"]

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = len(all_annotations)
            new_ann["image_id"] = old_id_to_new_image[ann["image_id"]]
            all_annotations.append(new_ann)

    combined = {
        "images": list(all_images.values()),
        "annotations": all_annotations,
        "categories": categories,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2))
    print(f"[INFO] Combined {len(all_annotations)} annotations -> {output_path}")


def export_merged_coco(merged_records, filepath, extra_records=None):
    """Build a clean COCO-segmentation JSON from the auto-merged instances
    (and, optionally, instances resolved during deliberation -- pass
    those in as extra_records once your team fills in `final_class`).
    """
    records = list(merged_records) + list(extra_records or [])

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    class_to_id = {name: i for i, name in enumerate(CLASSES)}

    images = {}
    for rec in records:
        if rec["image"] not in images:
            images[rec["image"]] = {
                "id": len(images),
                "file_name": rec["image"],
                "width": rec["image_w"],
                "height": rec["image_h"],
            }

    annotations = []
    for i, rec in enumerate(records):
        poly = rec["polygon"]
        xs, ys = poly.exterior.coords.xy
        flat_coords = [coord for pair in zip(xs, ys) for coord in pair]
        minx, miny, maxx, maxy = poly.bounds

        annotations.append({
            "id": i,
            "image_id": images[rec["image"]]["id"],
            "category_id": class_to_id[rec["class_name"]],
            "segmentation": [flat_coords],
            "area": poly.area,
            "bbox": [minx, miny, maxx - minx, maxy - miny],
            "iscrowd": 0,
            "contributing_members": rec.get("contributing_members", []),
            "status": rec.get("status", "DELIBERATED"),
        })

    coco_out = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories,
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(coco_out, indent=2))
    print(f"[INFO] Wrote {len(annotations)} final annotations -> {filepath}")
