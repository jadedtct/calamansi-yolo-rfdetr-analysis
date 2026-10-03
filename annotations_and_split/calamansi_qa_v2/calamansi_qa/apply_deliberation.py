"""
After your team reviews deliberation_queue.csv and fills in the
'final_class' column, run this to merge those decisions into the
final dataset:

    python -m calamansi_qa.apply_deliberation

This re-derives the same instance groups from the raw exports (matching
is deterministic given the same inputs), looks each one up by its
group_id, attaches the team's chosen class, and appends it alongside
merged_dataset.coco.json into the combined final dataset.
"""

import csv

from .config import MEMBER_FILES, OUTPUT_DIR
from .ingest import load_all_members, group_by_image
from .matching import build_instance_groups
from .consensus import pick_representative_polygon
from .reports import export_merged_coco, combine_coco_files


def _rebuild_group_lookup():
    all_annotations = load_all_members(MEMBER_FILES)
    by_image = group_by_image(all_annotations)

    lookup = {}
    for image_name, anns_for_image in by_image.items():
        groups = build_instance_groups(anns_for_image)
        for i, group in enumerate(groups):
            group_id = f"{image_name}__group{i}"
            lookup[group_id] = group
    return lookup


def apply(deliberation_csv_path, merged_dataset_path, output_path):
    group_lookup = _rebuild_group_lookup()

    deliberated_records = []
    skipped = 0

    with open(deliberation_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            final_class = (row.get("final_class") or "").strip()
            if not final_class:
                skipped += 1
                continue  # not decided yet -- leave it out for now

            group_id = row["group_id"]
            group = group_lookup.get(group_id)
            if not group:
                print(f"[WARN] group_id {group_id} not found, skipping")
                continue

            rep_ann = pick_representative_polygon(group, final_class)
            deliberated_records.append({
                "image": row["image"],
                "group_id": group_id,
                "class_name": final_class,
                "polygon": rep_ann["polygon"],
                "image_w": rep_ann["image_w"],
                "image_h": rep_ann["image_h"],
                "contributing_members": sorted({a["member"] for a in group}),
                "status": "DELIBERATED",
            })

    if skipped:
        print(f"[INFO] {skipped} rows had no final_class yet and were skipped.")

    # Re-load already-merged records isn't needed from JSON -- easier to
    # just re-run main.run() logic; but to keep this script standalone and
    # avoid recomputation, we simply re-export using the freshly merged
    # + deliberated records together. If you've already run main.py,
    # pass its merged_records back in via a small wrapper in your own
    # script if you want to avoid recompute -- shown simply here for clarity.
    export_merged_coco(deliberated_records, output_path, extra_records=[])
    print(f"[INFO] {len(deliberated_records)} deliberated instances written to {output_path}")

    final_path = OUTPUT_DIR / "final_dataset.coco.json"
    combine_coco_files([merged_dataset_path, output_path], final_path)
    print(f"[INFO] Training-ready dataset (merged + deliberated) -> {final_path}")


if __name__ == "__main__":
    apply(
        deliberation_csv_path=OUTPUT_DIR / "deliberation_queue.csv",
        merged_dataset_path=OUTPUT_DIR / "merged_dataset.coco.json",
        output_path=OUTPUT_DIR / "deliberated_dataset.coco.json",
    )
