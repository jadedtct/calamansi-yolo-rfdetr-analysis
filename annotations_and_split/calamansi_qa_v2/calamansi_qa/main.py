"""
Run the full QA pipeline end to end.

    python -m calamansi_qa.main

Reads each member's COCO segmentation export (see config.MEMBER_FILES),
and writes to OUTPUT_DIR:
    redo_queue.csv          -- individual fixes, no discussion needed
    deliberation_queue.csv  -- group disagreements, needs team decision
    count_mismatch.csv      -- per-image / per-member count sanity check
    merged_dataset.coco.json -- the auto-merged (>=3/4 agreement) portion
"""

from .config import MEMBER_FILES, OUTPUT_DIR
from .ingest import load_all_members, group_by_image
from .matching import find_self_duplicates, build_instance_groups
from .voting import decide_group
from .reports import (
    check_count_mismatches,
    write_redo_queue,
    write_deliberation_queue,
    write_count_mismatch_report,
    export_merged_coco,
)


def run():
    all_annotations = load_all_members(MEMBER_FILES)
    by_image = group_by_image(all_annotations)

    redo_flags = []
    conflict_flags = []
    mismatch_flags = []
    merged_records = []

    for image_name, anns_for_image in by_image.items():
        redo_flags.extend(find_self_duplicates(anns_for_image))

        groups = build_instance_groups(anns_for_image)

        for i, group in enumerate(groups):
            group_id = f"{image_name}__group{i}"
            merged, conflict, extra_flags = decide_group(group, image_name, group_id)
            redo_flags.extend(f for f in extra_flags if f["queue"] == "REDO")
            if merged:
                merged_records.append(merged)
            if conflict:
                conflict_flags.append(conflict)

        mismatch_flags.extend(check_count_mismatches(anns_for_image, groups, image_name))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_redo_queue(redo_flags, OUTPUT_DIR / "redo_queue.csv")
    write_deliberation_queue(conflict_flags, OUTPUT_DIR / "deliberation_queue.csv")
    write_count_mismatch_report(mismatch_flags, OUTPUT_DIR / "count_mismatch.csv")
    export_merged_coco(merged_records, OUTPUT_DIR / "merged_dataset.coco.json")

    print("\n===== SUMMARY =====")
    print(f"Images processed:        {len(by_image)}")
    print(f"Auto-merged instances:   {len(merged_records)}")
    print(f"Redo items:              {len(redo_flags)}")
    print(f"Deliberation items:      {len(conflict_flags)}")
    print(f"Count-mismatch rows:     {len(mismatch_flags)}")
    print("====================\n")
    print("Next steps:")
    print("1. Send redo_queue.csv items back to the listed member to fix individually.")
    print("2. Review deliberation_queue.csv as a team, fill in the 'final_class' column.")
    print("3. Run apply_deliberation.py with the completed CSV to fold those into the dataset.")


if __name__ == "__main__":
    run()
