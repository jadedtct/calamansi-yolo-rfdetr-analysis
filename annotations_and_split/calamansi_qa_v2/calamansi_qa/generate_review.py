"""
Run AFTER main.py has produced output/deliberation_queue.csv.

    python -m calamansi_qa.generate_review

Generates one cropped, annotated image per deliberation item into
output/review_images/, organized into subfolders by reason code, plus
output/review_gallery.html -- a single offline page to browse them all.

Open review_gallery.html in your browser, decide each one, then fill
in the 'final_class' column in deliberation_queue.csv as usual.
"""

import csv

from .config import MEMBER_FILES, OUTPUT_DIR
from .ingest import load_all_members, group_by_image
from .matching import build_instance_groups
from .review import generate_review_crop, build_html_gallery


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


def run():
    group_lookup = _rebuild_group_lookup()

    review_dir = OUTPUT_DIR / "review_images"
    gallery_entries = []
    missing_source = 0

    with open(OUTPUT_DIR / "deliberation_queue.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        group_id = row["group_id"]
        group = group_lookup.get(group_id)
        if not group:
            continue

        reason = row["reason"]
        safe_group_id = group_id.replace("/", "_").replace("\\", "_")
        out_path = review_dir / reason / f"{safe_group_id}.png"

        ok = generate_review_crop(group, MEMBER_FILES, out_path)
        if not ok:
            missing_source += 1
            continue

        gallery_entries.append({
            "image_rel_path": f"review_images/{reason}/{safe_group_id}.png",
            "group_id": group_id,
            "reason": reason,
            # sorted for a stable, consistent order across the gallery --
            # matches the deterministic color each member gets in review.py
            "member_votes": sorted((a["member"], a["class_name"]) for a in group),
        })

    build_html_gallery(gallery_entries, OUTPUT_DIR / "review_gallery.html")

    print(f"[INFO] Generated {len(gallery_entries)} review images -> {review_dir}")
    if missing_source:
        print(f"[WARN] Could not find the source photo for {missing_source} item(s) -- "
              "check that images are stored next to each member's JSON export.")
    print(f"[INFO] Open output/review_gallery.html in your browser to review everything.")


if __name__ == "__main__":
    run()
