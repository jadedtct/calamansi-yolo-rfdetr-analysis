"""
Stage 8 -- Split the final QA'd dataset into train/valid/test.

Per the thesis manuscript: 70/15/15 split, stratified so each split
holds a proportional share of each of the 4 grading classes, using a
fixed random seed for reproducibility. Splitting is done by IMAGE
(never by instance) so no single photo's fruits get scattered across
two splits -- see data_splitting_augmentation_summary.md for why.

    python -m calamansi_qa.split_dataset

Requires: pip install iterative-stratification

Reads:  output/final_dataset.coco.json
Writes: output/splits/train/_annotations.coco.json (+ images)
        output/splits/valid/_annotations.coco.json (+ images)
        output/splits/test/_annotations.coco.json  (+ images)

NOTE (per the manuscript): before splitting, images with insufficient
resolution, severe motion blur, or annotation errors should already
have been excluded (this is described as happening within Roboflow,
before this step). This script assumes that cleanup already happened
and does not re-check it.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .config import MEMBER_FILES, CLASSES, OUTPUT_DIR
from .ingest import load_all_members

RANDOM_SEED = 42  # fixed for reproducibility -- record this in your methodology
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def _build_class_matrix(data):
    """One row per image, one column per class -- instance counts.
    This is what the stratified splitter balances across train/val/test.
    """
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    images = data["images"]
    image_id_to_row = {img["id"]: i for i, img in enumerate(images)}
    matrix = np.zeros((len(images), len(CLASSES)), dtype=int)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    for ann in data["annotations"]:
        row = image_id_to_row[ann["image_id"]]
        col = class_to_idx[cat_id_to_name[ann["category_id"]]]
        matrix[row, col] += 1

    return matrix, images


def _subset_coco(data, keep_image_ids):
    """Builds a new COCO dict containing only the given images (and their
    annotations), with fresh sequential ids. Does not mutate `data`, so
    it's safe to call multiple times against the same source dict.
    """
    keep_image_ids = {int(i) for i in keep_image_ids}

    new_images = []
    old_to_new_img = {}
    for img in data["images"]:
        if img["id"] in keep_image_ids:
            new_id = len(new_images)
            old_to_new_img[img["id"]] = new_id
            new_img = dict(img)
            new_img["id"] = new_id
            new_images.append(new_img)

    new_annotations = []
    for ann in data["annotations"]:
        if ann["image_id"] in keep_image_ids:
            new_ann = dict(ann)
            new_ann["id"] = len(new_annotations)
            new_ann["image_id"] = old_to_new_img[ann["image_id"]]
            new_annotations.append(new_ann)

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"],
    }


def _print_class_counts(name, coco_subset):
    cat_id_to_name = {c["id"]: c["name"] for c in coco_subset["categories"]}
    counts = {cls: 0 for cls in CLASSES}
    for ann in coco_subset["annotations"]:
        counts[cat_id_to_name[ann["category_id"]]] += 1
    total = sum(counts.values())
    print(f"\n{name}: {len(coco_subset['images'])} images, {total} instances")
    for cls in CLASSES:
        pct = (counts[cls] / total * 100) if total else 0
        print(f"    {cls:<15} {counts[cls]:>5}  ({pct:.1f}%)")


def _find_any_source_image(canonical_name, all_annotations, member_files):
    for ann in all_annotations:
        if ann["file_name"] == canonical_name:
            json_path = Path(member_files[ann["member"]])
            candidate = json_path.parent / ann["export_file_name"]
            if candidate.exists():
                return candidate
    return None


def _copy_images(coco_subset, all_annotations, member_files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for img in coco_subset["images"]:
        src = _find_any_source_image(img["file_name"], all_annotations, member_files)
        if src is None:
            missing += 1
            continue
        shutil.copy(src, dest_dir / (img["file_name"] + src.suffix))
    if missing:
        print(f"[WARN] Could not find the source photo for {missing} image(s) in {dest_dir}")


def run():
    final_path = OUTPUT_DIR / "final_dataset.coco.json"
    data = json.loads(final_path.read_text())

    matrix, images = _build_class_matrix(data)
    image_ids = np.array([img["id"] for img in images])

    # Split 1: train (70%) vs temp (30% -> will become val+test)
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    train_idx, temp_idx = next(msss1.split(image_ids, matrix))

    # Split 2: temp -> val (50%) vs test (50%) == 15%/15% of the total
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=RANDOM_SEED
    )
    val_sub_idx, test_sub_idx = next(msss2.split(image_ids[temp_idx], matrix[temp_idx]))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    train_coco = _subset_coco(data, image_ids[train_idx])
    val_coco = _subset_coco(data, image_ids[val_idx])
    test_coco = _subset_coco(data, image_ids[test_idx])

    splits_dir = OUTPUT_DIR / "splits"
    for name, coco_subset in [("train", train_coco), ("valid", val_coco), ("test", test_coco)]:
        out_dir = splits_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_annotations.coco.json").write_text(json.dumps(coco_subset, indent=2))
        _print_class_counts(name, coco_subset)

    print(f"\n[INFO] Wrote train/valid/test annotation files -> {splits_dir}")
    print("[INFO] Copying source images into each split folder...")

    all_annotations = load_all_members(MEMBER_FILES)
    for name, coco_subset in [("train", train_coco), ("valid", val_coco), ("test", test_coco)]:
        _copy_images(coco_subset, all_annotations, MEMBER_FILES, splits_dir / name)

    print("[INFO] Done. Random seed used:", RANDOM_SEED)


if __name__ == "__main__":
    run()
