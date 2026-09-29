"""
Stage 1 -- Ingest COCO-segmentation exports.

Each member exports their own project from Roboflow as
"COCO Segmentation" and unzips it. The file you need is
`_annotations.coco.json` inside the split folder (train/valid/test).
"""

import json
import re
from pathlib import Path

from shapely.geometry import Polygon

from .config import CLASSES, CLASS_NAME_ALIASES

_ROBOFLOW_HASH_SUFFIX = re.compile(r"\.rf\.[0-9a-zA-Z]+\.\w+$")
_DUPLICATED_EXT_SUFFIX = re.compile(r"(_(?:png|jpe?g|bmp|tiff?))(?:\1)+$", re.IGNORECASE)

_CANONICAL_CLASS_LOOKUP = {c.lower(): c for c in CLASSES}
_CANONICAL_CLASS_LOOKUP.update({k.lower(): v for k, v in CLASS_NAME_ALIASES.items()})


def normalize_class_name(raw_name):
    """Map a raw category name from a member's export onto the canonical
    class list in config.CLASSES, using config.CLASS_NAME_ALIASES for
    known variants (case-insensitive). Unrecognized names pass through
    unchanged so they surface in the unknown-class warning at load time,
    instead of failing silently.
    """
    key = raw_name.strip().lower()
    return _CANONICAL_CLASS_LOOKUP.get(key, raw_name)


_ROBOFLOW_HASH_SUFFIX = re.compile(r"\.rf\.[0-9a-zA-Z]+\.\w+$")
_DUPLICATED_EXT_SUFFIX = re.compile(r"(_(?:png|jpe?g|bmp|tiff?))(?:\1)+$", re.IGNORECASE)


def canonical_image_key(file_name):
    """Roboflow appends a unique hash to every exported filename, e.g.
    'photo1_jpg.rf.a1b2c3d4e5f6789.jpg'. That hash differs between
    every export -- so even when 4 members annotate the exact same
    source photo, their raw file_names will NOT match. Strip the
    hash suffix so all members' exports resolve to the same key for
    the same source photo.

    Also collapses an accidentally-duplicated extension suffix, e.g.
    'clip_100_cropped_png_png' -> 'clip_100_cropped_png' -- this can
    happen if a source image already had a doubled extension (such as
    from being re-exported/re-uploaded) before ever reaching Roboflow.
    """
    key = _ROBOFLOW_HASH_SUFFIX.sub("", file_name)
    key = _DUPLICATED_EXT_SUFFIX.sub(r"\1", key)
    return key


def _reshape_to_xy_pairs(flat_coords):
    """[x1, y1, x2, y2, ...] -> [(x1, y1), (x2, y2), ...]"""
    return list(zip(flat_coords[0::2], flat_coords[1::2]))


def _make_polygon(flat_coords, member, ann_id):
    points = _reshape_to_xy_pairs(flat_coords)
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)  # standard fix for self-intersecting polygons
        print(f"[WARN] {member} ann {ann_id}: invalid polygon auto-fixed")
    return poly


def load_coco(member_name, filepath):
    """Load one member's COCO segmentation export.

    Returns a list of dicts, one per annotation:
      member, image_id, file_name, image_w, image_h,
      class_name, polygon (shapely Polygon), raw_ann_id
    """
    data = json.loads(Path(filepath).read_text())

    images_by_id = {img["id"]: img for img in data["images"]}
    categories_by_id = {cat["id"]: cat["name"] for cat in data["categories"]}

    annotations = []
    for ann in data["annotations"]:
        seg = ann.get("segmentation")
        if not seg or not seg[0]:
            continue  # skip empty/degenerate annotations

        img = images_by_id[ann["image_id"]]
        polygon = _make_polygon(seg[0], member_name, ann["id"])

        annotations.append({
            "member": member_name,
            "image_id": ann["image_id"],
            "file_name": canonical_image_key(img["file_name"]),
            "export_file_name": img["file_name"],  # kept for debugging/reference
            "image_w": img["width"],
            "image_h": img["height"],
            "class_name": normalize_class_name(categories_by_id[ann["category_id"]]),
            "polygon": polygon,
            "raw_ann_id": ann["id"],
        })

    return annotations


def load_all_members(member_files):
    """member_files: dict of {member_name: filepath}"""
    all_annotations = []
    for member_name, filepath in member_files.items():
        anns = load_coco(member_name, filepath)
        all_annotations.extend(anns)
        print(f"[INFO] Loaded {len(anns)} annotations from {member_name}")

    _warn_if_images_dont_align(all_annotations)
    _warn_if_unknown_classes(all_annotations)
    _warn_if_dimensions_mismatch(all_annotations)
    return all_annotations


def _warn_if_dimensions_mismatch(all_annotations):
    """Sanity check: if members' exports resized images differently,
    polygon coordinates are on different scales even for the 'same'
    photo, which silently corrupts every IoU calculation between them.
    """
    dims_by_image = {}
    for ann in all_annotations:
        dims_by_image.setdefault(ann["file_name"], set()).add((ann["image_w"], ann["image_h"]))

    mismatched = {k: v for k, v in dims_by_image.items() if len(v) > 1}
    if mismatched:
        print(f"[WARN] {len(mismatched)} image(s) have different width/height "
              "across members -- their IoU comparisons will be inaccurate:")
        for name, dims in list(mismatched.items())[:5]:
            print(f"    {name}: {dims}")
        if len(mismatched) > 5:
            print(f"    ...and {len(mismatched) - 5} more")
    else:
        print("[INFO] Image dimensions consistent across all members -- looks correct.")


def _warn_if_unknown_classes(all_annotations):
    unknown = {}
    for ann in all_annotations:
        if ann["class_name"] not in CLASSES:
            unknown.setdefault(ann["class_name"], set()).add(ann["member"])

    if unknown:
        print("[WARN] Found class names that don't match config.CLASSES or "
              "config.CLASS_NAME_ALIASES -- these will crash export until fixed:")
        for name, members in unknown.items():
            print(f"    '{name}'  (used by: {', '.join(sorted(members))})")
        print("    Add the missing variant to CLASS_NAME_ALIASES in config.py.")
    else:
        print("[INFO] All class names recognized -- looks correct.")


def _warn_if_images_dont_align(all_annotations):
    """Sanity check: after canonicalizing filenames, every member should
    reference the same set of images. If not, either the members
    annotated different photos, or the hash-stripping above didn't
    fully normalize this export's filename pattern.
    """
    images_per_member = {}
    for ann in all_annotations:
        images_per_member.setdefault(ann["member"], set()).add(ann["file_name"])

    all_key_sets = list(images_per_member.values())
    common = set.intersection(*all_key_sets) if all_key_sets else set()
    union = set.union(*all_key_sets) if all_key_sets else set()

    if common != union:
        print(f"[WARN] Members do not all reference the same images after "
              f"filename normalization: {len(common)} images shared by everyone, "
              f"{len(union)} images total across all members.")
        print("[WARN] If this number looks wrong, check that filenames really "
              "match across members' exports (see canonical_image_key in ingest.py).")
    else:
        print(f"[INFO] All members reference the same {len(common)} image(s) after "
              "filename normalization -- looks correct.")


def group_by_image(annotations):
    grouped = {}
    for ann in annotations:
        grouped.setdefault(ann["file_name"], []).append(ann)
    return grouped
