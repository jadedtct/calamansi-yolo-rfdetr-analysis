import json
from pathlib import Path
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset"

ANNOTATION_FILES = {
    "train": PROJECT_ROOT / "dataset"/"rf-detr_labels"/"train_annotations.coco.json",
    "valid": PROJECT_ROOT / "dataset"/"rf-detr_labels"/"valid_annotations.coco.json",
    "test": PROJECT_ROOT / "dataset"/"rf-detr_labels"/"test_annotations.coco.json",
}

IMAGE_DIRS = {
    "train": DATASET_DIR / "images" / "train",
    "valid": DATASET_DIR / "images" / "valid",
    "test": DATASET_DIR / "images" / "test",
}

LABEL_DIRS = {
    "train": DATASET_DIR / "labels" / "train",
    "valid": DATASET_DIR / "labels" / "valid",
    "test": DATASET_DIR / "labels" / "test",
}


# ============================================================
# EXPECTED CLASSES FROM YOUR COCO DATASET
# ============================================================

EXPECTED_CLASSES = {
    0: "Extra Class",
    1: "Class 1",
    2: "Class 2",
    3: "Reject Class",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(image_dir, coco_filename):
    """
    Match a COCO filename to an actual JPG/JPEG/PNG image.

    Your COCO JSON may contain names such as:
        clip_94_cropped_png

    while the actual file may be:
        clip_94_cropped_png.jpg
    """

    coco_stem = Path(coco_filename).stem.lower()

    possible_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ]

    # Direct extension attempts
    for ext in possible_extensions:
        candidate = image_dir / (Path(coco_filename).stem + ext)

        if candidate.exists():
            return candidate

    # More robust stem comparison
    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        if image_path.stem.lower() == coco_stem:
            return image_path

    return None


def polygon_to_yolo(segmentation, width, height):
    """
    Convert COCO polygon coordinates from pixels
    into normalized YOLO coordinates.
    """

    if not segmentation:
        return []

    polygons = []

    # COCO segmentation is normally:
    # [
    #   [x1, y1, x2, y2, ...]
    # ]
    #
    # Each polygon is converted separately.

    for polygon in segmentation:

        if len(polygon) < 6:
            print("WARNING: Polygon has fewer than 3 points. Skipping.")
            continue

        normalized = []

        for i in range(0, len(polygon), 2):

            x = polygon[i]
            y = polygon[i + 1]

            # Normalize
            x_norm = x / width
            y_norm = y / height

            # Keep coordinates safely inside 0-1
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))

            normalized.append(x_norm)
            normalized.append(y_norm)

        polygons.append(normalized)

    return polygons


# ============================================================
# CONVERT ONE SPLIT
# ============================================================

def convert_split(split_name):

    annotation_file = ANNOTATION_FILES[split_name]
    image_dir = IMAGE_DIRS[split_name]
    label_dir = LABEL_DIRS[split_name]

    print("\n" + "=" * 60)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 60)

    if not annotation_file.exists():
        print(f"ERROR: Annotation file not found:")
        print(annotation_file)
        return

    if not image_dir.exists():
        print(f"ERROR: Image directory not found:")
        print(image_dir)
        return

    label_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load COCO JSON
    # --------------------------------------------------------

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"COCO images:       {len(images)}")
    print(f"COCO annotations:  {len(annotations)}")
    print(f"COCO categories:   {len(categories)}")

    # --------------------------------------------------------
    # Display categories
    # --------------------------------------------------------

    print("\nCategories found:")

    category_mapping = {}

    for category in categories:

        coco_id = category["id"]
        name = category["name"]

        category_mapping[coco_id] = name

        print(f"  {coco_id} -> {name}")

    # --------------------------------------------------------
    # Verify classes
    # --------------------------------------------------------

    if category_mapping != EXPECTED_CLASSES:

        print("\nWARNING:")
        print("The categories in this JSON do not exactly match")
        print("the expected four classes.")

        print("\nFound:")
        print(category_mapping)

        print("\nExpected:")
        print(EXPECTED_CLASSES)

        print("\nThe conversion will still use the category IDs")
        print("provided in the COCO file.")

    # --------------------------------------------------------
    # Group annotations by image ID
    # --------------------------------------------------------

    annotations_by_image = defaultdict(list)

    for annotation in annotations:

        image_id = annotation["image_id"]

        annotations_by_image[image_id].append(annotation)

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    matched_images = 0
    missing_images = 0
    total_objects = 0

    class_counter = Counter()

    for image in images:

        image_id = image["id"]
        coco_filename = image["file_name"]

        width = image["width"]
        height = image["height"]

        # Find actual JPG
        image_path = find_image(
            image_dir,
            coco_filename
        )

        if image_path is None:

            print(
                f"WARNING: Could not find image: "
                f"{coco_filename}"
            )

            missing_images += 1
            continue

        matched_images += 1

        # Label filename follows actual image filename
        label_filename = image_path.stem + ".txt"

        label_path = label_dir / label_filename

        lines = []

        image_annotations = annotations_by_image.get(
            image_id,
            []
        )

        for annotation in image_annotations:

            category_id = annotation["category_id"]

            segmentation = annotation.get(
                "segmentation"
            )

            # Skip annotations without polygons
            if not segmentation:
                continue

            polygons = polygon_to_yolo(
                segmentation,
                width,
                height
            )

            for polygon in polygons:

                if len(polygon) < 6:
                    continue

                values = [
                    str(category_id)
                ]

                values.extend(
                    f"{value:.6f}"
                    for value in polygon
                )

                lines.append(
                    " ".join(values)
                )

                class_counter[category_id] += 1
                total_objects += 1

        # Write label file
        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("\n".join(lines))

            if lines:
                f.write("\n")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nRESULT")
    print("-" * 40)

    print(f"Images matched:    {matched_images}")
    print(f"Images missing:    {missing_images}")
    print(f"Objects converted: {total_objects}")

    print("\nObjects by class:")

    for class_id in sorted(class_counter):

        class_name = category_mapping.get(
            class_id,
            f"Unknown ({class_id})"
        )

        print(
            f"  {class_id}: "
            f"{class_name:<15} "
            f"{class_counter[class_id]}"
        )

    print(f"\nLabels saved to:")
    print(label_dir)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("COCO → YOLO26 SEGMENTATION CONVERTER")
    print("=" * 60)

    for split in ["train", "valid", "test"]:

        convert_split(split)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)