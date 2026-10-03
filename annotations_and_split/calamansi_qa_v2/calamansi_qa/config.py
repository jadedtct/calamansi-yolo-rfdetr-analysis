"""
Central configuration for the calamansi grading QA pipeline.
Edit the values in this file only -- the rest of the code should not
need to change for normal use.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Grading classes, in hierarchy order (best -> worst).
# Order matters: it is used to decide which class pairs are "adjacent".
# ----------------------------------------------------------------------
CLASSES = ["Extra Class", "Class 1", "Class 2", "Reject Class"]

ADJACENT_PAIRS = {
    frozenset(["Extra Class", "Class 1"]),
    frozenset(["Class 1", "Class 2"]),
    frozenset(["Class 2", "Reject Class"]),
}

# ----------------------------------------------------------------------
# IoU thresholds
# ----------------------------------------------------------------------
IOU_MATCH_THRESHOLD = 0.5      # two members' polygons = same fruit
IOU_SELFDUP_THRESHOLD = 0.5    # one member's own polygons overlap = duplicate
IOU_LOWCONF_THRESHOLD = 0.6    # matched group average IoU below this = shaky match

# ----------------------------------------------------------------------
# Member COCO-segmentation export files.
# key   = member name (used in every report so you know whose work it is)
# value = path to that member's _annotations.coco.json (exported from Roboflow)
# ----------------------------------------------------------------------
MEMBER_FILES = {
    "IRA": "C:/Users/Lenovo/Desktop/data/ira/train/_annotations.coco.json",
    "JADED": "C:/Users/Lenovo/Desktop/data/jaded/train/_annotations.coco.json",
    "KALOY": "C:/Users/Lenovo/Desktop/data/kaloy/train/_annotations.coco.json",
    "RENZO": "C:/Users/Lenovo/Desktop/data/renzo/train/_annotations.coco.json",
}

ALL_MEMBERS = set(MEMBER_FILES.keys())

OUTPUT_DIR = Path("output")

# ----------------------------------------------------------------------
# Reason codes -- kept as two separate, non-overlapping lists so redo
# issues (fixable by one person) never get mixed into the deliberation
# report (needs the whole team) in the output files.
# ----------------------------------------------------------------------
REDO_REASONS = [
    "SELF_DUPLICATE_SAME_CLASS",
    "SELF_DUPLICATE_DIFF_CLASS",
    "MISSING_ANNOTATION",
    "EXTRA_ANNOTATION",
    "LOW_CONFIDENCE_MATCH",
]

DELIBERATION_REASONS = [
    "FULL_DISAGREEMENT",
    "TIE",
    "ADJACENT_CLASS_CONFUSION",
    "SPLIT_MINORITY",
]

# ----------------------------------------------------------------------
# Class name aliases. Different members' Roboflow projects can end up
# with slightly different category names for the same class (e.g. one
# project has "Reject Class", another just has "Reject"). Add any
# variant you find here -- keys are matched case-insensitively.
# ----------------------------------------------------------------------
CLASS_NAME_ALIASES = {
    "extra": "Extra Class",
    "extra class": "Extra Class",
    "class 1": "Class 1",
    "class1": "Class 1",
    "class 2": "Class 2",
    "class2": "Class 2",
    "reject": "Reject Class",
    "rejected": "Reject Class",
    "reject class": "Reject Class",
}
