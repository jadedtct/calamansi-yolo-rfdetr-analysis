# Calamansi Grading QA Pipeline

Cross-checks 4 team members' Roboflow polygon (instance segmentation)
annotations against each other: finds conflicts, auto-merges clean
majority agreement, flags self-duplicates, and reports count mismatches.

## Setup

```
pip install -r requirements.txt
```

## 1. Export from Roboflow

Each member exports their **own** annotated project:
`Export Dataset` -> format **COCO Segmentation** -> download zip.

Unzip and grab the `_annotations.coco.json` file from the split folder
(e.g. `train/_annotations.coco.json`). All 4 members must have
annotated the **same set of images**.

Place the 4 files in `data/`, then update the paths in
`calamansi_qa/config.py` under `MEMBER_FILES` to match your filenames.

## 2. Run the pipeline

```
python -m calamansi_qa.main
```

This writes to `output/`:

| File | Contents |
|---|---|
| `redo_queue.csv` | Individual fixes -- send back to the listed member |
| `deliberation_queue.csv` | Group disagreements -- needs a team decision |
| `count_mismatch.csv` | Per-image, per-member annotation count sanity check |
| `merged_dataset.coco.json` | The auto-merged (>=3/4 agreement) portion only |

## 3. Handle the redo queue

Send `redo_queue.csv` rows to the named member. They fix their own
annotations in Roboflow, re-export, and you re-run step 2.

## 4. Handle the deliberation queue

Open `deliberation_queue.csv` (a spreadsheet works fine), discuss each
row as a team, and fill in the `final_class` column with the agreed
label. Leave it blank for anything not yet decided -- those rows are
simply skipped for now.

## 5. Apply the team's decisions

```
python -m calamansi_qa.apply_deliberation
```

This reads your completed `final_class` column and writes
`output/final_dataset.coco.json` -- the auto-merged instances combined
with the deliberated ones, ready for training.

Note that you should only fill the final_class column with: Extra Class, Class 1, Class 2, Reject Class
## Reason codes

See `REASON_CODE_CHEATSHEET.md` for the full list with explanations.

## Adjusting thresholds

All tunable values (IoU thresholds, class list, adjacency pairs, member
file paths) live in `calamansi_qa/config.py`.
