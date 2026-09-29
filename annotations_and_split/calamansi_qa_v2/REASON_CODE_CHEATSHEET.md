# Reason Code Cheat Sheet

Two separate queues. **Redo** items are fixable by one person alone,
no discussion needed. **Deliberation** items need the whole team to
decide the correct label.

## REDO queue (`redo_queue.csv`) -- individual fixes

| Code | Meaning | What to do |
|---|---|---|
| `SELF_DUPLICATE_SAME_CLASS` | One member traced the same calamansi twice with the same label | Delete the duplicate polygon |
| `SELF_DUPLICATE_DIFF_CLASS` | One member traced the same calamansi twice with two different labels | They weren't sure which class it was -- pick one and delete the other |
| `MISSING_ANNOTATION` | At least 2+ other members marked this fruit, but this member didn't | Go back and annotate it |
| `EXTRA_ANNOTATION` | Only one member marked this -- nobody else's polygon matched it | Verify it's actually a calamansi (not a leaf/shadow/duplicate); remove if not |
| `LOW_CONFIDENCE_MATCH` | The matched polygons barely cleared the overlap threshold | The tracing was imprecise -- redo the polygon more carefully |

## DELIBERATION queue (`deliberation_queue.csv`) -- team decision needed

| Code | Meaning | What it signals |
|---|---|---|
| `FULL_DISAGREEMENT` | All 4 members picked 4 different classes (1-1-1-1) | Strongest confusion signal -- review these first |
| `TIE` | Exactly 2-2 split between two classes | No majority at all |
| `ADJACENT_CLASS_CONFUSION` | 2-1-1 split, and at least one minority vote is next to the majority class in the grading hierarchy (Extra Class ↔ Class 1 ↔ Class 2 ↔ Reject Class) | Likely a genuinely blurry quality boundary -- may mean your team needs clearer shared examples of where one grade ends and the next begins |
| `SPLIT_MINORITY` | 2-1-1 split, but the minority votes are NOT adjacent to the majority class | More likely a real mistake (wrong click, missed defect, misunderstood class) than a boundary judgment call |

## Auto-merge rule (no flag at all -- happens silently)

Any class with **3 or more of 4 votes** merges automatically as that
class, regardless of what the 4th vote was (even if that 4th vote is
far from the majority in the hierarchy). Example: 3x Extra Class + 1x
Class 1 -> merges as **Extra Class**.
