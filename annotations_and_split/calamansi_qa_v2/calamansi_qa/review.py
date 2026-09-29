"""
Stage 7b -- Visual review images for the deliberation queue.

Draws each member's polygon -- color-coded, no text on the photo --
on a cropped region of the actual source photo. Labels (who voted
what) live in the gallery card's info panel below the image instead
of being burned onto the photo, so nothing covers the fruit itself.

Requires: pip install Pillow
"""

from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#9B5DE5", "#F4A261"]
_member_color_cache = {}


def get_member_color(member):
    """Deterministic per-member color, reused by both the drawn outlines
    and the gallery's legend so they always match.
    """
    if member not in _member_color_cache:
        _member_color_cache[member] = DEFAULT_PALETTE[len(_member_color_cache) % len(DEFAULT_PALETTE)]
    return _member_color_cache[member]


def _find_source_image(group, member_files):
    """Any member's actual exported image file works -- it's the same
    source photo. Roboflow keeps images in the same folder as the
    JSON, so we look next to each member's configured file path.
    """
    for ann in group:
        json_path = Path(member_files[ann["member"]])
        candidate = json_path.parent / ann["export_file_name"]
        if candidate.exists():
            return candidate
    return None


def generate_review_crop(group, member_files, output_path, padding=60):
    """Draws every member's polygon outline (color-coded, NO text) on a
    crop of the source photo around the disputed fruit. Who-voted-what
    is shown separately in the gallery card, not on the image itself,
    so outlines never cover part of the fruit. Returns True on
    success, False if the source image couldn't be found.
    """
    source_path = _find_source_image(group, member_files)
    if source_path is None:
        return False

    img = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    all_x, all_y = [], []
    for ann in group:
        xs, ys = ann["polygon"].exterior.coords.xy
        xs, ys = list(xs), list(ys)
        all_x.extend(xs)
        all_y.extend(ys)

        color = get_member_color(ann["member"])
        draw.polygon(list(zip(xs, ys)), outline=color, width=3)

    minx, maxx = min(all_x) - padding, max(all_x) + padding
    miny, maxy = min(all_y) - padding, max(all_y) + padding
    box = (max(0, minx), max(0, miny), min(img.width, maxx), min(img.height, maxy))
    crop = img.crop(box)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return True


def build_html_gallery(entries, output_path):
    """entries: list of dicts with keys:
        image_rel_path, group_id, reason,
        member_votes -- list of (member, class_name) tuples, in the
        SAME color order as the outlines drawn on the crop.
    Writes a single static HTML file you open directly in a browser --
    no server needed, works fully offline.
    """
    rows = []
    for e in entries:
        legend_items = "".join(
            f'<div class="vote-row">'
            f'<span class="swatch" style="background:{get_member_color(member)}"></span>'
            f'<span class="vote-member">{member}</span>'
            f'<span class="vote-class">{class_name}</span>'
            f'</div>'
            for member, class_name in e["member_votes"]
        )
        rows.append(f"""
        <div class="card">
          <img src="{e['image_rel_path']}" loading="lazy">
          <div class="meta">
            <div class="reason">{e['reason']}</div>
            <div class="group">{e['group_id']}</div>
            <div class="votes">{legend_items}</div>
          </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Deliberation Review</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:20px; }}
h1 {{ font-size: 20px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap:16px; }}
.card {{ background:#1c1c1c; border-radius:8px; overflow:hidden; transition:background-color 160ms ease, box-shadow 160ms ease; }}
.card:hover {{ background:#264653; box-shadow:0 0 0 2px #2A9D8F; }}
.card img {{ width:100%; display:block; background:#000; }}
.meta {{ padding:8px 10px; font-size:13px; }}
.reason {{ font-weight:600; color:#F4A261; }}
.group {{ color:#888; font-size:11px; margin:2px 0 8px; }}
.votes {{ display:flex; flex-direction:column; gap:3px; }}
.vote-row {{ display:flex; align-items:center; gap:6px; }}
.swatch {{ width:10px; height:10px; border-radius:2px; flex-shrink:0; }}
.vote-member {{ color:#ccc; min-width:70px; }}
.vote-class {{ color:#eee; font-weight:500; }}
</style></head>
<body>
<h1>Deliberation Review ({len(entries)} items)</h1>
<div class="grid">{''.join(rows)}</div>
</body></html>"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
