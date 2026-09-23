"""Contact sheet: the FIRST RGB frame of every clip in a raw pool, tiled, so a
person can confirm by eye that the object and the goal are visible before the
pool is labelled and trained on (visibility is necessary,
not sufficient -- pickcube's marker is visible and still unlocalisable).

Outline: the object's segmentation (manifest `obj_seg_id`) is drawn in green and
the goal's (`goal_seg_id`, when the task renders one) in magenta on frame 0.
Ids come from the manifest's `target_id` and the body named '*goal*'
(see `_ids`); clips with neither get no outline, not an error.

    $PM -m msgen.contact_sheet --raw data/raw/pickcube_wallsel200 \
        --out <dir>/sheet.jpg
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

GREEN, MAGENTA = (0, 220, 0), (230, 0, 200)


def _outline(rgb, seg, sid, color):
    m = seg == sid
    if not m.any():
        return rgb
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    rgb = rgb.copy()
    rgb[edge] = color
    return rgb


def _ids(c):
    """(object seg id, goal seg id) from a replay manifest entry. `replay.py`
    writes `target_id` plus parallel `body_names`/`body_ids`; the goal, when the
    task has a rendered one, is the body whose name contains 'goal'. Explicit
    `obj_seg_id` / `goal_seg_id` fields win when present."""
    oid = c.get("obj_seg_id", c.get("target_id"))
    gid = c.get("goal_seg_id")
    if gid is None and "body_names" in c:
        for n, i in zip(c["body_names"], c["body_ids"]):
            if "goal" in n:
                gid = i
                break
    return oid, gid


def _thumb(img, size):
    """Nearest-neighbour resize, no extra dependency."""
    h, w = img.shape[:2]
    ys = (np.arange(size) * h / size).astype(int)
    xs = (np.arange(size) * w / size).astype(int)
    return img[ys][:, xs]


def _label(tile, text):
    """A 14-px strip under the tile; the last three characters of the clip id
    (its index digits) are drawn as bars of length digit+1. imageio has no text
    rasteriser, so the exact ids also go to the sidecar json."""
    strip = np.full((14, tile.shape[1], 3), 255, np.uint8)
    for k, ch in enumerate(text[-3:]):
        v = int(ch) if ch.isdigit() else 0
        strip[3:11, 4 + 12 * k: 4 + 12 * k + v + 1] = 0
    return np.concatenate([tile, strip], axis=0)


def build_sheet(raw_dir, out_jpg, cols=8, thumb=192, outline=True):
    import imageio.v2 as imageio
    man = json.load(open(f"{raw_dir}/manifest.json"))
    meta = {c["clip"]: c for c in man.get("clips", [])}
    files = sorted(glob.glob(f"{raw_dir}/clip_*.npz"))
    tiles, ids = [], []
    for f in files:
        clip = os.path.basename(f)[:-4]
        z = np.load(f)
        rgb = z["rgb"][0]
        if outline and "seg" in z.files and clip in meta:
            seg = z["seg"][0]
            oid, gid = _ids(meta[clip])
            if oid is not None:
                rgb = _outline(rgb, seg, oid, GREEN)
            if gid is not None:
                rgb = _outline(rgb, seg, gid, MAGENTA)
        tiles.append(_label(_thumb(rgb, thumb), clip))
        ids.append(clip)
    if not tiles:
        raise SystemExit(f"no clip_*.npz under {raw_dir}")
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    os.makedirs(os.path.dirname(out_jpg) or ".", exist_ok=True)
    imageio.imwrite(out_jpg, sheet, quality=90)
    json.dump(dict(raw=raw_dir, cols=cols, thumb=thumb, clips=ids, task=man.get("task")),
              open(out_jpg[:-4] + ".json", "w"), indent=1)
    return dict(n_clips=len(tiles), rows=rows, out=out_jpg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True, help=".jpg; a .json sidecar lists clip order")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=192)
    ap.add_argument("--no-outline", action="store_true")
    a = ap.parse_args()
    r = build_sheet(a.raw, a.out, a.cols, a.thumb, not a.no_outline)
    print(f"{r['n_clips']} clips, {r['rows']} rows -> {r['out']}")


if __name__ == "__main__":
    main()
