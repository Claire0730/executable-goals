"""$PM tests/test_contact_sheet.py -- contact sheet on a synthetic 2-clip pool."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def make_pool(d):
    os.makedirs(d, exist_ok=True)
    clips = []
    for i in range(2):
        rgb = np.full((5, 384, 384, 3), 40 * (i + 1), np.uint8)
        rgb[0, 100:150, 100:150] = 255                      # a bright square on frame 0 only
        seg = np.zeros((5, 384, 384), np.uint16); seg[:, 100:150, 100:150] = 7
        np.savez(f"{d}/clip_{i:03d}.npz", rgb=rgb, seg=seg, body_ids=np.arange(43, dtype=np.int32))
        clips.append(dict(clip=f"clip_{i:03d}", obj_seg_id=7))
    json.dump(dict(task="pickcube", clips=clips), open(f"{d}/manifest.json", "w"))


def main():
    from msgen.contact_sheet import build_sheet
    import imageio.v2 as imageio
    with tempfile.TemporaryDirectory() as td:
        make_pool(f"{td}/raw")
        r = build_sheet(f"{td}/raw", f"{td}/sheet.jpg", cols=2, thumb=96)
        check("n_clips == 2", r["n_clips"] == 2, str(r))
        img = imageio.imread(f"{td}/sheet.jpg")
        check("sheet is 1 row x 2 cols of 96px + label strip",
              img.shape[1] == 2 * 96 and img.shape[0] >= 96, str(img.shape))
        check("uses FIRST frame", img[:96, :96].max() > 200, str(img[:96, :96].max()))
        check("sidecar lists clips", json.load(open(f"{td}/sheet.json"))["clips"] == ["clip_000", "clip_001"])
    print(f"\n{len(OK)} pass, {len(BAD)} fail"); sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
