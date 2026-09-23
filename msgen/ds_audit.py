"""Dataset-directory audit: what the loader WILL take vs what dataset.json CLAIMS.

WHY. TraceGen/dataio/datasets.py:277 `_collect_all_episodes` enumerates episodes with
`dataset_dir.iterdir()` and NEVER reads dataset.json. Any `dataset.json` sitting in the
directory is therefore documentation, not configuration -- and the two can disagree
silently. This actually happened: `data/ds/realcam_n2400` holds 2400 clip directories
(200 demos x 3 camera views x 4 tasks) while its four dataset.json files each list 200
`clip_000..clip_199` entries with `source_raw` pointing at the v0 pool only. The names do
not even match the directories on disk (`liftpeg_v0_clip_000`), so nothing errored: the
run trained on all 2400 clips while the json said 800, and a later reader concluded that
v1/v2 were a held-out split. They were not.

WHAT THIS CHECKS. For one dataset directory:
  * how many episode directories the loader will actually see
  * how many clips the dataset.json files in that directory claim
  * whether the claimed clip names exist on disk
Any disagreement is reported. `check(..., strict=True)` raises instead of warning.

An episode directory is one that contains a `samples/` subdirectory, which is the loader's
own requirement (`_scan_episode`). Directories without it are counted separately so a stray
folder is not mistaken for a data problem.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List


def scan(dataset_dir: str) -> Dict:
    d = os.path.abspath(dataset_dir)
    subdirs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    episodes = [x for x in subdirs if os.path.isdir(os.path.join(d, x, "samples"))]
    other = [x for x in subdirs if x not in set(episodes)]
    claims: List[Dict] = []
    for p in sorted(glob.glob(f"{d}/*.json")):
        try:
            j = json.load(open(p))
        except Exception as e:
            claims.append(dict(file=os.path.basename(p), error=str(e)))
            continue
        if "clips" not in j or not isinstance(j["clips"], (list, tuple)):
            continue          # e.g. t2k_summary.json carries a clip COUNT, not a list
        names = [c["clip"] if isinstance(c, dict) else str(c) for c in j["clips"]]
        missing = [n for n in names if not os.path.isdir(os.path.join(d, n))]
        claims.append(dict(file=os.path.basename(p), task=j.get("task"),
                           source_raw=j.get("source_raw"), n_clips=len(names),
                           n_samples=j.get("n_samples"), missing=len(missing),
                           example_missing=missing[0] if missing else None))
    return dict(dir=d, n_episode_dirs=len(episodes), n_other_dirs=len(other),
                other=other[:5], claims=claims,
                claimed_total=sum(c.get("n_clips", 0) for c in claims))


def report(dataset_dir: str) -> Dict:
    r = scan(dataset_dir)
    print(f"{r['dir']}")
    print(f"  loader will see : {r['n_episode_dirs']} episode dirs")
    if r["n_other_dirs"]:
        print(f"  non-episode dirs: {r['n_other_dirs']}  e.g. {r['other']}")
    if not r["claims"]:
        print("  dataset.json    : none in this directory")
    for c in r["claims"]:
        if "error" in c:
            print(f"  {c['file']:26s} UNREADABLE: {c['error']}")
            continue
        flag = ""
        if c["missing"]:
            flag = f"  <-- {c['missing']} of its clip names are NOT on disk (e.g. {c['example_missing']})"
        print(f"  {c['file']:26s} claims {c['n_clips']:5d} clips / {c['n_samples']} samples"
              f"  source_raw={c['source_raw']}{flag}")
    if r["claims"]:
        if r["claimed_total"] != r["n_episode_dirs"]:
            print(f"  MISMATCH: json total {r['claimed_total']} != {r['n_episode_dirs']} on disk. "
                  f"The loader uses the DISK count; the json is not read.")
        else:
            print("  OK: json total matches the disk count.")
    return r


def check(dataset_dir: str, strict: bool = False) -> Dict:
    r = report(dataset_dir)
    bad = bool(r["claims"]) and r["claimed_total"] != r["n_episode_dirs"]
    if bad and strict:
        raise SystemExit(
            f"ds_audit: {dataset_dir} has {r['n_episode_dirs']} episode directories but its "
            f"dataset.json files claim {r['claimed_total']}. The loader enumerates the "
            f"directory (datasets.py:277) and ignores the json, so training would silently "
            f"use {r['n_episode_dirs']}. Fix the json or pass MSGEN_DS_AUDIT=warn.")
    return r


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or sorted(glob.glob("data/ds/*"))
    bad = 0
    for t in targets:
        if not os.path.isdir(t):
            continue
        r = report(t)
        if r["claims"] and r["claimed_total"] != r["n_episode_dirs"]:
            bad += 1
        print()
    print(f"=== {bad} directory/directories disagree with their dataset.json ===")
