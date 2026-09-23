"""`msgen.run_train` with a provenance sidecar.

WHY. `msgen/run_train.py` prints `[patch_all] active: ...` BEFORE the Tee that captures
train.log is installed, and the run.json it writes has no MSGEN_* field, so which patches
(wall, depthnorm, depthscale, seed, qfeat, ...) and which camera overrides a planner run was
trained under cannot be recovered afterwards (`patch_all.py` even says "run.json records
the flag" -- it does not). Verified: `grep "\\[patch_all\\] active" runs/*/train.log`
is empty for every run.

WHAT. Same CLI as `msgen.run_train`. Before training it writes
`runs/<tag>/patches.json` = {active patches, all MSGEN_*/MSPPO_* env vars, git HEAD,
argv}; after training it re-reads run.json and appends the same record under a
"provenance" key (run.json is rewritten by run_train, so the sidecar is the durable copy).
Existing modules are not edited: `apply_all` is wrapped to capture its return value.

USE  (drop-in)
    python -m msgen.run_train_logged --tag <arm> --dataset ... [same flags]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    import msgen.patch_all as PA
    import msgen.run_train as RT

    tag = None
    for i, a in enumerate(sys.argv):
        if a == "--tag" and i + 1 < len(sys.argv):
            tag = sys.argv[i + 1]
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    if tag is None:
        raise SystemExit("run_train_logged: --tag is required to place patches.json")
    run_dir = os.path.abspath(f"runs/{tag}")
    os.makedirs(run_dir, exist_ok=True)
    rec = dict(tag=tag, argv=sys.argv[1:], git=_git(),
               env={k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("MSGEN_", "MSPPO_"))},
               active=None)

    orig = PA.apply_all

    def apply_all(*a, **kw):
        active = orig(*a, **kw)
        rec["active"] = {k: bool(v) for k, v in dict(active).items()}
        json.dump(rec, open(f"{run_dir}/patches.json", "w"), indent=2)
        print(f"[run_train_logged] patches.json written: active="
              f"{sorted(k for k, v in rec['active'].items() if v)}", flush=True)
        return active

    PA.apply_all = apply_all
    json.dump(rec, open(f"{run_dir}/patches.json", "w"), indent=2)   # even if apply_all never runs
    try:
        RT.main()
    finally:
        rj = f"{run_dir}/run.json"
        if os.path.exists(rj):
            d = json.load(open(rj))
            d["provenance"] = rec
            json.dump(d, open(rj, "w"), indent=2)


if __name__ == "__main__":
    main()
