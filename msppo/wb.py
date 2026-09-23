"""Optional Weights & Biases monitoring for the msppo trainers.

Runs online unless MSPPO_WANDB=0 (the release scripts export MSPPO_WANDB=0).
`init_timeout` bounds a hang of `wandb.init` on a slow network; any wandb failure
prints one line and the run proceeds without it. The full config dict is logged and
the run id is returned so the trainer can record it in run.json. Training metrics
are monitoring only and are never a result source; citable numbers come from the
independent evaluation jsons.
"""
from __future__ import annotations

import os

PROJECT = "tracegen-maniskill"


def maybe_init(tag: str, cfg: dict, group: str | None = None):
    """Returns a wandb run or None. Call once, after the tag and config are known."""
    if os.environ.get("MSPPO_WANDB", "1") == "0":
        return None
    os.environ.setdefault("WANDB_MODE", "online")
    try:
        import wandb
        run = wandb.init(project=PROJECT, name=tag, group=group, config=cfg,
                         settings=wandb.Settings(init_timeout=30))
        return run
    except Exception as e:                                    # noqa: BLE001
        print(f"[wb] wandb disabled for this run: {type(e).__name__}: {e}", flush=True)
        return None


def log(run, metrics: dict, step: int | None = None):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception:                                         # noqa: BLE001
        pass


def finish(run):
    if run is None:
        return ""
    try:
        rid = run.id
        run.finish()
        return rid
    except Exception:                                         # noqa: BLE001
        return ""
