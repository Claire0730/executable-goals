"""Make `predict_trajectory` deterministic, from outside the pristine checkout.

VERIFIED DEFECT. `models/model_flow.py:292-297` calls the decoder's `predict_trajectory`
without passing a generator, and the decoder's signature defaults it to `None`
(`cogvideox_flow.py:466`), so the initial latent at `cogvideox_flow.py:517`

    latents = torch.randn(latents_shape, generator=generator, device=..., dtype=...)

is drawn from the global RNG. The same checkpoint therefore produces different traces run to
run. This is the source of the 19-75% trace-metric variance recorded in
`<private-repo>/docs/HANDOFF_PLANNER_20260813.md`, and it is what makes a small MSE difference between two
arms unreadable.

WHY A COUNTER AND NOT ONE FIXED SEED. Seeding identically on every call would give every
batch the same initial noise -- reproducible, but a single draw replicated, which biases the
average. The counter gives independent noise across batches while keeping the SEQUENCE
reproducible, so arm A and arm B see the same noise on the same batch: a paired comparison,
which is exactly what a 1536-parameter change needs.

The counter lives on the module instance and resets when the process starts, so
reproducibility depends on the batch order being fixed -- true for `val_loader`, which is
built without shuffling.

Also seeds the global RNG per call, because `cogvideox_flow.py:495` draws
`negative_prompt_embeds` WITHOUT a generator. That tensor is only used when
`guidance_scale > 1.0`, and both our eval paths use 1.0 (`trainer.py:971`,
`msgen/predict.py:96`), so it does not currently affect any number -- seeding it costs
nothing and removes the trap if someone raises the guidance scale later.

    from msgen.patch_seed import maybe_patch
    maybe_patch()             # MSGEN_SEED=1234, before the trainer builds the model
"""
from __future__ import annotations

import os

_APPLIED = False


def patch_seed(seed: int = 1234):
    global _APPLIED
    if _APPLIED:
        return
    import torch
    from models.decoder.cogvideox_flow import CogVideoXDecoder_flow

    orig = CogVideoXDecoder_flow.predict_trajectory

    def predict_trajectory(self, trunk_conditioning, *a, **k):
        n = getattr(self, "_seed_calls", 0)
        self._seed_calls = n + 1
        s = seed + n
        dev = trunk_conditioning.device
        if k.get("generator") is None:
            g = torch.Generator(device=dev)
            g.manual_seed(s)
            k["generator"] = g
        torch.manual_seed(s)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(s)
        return orig(self, trunk_conditioning, *a, **k)

    CogVideoXDecoder_flow.predict_trajectory = predict_trajectory
    _APPLIED = True
    print(f"[patch_seed] predict_trajectory is now deterministic (base seed {seed}, "
          f"+1 per call, so batch k always gets noise {seed}+k)")


def seed_globals(seed: int = 1234):
    """Seed the process RNGs, which is what actually pins the DATA side.

    MEASURED: with `patch_seed` alone, two seeded repeats of the misclassification rate came
    out 11.4% and 11.1% -- reduced from the 0.9 pp unseeded range but not identical. The
    residue is not in the sampler: `dataio/datasets.py:559,578` choose one of the three
    instructions with Python's global `random.choice`, inside `__getitem__`, which runs in
    DataLoader WORKER processes. `patch_seed` seeds inside `predict_trajectory`, i.e. after
    the batch has already been fetched, so it cannot reach them.

    torch derives each worker's seed from the main process's RNG when the iterator is created
    (`base_seed + worker_id`, and `_worker_loop` seeds both `random` and `torch` with it), so
    seeding here -- immediately before iteration -- makes the instruction choice deterministic
    AND identical across arms. This is exactly why the run_eval path was already bit-identical:
    `test_benchmark.py:136` calls `setup_seed` before iterating.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[patch_seed] seeded process RNGs with {seed} (pins the random instruction choice)")


def seed_from_env():
    """Seed globals iff MSGEN_SEED is set. Call right before iterating a loader."""
    v = os.environ.get("MSGEN_SEED", "")
    if v and v != "0":
        seed_globals(1234 if v in ("1", "on", "true") else int(v))
        return True
    return False


def maybe_patch():
    """Apply only when MSGEN_SEED is set, so every existing run is unaffected."""
    v = os.environ.get("MSGEN_SEED", "")
    if v and v != "0":
        patch_seed(seed=1234 if v in ("1", "on", "true") else int(v))
        return True
    return False
