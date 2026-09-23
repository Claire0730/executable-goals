"""Make `lr_backbone` reachable. Enable with MSGEN_LRGROUPS.

THE DEFECT. `TraceGen/utils/misc.py` splits the optimizer into two groups by
substring:

    if any(backbone_name in name for backbone_name in ['vision_encoder', 'text_encoder']):

No parameter in this model contains either string. The four encoders are named
`dino_encoder`, `siglip_encoder`, `depth_encoder`, `t5_encoder`, so
`backbone_params` is always empty, the second param group is never created, and
`cfg.train.lr_backbone` (2.0e-5 in cfg/eval.yaml:45) is dead config.

WHY IT IS CURRENTLY HARMLESS BUT WILL NOT REMAIN SO. The loop skips every parameter with
`requires_grad=False`, and all four encoders are frozen, so the group would be
empty even with correct names -- this patch is an identity operation right now.
It bites the moment anything calls `model_flow.unfreeze_backbones()`: 67.7 M
pretrained parameters (last 2 blocks of all four encoders, measured) would train
at `lr_decoder` = 1.5e-4 instead of the 2.0e-5 the config asks for, a factor of
7.5, on 200 clips. That is the shape of catastrophic forgetting, and it would
look like "unfreezing does not work" rather than like a routing bug.

Install it before the unfreeze experiment, not with it, so the two changes are
never confounded.

    MSGEN_LRGROUPS=1 $PG -m msgen.run_train ...
"""
from __future__ import annotations

import os

# The real module names, from the model's own named_parameters().
BACKBONES = ("dino_encoder", "siglip_encoder", "depth_encoder", "t5_encoder")
_APPLIED = False


def maybe_patch() -> bool:
    """Apply when MSGEN_LRGROUPS is set. Returns whether it engaged."""
    global _APPLIED
    if _APPLIED or os.environ.get("MSGEN_LRGROUPS", "0") in ("0", "", "false", "False"):
        return _APPLIED
    import utils.misc as misc

    if not hasattr(misc, "create_optimizer"):
        print("[patch_lrgroups] utils.misc has no create_optimizer; not applied")
        return False
    orig = misc.create_optimizer

    def create_optimizer(model, cfg, *a, **kw):
        import torch.optim as optim
        dec, bb = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (bb if any(b in name for b in BACKBONES) else dec).append(p)
        groups = [{"params": dec, "lr": cfg.train.lr_decoder,
                   "weight_decay": cfg.train.weight_decay}]
        if bb:
            groups.append({"params": bb, "lr": cfg.train.lr_backbone,
                           "weight_decay": cfg.train.weight_decay})
        print(f"[patch_lrgroups] decoder group {len(dec)} tensors @ {cfg.train.lr_decoder}; "
              f"backbone group {len(bb)} tensors @ "
              f"{cfg.train.lr_backbone if bb else 'n/a (all frozen)'}")
        return optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

    misc.create_optimizer = create_optimizer
    _APPLIED = True
    print("[patch_lrgroups] active: backbone group matched on "
          + ", ".join(BACKBONES))
    return True
