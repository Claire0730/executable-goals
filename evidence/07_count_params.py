"""Read-only CPU parameter recount from checkpoint state_dicts (no model build, no GPU)."""
import sys, json, collections, torch
import os; sys.path.insert(0, os.environ.get("TRACEGEN_DIR", "third_party/TraceGen")); sys.path.insert(0, os.environ.get("EG_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CKPTS = {
    "mix4_realcam_n2400": os.environ.get("CKPT_DIR", "checkpoints") + "/planner/mix4_realcam_n2400.pth",
    "mix5_t2k_gmap": os.environ.get("CKPT_DIR", "checkpoints") + "/planner/mix5_t2k_gmap.pth",
    "mix5_t2k_n3000": os.environ.get("CKPT_DIR", "checkpoints") + "/planner/mix5_t2k_n3000.pth",
    "student_mt5_rcfz_gmpc_s0": "runs_rl/mt5_rcfz_gmpc_s0/student.pt",
    "student_best_mt5_rcfz_gmpc_s0": "runs_rl/mt5_rcfz_gmpc_s0/student_best.pt",
}


def find_sd(obj, path="root"):
    """Return (path, dict-of-tensors) for the largest tensor dict inside obj."""
    best = (None, None, -1)
    if isinstance(obj, dict):
        ten = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        n = sum(v.numel() for v in ten.values())
        if n > best[2]:
            best = (path, ten, n)
        for k, v in obj.items():
            if isinstance(v, dict):
                p, t, m = find_sd(v, f"{path}.{k}")
                if m > best[2]:
                    best = (p, t, m)
    elif hasattr(obj, "state_dict"):
        return find_sd(obj.state_dict(), path + ".state_dict()")
    return best


out = {}
for name, path in CKPTS.items():
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    top_keys = list(obj.keys()) if isinstance(obj, dict) else [type(obj).__name__]
    p, sd, n = find_sd(obj)
    by_prefix = collections.OrderedDict()
    by_prefix2 = collections.OrderedDict()
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "").replace("module.", "")
        parts = kk.split(".")
        by_prefix[parts[0]] = by_prefix.get(parts[0], 0) + v.numel()
        k2 = ".".join(parts[:2])
        by_prefix2[k2] = by_prefix2.get(k2, 0) + v.numel()
    buf_like = {k: v.numel() for k, v in sd.items() if any(s in k for s in ("running_mean", "running_var", "num_batches_tracked", "position_ids", "freqs", "rope", "pos_embed_buffer"))}
    out[name] = {
        "path": path,
        "top_level_keys": top_keys,
        "state_dict_path": p,
        "n_tensors": len(sd),
        "total_numel_state_dict": n,
        "by_top_prefix": by_prefix,
        "by_two_level_prefix": by_prefix2 if len(by_prefix2) < 200 else {k: v for k, v in by_prefix2.items() if v > 100000},
        "buffer_like_keys_numel": sum(buf_like.values()),
        "buffer_like_keys_sample": dict(list(buf_like.items())[:30]),
    }
    print(name, n, dict(by_prefix), flush=True)
    del obj, sd

json.dump(out, open(sys.argv[1], "w"), indent=1)
