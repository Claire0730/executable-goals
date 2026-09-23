"""$PM tests/test_relbank_k.py -- K-column relbank export/load/scale on a synthetic bank."""
import json, sys, tempfile
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def synth(td, K=3, N=6, seed=0):
    rng = np.random.default_rng(seed)
    obj = rng.normal(size=(N, 3)); goal = obj + rng.normal(size=(N, 3)) * 0.3
    q = np.tile([1.0, 0, 0, 0], (N, 1))
    cfg = dict(task="pickcube", num_envs=N, seed=997,
               configs=[dict(env_idx=i, clip=f"clip_{i:03d}", obj=list(obj[i]) + [1, 0, 0, 0],
                             goal=list(goal[i]) + [1, 0, 0, 0]) for i in range(N)])
    json.dump(cfg, open(f"{td}/configs.json", "w"))
    paths = []
    for k in range(K):
        gp = np.zeros((N, 33, 3)); gp[:, -1] = goal + rng.normal(size=(N, 3)) * 0.02
        gq = np.tile([1.0, 0, 0, 0], (N, 33, 1))
        solved = np.ones(N, bool); solved[k] = False           # each seed drops a different scene
        p = f"{td}/g{k}.npz"
        np.savez(p, goal_p=gp, goal_q=gq, solved=solved, true_goal=np.concatenate([goal, q], -1))
        paths.append(p)
    return paths


def main():
    from msppo.peg_relbank import export_k, load, scale_error
    with tempfile.TemporaryDirectory() as td:
        paths = synth(td)
        n = export_k(paths, td, f"{td}/rel.npz")
        z = np.load(f"{td}/rel.npz")
        check("N = intersection of solved (6 - 3 dropped)", n == 3 and z["frac"].shape == (3, 3), str(z["frac"].shape))
        check("scene_idx kept", list(z["scene_idx"]) == [3, 4, 5], str(z["scene_idx"]))
        d = load(f"{td}/rel.npz", device="cpu")
        check("load K", d["K"] == 3 and tuple(d["dq"].shape) == (3, 3, 4))
        idx = torch.tensor([0, 2]); a = torch.tensor([0.0, 1.0])
        f, lv, lw, dq = scale_error(d, idx, a)
        check("scale_error shapes [B,K]", tuple(f.shape) == (2, 3) and tuple(dq.shape) == (2, 3, 4), f"{tuple(f.shape)} {tuple(dq.shape)}")
        check("a=0 -> frac 1, lateral 0 across K", torch.allclose(f[0], torch.ones(3)) and float(lv[0].abs().max()) == 0)
        check("a=1 -> raw columns", torch.allclose(f[1], d["frac"][2]) and torch.allclose(lv[1], d["lat_v"][2]))
        np.savez(f"{td}/one.npz", frac=z["frac"][:, 0], lat_v=z["lat_v"][:, 0], lat_w=z["lat_w"][:, 0], dq=z["dq"][:, 0])
        d1 = load(f"{td}/one.npz", device="cpu")
        f1, _, _, dq1 = scale_error(d1, idx, a)
        check("legacy bank: K=1, shapes unchanged [B]", d1["K"] == 1 and tuple(f1.shape) == (2,) and tuple(dq1.shape) == (2, 4))
    print(f"\n{len(OK)} pass, {len(BAD)} fail"); sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
