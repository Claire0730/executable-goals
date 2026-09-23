"""$PM tests/test_ksample.py -- msgen.ksample on synthetic preds files."""
import sys, tempfile
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def write(path, pred, eps):
    np.savez_compressed(path, pred=pred.astype(np.float32), gt=pred.astype(np.float32),
                        mask=np.ones(pred.shape, bool), episode_id=eps,
                        frame_id=np.array(["00000"] * len(eps)))


def main():
    from msgen.ksample import mean_preds
    rng = np.random.default_rng(0)
    eps = np.array(["clip_000", "clip_001"])
    base = rng.normal(size=(2, 400, 33, 3))
    with tempfile.TemporaryDirectory() as td:
        ps = []
        for k in range(4):
            p = f"{td}/s{k}.npz"; write(p, base + k, eps); ps.append(p)
        r = mean_preds(ps, f"{td}/mean.npz")
        z = np.load(f"{td}/mean.npz", allow_pickle=True)
        check("mean == base + 1.5", np.allclose(z["pred"], base + 1.5, atol=1e-5))
        check("K recorded", r["K"] == 4 and z["seed_list"].item() != "")
        check("episode order kept", list(z["episode_id"]) == list(eps))
        write(f"{td}/dup.npz", base, eps)
        try:
            mean_preds([f"{td}/dup.npz", f"{td}/dup.npz"], f"{td}/x.npz"); dup_ok = False
        except SystemExit:
            dup_ok = True
        check("identical samples refused", dup_ok)
        write(f"{td}/swap.npz", base, eps[::-1])
        try:
            mean_preds([ps[0], f"{td}/swap.npz"], f"{td}/y.npz"); mis_ok = False
        except SystemExit:
            mis_ok = True
        check("misaligned episode_id refused", mis_ok)
    print(f"\n{len(OK)} pass, {len(BAD)} fail"); sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
