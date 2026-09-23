"""$PG tests/test_depth_patches.py -- the two depth arms' patches, on the real model/dataset (GPU ~8 GB).
Run twice via the wrapper below: once per flag, because each patch is process-global.
Requires a planner checkpoint and a dataset that are not part of the release; adapt the two paths in
build_trainer(...)."""
import os, subprocess, sys
PG = __import__("os").environ.get("PG", "python")
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
CHILD = r'''
import os, sys, torch; sys.argv = ["x"]
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from msgen.predict import build_trainer
trainer, model = build_trainer("data/ds/pickcube_wallsel200", "runs/mix4gvwallsel_n800/ckpt/20260825_093005/final_model.pth", 4, 0)
ts = []
for i, b in enumerate(trainer.val_loader):
    ts.append(b["trajectory"])
    if i >= 7: break
t = torch.cat(ts); mv = t[:, :, 1:, :2].abs().sum(dim=(2, 3)) > 20.0 / 384      # moving queries only (>20 px total travel), 8 batches
z_abs = float(t[:, :, 0, 2].abs().mean()); z_del = float(t[:, :, 1:, 2][mv].abs().mean()); xy_del = float(t[:, :, 1:, :2][mv].abs().mean())
tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
dep = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "depth_encoder" in n); oth = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "depth_encoder" not in n)
print(f"RESULT z_abs={z_abs:.5f} z_del={z_del:.6f} xy_del={xy_del:.6f} trainable={tr} depth_trainable={dep} other_trainable={oth}")
'''
def run(env):
    e = dict(os.environ, PYTHONPATH=ROOT, **env)
    out = subprocess.run([PG, "-c", CHILD], env=e, cwd=ROOT, capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT")][-1]
    return dict(kv.split("=") for kv in line.split()[1:])
OK, BAD = [], []
def check(name, cond, detail=""):
    (OK if cond else BAD).append(name); print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
base = run({})
sc = run({"MSGEN_DEPTH_SCALE": "4"})
un = run({"MSGEN_DEPTH_UNFREEZE": "2", "MSGEN_LRGROUPS": "1"})
r = float(sc["z_del"]) / float(base["z_del"])
check("depthscale x4 multiplies depth deltas by 4", abs(r - 4) < 0.01, f"ratio {r:.3f}")
check("depthscale leaves xy deltas unchanged", abs(float(sc["xy_del"]) - float(base["xy_del"])) < 1e-9)
check("stock: moving-point depth deltas 2-8x smaller than xy deltas", 2 < float(base["xy_del"]) / float(base["z_del"]) < 8,
      f"xy/z = {float(base['xy_del'])/float(base['z_del']):.2f}")
check("depthtower: trainable params increase, only in depth_encoder",
      int(un["trainable"]) > int(base["trainable"]) and int(un["depth_trainable"]) > 1e6 and int(base["depth_trainable"]) < 5e3,   # stock: 6-param stem + final layernorm (1536)
      f"{int(base['trainable'])/1e6:.1f}M -> {int(un['trainable'])/1e6:.1f}M (depth {int(un['depth_trainable'])/1e6:.1f}M)")
check("depthtower: stock trainable count unchanged without flag", int(sc["trainable"]) == int(base["trainable"]))
check("depthtower: non-depth trainable count unchanged", int(un["other_trainable"]) == int(base["other_trainable"]),
      f"{un['other_trainable']} vs {base['other_trainable']}")
print(f"\n{len(OK)} pass, {len(BAD)} fail"); sys.exit(1 if BAD else 0)
