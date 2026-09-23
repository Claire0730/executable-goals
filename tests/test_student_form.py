"""$PM tests/test_student_form.py -- goal_form slots/tokens on StudentTransformer (CPU)."""
import sys, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main():
    from msppo.student import StudentTransformer, goal_slices, goal_extra_slices
    base = dict(num_kp=64, act_dim=8, pose_obs=True, scene_kp=False, tcp=True, qdim=9, prev_action=True)
    s0 = StudentTransformer(**base)                         # mean == legacy
    s1 = StudentTransformer(goal_form="sd", **base)
    s2 = StudentTransformer(goal_form="k", goal_k=4, **base)
    # --no-scene pose layout: qpos 9 + qvel 9 + tcp 7 + action 8 + obj 7 + goal 7 + rel 9 = 56
    D = s0.obs_dim
    check("mean layout == legacy no-scene (56)", D == 56, str(D))
    check("sd adds 3", s1.obs_dim == D + 3 and s1.sl["goal_sd"] == slice(D, D + 3), str(s1.sl.get("goal_sd")))
    check("k adds 4x7", s2.obs_dim == D + 28 and s2.sl["goal_k"] == slice(D, D + 28), str(s2.sl.get("goal_k")))
    check("legacy prefix preserved", all(s0.sl[k] == s1.sl[k] == s2.sl[k] for k in s0.sl))
    B = 5
    check("sd token count 6", tuple(s1.tokens(torch.zeros(B, 59)).shape) == (B, 6, 128))
    check("k token count 5+4", tuple(s2.tokens(torch.zeros(B, 84)).shape) == (B, 9, 128))
    # K tokens are order-free: permuting the samples must not change the action
    x = torch.randn(B, 84); xp = x.clone(); ks = s2.sl["goal_k"]
    xp[:, ks] = x[:, ks].reshape(B, 4, 7)[:, [3, 0, 2, 1]].reshape(B, 28)
    s2.eval()
    with torch.no_grad():
        check("K tokens permutation-invariant", torch.allclose(s2.actor_mean(x), s2.actor_mean(xp), atol=1e-5))
        check("K tokens are READ (perturbing them moves the action)",
              not torch.allclose(s2.actor_mean(x), s2.actor_mean(x + torch.cat([torch.zeros(B, 56), torch.ones(B, 28)], 1))))
    g_sl, r_sl = goal_slices(s2)
    check("goal_slices unchanged 2-tuple; goal_extra_slices lists goal_k", goal_extra_slices(s2) == [slice(56, 84)] and g_sl == s2.sl["goal_pose"])
    check("goal_extra_slices sd", goal_extra_slices(s1) == [slice(56, 59)])
    check("legacy goal_extra_slices empty", goal_extra_slices(s0) == [])
    s3 = StudentTransformer(goal_form="ell", **base)
    check("ell adds 6", s3.obs_dim == 62 and s3.sl["goal_ell"] == slice(56, 62), str(s3.sl.get("goal_ell")))
    check("ell token count 6", tuple(s3.tokens(torch.zeros(B, 62)).shape) == (B, 6, 128))
    check("goal_extra_slices ell", goal_extra_slices(s3) == [slice(56, 62)])
    sd = s0.state_dict(); s0b = StudentTransformer(**base); s0b.load_state_dict(sd)
    check("legacy state_dict loads into mean-form student", True)
    try:
        StudentTransformer(goal_form="ndt", **base); bad_ok = False
    except ValueError:
        bad_ok = True
    check("unknown goal_form refused", bad_ok)
    print(f"\n{len(OK)} pass, {len(BAD)} fail"); sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
