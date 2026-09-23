"""Feed the TEACHER the same degraded view the student has, during distillation.

WHY. `multi_distill.py` labels with `e.kp.teacher_observation()`, which is built
from simulator truth. That is correct for a teacher trained on truth, and WRONG
for the v2 noise-trained teachers: their action then depends on information the
student's observation does not contain, so two different true states that
perception maps onto the SAME student observation get DIFFERENT labels. Least
squares answers that with the mean -- and the search behaviour we spent the
teacher runs installing is exactly the component that averages away.

HOW -- AND WHY NOT THE OBVIOUS WAY. The first version wrote the student's
perceived poses straight into the teacher's pose slots. That FAILED completely.
The reason is stated in `student_observation_perc`'s own docstring -- the
perceived pose lives in the PCA frame of the t=0 point set, not the CAD frame
the teacher was trained on, so the numbers are a different convention for the
same geometry.

What DOES transfer is the RELATIVE vector. Measurement showed the
frame offset cancels in goal-minus-object: exactly 0.0 mm at t=0 for PickCube and
StackCube. So the object stays at truth, in the teacher's own frame, and the GOAL
is shifted by the perception's relative error:

    goal_teacher = goal_true + [(goal_perc - obj_perc) - (goal_true - obj_true)]

The teacher then faces the same object-to-goal error the student faces, expressed
in the frame it understands. The derived difference fields are repaired so the
true goal is not recoverable, and LiftPegUpright -- whose goal is not in the
stock vector at all -- goes through `set_goal_offset`.

The teacher keeps its PRIVILEGE where it belongs: its PPO reward came from true
state, so its behaviour is correct with respect to the true world. Only the
observation it must express that behaviour through is now the student's.
"""
from __future__ import annotations

import torch

from msppo.obs_noise2 import DERIVED, TCP_P
from msppo.obs_noise import _qmul

NOISE_SL = {
    "peginsert": dict(obj_p=slice(25, 28), obj_q=slice(28, 32),
                      goal_p=slice(35, 38), goal_q=slice(38, 42)),
    "stack":     dict(obj_p=slice(25, 28), obj_q=slice(28, 32),
                      goal_p=slice(32, 35), goal_q=slice(35, 39)),
    "pickcube":  dict(obj_p=slice(29, 32), obj_q=slice(32, 36),
                      goal_p=slice(26, 29)),
    "liftpeg":   dict(obj_p=slice(25, 28), obj_q=slice(28, 32)),
}


def _qconj(q):
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def teacher_obs_from_perception(env, task, student_obs):
    """[B, teacher_obs_dim] with the goal carrying the student's RELATIVE error."""
    kp = env.kp
    sl0 = 2 * env.qdim + 7 + env.act_dim
    op_p = student_obs[:, sl0:sl0 + 3]
    gp_p, gq_p = student_obs[:, sl0 + 7:sl0 + 10], student_obs[:, sl0 + 10:sl0 + 14]

    raw = kp.raw_from_sim()
    sl = NOISE_SL[task]
    clean_goal_fn = getattr(kp, "_goal_from_obs_clean", kp._goal_from_obs)
    gp_true, gq_true = clean_goal_fn(raw)
    op_true = raw[:, sl["obj_p"]]

    # the relative error, frame-cancelling; the object itself stays at truth
    dp = (gp_p - op_p) - (gp_true - op_true)
    dq = _qmul(gq_p, _qconj(gq_true))

    o = raw.clone()
    gsl = sl.get("goal_p")
    if gsl is not None:
        o[:, gsl] = raw[:, gsl] + dp
        if "goal_q" in sl:
            o[:, sl["goal_q"]] = _qmul(dq, raw[:, sl["goal_q"]])
    pt = {"tcp_p": raw[:, TCP_P[task]], "obj_p": o[:, sl["obj_p"]]}
    if gsl is not None:
        pt["goal_p"] = o[:, gsl]
    for dst, a_, b_ in DERIVED[task]:
        if a_ in pt and b_ in pt:
            o[:, dst] = pt[a_] - pt[b_]

    if getattr(kp, "sig_obs", False):
        kp._sig_override = env.perc_sig_row()

    # `set_goal_offset` MUTATES the committed goal attributes; for LiftPegUpright
    # the goal lives ENTIRELY there. Apply, read, restore.
    setter = getattr(kp, "set_goal_offset", None)
    names = [a for a in ("_goal_xy", "_goal_q") if hasattr(kp, a)] if setter else []
    saved = [getattr(kp, a).clone() for a in names]
    if setter is not None:
        setter(dp, dq)
    out = kp.teacher_observation(o)
    kp._sig_override = None
    for a, v in zip(names, saved):
        setattr(kp, a, v)
    return out
