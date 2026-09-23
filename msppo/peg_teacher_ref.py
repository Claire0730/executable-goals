"""Use the external reference peg teacher as a frozen artifact.

Row 2 for peg is already answered: `integration/runs/peg_stock_anypose/final_ckpt.pt`
scores **0.98 native insertion** (N=100, clearance 0.01), verified independently
earlier. Retraining it here has only ablation value, so rows 3 and 4 --
the peg student and the peg student driven by TraceGen -- distil from that
checkpoint instead of waiting for one of ours.

This keeps the standing rule that `integration/` code is never merged into
`msppo/`. Two things cross the boundary and both are DATA, not code:

  * `final_ckpt.pt`, loaded into `msppo.ppe.FlatActorCritic`, whose keys match it
    exactly (actor_logstd + critic.{0,2,4,6} + actor_mean.{0,2,4,6}, 436 -> 256
    -> 256 -> 256 -> 8).
  * `assets/peg_ref_kp64.npy`, their canonical keypoint set, exported once. This
    one is NOT optional: their kp64 is a ONE-SIDE OCCLUSION-MASKED subset of 128
    box points (centroid (-21.6, 0.9, -7.5) mm, only 23 of 64 points with x > 0),
    while `peg_kp_env.unit_box_keypoints` is a symmetric sample. Feeding the
    teacher a symmetric point set is feeding it out-of-distribution input.
    Re-deriving their subset would mean reimplementing `mask_oneside_indices` and
    its RNG consumption order, which is exactly the kind of silent divergence the
    exported array avoids.

The observation is rebuilt here rather than imported, in their field order
(`franka_ap2ap.py:305-315`), and `check()` verifies it elementwise against their
own env before anything depends on it.

    python -m msppo.peg_teacher_ref            # run the gate
"""
from __future__ import annotations

import os

import numpy as np
import torch

from msppo.peg_kp_env import to_world

REF_CKPT = os.environ.get("PEG_REF_CKPT", "")
REF_KP64 = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "assets", "peg_ref_kp64.npy")
REF_OBS_DIM = 436
REF_ACT_DIM = 8


def ref_kp64(device="cuda") -> torch.Tensor:
    """[64,3] canonical keypoints in the peg's own frame, as the teacher saw them."""
    return torch.from_numpy(np.load(REF_KP64)).to(device)


def ref_obs(base, kp64: torch.Tensor) -> torch.Tensor:
    """[B,436] the observation vector the reference teacher was trained on.

    Field order is theirs: proprioception (qpos, qvel) then the extra dict in its
    literal insertion order -- is_grasped, tcp_pose, tcp_to_obj, obj_pose,
    obj_vel, goal_pose, obj_to_goal, obj_kp, goal_kp.

    Reads the live sim, so the caller must not have auto-reset since the action:
    `goal_pose` and `peg.pose` are live properties and would describe the NEXT
    episode for any env that just finished.
    """
    b = base
    n = b.num_envs
    canon = kp64[None].expand(n, -1, -1)
    op, oq = b.peg.pose.p, b.peg.pose.q
    gp = b.goal_pose
    tcp = b.agent.tcp_pose
    return torch.cat([
        b.agent.robot.get_qpos(), b.agent.robot.get_qvel(),          # 9 + 9
        b.agent.is_grasping(b.peg).float()[:, None],                  # 1
        tcp.raw_pose,                                                 # 7
        op - tcp.p,                                                   # 3
        b.peg.pose.raw_pose,                                          # 7
        b.peg.linear_velocity, b.peg.angular_velocity,                # 6
        gp.p, gp.q,                                                   # 7
        gp.p - op,                                                    # 3
        to_world(canon, op, oq).reshape(n, -1),                       # 192
        to_world(canon, gp.p, gp.q).reshape(n, -1),                   # 192
    ], dim=-1)


def load_ref_teacher(path: str = REF_CKPT, device="cuda"):
    """The frozen teacher, in eval mode with grads off. Never trained here."""
    from msppo.ppe import FlatActorCritic
    agent = FlatActorCritic(REF_OBS_DIM, REF_ACT_DIM,
                            slice(REF_OBS_DIM, REF_OBS_DIM), 0).to(device)
    sd = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(sd, dict) or "critic.0.weight" not in sd:
        sd = sd["agent"] if isinstance(sd, dict) and "agent" in sd else sd
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in the reference ckpt: {unexpected}"
    assert not [k for k in missing if "kp_backbone" not in k], f"missing: {missing}"
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent


# ------------------------------------------------------------------- gate --
def check(n=32, seed=0):
    """Elementwise: does `ref_obs` reproduce their env's own observation?

    Run against THEIR env, where the ground truth is the vector their teacher was
    fed. A field in the wrong slot is invisible in aggregate metrics and would
    show up only as a mysteriously weak teacher, so this is checked before rows 3
    and 4 depend on it.
    """
    import sys
    sys.path.insert(0, os.environ.get("INTEGRATION_DIR", "third_party/integration"))
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import retrain.franka_stock_tasks  # noqa: F401

    env = gym.make("PegInsertionSideAP2AP-v0", num_envs=n, obs_mode="state",
                   sim_backend="physx_cuda", reward_mode="normalized_dense")
    theirs, _ = env.reset(seed=seed)
    base = env.unwrapped
    kp = ref_kp64()
    mine = ref_obs(base, kp)

    ok = []
    d = (mine - theirs).abs()
    ok.append(("obs dim", mine.shape[-1] == theirs.shape[-1] == REF_OBS_DIM,
               f"{mine.shape[-1]} vs {theirs.shape[-1]}"))
    ok.append(("obs elementwise", float(d.max()) < 1e-4,
               f"max |diff| {float(d.max()):.3e}"))
    # locate any mismatch by field rather than reporting one scalar
    fields = [("qpos", 0, 9), ("qvel", 9, 18), ("is_grasped", 18, 19),
              ("tcp_pose", 19, 26), ("tcp_to_obj", 26, 29), ("obj_pose", 29, 36),
              ("obj_vel", 36, 42), ("goal_pose", 42, 49), ("obj_to_goal", 49, 52),
              ("obj_kp", 52, 244), ("goal_kp", 244, 436)]
    worst = [(f, float(d[:, a:b].max())) for f, a, b in fields]
    bad = [f"{f}={v:.2e}" for f, v in worst if v >= 1e-4]
    ok.append(("every field in its slot", not bad, "all < 1e-4" if not bad
               else "MISMATCHED: " + ", ".join(bad)))

    # the teacher must also ACT the same on it
    agent = load_ref_teacher()
    with torch.no_grad():
        a_mine = agent.actor_mean(agent.encode(mine))
        a_theirs = agent.actor_mean(agent.encode(theirs))
    ok.append(("teacher action identical", float((a_mine - a_theirs).abs().max()) < 1e-4,
               f"max |diff| {float((a_mine - a_theirs).abs().max()):.3e}"))
    env.close()

    for name, good, detail in ok:
        print(f"[{'PASS' if good else 'FAIL'}] {name}   {detail}")
    return 0 if all(g for _, g, _ in ok) else 1


if __name__ == "__main__":
    raise SystemExit(check())
