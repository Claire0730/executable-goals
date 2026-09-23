"""MSPPO_STACK_FRAME=1 -- StackCube teacher with the planner-derived PATH interface (psi) and the rewards that
make the teacher DEPEND on it (framework 2026-08-30, internal design note §12).

Observation: +4 dims [k1 approach unit vector (3), h carry height (1)], inserted right BEFORE the keypoint
block (kp_slice shifts by 4; every other block keeps its offset, so --init-from a padded checkpoint is
strict). Sampled once per episode in `_commit_goal` (frozen, like the goal):
    theta ~ U(0, THETA_MAX deg), phi ~ U(0, 2pi):  k1 = (sin th cos ph, sin th sin ph, -cos th)   (points DOWN onto cubeA)
    h ~ U(H_MIN, H_MAX)  metres above cubeA's start height
MSPPO_FRAME_FORCE="theta_deg=..,phi_deg=..,h=.." fixes them for every episode (dependence tests).

Rewards (TRUE simulator state, like every other shaping term here):
    push      -W_PUSH * tanh(|xy_A - xy_A0| / 0.02)                    while cubeA has never been grasped
              (the 37 pre-grasp sweeps: the gripper shoves cubeA into cubeB before grasping it)
    approach  +W_K1 * max(0, cos(v_tcp, k1))                           not grasped, |tcp - cubeA| < R_APP, tcp moving
    carry     -W_H * relu(| (z_A - z_A0) - h | - BAND) / h              grasped and 0.02 < |xy_A - xy_B| < R_NEAR
              (TWO-sided: a one-sided floor is satisfied by a constant carry and creates no dependence)
Everything else (stock reward, --w-kp, --w-bump) untouched. With the env var unset nothing is patched.
info[...]: frame_carry (max lift while grasped), frame_app_cos (mean cos in the approach window), frame_pushed
(max pre-grasp xy displacement), frame_h / frame_k1 (the commanded values) -- read by f2_dep_eval.py.
"""
from __future__ import annotations
import math, os
_APPLIED = False
THETA_MAX = float(os.environ.get("MSPPO_FRAME_THETA_MAX", "25"))
H_MIN, H_MAX = float(os.environ.get("MSPPO_FRAME_HMIN", "0.06")), float(os.environ.get("MSPPO_FRAME_HMAX", "0.13"))
W_PUSH = float(os.environ.get("MSPPO_FRAME_WPUSH", "0.5"))
W_K1 = float(os.environ.get("MSPPO_FRAME_WK1", "2.0"))   # one-time bonus at first grasp (was 0.2/step, farmable)
PUSH_DEAD = float(os.environ.get("MSPPO_FRAME_PUSH_DEAD", "0.005"))
W_H = float(os.environ.get("MSPPO_FRAME_WH", "1.0"))
BAND = float(os.environ.get("MSPPO_FRAME_BAND", "0.015"))
R_APP, R_NEAR = 0.08, 0.10
R_TRANSPORT = float(os.environ.get("MSPPO_FRAME_RTRANS", "0.03"))   # carry band applies while grasped and farther than this from cubeB (xy)


def _force():
    s = os.environ.get("MSPPO_FRAME_FORCE", "")
    if not s:
        return None
    d = {}
    for kv in s.split(","):
        k, v = kv.split("="); d[k.strip()] = float(v)
    return d


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get("MSPPO_STACK_FRAME", "") != "1":
        return False
    import torch, gymnasium as gym, numpy as np
    from msppo import stack_kp_env as SE
    KS = SE.KeypointStack
    orig_init, orig_commit, orig_step = KS.__init__, KS._commit_goal, KS.step
    force = _force()

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        n = self.num_envs; dev = "cuda"
        # WARM START: kp_teacher --init-from splices zero columns only for blocks it knows (ap2ap/last_action/sig/
        # contact/priv). Our 4-dim frame block sits exactly where a `sig` block would (right before the keypoints),
        # so during TRAINING we let the loader believe it is a sig block (MSPPO_FRAME_ASSIG=1); run.json still records
        # sig_obs=False (from the CLI), so evaluation never zeroes or re-creates it. `_augment_core` below uses the
        # real flag, never this alias.
        self._frame_real_sig = bool(self.sig_obs)
        if os.environ.get("MSPPO_FRAME_ASSIG", "") == "1":
            self.sig_obs = True
        s = self.kp_slice
        self.frame_slice = slice(s.start, s.start + 4)
        self.kp_slice = slice(s.start + 4, s.stop + 4)
        total = s.stop + 4
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n, total), dtype=np.float32)
        self._frame_k1 = torch.zeros(n, 3, device=dev); self._frame_k1[:, 2] = -1.0
        self._frame_h = torch.full((n,), 0.09, device=dev)
        self._f_zA0 = torch.zeros(n, device=dev); self._f_xyA0 = torch.zeros(n, 2, device=dev)
        self._f_tcp_prev = torch.zeros(n, 3, device=dev); self._f_grasped_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._f_carry = torch.zeros(n, device=dev); self._f_app_sum = torch.zeros(n, device=dev); self._f_app_n = torch.zeros(n, device=dev)
        self._f_pushed = torch.zeros(n, device=dev)
        # stats of the LAST COMPLETED episode per env (frozen at commit time, i.e. when the env is done and re-sampled):
        # the running trackers are reset inside orig_step's _commit_goal before an evaluator can read them.
        self._f_done_carry = torch.zeros(n, device=dev); self._f_done_app = torch.zeros(n, device=dev)
        self._f_done_pushed = torch.zeros(n, device=dev); self._f_done_h = torch.full((n,), 0.09, device=dev); self._f_done_k1 = torch.zeros(n, 3, device=dev)
        self._f_gen = torch.Generator(device=dev).manual_seed(int(os.environ.get("MSPPO_FRAME_SEED", "0")))

    def _sample(self, idx):
        n = int(idx.sum()) if idx.dtype == torch.bool else len(idx)
        dev = self._frame_h.device
        if force is not None:
            th = torch.full((n,), math.radians(force.get("theta_deg", 0.0)), device=dev)
            ph = torch.full((n,), math.radians(force.get("phi_deg", 0.0)), device=dev)
            h = torch.full((n,), force.get("h", 0.09), device=dev)
        else:
            th = torch.rand(n, generator=self._f_gen, device=dev) * math.radians(THETA_MAX)
            ph = torch.rand(n, generator=self._f_gen, device=dev) * 2 * math.pi
            h = H_MIN + torch.rand(n, generator=self._f_gen, device=dev) * (H_MAX - H_MIN)
        k1 = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), -torch.cos(th)], dim=-1)
        self._f_done_carry[idx] = self._f_carry[idx]; self._f_done_app[idx] = self._f_app_sum[idx] / self._f_app_n[idx].clamp_min(1)
        self._f_done_pushed[idx] = self._f_pushed[idx]; self._f_done_h[idx] = self._frame_h[idx]; self._f_done_k1[idx] = self._frame_k1[idx]
        self._frame_k1[idx] = k1; self._frame_h[idx] = h
        pa = self.base.cubeA.pose.p
        self._f_zA0[idx] = pa[idx, 2]; self._f_xyA0[idx] = pa[idx, :2]
        self._f_tcp_prev[idx] = self.base.agent.tcp.pose.p[idx]
        self._f_grasped_ever[idx] = False; self._f_carry[idx] = 0.0; self._f_app_sum[idx] = 0.0; self._f_app_n[idx] = 0.0; self._f_pushed[idx] = 0.0

    def _commit_goal(self, raw, idx=None):
        orig_commit(self, raw, idx)
        _sample(self, torch.ones(self.num_envs, dtype=torch.bool, device=self._frame_h.device) if idx is None else idx)

    def _augment_core(self, obs, clean=None):
        obj, goal, _ = self.keypoints(obs)
        b = obs.shape[0]
        parts = [obs]
        if self.ap2ap_fields: parts.append(self._extra_fields(obs))
        if self.last_action_obs: parts.append(self._last_action)
        if self._frame_real_sig: parts.append(self._sig_row(b))
        if self.contact_obs: parts.append(self._contact_row(b))
        if self.priv_obs: parts.append(self._priv_row(clean if clean is not None else obs))
        parts.append(torch.cat([self._frame_k1, self._frame_h[:, None]], dim=-1))
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, dim=-1), obj, goal

    def step(self, action):
        base = self.base
        tcp_prev = base.agent.tcp.pose.p.clone()
        obs, rew, term, trunc, info = orig_step(self, action)
        done = (term | trunc).bool()
        pa, pb, tcp = base.cubeA.pose.p, base.cubeB.pose.p, base.agent.tcp.pose.p
        gr = base.agent.is_grasping(base.cubeA).bool()
        live = ~done
        # --- pre-grasp push of cubeA
        never = ~self._f_grasped_ever
        disp = (pa[:, :2] - self._f_xyA0).norm(dim=-1)
        pen_push = W_PUSH * torch.tanh(torch.relu(disp - PUSH_DEAD) / 0.02) * (never & live).float()
        self._f_pushed = torch.where(never & live, torch.maximum(self._f_pushed, disp), self._f_pushed)
        # --- approach alignment with k1 (before grasp, near cubeA, moving)
        v = tcp - tcp_prev; sp = v.norm(dim=-1)
        near = (tcp - pa).norm(dim=-1) < R_APP
        win = never & near & (sp > 2e-3) & live
        cos = (v / sp.clamp_min(1e-9)[:, None] * self._frame_k1).sum(-1)
        first_grasp = gr & never & live
        r_app = W_K1 * ((self._f_app_sum + cos * win.float()) / (self._f_app_n + win.float()).clamp_min(1)).clamp(0.0, 1.0) * first_grasp.float()
        self._f_app_sum += cos * win.float(); self._f_app_n += win.float()
        # --- two-sided carry band during transport
        lift = pa[:, 2] - self._f_zA0
        xy = (pa[:, :2] - pb[:, :2]).norm(dim=-1)
        # iteration 2 (2026-08-30 14:20): the first arm enforced the band only inside 10 cm of cubeB, where the cube is
        # already descending, so carry stayed 76 mm at every h (slope 0.03). The band now covers the whole transport.
        transport = gr & (xy > R_TRANSPORT) & live
        pen_h = W_H * torch.relu((lift - self._frame_h).abs() - BAND) / self._frame_h * transport.float()
        self._f_carry = torch.where(gr & live, torch.maximum(self._f_carry, lift), self._f_carry)
        self._f_grasped_ever |= gr & live
        rew = rew + r_app - pen_push - pen_h
        info["frame_carry"] = self._f_carry; info["frame_app_cos"] = self._f_app_sum / self._f_app_n.clamp_min(1)
        info["frame_pushed"] = self._f_pushed; info["frame_h"] = self._frame_h; info["frame_k1"] = self._frame_k1
        info["frame_pen"] = pen_push + pen_h; info["frame_rapp"] = r_app
        info["frame_done_carry"] = self._f_done_carry; info["frame_done_app"] = self._f_done_app; info["frame_done_pushed"] = self._f_done_pushed
        info["frame_done_h"] = self._f_done_h; info["frame_done_k1"] = self._f_done_k1
        return obs, rew, term, trunc, info

    KS.__init__, KS._commit_goal, KS._augment_core, KS.step = __init__, _commit_goal, _augment_core, step
    _APPLIED = True
    print(f"[stack_frame] psi obs (+4 before kp) | push w={W_PUSH} approach w={W_K1} carry band w={W_H} band={BAND} rtrans={R_TRANSPORT} "
          f"| theta<={THETA_MAX} deg h in [{H_MIN},{H_MAX}] | force={force}", flush=True)
    return True
