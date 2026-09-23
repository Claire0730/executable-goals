"""MSPPO_FRAME_TASK=pickcube|liftpeg|peginsert -- the SAME planner-derived path interface (psi) and dependence-making
rewards as msppo/patch_stack_frame.py, applied to the other three teacher envs (framework 2026-08-30 §12).

psi (4 dims, before the keypoint block): k1 approach unit vector (3) + h carry height (1), sampled once per episode
in `_commit_goal` (frozen). MSPPO_FRAME_FORCE="theta_deg=..,phi_deg=..,h=.." fixes them (dependence tests).
Rewards (true state):  push  -W_PUSH*tanh(|xy_obj - xy_obj0|/0.02) while never grasped
                       approach +W_K1*max(0,cos(v_tcp,k1)) not grasped, |tcp-obj|<R_APP, moving
                       carry  -W_H*relu(|lift-h|-BAND)/h while grasped and in the task's transport window
Per-task table (object actor, transport window, h prior, default W_H):
  pickcube  cube  : window = grasped & |xy_obj - xy_goal| > 0.03 ; h in [0.06,0.14] ; W_H default 0 (goal is a free point in
                    the air whose height the task fixes -- a carry band would fight it; k1 is the consumed component)
  liftpeg   peg   : window = grasped & peg tilt from upright > 20 deg ; h in [0.02,0.08] (lift a little, then rotate)
  peginsert peg   : window = grasped & |xy_peg - xy_hole| > 0.05 ; h in [0.04,0.10]
MSPPO_FRAME_ASSIG=1 during training aliases the block to the loader's `sig` slot so --init-from splices zero columns
(run.json keeps sig_obs unset). With the env var unset nothing is patched.
"""
from __future__ import annotations
import math, os
_APPLIED = set()      # tasks patched in this process (one class each; several tasks may coexist, e.g. multi-task distillation)
THETA_MAX = float(os.environ.get("MSPPO_FRAME_THETA_MAX", "25"))
W_PUSH = float(os.environ.get("MSPPO_FRAME_WPUSH", "0.5"))
W_K1 = float(os.environ.get("MSPPO_FRAME_WK1", "2.0"))   # one-time bonus at first grasp (was 0.2/step, farmable)
PUSH_DEAD = float(os.environ.get("MSPPO_FRAME_PUSH_DEAD", "0.005"))
BAND = float(os.environ.get("MSPPO_FRAME_BAND", "0.015"))
R_APP = 0.08
TASKS = {
    "pickcube":  dict(mod="msppo.pickcube_kp_env", cls="KeypointPickCube", obj="cube", hmin=0.06, hmax=0.14, wh=0.0, k1_base="0,0,-1"),
    "liftpeg":   dict(mod="msppo.liftpeg_kp_env",  cls="KeypointLiftPeg",  obj="peg",  hmin=0.06, hmax=0.14, wh=1.0, k1_base="0,0,-1"),   # lp_paired carries 101 mm
    "peginsert": dict(mod="msppo.peginsert_kp_env", cls="KeypointPegInsertShared", obj="peg", hmin=0.06, hmax=0.12, wh=1.0, k1_base="-0.28,-0.56,-0.78"),  # kp_teacher's peg env; cone axis = measured natural approach (pi_v9_frame4)
    # distillation-only entries (2026-08-31): their teachers never saw psi, so psi is an inert channel on these tasks;
    # the entries exist so frame_patches.apply() can install the psi sampler for the student's obs. `obj` = the ATTRIBUTE
    # name on the base env (both expose the object as `obj`).
    "pushcube":    dict(mod="msppo.pushcube_kp_env",    cls="KeypointPushCube",    obj="obj", hmin=0.01, hmax=0.05, wh=0.0, k1_base="0,0,-1"),
    "placesphere": dict(mod="msppo.placesphere_kp_env", cls="KeypointPlaceSphere", obj="obj", hmin=0.04, hmax=0.12, wh=0.0, k1_base="0,0,-1"),
}


def _force(task=""):
    """MSPPO_FRAME_FORCE_<TASK> (per task, multi-task processes) beats MSPPO_FRAME_FORCE (all tasks)."""
    s = os.environ.get(f"MSPPO_FRAME_FORCE_{task.upper()}", "") or os.environ.get("MSPPO_FRAME_FORCE", "")
    if not s: return None
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in s.split(","))}


def maybe_patch(task=None) -> bool:
    """Patch ONE task's keypoint env class. `task` defaults to MSPPO_FRAME_TASK (the single-task chains); a multi-task
    process (distillation / multi_eval) calls it once per task. Cone axis: MSPPO_FRAME_K1_BASE_<TASK>, else MSPPO_FRAME_K1_BASE,
    else the task default in TASKS."""
    T = task or os.environ.get("MSPPO_FRAME_TASK", "")
    if T in _APPLIED: return True
    if T not in TASKS: return False
    import importlib, torch, gymnasium as gym, numpy as np
    spec = TASKS[T]; W_H = float(os.environ.get("MSPPO_FRAME_WH", str(spec["wh"])))
    K1_BASE = [float(v) for v in os.environ.get(f"MSPPO_FRAME_K1_BASE_{T.upper()}", os.environ.get("MSPPO_FRAME_K1_BASE", spec["k1_base"])).split(",")]
    H_MIN = float(os.environ.get("MSPPO_FRAME_HMIN", str(spec["hmin"]))); H_MAX = float(os.environ.get("MSPPO_FRAME_HMAX", str(spec["hmax"])))
    M = importlib.import_module(spec["mod"]); KS = getattr(M, spec["cls"])
    HAS_COMMIT = hasattr(KS, "_commit_goal")           # peginsert (Shared) has no per-episode goal commit: sample at reset / done
    orig_init, orig_commit, orig_step, orig_aug, orig_reset = KS.__init__, getattr(KS, "_commit_goal", None), KS.step, KS._augment, KS.reset
    force = _force(T)

    def obj_actor(self): return getattr(self.base, spec["obj"])

    def goal_xy(self):
        if T == "pickcube": return self.base.goal_site.pose.p[:, :2]
        if T == "peginsert": return self.base.goal_pose.p[:, :2]
        return obj_actor(self).pose.p[:, :2]   # liftpeg: in place

    def upright_tilt_deg(self):   # liftpeg: angle between the peg's long axis and world z
        q = obj_actor(self).pose.q; w, x, y, z = q.unbind(-1)
        ax = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)   # R @ e_x (peg long axis)
        return torch.rad2deg(torch.acos(ax[:, 2].abs().clamp(-1, 1)))

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        n = self.num_envs; dev = "cuda"; s = self.kp_slice
        self._frame_real_sig = bool(getattr(self, "sig_obs", False))
        alias = os.environ.get("MSPPO_FRAME_ASSIG", "")
        if alias in ("1", "sig"): self.sig_obs = True          # loader alias only; _augment uses _frame_real_sig
        if alias == "contact": self.contact_obs = True         # for envs whose sig block is real (peg v2 teachers)
        self.frame_slice = slice(s.start, s.start + 4); self.kp_slice = slice(s.start + 4, s.stop + 4)
        total = s.stop + 4
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n, total), dtype=np.float32)
        self._frame_k1 = torch.zeros(n, 3, device=dev); self._frame_k1[:, 2] = -1.0; self._frame_h = torch.full((n,), (H_MIN + H_MAX) / 2, device=dev)
        for nm in ("_f_zA0", "_f_carry", "_f_app_sum", "_f_app_n", "_f_pushed", "_f_done_carry", "_f_done_app", "_f_done_pushed", "_f_done_h"):
            setattr(self, nm, torch.zeros(n, device=dev))
        self._f_xyA0 = torch.zeros(n, 2, device=dev); self._f_grasped_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._f_done_k1 = torch.zeros(n, 3, device=dev)
        self._f_gen = torch.Generator(device=dev).manual_seed(int(os.environ.get("MSPPO_FRAME_SEED", "0")))

    def _sample(self, idx):
        n = int(idx.sum()); dev = self._frame_h.device
        self._f_done_carry[idx] = self._f_carry[idx]; self._f_done_app[idx] = self._f_app_sum[idx] / self._f_app_n[idx].clamp_min(1)
        self._f_done_pushed[idx] = self._f_pushed[idx]; self._f_done_h[idx] = self._frame_h[idx]; self._f_done_k1[idx] = self._frame_k1[idx]
        if force is not None:
            th = torch.full((n,), math.radians(force.get("theta_deg", 0.0)), device=dev); ph = torch.full((n,), math.radians(force.get("phi_deg", 0.0)), device=dev)
            h = torch.full((n,), force.get("h", (H_MIN + H_MAX) / 2), device=dev)
        else:
            th = torch.rand(n, generator=self._f_gen, device=dev) * math.radians(THETA_MAX); ph = torch.rand(n, generator=self._f_gen, device=dev) * 2 * math.pi
            h = H_MIN + torch.rand(n, generator=self._f_gen, device=dev) * (H_MAX - H_MIN)
        local = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), -torch.cos(th)], -1)   # cone around -z
        b = torch.tensor(K1_BASE, device=dev, dtype=torch.float32); b = b / b.norm()
        z = torch.tensor([0.0, 0.0, -1.0], device=dev)
        # rotate the cone from -z onto the base axis (Rodrigues); identity when base == -z
        v = torch.cross(z, b, dim=0); c = torch.dot(z, b)
        if v.norm() < 1e-6:
            R = torch.eye(3, device=dev) if c > 0 else torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev))
        else:
            vx = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=dev)
            R = torch.eye(3, device=dev) + vx + vx @ vx * (1 / (1 + c))
        self._frame_k1[idx] = local @ R.T; self._frame_h[idx] = h
        pa = obj_actor(self).pose.p
        self._f_zA0[idx] = pa[idx, 2]; self._f_xyA0[idx] = pa[idx, :2]
        self._f_grasped_ever[idx] = False; self._f_carry[idx] = 0.0; self._f_app_sum[idx] = 0.0; self._f_app_n[idx] = 0.0; self._f_pushed[idx] = 0.0

    def _commit_goal(self, raw, idx=None):
        orig_commit(self, raw, idx)
        _sample(self, torch.ones(self.num_envs, dtype=torch.bool, device=self._frame_h.device) if idx is None else idx)

    def reset(self, *, seed=None, options=None):
        out = orig_reset(self, seed=seed, options=options)
        if not HAS_COMMIT:
            _sample(self, torch.ones(self.num_envs, dtype=torch.bool, device=self._frame_h.device))
            # re-lay the reset observation with the freshly sampled psi (orig_reset built it with the old one)
            obs, info = out
            if torch.is_tensor(obs): obs = self._augment(self.raw_from_sim())[0]
            out = (obs, info)
        return out

    def _augment(self, obs):
        clean = obs
        obs = self._noisy(obs)
        obj, goal, _ = self.keypoints(obs)
        b = obs.shape[0]; parts = [obs]
        if self.ap2ap_fields: parts.append(self._extra_fields(obs))
        if self.last_action_obs: parts.append(self._last_action)
        if self._frame_real_sig: parts.append(self._sig_row(b))
        if getattr(self, "priv_obs", False): parts.append(self._priv_row(clean))
        parts.append(torch.cat([self._frame_k1, self._frame_h[:, None]], -1))
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, -1), obj, goal

    def step(self, action):
        base = self.base; tcp_prev = base.agent.tcp.pose.p.clone()
        obs, rew, term, trunc, info = orig_step(self, action)
        done = (term | trunc).bool(); live = ~done
        A = obj_actor(self); pa = A.pose.p; tcp = base.agent.tcp.pose.p; gr = base.agent.is_grasping(A).bool()
        never = ~self._f_grasped_ever
        disp = (pa[:, :2] - self._f_xyA0).norm(dim=-1)
        pen_push = W_PUSH * torch.tanh(torch.relu(disp - PUSH_DEAD) / 0.02) * (never & live).float()
        self._f_pushed = torch.where(never & live, torch.maximum(self._f_pushed, disp), self._f_pushed)
        v = tcp - tcp_prev; sp = v.norm(dim=-1); near = (tcp - pa).norm(dim=-1) < R_APP
        win = never & near & (sp > 2e-3) & live
        cos = (v / sp.clamp_min(1e-9)[:, None] * self._frame_k1).sum(-1)
        first_grasp = gr & never & live
        r_app = W_K1 * ((self._f_app_sum + cos * win.float()) / (self._f_app_n + win.float()).clamp_min(1)).clamp(0.0, 1.0) * first_grasp.float(); self._f_app_sum += cos * win.float(); self._f_app_n += win.float()
        lift = pa[:, 2] - self._f_zA0
        if T == "liftpeg": window = gr & (upright_tilt_deg(self) > 20) & live
        else: window = gr & ((pa[:, :2] - goal_xy(self)).norm(dim=-1) > (0.05 if T == "peginsert" else 0.03)) & live
        pen_h = W_H * torch.relu((lift - self._frame_h).abs() - BAND) / self._frame_h * window.float()
        self._f_carry = torch.where(gr & live, torch.maximum(self._f_carry, lift), self._f_carry); self._f_grasped_ever |= gr & live
        rew = rew + r_app - pen_push - pen_h
        if not HAS_COMMIT and bool(done.any()):
            # no _commit_goal hook in this env: freeze stats + resample psi for the envs that just finished (their first
            # new-episode observation carries the previous psi for one step; the forced-psi dependence tests are unaffected)
            _sample(self, done)
        info.update(frame_carry=self._f_carry, frame_app_cos=self._f_app_sum / self._f_app_n.clamp_min(1), frame_pushed=self._f_pushed,
                    frame_h=self._frame_h, frame_k1=self._frame_k1, frame_pen=pen_push + pen_h, frame_rapp=r_app,
                    frame_done_carry=self._f_done_carry, frame_done_app=self._f_done_app, frame_done_pushed=self._f_done_pushed,
                    frame_done_h=self._f_done_h, frame_done_k1=self._f_done_k1)
        return obs, rew, term, trunc, info

    KS.__init__, KS._augment, KS.step, KS.reset = __init__, _augment, step, reset
    if HAS_COMMIT: KS._commit_goal = _commit_goal
    _APPLIED.add(T)
    print(f"[frame:{T}] psi obs (+4 before kp) | push w={W_PUSH} approach w={W_K1} carry band w={W_H} band={BAND} | theta<={THETA_MAX} h in [{H_MIN},{H_MAX}] k1_base={K1_BASE} | force={force}", flush=True)
    return True
