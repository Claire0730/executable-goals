"""PPO for ManiSkill3 vectorized envs, CleanRL-style.

The pip package ships no `examples/baselines/ppo`, so this is written fresh
rather than vendored. For the hyperparameters see PPOConfig below.

Truncation is bootstrapped from `final_observation`. ManiSkill runs fixed-length
episodes, so almost every episode ends by truncation rather than termination --
treating those as terminal would tell the critic the world ends at step 100 and
systematically depress values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    # These follow ManiSkill's `examples/baselines/ppo/ppo.py` except epochs and
    # target_kl.
    total_steps: int = 25_000_000
    num_steps: int = 50            # one whole stock episode
    lr: float = 3e-4
    gamma: float = 0.8
    gae_lambda: float = 0.9
    clip: float = 0.2
    epochs: int = 8
    minibatches: int = 32
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # ManiSkill's reference defaults target_kl to 0.1; 0 here disables the stop.
    target_kl: float = 0.0
    norm_adv: bool = True
    anneal_lr: bool = False        # off in the reference; was a patch for the collapse
    clip_vloss: bool = False       # ditto


def train(env, agent, cfg: PPOConfig, device="cuda", log_every=10, logger=print,
          on_log=None,
          eval_fn=None, eval_every=50, save_best=None):
    n = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs_b = torch.zeros((cfg.num_steps, n, obs_dim), device=device)
    act_b = torch.zeros((cfg.num_steps, n, act_dim), device=device)
    logp_b = torch.zeros((cfg.num_steps, n), device=device)
    rew_b = torch.zeros((cfg.num_steps, n), device=device)
    done_b = torch.zeros((cfg.num_steps, n), device=device)
    val_b = torch.zeros((cfg.num_steps, n), device=device)
    # value of the true next state, kept separately so truncation can bootstrap
    nextval_b = torch.zeros((cfg.num_steps, n), device=device)

    # The reference steps `clip_action(action)`; an unclipped Gaussian sample
    # sends out-of-range targets to the controller and makes the executed action
    # differ from the one PPO computes log-probs for.
    act_low = torch.as_tensor(env.single_action_space.low, device=device)
    act_high = torch.as_tensor(env.single_action_space.high, device=device)

    obs, _ = env.reset(seed=0)
    obs = obs.to(device).float()
    done = torch.zeros(n, device=device)

    batch = n * cfg.num_steps
    iters = cfg.total_steps // batch
    succ_hist, grasp_hist, hist = [], [], []
    t0 = time.time()

    best = -1.0
    for it in range(1, iters + 1):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - (it - 1) / iters)
        for step in range(cfg.num_steps):
            obs_b[step], done_b[step] = obs, done
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(obs)
            act_b[step], logp_b[step], val_b[step] = a, lp, v

            obs, r, term, trunc, info = env.step(a.clamp(act_low, act_high))
            obs = obs.to(device).float()
            d = (term | trunc).float()
            rew_b[step] = r.to(device).float()

            with torch.no_grad():
                nv = agent.get_value(obs).squeeze(-1)
                # The env wrapper augments the terminal observation with the
                # keypoints that episode actually had; the raw `final_observation`
                # the vector env stores is the un-augmented base state.
                if "final_observation_kp" in info:
                    m = d.bool()
                    nv = nv.clone()
                    nv[m] = agent.get_value(info["final_observation_kp"].to(device).float()).squeeze(-1)
            # Bootstrap V(final_obs) at EVERY done, termination included -- this
            # is what ManiSkill's own ppo.py does (`real_next_values =
            # next_not_done * nextvalues + final_values[t]`, where final_values
            # is filled for the whole done_mask).
            #
            # The textbook rule "terminal => V = 0" is WRONG here and was the
            # single biggest defect in this file. StackCube sets
            # `terminated = success`, so zeroing the bootstrap tells the critic
            # that succeeding ends the reward stream. With gamma 0.8 and a
            # normalized_dense reward around 0.7/step, NOT succeeding is worth
            # ~0.7/(1-0.8) = 3.5, while succeeding is worth just the final
            # reward. The agent was being actively trained to avoid completing
            # the task -- a "release valley": place the cube, then refuse to
            # let go.
            nextval_b[step] = nv
            done = d

            if d.any():
                fi = info.get("final_info", info)
                sv = fi.get("success", info.get("success"))
                if sv is not None:
                    # Count EPISODES, not per-step batch means. StackCube
                    # terminates on success, so a step where 3 envs finish is a
                    # step where 3 envs succeeded -- mean() = 1.0. Averaging
                    # those step-means with the horizon step (where ~all 1024
                    # envs truncate) weights 3 successes as heavily as 1024
                    # ordinary endings and inflates the rate badly.
                    e = sv[d.bool()].float()
                    succ_hist.append((float(e.sum()), float(e.numel())))
            # StackCube success stays 0 for millions of steps, so grasp rate is
            # what distinguishes "learning slowly" from "stuck before transport".
            g = info.get("is_cubeA_grasped")
            if g is not None:
                grasp_hist.append(g.float().mean().item())

        # GAE using the stored next-state values (handles truncation correctly)
        adv = torch.zeros_like(rew_b)
        lastgae = 0
        for t in reversed(range(cfg.num_steps)):
            nonterm = 1.0 - done_b[t + 1] if t + 1 < cfg.num_steps else 1.0 - done
            delta = rew_b[t] + cfg.gamma * nextval_b[t] - val_b[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + val_b

        b_obs, b_act = obs_b.reshape(-1, obs_dim), act_b.reshape(-1, act_dim)
        b_logp, b_adv, b_ret = logp_b.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_val = val_b.reshape(-1)
        idx = np.arange(batch)
        mb = batch // cfg.minibatches
        kl = 0.0
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb):
                mi = idx[s:s + mb]
                _, newlp, ent, newv = agent.get_action_and_value(b_obs[mi], b_act[mi])
                ratio = (newlp - b_logp[mi]).exp()
                with torch.no_grad():
                    kl = ((ratio - 1) - (newlp - b_logp[mi])).mean().item()
                # ManiSkill's reference ppo.py breaks the MINIBATCH loop here,
                # BEFORE the optimiser step, so the offending update is not
                # applied at all, and then breaks the epoch loop. Dormant while
                # target_kl is 0.
                if cfg.target_kl and kl > cfg.target_kl:
                    break
                a_mb = b_adv[mi]
                if cfg.norm_adv:
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                if cfg.clip_vloss:
                    unclipped = (newv - b_ret[mi]) ** 2
                    vclip = b_val[mi] + (newv - b_val[mi]).clamp(-cfg.clip, cfg.clip)
                    vl = 0.5 * torch.max(unclipped, (vclip - b_ret[mi]) ** 2).mean()
                else:
                    vl = 0.5 * ((newv - b_ret[mi]) ** 2).mean()
                loss = pg - cfg.ent_coef * ent.mean() + cfg.vf_coef * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()
            # Reference: the threshold is target_kl, NOT 1.5x it.
            if cfg.target_kl and kl > cfg.target_kl:
                break

        if it % log_every == 0 or it == iters:
            win = succ_hist[-200:]
            tot = sum(c for _, c in win)
            sr = (sum(k for k, _ in win) / tot) if tot else float("nan")
            sps = int(it * batch / (time.time() - t0))
            gr = float(np.mean(grasp_hist[-500:])) if grasp_hist else float("nan")
            logger(f"it {it}/{iters} steps {it*batch:,} success {sr:.3f} grasp {gr:.3f} "
                   f"rew {rew_b.mean().item():.3f} kl {kl:.4f} {sps} sps")
            hist.append(dict(it=it, steps=it * batch, success=sr, grasp=gr,
                             rew=float(rew_b.mean()), kl=float(kl)))
            if on_log is not None:
                on_log(hist[-1])
            # Peak and final diverged badly, so the best policy is kept explicitly
            # rather than trusting whatever the last iteration happens to be.
            if sr == sr and sr > best and save_best:
                best = sr
                torch.save(agent.state_dict(), save_best)
                logger(f"   new best success {sr:.3f} -> {save_best}")
        if eval_fn is not None and (it % eval_every == 0 or it == iters):
            eval_fn(agent, it)
    return hist
