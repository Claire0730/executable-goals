"""Dex4D's Paired Point Encoding, ported to ManiSkill observations.

Faithful to `algorithms/rl/ppo/module.py:16-39, 293-313, 370-377`:
the object and goal keypoints are stacked into a 6-channel point cloud
[object_xyz, goal_xyz], pushed through a per-point MLP, and MEAN POOLED into one
feature that replaces the raw K*6 numbers in the observation vector.

One deliberate difference: upstream hardcodes `kp_start = 197`. Here the slice is
passed in from the env, because a hardcoded offset silently mis-slices the
observation the moment anything about the state layout changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SimplePointNetBackbone(nn.Module):
    """Per-point MLP + mean pooling. Permutation invariant, no neighbourhood ops."""

    def __init__(self, pc_dim: int = 6, feature_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pc_dim, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:   # [B,N,6] -> [B,F]
        return self.mlp(pc).mean(dim=1)


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class PairedKeypointActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, kp_slice, num_kp,
                 kp_feature_dim=128, hidden=(512, 512, 256), init_logstd=-0.5):
        super().__init__()
        self.kp_slice = kp_slice
        self.num_kp = num_kp
        self.kp_backbone = SimplePointNetBackbone(6, kp_feature_dim)
        # raw keypoints leave the vector, one pooled feature takes their place
        feat_dim = obs_dim - num_kp * 6 + kp_feature_dim

        def mlp(out, out_std):
            layers, d = [], feat_dim
            for h in hidden:
                layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
                d = h
            layers += [layer_init(nn.Linear(d, out), std=out_std)]
            return nn.Sequential(*layers)

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * init_logstd)

    def encode(self, obs):
        s = self.kp_slice
        k = self.num_kp
        kp = obs[:, s]
        object_kp = kp[:, : k * 3].reshape(-1, k, 3)
        goal_kp = kp[:, k * 3: k * 6].reshape(-1, k, 3)
        paired = torch.cat([object_kp, goal_kp], dim=-1)        # [B,K,6]
        feat = self.kp_backbone(paired)                          # [B,F]
        return torch.cat([obs[:, : s.start], feat, obs[:, s.stop:]], dim=1)

    def get_value(self, obs):
        return self.critic(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        z = self.encode(obs)
        mean = self.actor_mean(z)
        logstd = self.actor_logstd.expand_as(mean)
        dist = torch.distributions.Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(1), dist.entropy().sum(1),
                self.critic(z).squeeze(-1))


class PlainActorCritic(PairedKeypointActorCritic):
    """Control arm S: identical trunk, keypoints dropped entirely."""

    def __init__(self, obs_dim, act_dim, kp_slice, num_kp, **kw):
        super().__init__(obs_dim, act_dim, kp_slice, num_kp, **kw)
        feat_dim = obs_dim - num_kp * 6
        hidden = kw.get("hidden", (512, 512, 256))

        def mlp(out, out_std):
            layers, d = [], feat_dim
            for h in hidden:
                layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
                d = h
            layers += [layer_init(nn.Linear(d, out), std=out_std)]
            return nn.Sequential(*layers)

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01)
        self.kp_backbone = nn.Identity()

    def encode(self, obs):
        return torch.cat([obs[:, : self.kp_slice.start], obs[:, self.kp_slice.stop:]], dim=1)

# Flat MLP Method
class FlatActorCritic(PairedKeypointActorCritic):
    """The reference architecture: a plain MLP over the WHOLE flat observation.

    ManiSkill `examples/baselines/ppo/ppo.py` (`Agent` class). 256-256-256
    Tanh, orthogonal init, final actor layer scaled 0.01*sqrt(2), and
    actor_logstd initialised to -0.5. The 384 keypoint dimensions go straight in
    as numbers; nothing pools or drops them.

    Neither existing head does this. `PairedKeypointActorCritic` POOLS the points
    through a PointNet, and `PlainActorCritic` DROPS the keypoint block entirely
    (a 447-D observation reaches its trunk as 63-D). See the note in
    ppe_strict.py.
    """

    def __init__(self, obs_dim, act_dim, kp_slice, num_kp, **kw):
        super().__init__(obs_dim, act_dim, kp_slice, num_kp, **kw)
        hidden = kw.get("hidden", (256, 256, 256))

        def mlp(out, out_std):
            layers, d = [], obs_dim
            for h in hidden:
                layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
                d = h
            layers += [layer_init(nn.Linear(d, out), std=out_std)]
            return nn.Sequential(*layers)

        self.critic = mlp(1, 1.0)
        self.actor_mean = mlp(act_dim, 0.01 * (2 ** 0.5))
        self.kp_backbone = nn.Identity()
        # reference: torch.ones(1, act_dim) * -0.5
        with torch.no_grad():
            self.actor_logstd.fill_(-0.5)

    def encode(self, obs):
        return obs


class PrivCritic(nn.Module):
    """ASYMMETRIC ACTOR-CRITIC. The observation carries a block of
    TRUE object / goal positions (`priv_slice`, written by the env from the
    simulator state before any noise). The CRITIC reads it; the ACTOR sees it
    zeroed. So the policy still has to act on the noisy channels and the sig
    channel -- nothing about deployment changes -- but the value function no
    longer has to average over the noise distribution, which is where the
    return variance (and the KL blow-ups at 256 envs) came from. Pinto et al.
    2017; the same trick Dex4D's own teacher relies on, just at the critic.

    Wraps either head unchanged. `encode`/`actor_mean` are exposed with the
    mask applied, so `multi_distill.py` (`tc.actor_mean(tc.encode(obs))`)
    labels from exactly what the actor was trained on; a student can never
    inherit the privileged block through distillation.
    """

    def __init__(self, base, priv_slice):
        super().__init__()
        self.base, self.priv = base, priv_slice

    def _mask(self, obs):
        o = obs.clone(); o[:, self.priv] = 0.0
        return o

    def encode(self, obs):
        return self.base.encode(self._mask(obs))

    @property
    def actor_mean(self):
        return self.base.actor_mean

    def get_value(self, obs):
        return self.base.critic(self.base.encode(obs))

    def get_action_and_value(self, obs, action=None):
        za = self.base.encode(self._mask(obs))
        mean = self.base.actor_mean(za)
        logstd = self.base.actor_logstd.expand_as(mean)
        dist = torch.distributions.Normal(mean, logstd.exp())
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(1), dist.entropy().sum(1),
                self.base.critic(self.base.encode(obs)).squeeze(-1))
