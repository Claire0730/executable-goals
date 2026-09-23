"""Dex4D's teacher network, ported strictly (paper Fig.2(a), Fig.3(c), Table VII).

WHY A SECOND PPE FILE

`msppo/ppe.py`'s `PairedKeypointActorCritic` is already a PPE port and it is
already the project's best self-trained peg teacher: 0.750 +/- 0.175 over three
seeds against the flat head's 0.312 +/- 0.222, and 0.322 vs 0.057 at the tight
0.003 clearance.

  NOTE. An earlier `ppe.py` docstring said the opposite ("our paired-encoder run
  read 0.000 at 10.7M"). That comparison was against a `--reward pose_reach` run, and
  `peg_kp_ppo.py:44-47` records that THAT REWARD measures 0.000 at full budget.
  Reward and head were confounded. With `--reward stock` held fixed, paired wins.

So this file is not "try PPE"; it is "make the existing PPE match Dex4D exactly".
Four differences remain against the paper, and this class closes all four:

  | | ppe.PairedKeypointActorCritic | Dex4D Table VII | here |
  |---|---|---|---|
  | pooling      | mean only               | mean-max mixed        | mixed |
  | branch input | one flat concat         | per-branch MLP tokens | per-branch |
  | actor/critic | (512, 512, 256)         | [1024,1024,512,512]   | Dex4D |
  | points       | 64                      | 128                   | 128 |

WHAT IS DELIBERATELY *NOT* COPIED, and why

  * the reward stays the task's official one (`--reward stock`). The whole point
    of the comparison is one variable
  * PPO hyperparameters stay at `peg_kp_ppo.py`'s values (gamma 0.96, lambda 0.9,
    8 epochs, 32 minibatches, target_kl 0, 256 envs, 30M steps). Dex4D's 4096
    envs and KL 0.016 are exposed as flags but default OFF, because changing them
    would make the comparison multi-variable
  * domain randomisation (Table VI) and the three-stage curriculum. Both exist for
    sim-to-real transfer over 3,200 objects; peg is one object in sim. They belong
    on a separate line, not inside this comparison

SLICING. Every branch slice is PASSED IN, never derived from a hardcoded offset --
`ppe.py:8-10` records that a hardcoded `kp_start = 197` upstream silently
mis-slices the moment the state layout changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(dims, out_act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
        if out_act or i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MixedPoolPointNet(nn.Module):
    """Shared per-point MLP, then MEAN-MAX MIXED pooling (Fig.3(c)).

    The pooling is the one thing `ppe.SimplePointNetBackbone` gets wrong: it
    mean-pools only. Mean alone is a linear functional of the point set, so it
    cannot express "the furthest point along this direction" -- exactly the kind
    of extremal statistic an insertion depth or a tip position is made of.

    "Mixed" is read here as CONCATENATION of the two pooled vectors, which keeps
    both statistics at full width. `student.PointNetToken` uses the other reading
    (split the channels, max half and mean half); that one is left alone so the
    student's encoder stays byte-identical to existing checkpoints.
    """

    def __init__(self, pc_dim=6, hidden=(128, 128), feature_dim=128):
        super().__init__()
        self.mlp = _mlp([pc_dim, *hidden])
        self.head = layer_init(nn.Linear(2 * hidden[-1], feature_dim))

    def forward(self, pc):                      # [B,N,pc_dim] -> [B,feature_dim]
        f = self.mlp(pc)
        return self.head(torch.cat([f.mean(dim=1), f.amax(dim=1)], dim=-1))


class StrictPairedActorCritic(nn.Module):
    """Dex4D Fig.2(a): per-branch MLP tokenizers + paired-point PointNet, all
    concatenated into an MLP actor and an MLP critic.

    `encode` / `actor_mean` / `get_value` / `get_action_and_value` keep the same
    contract as `ppe.PairedKeypointActorCritic`, so `msppo/peg_eval.py` and
    `msppo/ppo.py` need no changes.

    branches: {name: slice} over the observation vector. Anything not covered by
    a branch and not inside `kp_slice` is an error, not silently dropped -- the
    constructor asserts full coverage.
    """

    def __init__(self, obs_dim, act_dim, kp_slice, num_kp, branches,
                 token_dim=128, pn_hidden=(128, 128), kp_feature_dim=128,
                 hidden=(1024, 1024, 512, 512), init_logstd=-0.5):
        super().__init__()
        self.kp_slice, self.num_kp = kp_slice, num_kp
        self.branches = dict(branches)

        covered = set()
        for name, sl in self.branches.items():
            covered |= set(range(sl.start, sl.stop))
        covered |= set(range(kp_slice.start, kp_slice.stop))
        missing = sorted(set(range(obs_dim)) - covered)
        if missing:
            raise ValueError(
                f"branches + kp_slice leave {len(missing)} observation dims "
                f"uncovered: {missing[:12]}{'...' if len(missing) > 12 else ''}. "
                "Every dim must be routed to a tokenizer; silently dropping one "
                "is how a state-layout change becomes an unexplained score.")

        self.tok = nn.ModuleDict({
            name: _mlp([sl.stop - sl.start, token_dim, token_dim])
            for name, sl in self.branches.items()})
        self.kp_backbone = MixedPoolPointNet(6, pn_hidden, kp_feature_dim)
        feat_dim = token_dim * len(self.branches) + kp_feature_dim
        self.feat_dim = feat_dim

        def trunk(out, out_std):
            layers, d = [], feat_dim
            for h in hidden:
                layers += [layer_init(nn.Linear(d, h)), nn.ELU()]
                d = h
            layers += [layer_init(nn.Linear(d, out), std=out_std)]
            return nn.Sequential(*layers)

        # ELU, not Tanh: Dex4D's actor/critic are 4 layers deep against ppe.py's
        # 3, and Tanh at that depth saturates. ManiSkill's own reference uses
        # Tanh at 3x256, which is the shallow regime where it is fine.
        self.actor_mean = trunk(act_dim, 0.01)
        self.critic = trunk(1, 1.0)
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * init_logstd)

    def encode(self, obs):
        s, k = self.kp_slice, self.num_kp
        kp = obs[:, s]
        paired = torch.cat([kp[:, : k * 3].reshape(-1, k, 3),
                            kp[:, k * 3: k * 6].reshape(-1, k, 3)], dim=-1)
        feats = [self.tok[n](obs[:, sl]) for n, sl in self.branches.items()]
        feats.append(self.kp_backbone(paired))
        return torch.cat(feats, dim=-1)

    def get_value(self, obs):
        return self.critic(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        z = self.encode(obs)
        mean = self.actor_mean(z)
        dist = torch.distributions.Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(1), dist.entropy().sum(1),
                self.critic(z).squeeze(-1))
