# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def forward_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                     nowhere_penalty_weight: float = 0.2) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    current_base_x = asset.data.body_pos_w[:, body_id, 0]

    if "prev_base_x" not in env.extras:
        env.extras["prev_base_x"] = current_base_x.clone()

    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)

    if len(reset_env_ids) > 0:
        env.extras["prev_base_x"][reset_env_ids] = current_base_x[reset_env_ids]

    delta_x = current_base_x - env.extras["prev_base_x"]

    forward_reward = delta_x - nowhere_penalty_weight

    env.extras["prev_base_x"].copy_(current_base_x)

    return forward_reward