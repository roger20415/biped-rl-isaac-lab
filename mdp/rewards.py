# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnv

from .config import Config
from .observations import get_phase


def com_error_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    sigma: float = 50000.0  # TODO
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]

    body_names = [
        "base_link", "back_1", "sacrum_1", "l_hip_1", "r_hip_1",
        "l_thigh_1", "r_thigh_1", "l_calf_1", "r_calf_1",
        "l_ankle_1", "r_ankle_1", "l_foot_1", "r_foot_1"
    ]
    body_ids, _ = asset.find_bodies(body_names)

    baselink_id, _ = asset.find_bodies("base_link")
    r_foot_id, _ = asset.find_bodies("r_foot_1")

    masses_list = [
        Config.BASELINK_MASS, Config.BACK_MASS, Config.SACRUM_MASS,
        Config.HIP_MASS, Config.HIP_MASS,
        Config.THIGH_MASS, Config.THIGH_MASS,
        Config.CALF_MASS, Config.CALF_MASS,
        Config.ANKLE_MASS, Config.ANKLE_MASS,
        Config.FOOT_MASS, Config.FOOT_MASS
    ]

    masses_tensor = torch.tensor(masses_list, device=asset.device, dtype=torch.float32).view(1, 13, 1)
    total_mass = torch.sum(masses_tensor)
    pos = asset.data.body_pos_w[:, body_ids, :]
    weighted_sum = torch.sum(pos * masses_tensor, dim=1)
    p_W_biped_com = weighted_sum / total_mass
    p_S_biped_com = p_W_biped_com.clone()
    p_S_biped_com[:, 2] = 0.0
    p_W_r_foot = asset.data.body_pos_w[:, r_foot_id[0], :]
    q_W_r_foot = asset.data.body_quat_w[:, r_foot_id[0], :]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=asset.device).repeat(env.num_envs, 1)
    xRFOOT_W_norm = math_utils.quat_apply(q_W_r_foot, x_axis)
    xRFOOT_W_norm = torch.nn.functional.normalize(xRFOOT_W_norm, dim=1)
    p_S_support = p_W_r_foot - Config.FOOT_LINK_X_SEMI_LENGTH * xRFOOT_W_norm
    p_S_support[:, 2] = 0.0
    vec_S_com_to_support = p_S_support - p_S_biped_com
    q_W_baselink = asset.data.body_quat_w[:, baselink_id[0], :]
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=asset.device).repeat(env.num_envs, 1)
    vec_W_yB = math_utils.quat_apply(q_W_baselink, y_axis)
    vec_S_yB = vec_W_yB.clone()
    vec_S_yB[:, 2] = 0.0
    vec_S_sacrum_proj_norm = torch.nn.functional.normalize(vec_S_yB, dim=1)
    err_signed = torch.sum(vec_S_com_to_support * vec_S_sacrum_proj_norm, dim=1)
    reward = torch.exp(-sigma * torch.square(err_signed))
    phase = get_phase(env).squeeze(-1)
    phase_mask = (phase > 0.0).float()

    return reward * phase_mask

def base_upright_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    sigma: float = 40.0
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    phase = get_phase(env).squeeze(-1)
    phase_mask = (phase <= 0.25).float()

    baselink_id, _ = asset.find_bodies("base_link")
    q_W_baselink = asset.data.body_quat_w[:, baselink_id[0], :]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=asset.device).repeat(env.num_envs, 1)
    base_up_W = math_utils.quat_apply(q_W_baselink, z_axis)
    upright_dot = base_up_W[:, 2]
    error = 1.0 - upright_dot
    reward = torch.exp(-sigma * error)

    return reward * phase_mask

def foot_lift_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    threshold: float = Config.FOOT_CONTACT_THRESHOLD
) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]

    body_ids, _ = asset.find_bodies(["l_foot_1", "r_foot_1"])
    foot_z_height = asset.data.body_pos_w[:, body_ids, 2]

    height_error = foot_z_height - threshold
    lift_violation = torch.clamp(height_error, min=0.0)
    total_violation = torch.sum(lift_violation, dim=1)
    phase = get_phase(env).squeeze(-1)
    phase_mask = (phase <= 0.25).float()
    penalty = total_violation
    
    return penalty * phase_mask

def action_rate_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    penalty = torch.sum(torch.square(action - prev_action), dim=1)

    return penalty

def action_l2_penalty(
    env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    
    action = env.action_manager.action
    penalty = torch.sum(torch.square(action), dim=1)
    return penalty