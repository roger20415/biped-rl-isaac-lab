from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import root_quat_w
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
        # quat: (num_envs, 4) format [x, y, z, w]
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (X-axis)
        sinr = 2 * (w * x + y * z)
        cosr = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr, cosr)

        # Pitch (Y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        # Yaw (Z-axis)
        siny = 2 * (w * z + x * y)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny, cosy)

        return torch.stack((roll, pitch, yaw), dim=1)

def get_euler_W_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = root_quat_w(env, asset_cfg)
    euler = quat_to_euler_xyz(quat)
    return euler[:, :2]  # roll, pitch

def has_foot_contact(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0014) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    z = asset.data.body_pos_w[:, body_ids, 2]  
    contact = (z < threshold).float()
    return contact
