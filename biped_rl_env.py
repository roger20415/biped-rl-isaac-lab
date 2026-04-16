import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from .biped_rl_env_cfg import BipedRlEnvCfg


class BipedRlEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BipedRlEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.history_length = self.cfg.state_history_length
        self.state_dim = self.cfg.state_dim
        self.custom_state_history = torch.zeros(
            (self.num_envs, self.history_length, self.state_dim),
            device=self.device,
            dtype=torch.float32
        )

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, truncated, extras = super().step(action)

        if isinstance(obs, dict) and "policy" in obs:
            current_state = obs["policy"][:, 2 * self.state_dim : 3 * self.state_dim]
        else:
            current_state = obs[:, 2 * self.state_dim : 3 * self.state_dim]
        
        self.custom_state_history[:, 0:-1, :] = self.custom_state_history[:, 1:, :].clone()
        self.custom_state_history[:, -1, :] = current_state.clone()

        return obs, rewards, dones, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        obs, extras = super().reset(env_ids)
        if env_ids is None:
            self.custom_state_history.zero_()
        else:
            self.custom_state_history[env_ids] = 0.0

        return obs, extras
