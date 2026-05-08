import torch
from isaaclab.envs import ManagerBasedRLEnv
from .biped_rl_env_cfg import BipedRlEnvCfg
from .training_config import TrainingConfig
from .bc.preprocess_cfg import PreprocessCfg
from . import mdp
# TODO review and check


class BipedRlEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: BipedRlEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.state_history = torch.zeros(
            (self.num_envs, self.cfg.state_history_length, self.cfg.state_dim),
            device=self.device,
            dtype=torch.float32
        )

        self.action_history = torch.zeros(
            (self.num_envs, self.cfg.action_history_length, self.cfg.action_dim), 
            device=self.device, 
            dtype=torch.float32
        )

        self._state_obs_mean = torch.as_tensor(
            PreprocessCfg.OBS_MEAN,
            device=self.device,
            dtype=torch.float32,
        )
        self._state_obs_std = torch.as_tensor(
            PreprocessCfg.OBS_STD,
            device=self.device,
            dtype=torch.float32,
        )

        self._action_scales = torch.as_tensor(
            mdp.load_action_scales(),
            device=self.device,
            dtype=torch.float32,
        )

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:

        #action = self._apply_phase0_action_mask(action)
        self._update_action_history_normal(action)
        #action = torch.clamp(action, min=-1.0, max=1.0)
        self._print_policy_action(action)
        print("Stepping environment with action now")
        obs, rewards, dones, truncated, extras = super().step(action)
        print("dones:", dones)
        self._print_policy_obs_partitions(obs)

        if isinstance(obs, dict) and "policy" in obs:
            self._current_state = obs["policy"][:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim]
        else:
            self._current_state = obs[:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim]

        self._update_state_history_normal(self._current_state, dones)
        #self._reset_history_buffers_on_dones(obs, dones, self._current_state)

        return obs, rewards, dones, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:

        obs, extras = super().reset(env_ids=env_ids, seed=seed, options=options)
        if isinstance(obs, dict) and "policy" in obs:
            self._current_state = obs["policy"][:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim].clone()
        else:
            self._current_state = obs[:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim].clone()

        if env_ids is None:
            self.state_history[:] = self._current_state.unsqueeze(1).expand_as(self.state_history)
            self.action_history.zero_()
        else:
            print(f"Resetting environments with IDs: {env_ids.tolist()}")
            s_0 = self._current_state[env_ids]
            self.state_history[env_ids] = s_0.unsqueeze(1).expand(-1, self.cfg.state_history_length, -1)
            self.action_history[env_ids] = 0.0

        if isinstance(obs, dict) and "policy" in obs:
            if env_ids is None:
                obs["policy"][:, 0 : self.cfg.state_dim] = self._current_state
                obs["policy"][:, self.cfg.state_dim : 2 * self.cfg.state_dim] = self._current_state
            else:
                obs["policy"][env_ids, 0 : self.cfg.state_dim] = s_0
                obs["policy"][env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0
        else:
            if env_ids is None:
                obs[:, 0 : self.cfg.state_dim] = self._current_state
                obs[:, self.cfg.state_dim : 2 * self.cfg.state_dim] = self._current_state
            else:
                obs[env_ids, 0 : self.cfg.state_dim] = s_0
                obs[env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0

        self._print_policy_obs_partitions(obs)

        # Denormalize history buffers for printing
        denorm_state_history = self.state_history * self._state_obs_std + self._state_obs_mean
        denorm_action_history = self.action_history * self._action_scales
        print("self.state_history (denormalized)\n", denorm_state_history)
        print("self.action_history (denormalized)\n", denorm_action_history)

        print("==========Environment reset completed==========")
        return obs, extras
    
    def _apply_phase0_action_mask(self, action: torch.Tensor, tolerance: float = 1e-10) -> torch.Tensor:
        if isinstance(self.obs_buf, dict) and "policy" in self.obs_buf:
            prev_obs = self.obs_buf["policy"]
        else:
            prev_obs = self.obs_buf
        mask = torch.abs(prev_obs[:, -1]) < tolerance

        if mask.any():
            action[mask, 0] = 0.0
            action[mask, 2:5] = 0.0
            action[mask, 7:10] = 0.0

            action[mask, 1] = action[mask, 5]
            action[mask, 6] = action[mask, 5]
            action[mask, 10] = action[mask, 5]

        return action

    def _print_policy_obs_partitions(self, obs: torch.Tensor | dict) -> None:
        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"][0]
        else:
            policy_obs = obs[0]

        print_denormalized = True

        s_t2 = policy_obs[0:TrainingConfig.STATE_DIM]
        s_t1 = policy_obs[TrainingConfig.STATE_DIM:2 * TrainingConfig.STATE_DIM]
        s_t = policy_obs[2 * TrainingConfig.STATE_DIM:3 * TrainingConfig.STATE_DIM]
        a_t2 = policy_obs[3 * TrainingConfig.STATE_DIM:3 * TrainingConfig.STATE_DIM + TrainingConfig.ACTION_DIM]
        a_t1 = policy_obs[3 * TrainingConfig.STATE_DIM + TrainingConfig.ACTION_DIM:3 * TrainingConfig.STATE_DIM + 2 * TrainingConfig.ACTION_DIM]

        if print_denormalized:
            s_t2 = s_t2 * self._state_obs_std + self._state_obs_mean
            s_t1 = s_t1 * self._state_obs_std + self._state_obs_mean
            s_t = s_t * self._state_obs_std + self._state_obs_mean
            a_t2 = a_t2 * self._action_scales
            a_t1 = a_t1 * self._action_scales

        def _format_rows(values: list[float], row_size: int = 5, precision: int = 6) -> str:
            rows = []
            for i in range(0, len(values), row_size):
                chunk = values[i : i + row_size]
                rows.append(" ".join(f"{v:.{precision}f}" for v in chunk))
            return "\n".join(rows)

        print("Env 0 Policy Obs Partitions:")
        print("S(t-2):")
        print(_format_rows(s_t2.tolist(), row_size=5))
        print("S(t-1):")
        print(_format_rows(s_t1.tolist(), row_size=5))
        print("S(t):")
        print(_format_rows(s_t.tolist(), row_size=5))
        print("A(t-2):")
        print(_format_rows(a_t2.tolist(), row_size=5))
        print("A(t-1):")
        print(_format_rows(a_t1.tolist(), row_size=5))
        print("phase")
        print(f"{policy_obs[-1]:.3f}")
        print("\n")

    def _print_policy_action(self, action: torch.Tensor) -> None:

        print_denormalized = True

        a_t = action[0]
        if print_denormalized:
            a_t = a_t * self._action_scales

        def _format_rows(values: list[float], row_size: int = 5, precision: int = 6) -> str:
            rows = []
            for i in range(0, len(values), row_size):
                chunk = values[i : i + row_size]
                rows.append(" ".join(f"{v:.{precision}f}" for v in chunk))
            return "\n".join(rows)

        print("Env 0 Action:")
        print(_format_rows(a_t.tolist(), row_size=5))
        print("\n")

    def _reset_history_buffers_on_dones(self, obs: torch.Tensor | dict, dones: torch.Tensor, current_state: torch.Tensor) -> None:        

        if not torch.any(dones):
            return
        
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        s_0 = current_state[reset_env_ids]
        
        self.state_history[reset_env_ids] = s_0.unsqueeze(1).expand(-1, self.cfg.state_history_length, -1)
        self.action_history[reset_env_ids] = 0.0
        action_start_idx = 3 * self.cfg.state_dim
        
        if isinstance(obs, dict) and "policy" in obs:
            obs["policy"][reset_env_ids, 0 : self.cfg.state_dim] = s_0
            obs["policy"][reset_env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0

            obs["policy"][reset_env_ids, action_start_idx : action_start_idx + self.cfg.action_dim] = 0.0
            obs["policy"][reset_env_ids, action_start_idx + self.cfg.action_dim : action_start_idx + 2 * self.cfg.action_dim] = 0.0
        else:
            obs[reset_env_ids, 0 : self.cfg.state_dim] = s_0
            obs[reset_env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0

            obs[reset_env_ids, action_start_idx : action_start_idx + self.cfg.action_dim] = 0.0
            obs[reset_env_ids, action_start_idx + self.cfg.action_dim : action_start_idx + 2 * self.cfg.action_dim] = 0.0

        print(f"Reset history buffers for {len(reset_env_ids)} environments: {reset_env_ids.tolist()}")

    def _update_state_history_normal(self, current_state: torch.Tensor, dones: torch.Tensor) -> None:
        
        running_env_ids = (~dones).nonzero(as_tuple=False).squeeze(-1)
        if len(running_env_ids) > 0:
            self.state_history[running_env_ids, 0:-1, :] = self.state_history[running_env_ids, 1:, :].clone()
            self.state_history[running_env_ids, -1, :] = current_state[running_env_ids].clone()

    def _update_action_history_normal(self, action: torch.Tensor) -> None:
        
        self.action_history[:, 0:-1, :] = self.action_history[:, 1:, :].clone()
        self.action_history[:, -1, :] = action.clone()