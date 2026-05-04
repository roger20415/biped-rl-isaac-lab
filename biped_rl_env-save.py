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

        self.custom_state_history = torch.zeros(
            (self.num_envs, self.cfg.state_history_length, self.cfg.state_dim),
            device=self.device,
            dtype=torch.float32
        )

        self.custom_action_history = torch.zeros(
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

    def _print_policy_obs_partitions(self, obs: torch.Tensor | dict) -> None:
        if isinstance(obs, dict) and "policy" in obs:
            policy_obs = obs["policy"][0]
        else:
            policy_obs = obs[0]

        print_denormalized = True
        # print_denormalized = False

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
        print("\n")

    def _print_policy_action(self, action: torch.Tensor) -> None:
        print_denormalized = True
        #print_denormalized = False

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

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:

        print("Stepping environment with action now")
        action = torch.clamp(action, min=-1.0, max=1.0)
        self._print_policy_action(action)
        # 1. 執行底層 step，取得最新觀測值與 dones 訊號
        obs, rewards, dones, truncated, extras = super().step(action)
        
        # ==========================================
        # [修改處] NaN 攔截防護網 (NaN Interception Safety Net)
        # 檢查 obs 中是否含有 NaN。若有，強制終止該環境、給予懲罰，並「重置底層物理引擎」。
        # ==========================================
        if isinstance(obs, dict) and "policy" in obs:
            has_nans = torch.isnan(obs["policy"]).any(dim=-1)
        else:
            has_nans = torch.isnan(obs).any(dim=-1)

        if torch.any(has_nans):
            # 取得發生 NaN 的環境 ID
            nan_env_ids = has_nans.nonzero(as_tuple=False).squeeze(-1)
            print(f"⚠️ [警告] 偵測到環境 {nan_env_ids.tolist()} 產生 NaN！強制物理重置...")
            
            # 強制將這些環境標記為 done
            dones[nan_env_ids] = True
            
            # 給予額外的懲罰，避免 Agent 學會故意引發 NaN 來逃避正常訓練
            rewards[nan_env_ids] -= 10.0 
            
            # # [新增] 強制呼叫環境的 reset，讓 Isaac Sim 底層物理引擎把機器人修復並歸位
            # reset_obs, _ = self.reset(env_ids=nan_env_ids, seed=None, options=None)
            
            # # [新增] 將重置後乾淨、安全的 obs 覆寫回目前的 obs
            # if isinstance(obs, dict) and "policy" in obs:
            #     obs["policy"][nan_env_ids] = reset_obs["policy"][nan_env_ids]
            # else:
            #     obs[nan_env_ids] = reset_obs[nan_env_ids]
        # ==========================================

        # 2. 提取當前最新狀態 S(t)
        if isinstance(obs, dict) and "policy" in obs:
            current_state = obs["policy"][:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim]
        else:
            current_state = obs[:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim]

        # ==========================================
        # History Buffer 管理流程
        # ==========================================
        # 3. 先對「所有」環境進行正常的 History 更新 (不管有沒有 done)
        self._update_history_buffers_normal(current_state, action)

        # 4. 再針對「發生重置 (done=True)」的環境進行覆寫與清理 (這會覆蓋掉剛才步驟 3 的結果)
        self._reset_history_buffers_on_dones(obs, dones, current_state)
        
        # 5. Debug 輸出，確認最終餵給 Policy 的資料是否乾淨
        self._print_policy_obs_partitions(obs)

        return obs, rewards, dones, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        #　TODO check and read
        # 1. 先執行預設的 reset 來更新物理引擎狀態，並取得尚未更新 history 的 obs
        obs, extras = super().reset(env_ids=env_ids, seed=seed, options=options)
        
        # 2. 從目前的 obs 中擷取出初始狀態 S(0)
        if isinstance(obs, dict) and "policy" in obs:
            current_state = obs["policy"][:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim].clone()
        else:
            current_state = obs[:, 2 * self.cfg.state_dim : 3 * self.cfg.state_dim].clone()

        # 3. 將 history 緩衝區填滿 S(0)，而不是使用 0.0 歸零
        if env_ids is None:
            # 針對所有環境：使用 unsqueeze 與 expand 將 S(0) 複製填滿整個 history length
            self.custom_state_history[:] = current_state.unsqueeze(1).expand_as(self.custom_state_history)
            self.custom_action_history.zero_()  # 動作歷史 (action) 在剛開始還沒有發生，維持為 0 即可
        else:
            # 針對特定重置的環境：擴展該環境的 S(0)
            s_0 = current_state[env_ids]
            self.custom_state_history[env_ids] = s_0.unsqueeze(1).expand(-1, self.cfg.state_history_length, -1)
            self.custom_action_history[env_ids] = 0.0

        # 4. 手動覆寫回傳的 obs，將 t-2 與 t-1 的位置替換為 S(0)，確保第一步的輸入格式為 (s(0), s(0), s(0))
        if isinstance(obs, dict) and "policy" in obs:
            if env_ids is None:
                obs["policy"][:, 0 : self.cfg.state_dim] = current_state
                obs["policy"][:, self.cfg.state_dim : 2 * self.cfg.state_dim] = current_state
            else:
                obs["policy"][env_ids, 0 : self.cfg.state_dim] = s_0
                obs["policy"][env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0
        else:
            if env_ids is None:
                obs[:, 0 : self.cfg.state_dim] = current_state
                obs[:, self.cfg.state_dim : 2 * self.cfg.state_dim] = current_state
            else:
                obs[env_ids, 0 : self.cfg.state_dim] = s_0
                obs[env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0

        # Debug
        self._print_policy_obs_partitions(obs)
        # Debug end

        print("Environment reset complete!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return obs, extras
    
    def _reset_history_buffers_on_dones(self, obs: torch.Tensor | dict, dones: torch.Tensor, current_state: torch.Tensor) -> None:
        if not torch.any(dones):
            return
        
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        s_0 = current_state[reset_env_ids]
        
        self.custom_state_history[reset_env_ids] = s_0.unsqueeze(1).expand(-1, self.cfg.state_history_length, -1)
        self.custom_action_history[reset_env_ids] = 0.0

        action_start_idx = 3 * self.cfg.state_dim
        
        if isinstance(obs, dict) and "policy" in obs:
            # 覆寫 S(t-2) 和 S(t-1) 為重置後的初始狀態 S(0)
            obs["policy"][reset_env_ids, 0 : self.cfg.state_dim] = s_0
            obs["policy"][reset_env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0
            # 覆寫 A(t-2) 和 A(t-1) 為零
            obs["policy"][reset_env_ids, action_start_idx : action_start_idx + self.cfg.action_dim] = 0.0
            obs["policy"][reset_env_ids, action_start_idx + self.cfg.action_dim : action_start_idx + 2 * self.cfg.action_dim] = 0.0
        else:    
            # 覆寫 S(t-2) 和 S(t-1) 為重置後的初始狀態 S(0)
            obs[reset_env_ids, 0 : self.cfg.state_dim] = s_0
            obs[reset_env_ids, self.cfg.state_dim : 2 * self.cfg.state_dim] = s_0
            # 覆寫 A(t-2) 和 A(t-1) 為零
            obs[reset_env_ids, action_start_idx : action_start_idx + self.cfg.action_dim] = 0.0
            obs[reset_env_ids, action_start_idx + self.cfg.action_dim : action_start_idx + 2 * self.cfg.action_dim] = 0.0
        print(f"Reset history buffers for {len(reset_env_ids)} environments: {reset_env_ids.tolist()}")

    def _update_history_buffers_normal(self, current_state: torch.Tensor, action: torch.Tensor) -> None:
        # 更新動作歷史：整體往左平移一步，並在最後一個位置填入當前 action
        self.custom_action_history[:, 0:-1, :] = self.custom_action_history[:, 1:, :].clone()
        self.custom_action_history[:, -1, :] = action.clone()

        # 更新狀態歷史：整體往左平移一步，並在最後一個位置填入當前 state
        self.custom_state_history[:, 0:-1, :] = self.custom_state_history[:, 1:, :].clone()
        self.custom_state_history[:, -1, :] = current_state.clone()