# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.biped_rl.assets.biped_config import BIPED_CFG  # isort:skip
from .bc.preprocess_cfg import PreprocessCfg


JOINTS: list[str] = ["sacrum",
                     "l_hip", "l_thigh",
                     "l_calf", "l_ankle",
                     "l_foot",
                     "r_hip", "r_thigh",
                     "r_calf", "r_ankle",
                     "r_foot"]
FOOT_CONTACT_THRESHOLD: float = 0.0014  # meters

ACTION_SCALES = mdp.load_action_scales()

##
# Scene definition
##


@configclass
class BipedRlSceneCfg(InteractiveSceneCfg):
    """Configuration for a biped robot scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
   
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    (sacrum_position, l_hip_position, l_thigh_position, l_calf_position, l_ankle_position, l_foot_position,
    r_hip_position, r_thigh_position, r_calf_position, r_ankle_position, r_foot_position) = [
        mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINTS[i], scale=float(ACTION_SCALES[i])) for i in range(len(JOINTS))
    ]

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        state_t_minus_2 = ObsTerm(
            func=mdp.get_past_state,
            params={"step_back": 2}
        )

        state_t_minus_1 = ObsTerm(
            func=mdp.get_past_state,
            params={"step_back": 1}
        )


        baselink_W_height = ObsTerm(
            func=mdp.get_norm_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "getter" : mdp.base_pos_z,
                "mean": [float(PreprocessCfg.OBS_MEAN[0])],
                "std": [float(PreprocessCfg.OBS_STD[0])],
            }
        )
        baselink_W_euler_xyz = ObsTerm(
            func=mdp.get_norm_euler_W_xyz,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "mean": PreprocessCfg.OBS_MEAN[1:4].tolist(),
                "std": PreprocessCfg.OBS_STD[1:4].tolist(),
            }
        )
        baselink_lin_vel = ObsTerm(
            func=mdp.get_norm_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "getter": mdp.root_lin_vel_w,
                "mean": PreprocessCfg.OBS_MEAN[4:7].tolist(),
                "std": PreprocessCfg.OBS_STD[4:7].tolist(),
            }
        )
        baselink_ang_vel = ObsTerm(
            func=mdp.get_norm_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "getter": mdp.root_ang_vel_w,
                "mean": PreprocessCfg.OBS_MEAN[7:10].tolist(),
                "std": PreprocessCfg.OBS_STD[7:10].tolist(),
            }
        )
        joint_pos = ObsTerm(
            func=mdp.get_norm_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
                "getter": mdp.joint_pos_rel,
                "mean": PreprocessCfg.OBS_MEAN[10:21].tolist(),
                "std": PreprocessCfg.OBS_STD[10:21].tolist(),
            }
        )
        joint_vel = ObsTerm(
            func=mdp.get_norm_vector,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
                "getter": mdp.joint_vel_rel,
                "mean": PreprocessCfg.OBS_MEAN[21:32].tolist(),
                "std": PreprocessCfg.OBS_STD[21:32].tolist(),
            }
        )
        l_foot_contact = ObsTerm(
            func=mdp.has_foot_contact,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["l_foot_1"]), "threshold": FOOT_CONTACT_THRESHOLD}
        )
        r_foot_contact = ObsTerm(
            func=mdp.has_foot_contact,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["r_foot_1"]), "threshold": FOOT_CONTACT_THRESHOLD}
        )

        # ==========================================
        # [4] Action History: A_{t-2} (11 dim)
        # ==========================================
        # TODO: action t-2

        # ==========================================
        # [5] Action History: A_{t-1} (11 dim)
        # ========================================== 
        # TODO: action t-1

        # ==========================================
        # [6] Phase (1 dim)
        # ==========================================
        # TODO: phase num

        # TODO obs must be 34*3 + 11*2 + 1 (phase) = 144
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # TODO: modify reward weights
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    move_forward = RewTerm(
            # TODO: should be modified to biped v3
            func=mdp.forward_velocity_reward,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]), 
                "nowhere_penalty_weight": 0.2
            }
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # TODO: modify termination conditions
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Base height out of bounds
    base_height_limit = DoneTerm(
        func=mdp.base_height_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "bounds": (0.0198, 0.0212),   # (min_z, max_z)
        },
    )


##
# Environment configuration
##


@configclass
class BipedRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BipedRlSceneCfg = BipedRlSceneCfg(num_envs=10, env_spacing=0.2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    state_history_length: int = 2
    state_dim: int = 34

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.episode_length_s = 30.0  # seconds
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1/120
        target_control_dt = 0.05
        self.decimation = int(round(target_control_dt / self.sim.dt))
        self.sim.render_interval = 12