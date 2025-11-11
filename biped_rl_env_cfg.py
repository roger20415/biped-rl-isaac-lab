# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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


JOINTS: list[str] = ["sacrum","l_hip", "l_thigh",
                     "l_calf", "l_ankle",
                     "l_foot",
                     "r_hip", "r_thigh",
                     "r_calf", "r_ankle",
                     "r_foot"]
REVOLUTE_JOINTS: list[str] = JOINTS[1:]  # exclude sacrum
FOOT_CONTACT_THRESHOLD: float = 0.0014  # meters

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

    sacrum_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["sacrum"],
        scale=0.007) #0.009
    revolute_joints_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=REVOLUTE_JOINTS,
        scale=math.radians(10.0)) #40


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        baselink_W_height = ObsTerm(func=mdp.base_pos_z)
        baselink_W_euler_xy = ObsTerm(
            func=mdp.get_euler_W_xy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])})
        baselink_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        baselink_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)}
        )

        l_foot_contact = ObsTerm(
            func=mdp.has_foot_contact,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["l_foot_1"]), "threshold": FOOT_CONTACT_THRESHOLD}
        )
        r_foot_contact = ObsTerm(
            func=mdp.has_foot_contact,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["r_foot_1"]), "threshold": FOOT_CONTACT_THRESHOLD}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    move_forward = RewTerm(
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

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Base height out of bounds
    base_height_limit = DoneTerm(
        func=mdp.base_height_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "bounds": (0.0137, 0.022),   # (min_z, max_z)
        },
    )


##
# Environment configuration
##


@configclass
class BipedRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BipedRlSceneCfg = BipedRlSceneCfg(num_envs=100, env_spacing=0.2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1/120
        target_control_dt = 0.05 # s
        self.decimation = int(round(target_control_dt / self.sim.dt))
        self.sim.render_interval = self.decimation