import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import UsdFileCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "biped_v3-IsaacLab.usd")


def get_optimized_actuators_cfg() -> dict:
    """
    Returns an optimized actuator configuration for the biped robot.
    Groups joints with identical stiffness and damping properties.
    """
    
    return {
        "back_actuator": ImplicitActuatorCfg(
            joint_names_expr=["back"],
            stiffness=150.0,
            damping=1.0
        ),
        
        "legs_and_lower_body": ImplicitActuatorCfg(
            joint_names_expr=["sacrum", ".*_hip", ".*_thigh", ".*_calf", ".*_ankle", ".*_foot"],
            stiffness=10.0,
            damping=0.0
        ),
    }

BIPED_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=None,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=None,
            solver_velocity_iteration_count=None,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.023701), # must be the same as Isaac Sim (0.0, 0.0, 0.023701)
    ),

    actuators=get_optimized_actuators_cfg(),
)