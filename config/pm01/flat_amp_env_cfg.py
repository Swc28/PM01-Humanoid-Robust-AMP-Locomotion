"""
PM01 人形机器人 AMP 平地运动训练环境配置。

主要功能：
1. 配置 PM01 教育版机器人模型及关节执行器参数。
2. 定义策略网络和 AMP 判别器所需的观测信息。
3. 配置关节位置控制及动作缩放参数。
4. 设置机器人线速度和角速度指令，实现速度跟踪控制。
5. 配置运动奖励、姿态约束、足底接触及动作平滑性奖励。
6. 设置机器人摔倒和异常接触时的终止条件。

训练目标：
利用 AMP 学习参考动作中的自然运动风格，
结合 PPO 和速度跟踪奖励，实现 PM01 在平地环境下的稳定双足行走与转向。
"""
from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from engineai_rl_lab.tasks.mytask import mdp
from engineai_rl_lab.tasks.mytask.amp_env_cfg import (
    DISC_HISTORY_LENGTH,
    AmpObservationsCfg,
    LocomotionAmpEnvCfg,
    PM01_DISC_TERM_ORDER,
    pin_observation_term_order,
)
from engineai_rl_lab.tasks.mytask.robots.pm01 import (
    PM01_CFG,
    PM01_DFS_JOINT_ORDER_ASSET_CFG,
    PM_WAIST_DFS_JOINT_NAMES,
)
from engineai_rl_lab.tasks.mytask.utils.actuators import build_delayed_actuators

ACTUATOR_DELAY_RANGE = (2, 8)

ACTION_SCALE = {
    ".*_HIP_PITCH_.*": 0.5,
    ".*_HIP_ROLL_.*": 0.15, # 0.2
    ".*_HIP_YAW_.*": 0.15, # 0.2
    ".*_KNEE_PITCH_.*": 0.5,
    ".*_ANKLE_PITCH_.*": 0.5,
    ".*_ANKLE_ROLL_.*": 0.15, # 0.2
    ".*WAIST_YAW.*": 0.15, # 0.2
    ".*_SHOULDER_PITCH_.*": 0.2,
    ".*_SHOULDER_ROLL_.*": 0.2,
    ".*_SHOULDER_YAW_.*": 0.2,
    ".*_ELBOW_PITCH_.*": 0.2,
    ".*_ELBOW_YAW_.*": 0.2,
}


@configclass
class PM01AmpObservationsCfg(AmpObservationsCfg):
    """PM01 observations. Disc terms are declared in expert-frame order, not inherited."""
    """
    PM01 机器人观测配置类。

    继承通用 AMP 观测配置，针对 PM01 定义判别器所需的观测数据，
    包括基座角速度、关节位置、关节速度、基座线速度和投影重力。
    通过统一观测顺序和历史帧长度，保证机器人生成的动作数据
    与专家参考动作数据具有一致的特征结构。
    """
    @configclass
    class DiscriminatorCfg(ObsGroup):
        """
        AMP 判别器观测组配置类。

        定义输入判别器的单帧运动特征，包括基座角速度、
        关节位置与速度、基座线速度和投影重力。
        通过历史帧堆叠提供连续运动信息，帮助判别器区分
        机器人生成的动作与专家参考动作。
        """
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.discriminator_joint_pos)
        joint_vel = ObsTerm(func=mdp.discriminator_joint_vel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = DISC_HISTORY_LENGTH
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()


@configclass
class PM01AMPFlatEnvCfg(LocomotionAmpEnvCfg):
    """AMP flat environment configuration for the PM01 robot."""
    """
    PM01 AMP 平地运动训练环境配置类。

    继承通用 AMP 运动环境配置，加载 PM01 机器人模型，
    设置关节控制、动作缩放、执行器延迟和观测顺序。
    同时配置线速度与角速度指令、运动奖励及终止条件，
    为 PM01 在平地环境下学习自然双足行走和转向动作
    提供完整的仿真训练环境。
    """
    observations: PM01AmpObservationsCfg = PM01AmpObservationsCfg()

    def __post_init__(self):
        """
        环境初始化方法。

        首先继承父类的默认环境配置，然后根据 PM01 的机器人结构
        设置平地地形、机器人模型、关节动作控制和观测参数。
        随后配置速度指令范围、运动奖励权重以及异常接触终止条件，
        完成 PM01 AMP 训练环境的初始化。
        """
        super().__post_init__()

        # PM01 trains on a plain ground plane rather than the generated rough terrain
        # configured by the shared AMP environment.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.robot = PM01_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators=build_delayed_actuators(PM01_CFG.actuators, *ACTUATOR_DELAY_RANGE),
        )

        self.actions.joint_pos.joint_names = PM_WAIST_DFS_JOINT_NAMES
        self.actions.joint_pos.scale = ACTION_SCALE

        pin_observation_term_order(self.observations.disc, PM01_DISC_TERM_ORDER)
        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG
            group.joint_vel.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG
        self.observations.policy.joint_vel.noise.joint_names = PM_WAIST_DFS_JOINT_NAMES

        self.events.add_joint_default_pos.params["asset_cfg"] = PM01_DFS_JOINT_ORDER_ASSET_CFG

        self.commands.base_velocity.rel_standing_envs = 0.0 # 0.1 # 0.0
        # 允许 lin_vel_x=0（站立）配合 ang_vel_z，形成纯原地转身命令
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5) # 0.8 (0.3, 0.3) # (0.0, 0.8) # (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0) # (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) # (-1.2,  1.2) 

        #奖励函数设置
        self.rewards.track_lin_vel_xy_exp.weight = 3.5 # 3.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.35 
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.lin_vel_z_l2.weight = -1.0 
        self.rewards.ang_vel_xy_l2.weight = -0.15 
        self.rewards.dof_torques_l2.weight = -1.0e-6
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.action_rate_l2.weight = -0.02 # -0.01
        self.rewards.action_smoothness.weight = 0.0
        self.rewards.dof_pos_limits.weight = -0.1
        self.rewards.joint_deviation_hip.weight = -0.02 # -0.01
        self.rewards.joint_deviation_arms.weight = -0.01
        self.rewards.joint_deviation_waist.weight = -1.0
        self.rewards.feet_air_time.weight = 1.5 # 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_slide.weight = -0.7 # -0.4
        # 原地转身相关 reward
         # self.rewards.feet_clearance_turning.weight = 1.0
         # self.rewards.command_stall_penalty.weight = -0.5
         # self.rewards.termination_penalty.weight = -30.0 # -50.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "LINK_BASE",
            "LINK_KNEE_PITCH.*",
            ".*SHOULDER.*",
            ".*ELBOW.*",
            "LINK_TORSO_YAW",
        ]
