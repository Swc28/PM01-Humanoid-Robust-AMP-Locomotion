# PM01 AMP 运动控制（MyTask）

> 本文档是 **PM01 单机型** 的维护文档。T800 相关代码仍保留在仓库中，但不在本文档的记录与维护范围内，避免文档与实机状态脱节。
>
> 文档结构按「概述 → 溯源 → 适配 → 部署 → 结构 → 修改日志 → 问题记录 → 注意事项」固定分节，后续新增内容一律在对应小节内追加，不重构整体框架。

---

## 1. 项目概述

| 项目 | 内容 |
|------|------|
| 定位 | 基于 Adversarial Motion Priors (AMP) 的 PM01 人形机器人运动控制训练环境 |
| 任务入口 | GYM ID：`MyTask-PM01-v1` |
| 训练框架 | Isaac Lab + RSL-RL + 自研 AMP 算法实现 |
| 实机型号 | PM01（当前唯一上线机型） |
| 代码位置 | `source/engineai_rl_lab/engineai_rl_lab/tasks/mytask/` |
| 上游官方 AMP 任务 | `tasks/locomotion/amp/`，GYM ID：`AMP-Flat-PM01-v1`（只读参考基线，不修改） |

**职责边界**

- `locomotion/amp/`：官方 AMP 任务，保持与上游一致，仅作对照与回归基线。
- `mytask/`：PM01 专属算法与配置的迭代场所，允许自由修改。
- 个人算法仓库：承载 PM01 运动控制算法的独立演进，与本仓库解耦（见第 7 节）。

---

## 2. 代码溯源与许可

### 2.1 溯源链路

```
AMP for Hardware / MimicKit / Legged Lab   （AMP 判别器、回放、参考动作采样思路）
        └── engineai_rl_lab（上游官方仓库，BSD 3-Clause，含官方 AMP 任务）
                └── 本仓库 fork（基线提交见下）
                        └── tasks/mytask（PM01 专属迭代层）
```

| 层级 | 来源 | 版本 / 提交 | 协议 |
|------|------|-------------|------|
| 仿真框架 | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `2.3.2`，commit `c22775241e28f465fe345fa1a482ad6d29d712b0` | BSD 3-Clause |
| 仿真器 | Isaac Sim | `5.1.0` | NVIDIA Omniverse License |
| RL 库 | [rsl_rl](https://github.com/leggedrobotics/rsl_rl) | 随 `isaaclab_rl` 安装 | BSD 3-Clause |
| 上游任务仓库 | [engineai-robotics/engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab) | fork 基线 `2af75ec` | BSD 3-Clause |
| AMP 结构参考 | [Legged Lab](https://github.com/zitongbai/legged_lab) | 结构与任务组织参考 | 以仓库声明为准 |
| AMP 训练参考 | [MimicKit](https://github.com/xbpeng/MimicKit) | 判别器训练 / 经验回放 / 动作采样 | 以仓库声明为准 |
| AMP 实机参考 | [AMP for Hardware](https://github.com/escontra/AMP_for_hardware) | 论文 *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions* | 以仓库声明为准 |
| 推理部署 | [MNN](https://github.com/alibaba/mnn) | 端侧推理 | Apache-2.0 |

### 2.2 引用协议要求

- 本仓库整体沿用上游 **BSD 3-Clause License**，详见根目录 `LICENSE.txt`。
- **不得删除**任何源文件头部的版权与 SPDX 声明（如 `SPDX-License-Identifier: BSD-3-Clause`）。
- 参考项目（Legged Lab / MimicKit / AMP for Hardware）的许可证以各自仓库声明为准；搬运其代码片段时，必须在该文件内注明来源仓库、文件路径和协议，不得改写为自有版权。
- 对外发布模型或论文时，需同时引用 AMP 原始论文与上述参考实现。

### 2.3 溯源维护规则

- 每次同步上游时，在第 7 节「修改日志」记录：上游 commit hash、同步范围、冲突处理方式。
- 上游官方 AMP 任务的代码**只读**。确需修改时，复制到 `mytask/` 内实现，不直接改 `locomotion/amp/`。

---

## 3. PM01 硬件适配说明

### 3.1 关键适配项

| 项目 | 取值 / 做法 | 对应位置 |
|------|-------------|----------|
| 机器人资产 | `PM01_CFG`（含关节名、默认位姿、关节限位） | `robots/pm01.py` |
| 关节顺序 | DFS 顺序，作为 policy / critic / 动作的统一排列 | `PM01_DFS_JOINT_ORDER_ASSET_CFG` |
| 动作维度 | 腰部及以下 + 双臂（`PM_WAIST_DFS_JOINT_NAMES`） | `robots/pm01.py` |
| 动作缩放 | 髋/膝/踝 pitch `0.5`，roll/yaw `0.2`，手臂 `0.2` | `config/pm01/flat_amp_env_cfg.py` |
| 腰部关节命名 | PM01 为 `WAIST_YAW`，T800 为 `TORSO_YAW`，统一用正则 `.*(WAIST\|TORSO)_YAW.*` 匹配 | `amp_env_cfg.py` |
| 执行器延迟 | 2~8 个物理步（4~16 ms），模拟通信延迟 | `ACTUATOR_DELAY_RANGE` |
| 控制频率 | 物理 `dt=0.002`，`decimation=5` → 控制步 `0.01s`（100 Hz） | `LocomotionAmpEnvCfg.__post_init__` |
| Episode 长度 | 20 s（2000 个控制步） | 同上 |
| 摔倒判定 | `LINK_BASE`、膝盖、肩、肘、`LINK_TORSO_YAW` 触地即终止 | `base_contact.params` |
| 地形 | 平地（±2 cm 起伏），不使用粗糙地形生成器 | `config/pm01/flat_amp_env_cfg.py` |
| 基座高度终止 | PM01 **不启用**（T800 才有 0.65 m 约束） | — |

### 3.2 与判别器对齐的硬件约束

专家数据集对 PM01 做过两处裁剪，判别器观测必须一致，否则 AMP 学不到正确风格：

| 规则 | 常量 | 位置 |
|------|------|------|
| 腰部 / 躯干 YAW 关节置零 | `DISC_ZEROED_JOINT_PATTERN` | `mdp/observations.py` |
| 头部关节整列剔除 | `DISC_DROPPED_JOINT_SUBSTRING` | `mdp/observations.py` |

**单帧特征维度 `frame_dim=55`**（PM01 当前配置：关节位置 + 关节速度 + 基座线速度 + 投影重力 + 基座角速度），修改任一 `include_*` 开关都必须同步重算该值。

---

## 4. 环境部署

### 4.1 依赖版本

| 依赖 | 版本 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| GPU 驱动 | 满足 Isaac Sim 5.1.0 要求 |
| Python | 3.11（conda 环境名 `robot`） |
| Isaac Lab | 2.3.2 @ `c2277524` |
| Isaac Sim | 5.1.0 |

### 4.2 部署步骤

```bash
# 1. 克隆仓库（含 LFS 资产）
git clone https://github.com/engineai-robotics/engineai_rl_lab.git
cd engineai_rl_lab
git lfs install && git lfs pull

# 2. 安装扩展（需已安装 Isaac Lab 2.3.2 并激活其环境）
conda activate robot
pip install -e source/engineai_rl_lab

# 3. 验证任务注册（能看到 MyTask-PM01-v1 即成功）
python -c "import gymnasium as gym, engineai_rl_lab.tasks; print([k for k in gym.registry if 'MyTask' in k])"
```

### 4.3 数据集准备

```bash
# 目录约定
datasets/amp/pm01/
├── config/dataset.yaml   # runner 中 dataset_path 指向此处
└── data/*.npz            # 专家动作片段
```

训练前先重放检查根节点位姿、关节顺序、帧率与脚底接触：

```bash
python scripts/amp/replay_npz.py --robot pm01 --input_folder datasets/amp/pm01/data
```

---

## 5. 代码结构说明

```
mytask/
├── amp_env_cfg.py                    # 环境骨架：Scene/Action/Observation/Command/Reward/Termination/Event
├── agents/amp_runner_cfg.py          # AMP 超参（RslRlAmpCfg）+ PPO runner 基类
├── algorithms/amp.py                 # AMP 算法主体（policy 更新 / 判别器更新 / style reward）
├── utils/
│   ├── AMP_data_loader.py            # 专家数据加载与 frame 拼接（决定 frame_dim 与顺序）
│   ├── AMP_discriminator.py          # 判别器网络
│   ├── disc_replay_buffer.py         # 判别器负样本回放
│   └── actuators.py                  # 延迟执行器构建
├── mdp/                              # MDP 原子函数
│   ├── rewards.py                    # 全部 reward term
│   ├── observations.py               # 观测 term + 判别器关节对齐
│   ├── command.py / events.py / noise.py / symmetry.py
├── robots/pm01.py                    # PM01 资产与关节常量
└── config/pm01/
    ├── __init__.py                   # 注册 GYM ID：MyTask-PM01-v1
    ├── flat_amp_env_cfg.py           # PM01 特化配置（填 MISSING、覆盖权重）
    └── agents/amp_ppo_cfg.py         # 实验名、训练轮次、AMP 超参覆盖
```

**分层原则**

| 层 | 可改范围 | 说明 |
|----|----------|------|
| `config/pm01/` | 自由修改 | 调参只改这一层，优先在 `__post_init__` 覆盖 |
| `mdp/` | 新增函数为主 | 被多处复用的原子逻辑放这里 |
| `amp_env_cfg.py` | 谨慎 | 改动影响所有机型配置 |
| `algorithms/` `utils/` | 谨慎 | 改动影响算法正确性，须记录版本与实验对比 |
| `locomotion/amp/` | 只读 | 官方基线，禁止修改 |

---

## 6. 训练与仿真验证

```bash
conda activate robot
cd /home/talos/engineai_rl_lab

# 训练（轮次由 config/pm01/agents/amp_ppo_cfg.py 的 max_iterations 决定）
python scripts/amp/train.py --task MyTask-PM01-v1 --num_envs 4096 --headless

# 训练并录制片段
python scripts/amp/train.py --task MyTask-PM01-v1 --num_envs 4096 --headless \
  --video --video_length 200 --video_interval 2000

# 断点续训
python scripts/amp/train.py --task MyTask-PM01-v1 --num_envs 4096 --headless \
  --resume --load_run <时间戳目录> --checkpoint model_19999.pt

# 训练曲线
tensorboard --logdir logs/rsl_rl/mytask_pm01

# 仿真回放（GUI，关闭窗口结束）
python scripts/amp/play.py --task MyTask-PM01-v1 --num_envs 16 \
  --load_run <时间戳目录> --checkpoint <编号最大的 model_*.pt>
```

**产物位置**：`logs/rsl_rl/mytask_pm01/<时间戳>/`

| 文件 | 用途 |
|------|------|
| `model_*.pt` | 训练权重（默认每 2000 轮保存） |
| `params/env.yaml` `params/agent.yaml` | 该次训练完整配置快照 |
| `git/engineai_rl_lab.diff` | 该次训练时的代码差异（版本追溯关键） |
| `export/*.onnx` `export/*.mnn` | 部署导出 |

> `play.py` 默认按字典序取 checkpoint，同目录存在 `model_19999_converted.pt` 时会被误选，务必显式指定 `--checkpoint`。

---

## 7. 修改日志

格式：**修改时间 - 修改内容 - Training**

`Training` 列填写对应的训练 run 时间戳（`logs/rsl_rl/mytask_pm01/` 下的目录名）；未训练填「未训练」，仅验证链路填「仅验证」。

| 修改时间 | 修改内容 | Training |
|----------|----------|----------|
| 2026-11-11 | 新增 `feet_clearance_turning`、`command_stall_penalty` 两个 reward 项，基础配置中默认 `weight=0`（不启用） | 未训练 |
| 2026-11-11 | PM01 命令范围改为 `lin_vel_x=(0.0, 0.8)`、`ang_vel_z=(-1.2, 1.2)`；启用两项转身 reward（`1.0` / `-0.5`） | 未训练（旧 run `2026-09-14_22-46-03` 仅验证链路） |
| 2026-11-12 | README 重构：删除算法迭代与版本管理章节，改为修改日志、问题记录、注意事项三节 | — |

> 新增记录时在表格**最后一行**追加，不改动历史行；涉及上游同步或算法逻辑改动时，在「修改内容」中注明 commit hash。

---

## 8. 遇到的问题 - 解决办法

| 问题现象 | 原因 | 解决办法 |
|----------|------|----------|
| AMP 训练不收敛、动作风格异常 | `DiscriminatorCfg.__dict__` 的顺序被子类覆盖或后赋值打乱，判别器输入特征错位 | 用 `pin_observation_term_order()` 固定观测项顺序；不要在子类里单独覆盖 `disc` 的观测项 |
| 判别器报维度不匹配 | `frame_dim` 与实际观测宽度不一致（改动了 `include_*` 开关但没重算） | 三处同步：`amp_env_cfg.py` 的 `DiscriminatorCfg`、runner cfg 的 `frame_length`/`frame_dim`/`flatten_history_dim`、`AMP_data_loader.py` 的 `_gather_frame_features` |
| 仿真回放加载了错误权重 | `play.py` 默认按字典序选 checkpoint，`model_19999_converted.pt` 排在 `model_19999.pt` 之后被误选 | 显式指定 `--checkpoint model_19999.pt`，不要依赖默认值 |
| 训练录制的视频全黑 | 无显示环境下 `--headless` 配合 `--video` 走视口截图失败 | 去掉 `--headless`，用 GUI 模式录制 |
| 改了配置但仿真行为没变 | 用的是旧配置训练出的 checkpoint，新 reward / 命令范围不会自动生效 | 重新训练生成新的 run 目录，再用该目录下编号最大的 checkpoint 回放 |

---

## 9. 注意事项

1. **调参只改 `config/pm01/flat_amp_env_cfg.py` 的 `__post_init__`**，不要在 `amp_env_cfg.py` 里直接改默认值，也不要动 `locomotion/amp/`。前者会影响所有机型配置，后者是官方基线，改了就无法与上游做回归对比。

2. **判别器观测的维度与顺序是三处强耦合的**：`DiscriminatorCfg` 的观测项、runner 里的 `frame_length/frame_dim/flatten_history_dim`、`AMP_data_loader.py` 的 frame 拼接顺序。任何一处改动都必须同时改另外两处，否则报错或静默学错。

3. **每次训练都要保留 run 目录下的 `params/*.yaml` 与 `git/*.diff`**，并在第 7 节「修改日志」登记「修改时间 - 修改内容 - Training」三元组；否则后续无法从权重反查到当时的代码与配置，实验不可复现。
