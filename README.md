# RoboSynChallenge ACT Baseline

[English](README_EN.md) · [实验档案](experiments/click_bell_80k/README.md) · [环境](docs/environment.md) · [4090验证步骤](docs/validation.md)

JXNU SynDex 的 RoboSynChallenge 2026（NeurIPS 2026）ACT 复现项目，面向 CobotMagic 双臂操作与机器人模仿学习。复用指定版本 LeRobot 的 Action Chunking Transformer，提供官方训练/仿真入口、数据适配契约、统一配置、CPU smoke 和实验记录。

## Local Simulation Evaluation (10 episodes)

| Task | Model | Steps | Batch | Success |
| --- | --- | ---: | ---: | --- |
| click_bell | ACT | 80k | 24 | 7/10 (70%) |

**这是本地仿真评测，不是官方比赛成绩、排行榜分数或最终名次。** 团队未在截止时间前提交有效官方评测申请。以上来自所有者提供的历史实验记录（2026-10-04 21:26:49），本次维护尚未独立取得原始 metrics、日志、视频和权重复核。10局是小样本，不能当作充分验证的总体成功率。

已报告完成1000 episodes数据下载、80k训练、10k至80k的8个checkpoint、自训练80k权重加载与本地random评测。仅click_bell有本项目实验记录；CLI可访问其他上游任务，不代表都已复现。自训练权重**待上传，尚无公开下载地址**。

## 模型与数据

ACT 使用视觉/状态条件和CVAE训练目标预测action chunk，再通过动作队列执行。实现来自LeRobot，不是早期合成数据MLP smoke。参考[ACT论文](https://arxiv.org/abs/2304.13705)及[架构/API/许可核查](docs/architecture.md)。

```text
三相机 + 14维qpos → LeRobot alias / normalization → ACTPolicy
→ 50×14 action chunk → action queue → CobotMagic / DexSim → 本地metrics与视频
```

数据：[RoboSynChallenge/cobotmagic_Sim_click_bell](https://huggingface.co/datasets/RoboSynChallenge/cobotmagic_Sim_click_bell)，LeRobot v2.1，历史记录为1000 episodes、74000 frames、25 FPS、1 task、3×640×480图像，状态/动作14维。

| 原始字段 | ACT字段 |
| --- | --- |
| observation.qpos | observation.state |
| cam_high.color | observation.images.cam_high |
| cam_left_wrist.color | observation.images.cam_left_wrist |
| cam_right_wrist.color | observation.images.cam_right_wrist |

qvel/qf各14维，未拼入状态。官方训练包装器处理features/stats/items/padding别名；不要假设原始数据天然使用新版命名。详见[数据说明](docs/dataset.md)。

## 安装与CPU检查

工具支持Python≥3.9，依赖PyYAML与NumPy；ACT/仿真环境单独使用Python3.11和Linux NVIDIA GPU。

```bash
git clone https://github.com/xiejunxiong/robosyn-act-baseline.git
cd robosyn-act-baseline
# PR未合并时，切到实现分支
git checkout codex/complete-act-baseline
bash scripts/setup_env.sh
source .venv-tools/bin/activate
python -m robosyn_act --help
bash scripts/smoke_test.sh
```

默认CPU smoke使用NumPy合成数据检查YAML、aliases、14维状态、[1,50,14]动作和三路[1,3,480,640]图像，**不执行模型**。`python -m robosyn_act smoke --act` 才会要求指定Git版LeRobot并执行真实ACTPolicy forward/loss/backward/select_action（缩小hidden layers、随机权重、无预训练下载），缺依赖时明确失败，不降级冒充通过。这也不是80k权重或仿真性能测试。

## 环境与版本边界

| 组件 | 官方参考 | 历史成功运行记录 |
| --- | --- | --- |
| Python | ACT要求>=3.10,<3.12 | 3.11.15 |
| PyTorch | uv.lock 2.7.1 | 2.10.0+cu128 |
| torchvision / NumPy | 0.22.1 / 1.26.4 | 未提供实际版本 |
| LeRobot | b883328e6c95681ca90a18b102e4ae5e1f91e2bf | 同commit，0.3.3 |
| datasets | 3.6.0 | 3.6.0 |
| EmbodiChain | v0.2.4.post1 | 89c14da8d |
| DexSim | 0.4.3 | 0.4.3 |

历史硬件RTX4090 24GB、Xeon Gold6430、Ubuntu22.04/Linux5.15，Torch构建CUDA12.8。实际Torch超出官方声明范围；私有dexsim-engine wheel hash问题**未解决**。历史运行直接使用已有venv，没有证明`uv sync --frozen`能从零复现。详见[环境差异](docs/environment.md)与[故障排查](docs/troubleshooting.md)。

当前启动接口核查固定RoboSynChallenge `64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6`；历史训练的RoboSynChallenge commit、数据revision和完整环境快照待补。不要在原实验目录reset源码。

## 最小训练与评测

已有可运行比赛镜像时，填入本机路径。在本仓库根目录执行，工具venv与ACT解释器可以分开：

```bash
export ROBOSYN_ROOT=/path/to/pinned/RoboSynChallenge
export ACT_PYTHON=/path/to/existing/venv/bin/python
export EMBODICHAIN_ROOT=/path/to/EmbodiChain
export DATASET_ROOT=/path/to/cobotmagic_Sim_click_bell
export RUN_DIR="$PWD/outputs/train/act_click_bell_bs24_80k"

# 没有数据时先按docs/dataset.md下载；检查真实数据
python -m robosyn_act check-data "$DATASET_ROOT" --config robosyn_act/configs/act_click_bell.yaml
python -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR" --dry-run
# 去掉--dry-run才启动80k训练；RUN_DIR必须不存在
python -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR"

# 如使用已有历史80k权重，将CHECKPOINT设为其真实位置，无需重新训练
export CHECKPOINT="$RUN_DIR/checkpoints/080000/pretrained_model"
# 确认Xvfb :99真实运行，见docs/evaluation.md
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display :99 --dry-run
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display :99
```

默认训练80k/batch24/workers4/chunk50/n_action_steps50/save10000/log100，训练seed1000是当前上游默认；评测默认click_bell/random/seed0/10局/361步。YAML可编辑，CLI参数优先，其次显式profile，再其次YAML。现有输出目录不会覆盖。`--profile smoke`是需要真实数据与ACT依赖的10步训练；不同于CPU `smoke`。

[训练与checkpoint](docs/training.md)解释推理文件、训练状态、nohup和指定step加载。上游包装器resume链路不完整，本项目不提供未经验证的恢复命令。[评测说明](docs/evaluation.md)解释专家筛选、随机seed、视频和计时范围。

官方权重可用于排查环境：`python -m robosyn_act download checkpoint --task click_bell --destination ./checkpoints/official_click_bell`。它**不是**团队80k权重；自训权重将上传HF模型仓库并补充下载地址/hash，不上传到普通Git。

## 实验结果与后续工作

历史10局：7成功、3超时；平均141.60/361 action steps，41.241 ms/model call，127.848 ms/episode，总31次调用，平均3.10次/局，专家筛选接受10/12 seeds。完整逐局表与来源见[实验档案](experiments/click_bell_80k/README.md)。

初步观察是能接近按钮但部分按压/对齐失败，失败后纠偏不足；这不是已验证因果结论。后续计划：20k/40k/60k/80k checkpoint对比；同权重n_action_steps=50/25/10（需先验证推理配置覆盖）；成功/失败视频分析；训练L1/KL/total曲线。历史loss数据目前 **Not recorded / Pending**，不生成虚构曲线。详见[Preliminary Failure Analysis与Roadmap](docs/failure_analysis.md)。

剩余限制：原始证据/公开权重待接入，10局样本有限，历史源代码及完整依赖未锁定，私有包安装受限，GPU训练/仿真尚未用本次包装器重跑。测试和dry-run只能验证它们实际覆盖的接口。

## 导航与许可证

目录保留轻量平铺Python包；统一YAML放在包内以保证wheel安装后可读取，未创建无用途的train/evaluate占位模块。完整目录和验证记录见[交付与服务器验证](docs/validation.md)。

原创代码保留[MIT License](LICENSE)，上游RoboSynChallenge、LeRobot、EmbodiChain代码及DexSim、数据、模型资产遵守各自许可。本仓库不重新分发上游资产。感谢[RoboSynChallenge](https://github.com/EDEM-AI/RoboSynChallenge)、[LeRobot](https://github.com/huggingface/lerobot)、[EmbodiChain](https://github.com/DexForce/EmbodiChain)与ACT作者。
