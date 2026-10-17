# robosyn-act-baseline

RoboSynChallenge ACT baseline 的复现入口：下载数据、检查 LeRobot 数据集、启动训练与仿真评测。

本项目调用比赛官方的 ACT 训练脚本及部署适配器，模型来自 LeRobot。官方源码固定到 [`64b0775`](https://github.com/EDEM-AI/RoboSynChallenge/tree/64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6)，LeRobot 版本由该源码的 `policy/act/uv.lock` 固定。

当前状态：启动工具与离线测试已提供；**尚未在真实数据与 GPU 仿真环境完成训练/成功率验证，也没有本项目训练出的权重**。`4090` 配置是保守起点，显存用量需要实测。单 GPU 入口不提供训练恢复或覆盖已有输出。

## 环境

实际训练和评测需要 Linux NVIDIA GPU 比赛镜像。本项目的启动工具仅依赖 Python 标准库（≥3.9）；macOS 可执行帮助、数据检查、命令生成和测试。

```bash
git clone https://github.com/xiejunxiong/robosyn-act-baseline.git
cd robosyn-act-baseline
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m robosyn_act --help
```

ACT 环境使用 Python 3.11 以及官方锁文件中的依赖，详见[环境安装](docs/environment.md)。已有比赛镜像时，先检查源码版本和解释器。

```bash
# 官方源码目录，不是本仓库
export ROBOSYN_ROOT=/root/workspace/RoboSynChallenge
export ACT_PYTHON="$ROBOSYN_ROOT/policy/act/.venv/bin/python"
```

所有相对数据和检查点路径相对于执行命令时的目录解析，再以绝对路径传给官方程序。

## 下载和检查数据

安装 Hugging Face CLI（可放在单独工具环境中）：`python3 -m pip install 'huggingface_hub[cli]>=0.34,<1'`。下载官方仿真数据；需要真实数据时把 `sim` 改为 `real`。

```bash
python3 -m robosyn_act download sim --task click_bell \
  --destination ./data/cobotmagic_Sim_click_bell --dry-run
# 去掉 --dry-run 执行下载
python3 -m robosyn_act download sim --task click_bell \
  --destination ./data/cobotmagic_Sim_click_bell
python3 -m robosyn_act check-data ./data/cobotmagic_Sim_click_bell
```

要求 LeRobot **v2.1** 数据集，包含 `meta/info.json`、`episodes.jsonl`、`tasks.jsonl`、`episodes_stats.jsonl`、Parquet 数据，以及视频特征对应的 MP4。检查验证元数据与必要文件存在性，不能检测每帧损坏。v3 数据先参考官方 `scripts/convert_lerobot3.0_to_2.1.py --help` 转换到新目录。

状态支持 `observation.qpos` 或 `observation.state`；三路相机支持 `cam_*.color` 或 `observation.images.cam_*`。字段映射由官方训练适配器完成。输入和动作维度由数据决定。

## 训练

```bash
# 检查路径、数据和参数，并打印命令
python3 -m robosyn_act train --python "$ACT_PYTHON" \
  --dataset-root ./data/cobotmagic_Sim_click_bell \
  --output-dir ./outputs/click_bell_smoke --profile smoke --dry-run

# 10 步 smoke run，仍需真实数据、完整 ACT 环境和 GPU
python3 -m robosyn_act train --python "$ACT_PYTHON" \
  --dataset-root ./data/cobotmagic_Sim_click_bell \
  --output-dir ./outputs/click_bell_smoke --profile smoke

# 80,000 步单 GPU 实验，seed=1000
python3 -m robosyn_act train --python "$ACT_PYTHON" \
  --dataset-root ./data/cobotmagic_Sim_click_bell \
  --output-dir ./outputs/click_bell_seed1000 --profile 4090 --gpu 0
```

| 配置 | steps | batch size | chunk size | n action steps | save freq |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 10 | 1 | 16 | 8 | 10 |
| 4090 | 80000 | 8 | 50 | 50 | 10000 |

可覆盖 `--batch-size`、`--steps`、`--chunk-size`、`--n-action-steps`、`--save-freq`、`--log-freq`，并设置 `--num-workers`、`--seed`、`--use-amp`。显存不足时减小 batch size。多 GPU 使用[官方 torchrun 流程](https://github.com/EDEM-AI/RoboSynChallenge/blob/64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6/docs/tutorials/policy/act.md)。

W&B 默认关闭；显式传入 `--wandb` 才会启用官方日志上传，使用前自行登录。训练输出已存在时命令报错，请使用新 run 名称。

## 评测

可下载官方权重检查仿真环境；这些权重不代表本仓库训练结果。

```bash
python3 -m robosyn_act download checkpoint --task click_bell \
  --destination ./checkpoints/ACT_sim_click_bell
python3 -m robosyn_act eval --python "$ACT_PYTHON" \
  --task click_bell --setting random \
  --checkpoint ./checkpoints/ACT_sim_click_bell --episodes 20 --gpu 0 --dry-run
# 去掉 --dry-run 后执行，默认 headless，交互显示可加 --gui
```

自训权重选择 `outputs/<run>/checkpoints/<step>/pretrained_model`，目录须包含 `config.json` 和 `model.safetensors`。结果和视频由官方脚本写入 `$ROBOSYN_ROOT/eval_result/`。

`python3 -m robosyn_act tasks` 列出 11 个任务。`open_pan` 会映射到该版本实际存在的 `configs/other_tasks/open_pan/`，其结果目录也包含 `other_tasks/open_pan`。

## 结果记录

尚无本项目实测结果。完成实验后记录 task、clear/random、数据来源及 revision、训练步数、seed、checkpoint、成功数/总 episode 数、GPU 和软件版本、官方结果 JSON 与日志路径。请分别报告官方权重验证和自训复现结果，不以训练 loss 代替任务成功率。

## 开发验证

```bash
python3 -m unittest discover -s tests -v
python3 -m robosyn_act tasks
```

离线测试不下载模型/数据、不需要 Torch 或 GPU；覆盖参数传递、数据契约、错误格式、检查点、输出保护、带空格路径和退出码。GPU 训练、视频解码和仿真需另行执行 smoke workflow。

```text
robosyn_act/         CLI、数据检查、官方入口
tests/              离线单元与进程测试
docs/environment.md 比赛镜像/源码环境说明
.github/workflows/  Python 测试
```

## 来源与许可

新增代码采用 MIT。官方代码、数据与权重分别遵守其上游许可，本仓库不重新分发这些资产。

- [RoboSynChallenge](https://github.com/EDEM-AI/RoboSynChallenge)
- [ACT 论文](https://arxiv.org/abs/2304.13705)
- [LeRobot](https://github.com/huggingface/lerobot)
- [比赛主页](https://robosyn-bench.net/)
