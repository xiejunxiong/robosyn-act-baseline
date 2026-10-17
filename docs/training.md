# Training and checkpoints

启动器调用固定官方 `policy/act/scripts/train.py`，由它适配数据并调用 LeRobot ACTPolicy。统一 YAML 在 [act_click_bell.yaml](../robosyn_act/configs/act_click_bell.yaml)，随 wheel 一起安装。历史归档配置在 experiments 中，包含 provenance，不能当作启动 YAML 使用。

```bash
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
export ACT_PYTHON=/path/to/existing/venv/bin/python
export DATASET_ROOT=/path/to/cobotmagic_Sim_click_bell
export RUN_DIR="$PWD/outputs/train/act_click_bell_bs24_80k"
# 必须是尚不存在的新 run 目录
python -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR" --dry-run
python -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR"
```

默认 steps=80000、batch=24、workers=4、chunk=50、n_action_steps=50、log_freq=100、save_freq=10000、seed=1000、device=cuda、GPU=0；AMP/W&B 默认关闭。seed 是核查版本上游默认值，历史命令没有显式设置它。CLI 参数 > 显式 profile > YAML；默认不应用 profile。`--profile smoke` 改为 10 步、batch=1、chunk=16、n_action_steps=8、save=10、log=1，仍需要真实数据和 ACT 环境；它不同于 CPU `smoke` 子命令。

显式 dataset-root 已足以定位数据。若需配置其他 LeRobot 缓存，可按本机目录设置 `HF_LEROBOT_HOME`，无需原服务器的固定绝对路径。相对数据、输出、checkpoint 路径以调用 cwd 解析后传绝对路径。也可用 `bash scripts/train.sh ...`；`BASELINE_PYTHON` 选择工具解释器，`ACT_PYTHON` 选择实际训练解释器，两者可以不同。

历史显存测试来自所有者记录，非本次重跑：batch16 ≈12.6GB，batch32 ≈24.1GB，batch24 ≈18.4GB。最终选24；显存随软件版本、图像、AMP等变化，不能保证任何 4090 都稳定。历史记录提到官方示例batch64；核查版本 train.py 的默认实际是8，应区分示例与脚本默认。

后台运行示例（先执行 dry-run；日志放在 run 目录之外，以免提前创建输出目录）：

```bash
mkdir -p logs
nohup python -u -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR" \
  > logs/act_click_bell_80k.log 2>&1 &
```

历史命令使用过 `--overwrite`，上游会删除已有输出。这里不提供该开关，已有目录直接报错；换一个 run 名称即可。

## Checkpoint semantics

LeRobot [train_utils.py](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/src/lerobot/utils/train_utils.py) 按保存频率及最终 step 保存，并更新 `last` 符号链接。80k run 的目录是六位数字：010000、020000、030000、040000、050000、060000、070000、080000。

```text
checkpoints/080000/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   └── train_config.json
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── rng_state.safetensors
    └── training_step.json
```

推理加载 pretrained_model 的 policy config 和 weights；train_config 是训练配置。恢复完整训练还需同级 training_state 中的 optimizer、RNG、step，以及有 scheduler 时的 scheduler 状态。只加载模型做新训练是 warm start，不等于恢复 optimizer/RNG 的 resume。

指定 step 的评测可将 `--checkpoint` 指向 `"$RUN_DIR/checkpoints/040000/pretrained_model"`。不依赖不明确的 latest。

## Resume limitation

当前官方包装器有 `--resume`，但其 argparse 没有 `config_path`。指定 LeRobot 的 [TrainPipelineConfig.validate](https://github.com/huggingface/lerobot/blob/b883328e6c95681ca90a18b102e4ae5e1f91e2bf/src/lerobot/configs/train.py) 明确要求恢复时提供该路径。直接添加 `--resume` 不构成可运行的恢复流程；直接绕过包装器还会丢掉 legacy alias patch。本项目暂不开放 resume。接入需同时恢复保存配置、模型、optimizer/RNG/step、保留数据适配，再用中断前后数值连续性测试验证，不能覆盖现有80k成果。

## Weights publication

自训练80k权重：**待上传，公开 URL 未提供**。普通 Git 忽略 safetensors/pt/pth。建议将 pretrained_model 上传到 Hugging Face 模型仓库，发布 model card、训练配置、许可证、SHA256和本地评测限定；需要恢复训练时另行保管 training_state。拿到真实模型仓库 ID 后才能填入下载示例，当前不虚构地址。

现有 `download checkpoint --task click_bell` 获取的是官方 `RoboSynChallenge/ACT_sim_click_bell`，不是本团队80k权重，不可用它声称复现本次7/10结果。
