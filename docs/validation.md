# Delivery and validation record

维护日期：2026-10-17。本次只在本地CPU环境验证工程接口；历史80k训练/7-of-10评测来自所有者记录，来源和未核实项保存在 experiments 中。

## Complete source tree

```text
robosyn-act-baseline/
├── README.md
├── README_EN.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── robosyn_act/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset_adapter.py
│   ├── smoke.py
│   └── configs/act_click_bell.yaml
├── scripts/
│   ├── setup_env.sh
│   ├── train.sh
│   ├── eval.sh
│   ├── smoke_test.sh
│   └── inspect_dataset.py
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_dataset_adapter.py
│   └── test_smoke.py
├── experiments/click_bell_80k/
│   ├── config.yaml
│   ├── results.json
│   └── README.md
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── environment.md
│   ├── training.md
│   ├── evaluation.md
│   ├── troubleshooting.md
│   ├── failure_analysis.md
│   └── validation.md
└── .github/workflows/tests.yml
```

保留已有平铺包结构；CLI承载train/eval入口，避免创建只转发调用的空模块。YAML在包内以保证wheel安装后可访问。

## Changes relative to the earlier launcher commit

- 更新：README.md、docs/environment.md、pyproject.toml、robosyn_act/__init__.py、robosyn_act/cli.py、tests/test_cli.py、.gitignore、.github/workflows/tests.yml。
- 新增：README_EN.md；config.py、dataset_adapter.py、smoke.py及包内configs/act_click_bell.yaml；5个scripts；test_config.py、test_dataset_adapter.py、test_smoke.py；3个experiment文件；architecture/dataset/training/evaluation/troubleshooting/failure_analysis/validation文档。
- LICENSE、__main__.py保留。未上传真实数据、大权重、视频、个人绝对路径或凭证。

## Actual local verification

| Check | Result | Scope |
| --- | --- | --- |
| Editable install in isolated Python3.9 venv | PASS | PyYAML6.0.3、NumPy1.26.4、CLI |
| python -m unittest discover -s tests -v | PASS, 28 tests | YAML/CLI优先级、数据契约/aliases、旧新相机混合拒绝、形状、metadata、输出保护、退出码、记录算术 |
| python -m robosyn_act smoke | PASS | NumPy合成数据/接口检查；real_act_forward=false |
| CLI help、config、tasks | PASS | 无GPU |
| bash -n on 4 shell scripts | PASS | shell语法，不是GPU运行 |
| Actual pinned upstream API audit and eval dry-run | PASS | 训练flags/aliases比对源码；真实上游checkout + 合成metadata/checkpoint文件，未加载权重 |
| Wheel build + packaged YAML presence | PASS | 补齐临时环境setuptools/wheel后完成 |
| python -m robosyn_act smoke --act | NOT RUN: dependency gate failed, exit2 | 本地缺少LeRobot/Torch，未执行真实ACT forward |
| GPU训练、DexSim评测、80k权重加载 | NOT RUN | 本次未连接GPU服务器，原始资产未提供 |
| Fresh uv/DexSim installation | NOT VERIFIED | 历史私有wheel hash问题未解决 |

首轮测试有一次macOS `/var`与`/private/var`临时目录断言失败，修正测试为比较resolved路径后28项全通过。首次非隔离wheel构建缺少bdist_wheel，补齐临时构建工具后成功。没有将这些初始失败或依赖限制计为模型验证。

GitHub Actions配置Python3.9和3.11，实际每次运行结果见PR Checks；本地Python3.9结果不等同于CI3.11结果。

## Minimum final checks on the RTX4090 server

前提：已有可工作的ACT venv、真实数据、80k checkpoint、EmbodiChain/DexSim资源。保留原训练目录；将ROBOSYN_ROOT指向已核查commit的独立源码副本。不要为验证重新跑80k。

```bash
# 在本仓库根目录，PR合并前使用codex/complete-act-baseline分支
bash scripts/setup_env.sh
source .venv-tools/bin/activate
export ROBOSYN_ROOT=/path/to/pinned/RoboSynChallenge
export ACT_PYTHON=/path/to/working/venv/bin/python
export EMBODICHAIN_ROOT=/path/to/EmbodiChain
export DATASET_ROOT=/path/to/cobotmagic_Sim_click_bell
export CHECKPOINT=/path/to/checkpoints/080000/pretrained_model
export DISPLAY=:99

nvidia-smi
"$ACT_PYTHON" -c 'import torch,datasets,lerobot.scripts.train; print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),datasets.__version__)'
git -C "$ROBOSYN_ROOT" rev-parse HEAD
python -m robosyn_act check-data "$DATASET_ROOT" --config robosyn_act/configs/act_click_bell.yaml
python -m robosyn_act smoke
# 仅当:99已确认空闲且无残留问题时，在另一个终端运行：
# Xvfb :99 -screen 0 1280x720x24
xdpyinfo -display "$DISPLAY"
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display "$DISPLAY" --dry-run
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display "$DISPLAY"
```

这会运行10局本地评测。结果在官方源码eval_result下，保留新JSON和视频，与历史转录独立存档。若想先测试训练链路，再增加10步训练（新目录，绝不覆盖历史run）：

```bash
python -m robosyn_act train --config robosyn_act/configs/act_click_bell.yaml \
  --profile smoke --dataset-root "$DATASET_ROOT" --output-dir ./outputs/validation_smoke
```

完整80k新实验命令见README或training.md。真正CPU ACT forward需要在有固定LeRobot和本项目的兼容环境中执行`python -m robosyn_act smoke --act`；当前工具venv本身没有模型依赖。

## Remaining technical work

1. 接入原始evaluation_metrics.json、训练日志、视频、模型config和权重hash，核对历史源码、数据revision和所有seeds。
2. 在HF发布真实80k模型及model card、补下载链接；当前没有可提供的自训练模型公开URL。
3. 解决供应方DexSim wheel hash/依赖组合，取得可从零复现的环境快照。
4. 验证真实ACT smoke、GPU10步训练和10局仿真；完成后如实附日志。
5. 实现并验证resume和安全n_action_steps覆盖；完成checkpoint/horizon对比、更大样本评测、视频分析和可靠loss采集。
