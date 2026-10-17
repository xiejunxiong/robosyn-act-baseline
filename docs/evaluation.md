# Local simulation evaluation

本地评测复用官方 ACT `eval.sh → scripts/eval_policy.py → deploy_policy.py`。这些程序在本地运行，不会提交比赛官方评测申请。已有7/10记录详见[实验档案](../experiments/click_bell_80k/README.md)。

先准备固定源码、可用的 ACT Python、EmbodiChain/DexSim、NVIDIA驱动与Vulkan。无桌面服务器需要真实运行的 Xvfb；设置 DISPLAY 字符串本身不会启动显示服务。

```bash
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
export ACT_PYTHON=/path/to/existing/venv/bin/python
export EMBODICHAIN_ROOT=/path/to/EmbodiChain
export CHECKPOINT=/path/to/checkpoints/080000/pretrained_model
export DISPLAY=:99
# 先检查；仅在确认 :99 没有正在使用且没有待排查的残留 socket 时启动
pgrep -a Xvfb
xdpyinfo -display "$DISPLAY"
# 若需要新建服务，在另一个终端保持运行：
# Xvfb :99 -screen 0 1280x720x24

python -m robosyn_act eval --checkpoint "$CHECKPOINT" \
  --embodichain-root "$EMBODICHAIN_ROOT" --display "$DISPLAY" --dry-run
python -m robosyn_act eval --checkpoint "$CHECKPOINT" \
  --embodichain-root "$EMBODICHAIN_ROOT" --display "$DISPLAY"
```

默认 click_bell/random/seed0/10 episodes/max_steps361/GPU0；与训练seed分开。默认 headless，`--gui` 可开启显示。headless 不代表当前 DexSim 绝不需要有效的显示/Vulkan上下文。默认开启上游专家可行性筛选和视频记录，筛掉的 seed 不计入成功率分母。不要只比较百分比而忽略筛选设置和具体 seeds。

包装器检查 checkpoint 的 type=act 和 model.safetensors 存在，以绝对路径传入。`--embodichain-root` 加入历史运行需要的 task PYTHONPATH，上游脚本再加入官方仓库/policy路径；未修改安装好的 ACT venv。dry-run 只验证路径和参数，不导入仿真、不读取权重张量、不证明GPU可用。

视频/JSON由官方代码写入 `$ROBOSYN_ROOT/eval_result/`。用以下命令找到新结果，并保留控制台日志和软件版本：

```bash
find "$ROBOSYN_ROOT/eval_result/click_bell" -name evaluation_metrics.json -print
```

单局数据包含成功标记、action steps、模型调用、seed；视频 episode 编号从0开始，因此 `episode_001` 对应报告中的第2局。当前代码推理计时包含观测预处理/传输到可执行动作，排除 env.step；不能把 inference ms 当完整模拟一局的耗时。当前源码对部分 seed 重置/计时已有实现；历史版本未知，精确复现实验还需原始 commit 和 metrics。

CLI 仍可列出上游11个任务（`python -m robosyn_act tasks`），但**仅 click_bell 有本项目历史实验记录**。open_pan 配置路径映射为 other_tasks/open_pan，其他任务使用上游时间上限或显式 `--max-steps`，不要把click_bell的361步套给所有任务。
