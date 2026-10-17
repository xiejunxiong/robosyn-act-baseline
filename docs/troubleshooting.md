# Troubleshooting

| 现象 | 可核对的处理方向与限制 |
| --- | --- |
| uv.lock 私有 dexsim-engine wheel hash 不一致 | 尚未解决。保存完整错误、wheel来源和环境；找供应方核对发布文件/锁文件。不要篡改hash或关闭校验。历史训练用已可用venv直接调用入口。 |
| lerobot.scripts.train 缺失 | 检查实际ACT解释器；固定Git commit b883328e6c95681ca90a18b102e4ae5e1f91e2bf，不能仅安装最新LeRobot。 |
| datasets 4.x 不兼容 | 历史成功环境与官方锁文件均为3.6.0；在独立环境按对应版本恢复，避免升级已运行的镜像环境。 |
| ACT .venv未初始化 | 显式设置ACT_PYTHON或--python指向已验证venv；不要让shell恰好找到的python替代它。 |
| SDL version unsupported | 核对镜像SDL、Vulkan、DexSim构建与显示环境；没有证据表明某个pip升级能普遍修复。保留完整异常。 |
| Vulkan/DexSim初始化失败，Xvfb不在运行 | `pgrep -a Xvfb`、`xdpyinfo -display :99`确认服务；仅设置DISPLAY不够。 |
| /tmp/.X11-unix/X99残留 | 检查Xvfb/其他X server、端口、socket所有者及使用者；不要自动删除。无法确认是否在用时找管理员处理或选择经检查空闲的新display。 |
| Warp CUDA driver entry point警告 | 同时检查nvidia-smi、Torch CUDA及后续异常。警告是否影响功能需运行结果确认，不建议忽略所有CUDA错误。 |
| 显存不足 | 历史batch24稳定不保证跨版本一致；先降低batch到16/8，单独记录该实验配置。 |
| checkpoint找不到/路径歧义 | 指向具体step的pretrained_model；包含config.json和model.safetensors。启动器转换为绝对路径。 |
| embodichain_tasks导入失败 | 指定EMBODICHAIN_ROOT并通过--embodichain-root加入task目录；检查镜像任务包布局/安装情况。 |
| Output already exists | 使用新的RUN_DIR；不自动覆盖已有模型。不要提前mkdir RUN_DIR。 |
| Upstream revision differs | 保留原训练源码，另建固定commit副本；不要reset原实验目录。历史commit待核验。 |

最小诊断（不输出token或完整环境变量）：

```bash
nvidia-smi
"$ACT_PYTHON" -c 'import sys,torch,datasets,importlib.metadata as m; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print("datasets",datasets.__version__); print("lerobot",m.version("lerobot"))'
"$ACT_PYTHON" -c 'import lerobot.scripts.train; print("LeRobot train import OK")'
git -C "$ROBOSYN_ROOT" rev-parse HEAD
git -C "$EMBODICHAIN_ROOT" rev-parse HEAD
xdpyinfo -display "$DISPLAY"
```

`torch.version.cuda`是PyTorch构建版本，不等于本机nvcc或驱动版本。依赖探测成功也不能代替实际仿真评测。
