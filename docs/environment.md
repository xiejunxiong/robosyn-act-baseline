# ACT 环境与上游版本

接口核对日期：2026-10-17。

| 组件 | 本项目依据 |
| --- | --- |
| RoboSynChallenge | `64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6` |
| LeRobot | 官方 ACT 配置中的 `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` |
| Python（训练/仿真） | 推荐 3.11，官方 ACT 约束 `>=3.10,<3.12` |
| EmbodiChain | 官方安装文档的 `v0.2.4.post1`，另须满足 ACT uv.lock |
| DexSim / Torch / torchvision | 由官方 ACT `uv.lock` 解析，勿单独升级 |

启动工具没有 Torch 依赖，在 Mac 上也可检查数据和命令；实际训练在 NVIDIA Linux 环境执行。

## 已有比赛镜像

1. 找到镜像内的 RoboSynChallenge、EmbodiChain 目录及 ACT Python。
2. 用 `git -C /path/to/RoboSynChallenge rev-parse HEAD` 检查源码版本。若不匹配，保留旧目录和数据，另建固定版本副本。启动器会拒绝不匹配的版本，因为命令行和适配器可能变化。
3. 指定解释器并检查依赖：

```bash
export ROBOSYN_ROOT=/root/workspace/RoboSynChallenge
export ACT_PYTHON="$ROBOSYN_ROOT/policy/act/.venv/bin/python"
"$ACT_PYTHON" -c "import sys, torch, lerobot, embodichain, dexsim; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

平台“无卡模式”可用于准备文件，但不能运行 CUDA 训练或 GPU 仿真。带卡实例启动后检查 `nvidia-smi` 和上述 CUDA 输出。

## 新的 Linux GPU 工作区

在新目录操作，不在已有实验目录 checkout/reset。需要 Git、Python 3.11、uv、适配的 NVIDIA 驱动及 Vulkan/仿真依赖。先阅读[官方安装说明](https://github.com/EDEM-AI/RoboSynChallenge/blob/64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6/docs/getting_started/installation.md)完成系统和 EmbodiChain 依赖。

官方 ACT 锁文件引用相邻的 EmbodiChain 目录：

```text
workspace/
├── EmbodiChain/
├── RoboSynChallenge/policy/act/.venv/
└── robosyn-act-baseline/
```

```bash
# 从空工作区开始
git clone https://github.com/DexForce/EmbodiChain.git
git -C EmbodiChain checkout v0.2.4.post1
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
git -C RoboSynChallenge checkout 64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6
cd RoboSynChallenge/policy/act
uv sync --frozen --python 3.11
```

官方配置通过供应商 HTTP 包索引提供 DexSim；请在受信任网络中依照官方说明安装并核对锁文件，不要将无关环境全局配置为不验证 TLS。该流程依赖外部二进制包服务可用性，离线 CI 不验证安装过程。

安装后运行依赖检查及 README 的 smoke training。若锁文件安装失败，请保留错误信息，对照固定版本官方说明排查，避免随意升级 LeRobot。

## 常见问题

- `Cannot read JSON .../meta/info.json`：数据路径应指向数据集本身，而非外层 `data/`。
- `ACT requires LeRobot v2.1`：先运行官方转换脚本，不要只修改版本号。
- `Python executable not found`：用 `--python /absolute/path/to/act/env/bin/python` 指定已有环境。
- `Output already exists`：换一个 run 目录，启动器不会删除已有文件。
- `Missing model.safetensors`：选择 `pretrained_model/` 或 HF 模型根目录。
- `CUDA out of memory`：减小 batch size，再考虑减小 chunk size 或官方多 GPU 训练。
- 本入口不提供 `--resume`：恢复涉及已保存配置和优化器状态，请备份后按官方流程操作。
