# Environment: reference, historical run, unresolved gaps

接口核查日期2026-10-17；历史环境来自项目所有者记录，未取得服务器完整pip freeze。工具安装与ACT/仿真安装分开，`scripts/setup_env.sh`仅安装轻量工具依赖，不安装或修改DexSim/ACT环境。

| Component | 官方参考 | 历史运行记录 |
| --- | --- | --- |
| RoboSynChallenge | 本工具核查64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6 | 历史commit未提供 |
| Python | >=3.10,<3.12 | 3.11.15 |
| Torch | uv.lock 2.7.1，pyproject要求<2.8 | 2.10.0+cu128 |
| torchvision | 0.22.1 | Pending |
| NumPy | 1.26.4 | Pending |
| LeRobot | b883328e6c95681ca90a18b102e4ae5e1f91e2bf | 同commit，0.3.3 |
| datasets | 3.6.0 | 3.6.0 |
| EmbodiChain | v0.2.4.post1 | 89c14da8d |
| DexSim | 0.4.3 | 0.4.3 |

历史服务器：NVIDIA RTX4090 24GB、Intel Xeon Gold6430、Ubuntu22.04、Linux5.15。CUDA12.8是Torch构建版本，驱动版本需另存。Torch2.10不在当前官方ACT约束内；这是历史成功运行记录，不应将它描述为官方支持矩阵。

## Existing competition image — historical route

保留已经训练成功的venv，显式设置`ACT_PYTHON=/path/to/existing/venv/bin/python`。检查`import lerobot.scripts.train`、Torch CUDA、datasets3.6.0以及任务包导入。不要盲目对现有环境执行pip升级、uv sync或源码reset。工具CLI运行在单独venv，`--python`选择训练/仿真解释器。

历史遇到官方uv.lock的私有dexsim-engine包hash校验不一致；实际用已有venv直接执行官方训练入口。**哈希问题未解决；从零安装的完整复现尚未验证。** 不能篡改锁文件hash、忽略校验或宣称私有wheel公开可安装。

## Fresh workspace — reference only, not a verified installer

在新目录取得源码，不覆盖历史实验目录：

```bash
git clone https://github.com/DexForce/EmbodiChain.git
# 新clone内切换记录中的commit
git -C EmbodiChain checkout 89c14da8d
git clone https://github.com/EDEM-AI/RoboSynChallenge.git
git -C RoboSynChallenge checkout 64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6
```

参考[官方安装文档](https://github.com/EDEM-AI/RoboSynChallenge/blob/64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6/docs/getting_started/installation.md)配置NVIDIA/Vulkan、仿真资源和供应方授权的DexSim。官方ACT环境位于RoboSynChallenge/policy/act，uv配置引用相邻EmbodiChain。官方参考命令为该目录中的`uv sync --frozen --python 3.11`，但历史hash失败意味着这里不能承诺它能完成。

进一步冲突：核查EmbodiChain commit89c14da8d的pyproject声明`lerobot>=0.4.4`，而ACT通过uv override固定0.3.3的Git commit。直接把所有仓库逐个pip install可能升级LeRobot，导致旧入口缺失。不要依赖普通pip自动解析出历史环境。需供应方协助确认组合、核对完整环境快照，并在独立新环境验证。

## Optional real ACT CPU smoke

`python -m robosyn_act smoke`无Torch要求。`--act`需要已经装有指定LeRobot Git commit的兼容Torch环境；验证pip的direct_url.json中的commit ID，不能只凭0.3.3版本字符串判断。此测试无需DexSim，不下载预训练ResNet权重，采用缩小的Transformer维度但真实ACTPolicy/CVAE与loss。editable本地LeRobot安装缺少VCS元数据时也会拒绝，需先明确安装来源，不能伪造元数据。

若选用独立CPU测试环境，可以从固定源码VCS安装LeRobot（不在历史GPU环境升级）：

```bash
# 在单独、兼容的Python >=3.10 venv中；依赖安装仍受网络/平台可用性影响
python -m pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git@b883328e6c95681ca90a18b102e4ae5e1f91e2bf'
python -m pip install -e /path/to/robosyn-act-baseline
python -m robosyn_act smoke --act
```

Mac的CPU smoke不代表Linux仿真已验证。实际训练/仿真仍按Python3.11镜像及已有venv操作。当前执行结果详见[validation](validation.md)。
