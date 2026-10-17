# click_bell / ACT / 80k

## Local Simulation Evaluation (10 episodes)

| Task | Model | Steps | Batch | Success |
| --- | --- | ---: | ---: | --- |
| click_bell | ACT | 80k | 24 | 7/10 (70%) |

来源：项目所有者于 2026-10-17 提供的历史实验记录，实验记录时间为 **2026-10-04 21:26:49**（原记录未注明时区）。目前尚未取得原始 metrics、日志、权重和视频进行独立复核；`results.json` 是带来源标注的转录文件，**不是原始 evaluation_metrics.json**。

团队 JXNU SynDex 已报告完成真实训练与本地仿真评测；未在截止时间前提交有效官方评测申请，因此没有可据此报告的官方比赛成绩、排行榜分数或最终名次。之前云端的合成数据 MLP/CVAE smoke 是另一次实验，不能与此 ACT Transformer 实验混用。

配置：random，80k checkpoint，seed=0，最大 361 action steps。专家可行性筛选接受 12 个候选 seed 中的 10 个；成功率的分母是接受的 10 局，不是 12 个候选。仅 10 局的小样本不支持对总体成功率作强结论。

| Episode（1-based） | Result | Action steps | Model calls |
| ---: | --- | ---: | ---: |
| 1 | SUCCESS | 48 | 1 |
| 2 | FAIL | 361 | 8 |
| 3 | SUCCESS | 48 | 1 |
| 4 | SUCCESS | 47 | 1 |
| 5 | FAIL | 361 | 8 |
| 6 | SUCCESS | 47 | 1 |
| 7 | SUCCESS | 48 | 1 |
| 8 | FAIL | 361 | 8 |
| 9 | SUCCESS | 48 | 1 |
| 10 | SUCCESS | 47 | 1 |

平均 action steps **141.60 / 361**；平均推理延迟 **41.241 ms / model call**；平均每局总推理时间 **127.848 ms**；共 **31** 次模型调用，平均 **3.10 / episode**。延迟的历史计时范围待原始代码与 metrics 核对；当前固定版本包含观察预处理/传输到可执行 action，排除 `env.step`，不是整局 wall-clock 时间。

记录称已生成 `010000` 至 `080000` 的 8 个 checkpoint。最终目录为 ACT 工作目录下 `outputs/train/act_click_bell_bs24_80k/checkpoints/080000/pretrained_model`，包含 `config.json`、`model.safetensors`、`train_config.json`；同级 `training_state/` 包含优化器、RNG 和 step 状态。**自训练权重尚无公开地址，待上传 Hugging Face 模型仓库。**

原 metrics 相对官方仓库的位置：

```text
eval_result/click_bell/act/random/None/pretrained_model/2026-10-04_21-26-49/evaluation_metrics.json
```

原日志文件名 `act_click_bell_80k.log`，训练曾用 nohup 后台执行。用户服务器的绝对路径不作为本项目运行前提。

原始材料接入清单：保留原 JSON/完整日志、所有 accepted/skipped seeds、GPU 驱动与 `pip freeze`、上游 Git commit/本地修改、dataset revision、checkpoint SHA256。原 JSON 可放入 `experiments/click_bell_80k/raw/`，先检查个人路径与凭证；大权重和视频放外部资产仓库并记录地址与 hash。历史记录暂不补写未提供的 torchvision、NumPy、训练 seed 或训练 loss；loss 状态为 **Not recorded / Pending**。

下一轮采用[统一运行配置](../../robosyn_act/configs/act_click_bell.yaml)，它固定训练 seed=1000（当前上游默认），并不证明历史 seed 已经核验。当前工具核对的 RoboSynChallenge revision 也不等于已知历史训练 revision。详见[失败分析与实验计划](../../docs/failure_analysis.md)。
