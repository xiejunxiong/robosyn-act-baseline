# Dataset and feature mapping

使用 [RoboSynChallenge/cobotmagic_Sim_click_bell](https://huggingface.co/datasets/RoboSynChallenge/cobotmagic_Sim_click_bell)。历史记录：LeRobot v2.1，1000 episodes、74000 frames、25 FPS、1 task、三相机各 640×480。dataset revision 尚未提供；下载同名仓库不保证字节级复现原始数据。

```bash
python -m pip install 'huggingface_hub[cli]>=0.34,<1'
python -m robosyn_act download sim --task click_bell --destination ./data/cobotmagic_Sim_click_bell
python -m robosyn_act check-data ./data/cobotmagic_Sim_click_bell \
  --config robosyn_act/configs/act_click_bell.yaml
# 等价检查脚本
python scripts/inspect_dataset.py ./data/cobotmagic_Sim_click_bell
```

Hugging Face 登录使用个人本地凭证；不要把 token 放入代码或命令示例。离线查看下载命令可加 `--dry-run`。

| 原始字段 | LeRobot ACT 输入 | 形状 |
| --- | --- | --- |
| observation.qpos | observation.state | [B, 14] |
| cam_high.color | observation.images.cam_high | [B, 3, 480, 640] |
| cam_left_wrist.color | observation.images.cam_left_wrist | [B, 3, 480, 640] |
| cam_right_wrist.color | observation.images.cam_right_wrist | [B, 3, 480, 640] |
| action | action | [B, 50, 14]，训练目标 chunk |
| action_is_pad | action_is_pad | [B, 50]，尾部 padding mask |

`observation.qvel`、`observation.qf` 各 14 维，保留在数据中，不自动拼入 ACT 的 14 维状态。磁盘视频元数据常为 HWC；LeRobot 解码后是浮点 CHW。alias 只处理名字，不对已经解码的图像再次转置/缩放；仿真原始图像到张量的转换由官方 deploy adapter 完成。

```text
cobotmagic_Sim_click_bell/
├── data/chunk-000/*.parquet
├── videos/chunk-000/...
└── meta/
    ├── info.json
    ├── stats.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

历史数据含上述文件。检查器验证 v2.1、必要 metadata 文件、至少一个 Parquet/视频、状态/动作和相机形状；显式 `--config` 还验证 fps/尺寸。不声称读过所有帧或检查过每个 episode 文件。LeRobot v2.1 使用 episodes_stats 聚合统计，因此不把额外 stats.json 作为所有 v2.1 数据集的硬性条件。

官方 adapter 会映射 features、stats、item 和对应 padding key；本地工具测试同一映射契约。已有 canonical key 优先；不要在混合命名数据上盲目重命名 metadata 或只改 codebase_version。核查版本的官方 metadata adapter 在需要任何 legacy alias 时会先移除图像字段，再加入 alias；已有 canonical 图像可能因此丢失。检查器拒绝这种混合数据，避免启动后无声丢相机。使用原始 legacy 命名或完全 canonical 的数据。

不支持直接训练 LeRobot v3。需要时参考官方 `scripts/convert_lerobot3.0_to_2.1.py --help` 转换至新目录，再检查。真实数据和下载目录不会提交到 Git。
