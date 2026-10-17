# Preliminary Failure Analysis

以下来自所有者历史观察，原视频尚待接入核验，属于初步假设而非经消融验证的因果结论。

| 报告局号 | 历史失败视频 | Action steps | Model calls |
| ---: | --- | ---: | ---: |
| 2 | episode_001_seed_924231285_fail.mp4 | 361 | 8 |
| 5 | episode_004_seed_192771779_fail.mp4 | 361 | 8 |
| 8 | episode_007_seed_530702035_fail.mp4 | 361 | 8 |

三个失败都超时。观察记录称模型能到按钮附近，但末端位置误差、按压深度不足或对齐偏差可能影响触发，失败后通常未有效纠偏。成功局通常47–48步、一次模型调用。不能仅凭调用数判定视觉、控制或训练分布中的哪一项是唯一原因。

值得检验的解释包括 action chunk 开环执行误差、闭环重规划频率、模仿学习 covariate shift、失败恢复不足和视觉随机化。需要同一组可行 seeds、配对评测及更大样本来比较。

## Roadmap — not yet completed

- **A / checkpoint comparison**：20k、40k、60k、80k，同一批 seeds与设置，观察成功率是否单调提高；不预设更晚模型更好。
- **B / action execution horizon**：同一权重和chunk_size=50，比较n_action_steps=50/25/10。指定LeRobot在select_action里裁切chunk并建立有长度上限的队列，理论上这些值满足约束；但当前官方get_model只转交device，未开放此override。后续需在模型初始化前加载修改后的配置、重建/reset队列并确认temporal ensemble关闭，验证数值/动作后才能评测。不能把act_step当作n_action_steps，也不能在线只改一个属性就声称正确。
- **C / videos**：对比目标定位、末端对齐、按压深度、动作连续性和纠偏；记录可见证据与反例。
- **D / training loss**：action reconstruction/L1、KL、total，当前均为 **Not recorded / Pending**。保留原日志后再提取；没有数据不生成曲线。固定代码暴露l1_loss/kld_loss不等于历史日志一定记录了它们。

所有新实验应同时记录checkpoint hash、dataset revision、上游commit、seed列表、筛选设置、软件/驱动版本。原10局70%是局部样本结果，不足以证明总体性能或比赛名次。
