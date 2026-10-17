# RoboSynChallenge ACT Baseline

[中文](README.md) · [Experiment record](experiments/click_bell_80k/README.md) · [Environment](docs/environment.md)

JXNU SynDex's ACT reproduction project for RoboSynChallenge 2026 (NeurIPS 2026), using the CobotMagic dual-arm robot. This repository provides validated launchers, a shared YAML config, feature-contract helpers, CPU smoke checks, and historical experiment records. The model is the pinned LeRobot ACTPolicy, not a newly implemented Transformer or the earlier synthetic MLP prototype.

## Local Simulation Evaluation (10 episodes)

| Task | Model | Steps | Batch | Success |
| --- | --- | ---: | ---: | --- |
| click_bell | ACT | 80k | 24 | 7/10 (70%) |

**Local simulation only, not an official competition score, leaderboard result, or ranking.** The team did not submit a valid official evaluation request before the deadline. These numbers are transcribed from owner-provided historical notes, recorded at 2026-10-04 21:26:49 (timezone unspecified). Original metrics, logs, weights, and videos have not yet been independently inspected in this maintenance session. Ten episodes are not a high-confidence estimate of population performance.

The owner reports completing 80,000 steps, saving checkpoints at every 10k step, loading the self-trained 80k model, and evaluating ten random episodes. Only click_bell has a recorded project experiment. Other upstream tasks being exposed by the CLI does not mean they have been reproduced.

ACT predicts action chunks from camera observations and robot state, with a CVAE objective during training. See the [ACT paper](https://arxiv.org/abs/2304.13705) and [architecture/API audit](docs/architecture.md). Production training and simulation call the official adapters.

## Environment and data

The launcher uses Python >=3.9, PyYAML and NumPy. GPU training/simulation uses a separate Linux/Python3.11 environment. Historical hardware: RTX4090 24GB, Xeon Gold6430, Ubuntu22.04/Linux5.15.

| Component | Official reference | Historical run |
| --- | --- | --- |
| Python | >=3.10,<3.12 | 3.11.15 |
| PyTorch | lock: 2.7.1 | 2.10.0+cu128 |
| torchvision / NumPy | 0.22.1 / 1.26.4 | Versions pending |
| LeRobot | b883328e6c95681ca90a18b102e4ae5e1f91e2bf | Same commit, 0.3.3 |
| datasets | 3.6.0 | 3.6.0 |
| EmbodiChain | v0.2.4.post1 | 89c14da8d |
| DexSim | 0.4.3 | 0.4.3 |

Historical Torch is outside the official ACT dependency range. The private dexsim-engine wheel hash mismatch is **unresolved**. The working run used an existing venv directly; a clean `uv sync --frozen` installation is not claimed. EmbodiChain's declared LeRobot dependency also conflicts with the ACT pin; see [environment notes](docs/environment.md) before installing anything in the working GPU environment.

The launcher pins RoboSynChallenge to `64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6` for its audited interface. The historical RoboSynChallenge commit, dataset revision, full environment snapshot, and checkpoint hashes remain pending.

Dataset: [RoboSynChallenge/cobotmagic_Sim_click_bell](https://huggingface.co/datasets/RoboSynChallenge/cobotmagic_Sim_click_bell), LeRobot v2.1; reported 1,000 episodes, 74,000 frames, 25 FPS, one task, three 640×480 cameras, and 14-dimensional states/actions.

| Original feature | ACT feature |
| --- | --- |
| observation.qpos | observation.state |
| cam_high.color | observation.images.cam_high |
| cam_left_wrist.color | observation.images.cam_left_wrist |
| cam_right_wrist.color | observation.images.cam_right_wrist |

qvel/qf are retained but not concatenated into the ACT state. The official adapter aliases metadata, statistics, samples, and padding masks. LeRobot decodes images to CHW; aliases do not transpose them again. [Dataset details](docs/dataset.md).

## Install and run

```bash
git clone https://github.com/xiejunxiong/robosyn-act-baseline.git
cd robosyn-act-baseline
git checkout codex/complete-act-baseline  # Until the PR is merged.
bash scripts/setup_env.sh
source .venv-tools/bin/activate
bash scripts/smoke_test.sh
```

The default `smoke` checks synthetic NumPy arrays/config/aliases/shapes only. `python -m robosyn_act smoke --act` requires a pip VCS installation of the exact LeRobot commit and compatible Torch. It runs actual ACTPolicy forward, loss, backward and select_action with reduced CPU architecture, random weights, and no pretrained downloads. Missing dependencies cause an explicit error; no mock result is presented as a real model test.

For an already functional competition environment, substitute local paths:

```bash
export ROBOSYN_ROOT=/path/to/pinned/RoboSynChallenge
export ACT_PYTHON=/path/to/existing/venv/bin/python
export EMBODICHAIN_ROOT=/path/to/EmbodiChain
export DATASET_ROOT=/path/to/cobotmagic_Sim_click_bell
export RUN_DIR="$PWD/outputs/train/act_click_bell_bs24_80k"
python -m robosyn_act check-data "$DATASET_ROOT" --config robosyn_act/configs/act_click_bell.yaml
python -m robosyn_act train --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR" --dry-run
python -m robosyn_act train --dataset-root "$DATASET_ROOT" --output-dir "$RUN_DIR"
export CHECKPOINT="$RUN_DIR/checkpoints/080000/pretrained_model"
# Ensure a real Xvfb service is running on :99 first.
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display :99 --dry-run
python -m robosyn_act eval --checkpoint "$CHECKPOINT" --display :99
```

Defaults: 80k steps, batch24, workers4, chunk50, n_action_steps50, log100, save10000; train seed1000 is the audited wrapper default, not an independently verified historical seed. Evaluation: click_bell/random/seed0/10 episodes/361 action steps. CLI flags override an explicitly selected profile, which overrides YAML. Existing output directories are rejected. A dry-run checks metadata/paths and prints commands, without running the model or simulator.

Download data with `python -m robosyn_act download sim --task click_bell --destination ./data/cobotmagic_Sim_click_bell` after installing the Hugging Face CLI. Self-trained 80k weights have **no public URL yet; upload pending**. The `download checkpoint` command downloads official weights, not this team's checkpoint. Large models/data/videos are excluded from ordinary Git.

See [training/checkpoints](docs/training.md), [evaluation](docs/evaluation.md), [troubleshooting](docs/troubleshooting.md), and [minimal RTX4090 validation](docs/validation.md). The pinned wrapper's resume flag lacks the config-path chain required by LeRobot, so this project does not advertise working resume support.

## Findings and roadmap

Reported metrics: mean141.60/361 action steps, 41.241ms/model call, 127.848ms total inference/episode, 31 calls in total, 3.10calls/episode, expert feasibility filter accepting10/12 seeds. Seven successes took47–48 steps and one call each; three failures timed out at361 steps with eight calls each. [Per-episode archive](experiments/click_bell_80k/results.json) is a provenance-labelled transcription, not the original metrics file.

Preliminary video observations suggest possible alignment/press-depth errors and weak recovery; no unique cause is established. Planned, not completed: compare20k/40k/60k/80k; compare execution horizons50/25/10 with unchanged weights after validating safe inference configuration; analyze failure videos; recover L1/KL/total training losses. Loss data are **Not recorded / Pending**, with no fabricated curves. [Failure analysis](docs/failure_analysis.md).

Original metrics/videos/weights, historical source versions, clean environment reproduction, and real GPU validation of these launchers remain pending. The original code in this repository remains [MIT](LICENSE); upstream code and proprietary simulation/data/model assets retain their own licenses. Thanks to [RoboSynChallenge](https://github.com/EDEM-AI/RoboSynChallenge), [LeRobot](https://github.com/huggingface/lerobot), [EmbodiChain](https://github.com/DexForce/EmbodiChain), and the ACT authors.
