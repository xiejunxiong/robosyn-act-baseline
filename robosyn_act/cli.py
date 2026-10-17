"""Validated launchers for the pinned official ACT implementation (stdlib only)."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

UPSTREAM_REVISION = "64b0775eddb3ed8c4c3d1212f322e8a9d036e9d6"
TASKS = (
    "click_bell", "handle_basket", "water_pouring", "table_rearrangement",
    "items_handover", "drawer_open_place", "mixer_operating", "item_assembly",
    "manipulate_pipette", "sample_loading", "open_pan",
)
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
PROFILES = {
    "smoke": {"steps": 10, "batch_size": 1, "chunk_size": 16,
              "n_action_steps": 8, "save_freq": 10, "log_freq": 1},
    "4090": {"steps": 80000, "batch_size": 8, "chunk_size": 50,
             "n_action_steps": 50, "save_freq": 10000, "log_freq": 100},
}


def resolved(value):
    return Path(value).expanduser().resolve()


def read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def validate_dataset(root):
    """Check the v2.1 metadata contract, not the contents of every video/frame."""
    root = resolved(root)
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("ACT requires LeRobot v2.1 data; use the official converter for v3 data.")
    features = info.get("features", {})
    if not isinstance(features, dict) or not all(isinstance(ft, dict) for ft in features.values()):
        raise ValueError("Dataset features must map names to feature objects.")
    state_key = next((key for key in ("observation.state", "observation.qpos")
                      if key in features), None)
    if state_key is None or "action" not in features:
        raise ValueError("Dataset requires observation.state (or observation.qpos) and action.")
    for key in (state_key, "action"):
        shape = features[key].get("shape", [])
        if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or shape[0] <= 0:
            raise ValueError(f"{key} must have a positive one-dimensional feature shape.")
    for camera in CAMERAS:
        key = next((k for k in (f"observation.images.{camera}", f"{camera}.color")
                    if k in features), None)
        if key is None or features[key].get("dtype") not in ("video", "image"):
            raise ValueError(f"Missing image/video feature for {camera}.")
    if type(info.get("fps")) not in (int, float) or not math.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("Dataset fps must be positive.")
    for key in ("total_episodes", "total_frames"):
        if type(info.get(key)) is not int or info[key] <= 0:
            raise ValueError(f"Dataset {key} must be positive.")
    if not any((root / "data").rglob("*.parquet")):
        raise ValueError("No data/**/*.parquet found; download the complete dataset.")
    if any(ft.get("dtype") == "video" for ft in features.values()):
        if not any((root / "videos").rglob("*.mp4")):
            raise ValueError("Video features declared but no videos/**/*.mp4 found.")
    for filename in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        if not (root / "meta" / filename).is_file():
            raise ValueError(f"Missing meta/{filename}; download the complete v2.1 dataset.")
    return {"root": str(root), "episodes": info["total_episodes"],
            "frames": info["total_frames"], "fps": info["fps"],
            "state_dim": features[state_key]["shape"][0],
            "action_dim": features["action"]["shape"][0]}


def validate_checkpoint(path):
    path = resolved(path)
    config = read_json(path / "config.json")
    if config.get("type") != "act":
        raise ValueError("Checkpoint config.json must have type='act'.")
    if not (path / "model.safetensors").is_file():
        raise ValueError("Missing model.safetensors; pass the pretrained_model directory.")
    return path


def upstream_root(path):
    if not path:
        raise ValueError("Set --upstream or ROBOSYN_ROOT to your RoboSynChallenge checkout.")
    root = resolved(path)
    for name in ("policy/act/scripts/train.py", "policy/act/eval.sh", "policy/act/uv.lock"):
        if not (root / name).is_file():
            raise ValueError(f"Incomplete upstream checkout: missing {root / name}")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Upstream revision {revision} differs from tested interface {UPSTREAM_REVISION}. "
                         "Use a separate checkout at the pinned commit; do not reset your existing work.")
    return root


def policy_python(root, override):
    candidate = override or str(root / "policy/act/.venv/bin/python")
    executable = shutil.which(candidate)
    if executable is None:
        raise ValueError(f"Python executable not found: {candidate}. Set up the official ACT environment first.")
    # Preserve a virtualenv's executable path: resolving its symlink loses venv isolation.
    return os.path.abspath(executable)


def build_command(args):
    root = upstream_root(args.upstream)
    python = policy_python(root, args.python)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.command == "train":
        summary = validate_dataset(args.dataset_root)
        options = dict(PROFILES[args.profile])
        for key in options:
            if getattr(args, key) is not None:
                options[key] = getattr(args, key)
        if options["n_action_steps"] > options["chunk_size"]:
            raise ValueError("n-action-steps must not exceed chunk-size.")
        output = resolved(args.output_dir)
        dataset = resolved(args.dataset_root)
        if output == dataset or dataset in output.parents or output in dataset.parents:
            raise ValueError("Output directory must be separate from the dataset directory.")
        if output.exists():
            raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
        cmd = [python, str(root / "policy/act/scripts/train.py"),
               "--dataset-root", summary["root"], "--output-dir", str(output),
               "--device", args.device, "--num-workers", str(args.num_workers),
               "--seed", str(args.seed)]
        for key, value in options.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]
        if args.use_amp:
            cmd.append("--use-amp")
        if args.wandb:
            cmd += ["--wandb", "--wandb-project", "robosyn-act-baseline"]
    else:
        checkpoint = validate_checkpoint(args.checkpoint)
        task = "other_tasks/open_pan" if args.task == "open_pan" else args.task
        config_path = root / "configs" / task / args.setting / "gym_config.json"
        if not config_path.is_file():
            raise ValueError(f"Task config not found: {config_path}")
        # The official eval.sh accepts a policy interpreter through PYTHON_BIN.
        env["PYTHON_BIN"] = python
        cmd = ["bash", str(root / "policy/act/eval.sh"), task, args.setting,
               str(checkpoint), args.gpu, "--pytorch_device", args.device,
               "--headless", "False" if args.gui else "True",
               "--max_episodes", str(args.episodes), "--seed", str(args.seed)]
    return cmd, root, env


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("tasks", help="List the 11 supported task names")
    check = commands.add_parser("check-data", help="Validate local LeRobot v2.1 metadata and files")
    check.add_argument("dataset_root")
    download = commands.add_parser("download", help="Download official data or a released ACT checkpoint")
    download.add_argument("kind", choices=("sim", "real", "checkpoint"))
    download.add_argument("--task", choices=TASKS, required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--dry-run", action="store_true")
    for name in ("train", "eval"):
        sub = commands.add_parser(name)
        sub.add_argument("--upstream", default=os.environ.get("ROBOSYN_ROOT"))
        sub.add_argument("--python", help="Python from the official ACT environment")
        sub.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES; single-GPU launcher")
        sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        sub.add_argument("--seed", type=nonnegative, default=1000)
        sub.add_argument("--dry-run", action="store_true")
        if name == "train":
            sub.add_argument("--dataset-root", required=True)
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--profile", choices=PROFILES, default="4090")
            for key in PROFILES["4090"]:
                sub.add_argument("--" + key.replace("_", "-"), type=positive)
            sub.add_argument("--num-workers", type=nonnegative, default=4)
            sub.add_argument("--use-amp", action="store_true")
            sub.add_argument("--wandb", action="store_true")
        else:
            sub.add_argument("--task", choices=TASKS, required=True)
            sub.add_argument("--setting", choices=("clear", "random"), default="random")
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--episodes", type=positive, default=20)
            sub.add_argument("--gui", action="store_true")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "tasks":
            print("\n".join(TASKS))
            return 0
        if args.command == "check-data":
            print(json.dumps(validate_dataset(args.dataset_root), indent=2))
            return 0
        if args.command == "download":
            prefix = {"sim": "cobotmagic_Sim_", "real": "cobotmagic_Real_",
                      "checkpoint": "ACT_sim_"}[args.kind]
            cmd = ["hf", "download", "RoboSynChallenge/" + prefix + args.task,
                   "--repo-type", "model" if args.kind == "checkpoint" else "dataset",
                   "--local-dir", str(resolved(args.destination))]
            cwd, env = Path.cwd(), os.environ.copy()
        else:
            if "," in args.gpu or not args.gpu.isdigit():
                raise ValueError("--gpu must be one numeric GPU id; use the official torchrun workflow for DDP.")
            cmd, cwd, env = build_command(args)
        print(f"cwd: {cwd}", flush=True)
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            return 0
        return subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
