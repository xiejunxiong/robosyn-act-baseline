"""One packaged YAML configuration shared by launchers and CPU smoke checks."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "configs/act_click_bell.yaml"


def load_config(path=None):
    path = Path(path).expanduser() if path else DEFAULT_CONFIG
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load config {path}: {exc}") from exc
    keys = {
        "dataset": {"repo_id", "state_dim", "action_dim", "fps", "image_shape"},
        "training": {"steps", "batch_size", "num_workers", "chunk_size", "n_action_steps",
                     "log_freq", "save_freq", "seed"},
        "evaluation": {"setting", "episodes", "seed", "max_steps"},
    }
    if not isinstance(config, dict) or set(config) != {"schema_version", "task", *keys}:
        raise ValueError("Config must contain schema_version, task, dataset, training, evaluation only.")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported config schema_version; expected 1.")
    if not isinstance(config["task"], str) or not config["task"]:
        raise ValueError("Config task must be a nonempty string.")
    for section, required in keys.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError(f"Config {section} must contain exactly {sorted(required)}.")
        for key, value in values.items():
            if key in {"repo_id", "image_shape", "setting"}:
                continue
            minimum = 0 if key in {"num_workers", "seed"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{section}.{key} must be an integer >= {minimum}.")
    data, train, evaluation = config["dataset"], config["training"], config["evaluation"]
    if not isinstance(data["repo_id"], str) or not data["repo_id"].strip():
        raise ValueError("dataset.repo_id must be a nonempty string.")
    shape = data["image_shape"]
    if (not isinstance(shape, list) or len(shape) != 3 or shape[0] != 3
            or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError("dataset.image_shape must be [3, positive height, positive width].")
    if train["n_action_steps"] > train["chunk_size"]:
        raise ValueError("n-action-steps must not exceed chunk-size.")
    if evaluation["setting"] not in ("clear", "random"):
        raise ValueError("evaluation.setting must be clear or random.")
    return config
