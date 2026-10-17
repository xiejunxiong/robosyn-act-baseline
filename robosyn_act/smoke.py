"""CPU contract smoke, with an explicit optional real LeRobot ACT forward/backward."""

import importlib.metadata
import json

import numpy as np

from .dataset_adapter import CAMERAS, validate_batch

LEROBOT_REVISION = "b883328e6c95681ca90a18b102e4ae5e1f91e2bf"


def synthetic_batch(config):
    rng = np.random.default_rng(7)
    data, train = config["dataset"], config["training"]
    # B=1 keeps CPU checks small; dimensions/chunk length come from the run YAML.
    batch = {
        "observation.qpos": rng.standard_normal((1, data["state_dim"])).astype("float32"),
        "action": rng.standard_normal((1, train["chunk_size"], data["action_dim"])).astype("float32"),
        "action_is_pad": np.zeros((1, train["chunk_size"]), dtype=bool),
        **{f"{camera}.color": rng.random((1, *data["image_shape"]), dtype=np.float32)
           for camera in CAMERAS},
    }
    return validate_batch(batch, config)


def run_smoke(config, act=False):
    batch = synthetic_batch(config)
    report = {"mode": "contract", "real_act_forward": False,
              "shapes": {key: list(value.shape) for key, value in batch.items()}}
    if not act:
        report["note"] = "Synthetic NumPy arrays: config/alias/shape checks only; no model or simulator."
        return report
    try:
        distribution = importlib.metadata.distribution("lerobot")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != LEROBOT_REVISION:
            raise ValueError("ACT smoke requires a pip VCS install of the pinned LeRobot commit; "
                             "a version string alone does not identify its source.")
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError("Real ACT smoke unavailable: install the pinned LeRobot in a compatible "
                         "Torch environment. Contract mode is separate and is not ACT inference.") from exc
    torch.manual_seed(7)
    torch.set_num_threads(2)
    data, train = config["dataset"], config["training"]
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(data["state_dim"],))}
    inputs.update({f"observation.images.{camera}": PolicyFeature(
        type=FeatureType.VISUAL, shape=tuple(data["image_shape"])) for camera in CAMERAS})
    # Real ACTPolicy, deliberately reduced hidden sizes for a CPU smoke. No pretrained downloads.
    policy_config = ACTConfig(
        device="cpu", push_to_hub=False, input_features=inputs,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(data["action_dim"],))},
        chunk_size=train["chunk_size"], n_action_steps=train["n_action_steps"],
        pretrained_backbone_weights=None, dim_model=64, dim_feedforward=128,
        n_heads=4, n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
        latent_dim=8, normalization_mapping={key: NormalizationMode.IDENTITY
                                           for key in ("VISUAL", "STATE", "ACTION")},
    )
    policy = ACTPolicy(policy_config).train()
    tensors = {key: torch.from_numpy(value) for key, value in batch.items()}
    loss, metrics = policy(tensors)
    if not torch.isfinite(loss).item():
        raise ValueError("ACT smoke produced a non-finite loss.")
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise ValueError("ACT smoke produced missing or non-finite gradients.")
    action = policy.select_action(tensors)
    if tuple(action.shape) != (1, data["action_dim"]):
        raise ValueError("ACT select_action returned an unexpected shape.")
    report.update(mode="act", real_act_forward=True, loss=loss.item(), metrics=metrics,
                  backward=True, action_shape=list(action.shape),
                  note="Real pinned ACTPolicy, reduced CPU architecture, synthetic data; not the 80k experiment.")
    return report
