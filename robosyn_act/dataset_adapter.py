"""Non-mutating feature contract helpers; production loading uses the upstream adapter.

Values are shared, not copied. LeRobot already decodes video to float32 CHW;
this alias operation must not transpose or normalize it a second time.
"""

from copy import deepcopy

CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
FEATURE_ALIASES = {
    "observation.qpos": "observation.state",
    **{f"{camera}.color": f"observation.images.{camera}" for camera in CAMERAS},
}


def alias_features(values):
    """Add canonical keys (and padding masks) without overwriting canonical data."""
    result = dict(values)
    for source, target in FEATURE_ALIASES.items():
        if target not in result and source in values:
            result[target] = values[source]
            if f"{source}_is_pad" in values:
                result[f"{target}_is_pad"] = values[f"{source}_is_pad"]
    return result


def adapt_metadata(features, stats):
    """Return independent metadata dictionaries with the same feature/stat aliases."""
    return deepcopy(alias_features(features)), deepcopy(alias_features(stats))


def validate_batch(batch, config):
    """Validate decoded, batched NumPy arrays or Torch tensors without importing Torch."""
    batch = alias_features(batch)
    data, train = config["dataset"], config["training"]
    state = batch.get("observation.state")
    if state is None or len(state.shape) != 2:
        raise ValueError("observation.state must have shape [B, state_dim].")
    size = state.shape[0]
    expected = {
        "observation.state": (size, data["state_dim"]),
        "action": (size, train["chunk_size"], data["action_dim"]),
        "action_is_pad": (size, train["chunk_size"]),
        **{f"observation.images.{camera}": (size, *data["image_shape"]) for camera in CAMERAS},
    }
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    for key, shape in expected.items():
        if key not in batch or tuple(batch[key].shape) != shape:
            raise ValueError(f"{key} must have shape {shape}.")
    return batch
