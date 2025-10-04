import dataclasses
import logging
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import jax.tree_util as jtu
import orbax.checkpoint as ocp
from google.protobuf import text_format

from lczero_training.training.state import TrainingState
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LeafMetadata:
    type: str
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None

    def format(self) -> str:
        parts = [self.type]
        if self.dtype is not None:
            parts.append(f"dtype={self.dtype}")
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        return " ".join(parts)


def _describe_leaf(value: object) -> LeafMetadata:
    """Return metadata describing the type and shape of a leaf value."""

    if value is None:
        type_name = "NoneType"
    else:
        type_name = (
            f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        )
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None

    if hasattr(value, "dtype") and hasattr(value, "shape"):
        try:
            dtype = str(getattr(value, "dtype"))
            shape_attr = getattr(value, "shape")
            if isinstance(shape_attr, Iterable):
                shape = tuple(int(dim) for dim in shape_attr)
        except Exception:  # pragma: no cover - defensive, should not happen
            dtype = None
            shape = None

    return LeafMetadata(type=type_name, dtype=dtype, shape=shape)


def _path_to_string(path: Iterable[object]) -> str:
    result = ""
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            if result:
                result += "."
            result += key.name
        elif isinstance(key, jtu.DictKey):
            if isinstance(key.key, str):
                if result:
                    result += "."
                result += key.key
            else:
                result += f"[{key.key!r}]"
        elif isinstance(key, jtu.SequenceKey):
            result += f"[{key.index}]"
        else:
            if result:
                result += "."
            result += str(key)

    return result or "<root>"


def _flatten_metadata(tree: object) -> Dict[str, LeafMetadata]:
    leaves, _ = jtu.tree_flatten_with_path(tree)
    metadata: Dict[str, LeafMetadata] = {}
    for path, value in leaves:
        path_str = _path_to_string(path)
        metadata[path_str] = _describe_leaf(value)
    return metadata


def _restore_checkpoint(
    path: str, template_state: TrainingState
) -> TrainingState:
    checkpoint_mgr = ocp.CheckpointManager(
        path,
        options=ocp.CheckpointManagerOptions(create=False),
    )
    restored = checkpoint_mgr.restore(
        None,
        args=ocp.args.PyTreeRestore(template_state),
    )
    if restored is None:
        raise ValueError(f"No checkpoint available at {path}.")
    assert isinstance(restored, TrainingState)
    return restored


def _resolve_spec(
    spec: str,
    config: RootConfig,
    template_state: TrainingState,
) -> Tuple[TrainingState, str]:
    normalized = spec.lower()

    if normalized == "model":
        return template_state, "model"

    restore_template = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    if normalized == "checkpoint":
        checkpoint_path = config.training.checkpoint.path
        if not checkpoint_path:
            logger.error(
                "Checkpoint path must be set in the configuration to load a checkpoint."
            )
            sys.exit(1)
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            logger.error(
                "Checkpoint path %s does not exist.",
                checkpoint_path,
            )
            sys.exit(1)
        try:
            restored = _restore_checkpoint(checkpoint_path, restore_template)
        except Exception as exc:  # pragma: no cover - orbax specific failures
            logger.error(
                "Failed to restore checkpoint from %s: %s",
                checkpoint_path,
                exc,
            )
            sys.exit(1)
        return restored, f"checkpoint({checkpoint_path})"

    checkpoint_path = os.path.expanduser(spec)
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint path %s does not exist.", checkpoint_path)
        sys.exit(1)
    try:
        restored = _restore_checkpoint(checkpoint_path, restore_template)
    except Exception as exc:  # pragma: no cover - orbax specific failures
        logger.error(
            "Failed to restore checkpoint from %s: %s", checkpoint_path, exc
        )
        sys.exit(1)
    return restored, checkpoint_path


def _diff_metadata(
    old_metadata: Dict[str, LeafMetadata],
    new_metadata: Dict[str, LeafMetadata],
    old_label: str,
    new_label: str,
) -> None:
    differences = False
    all_paths = sorted(set(old_metadata.keys()) | set(new_metadata.keys()))

    for path in all_paths:
        old_value = old_metadata.get(path)
        new_value = new_metadata.get(path)

        if old_value is None:
            differences = True
            if new_value is None:
                continue
            print(f"{path}: only in {new_label} -> {new_value.format()}")
        elif new_value is None:
            differences = True
            print(f"{path}: only in {old_label} -> {old_value.format()}")
        elif old_value != new_value:
            differences = True
            print(
                f"{path}: {old_label} -> {old_value.format()} | "
                f"{new_label} -> {new_value.format()}"
            )

    if not differences:
        print(f"No structural differences between {old_label} and {new_label}.")


def statediff(
    *,
    config_filename: str,
    old_spec: str,
    new_spec: str,
) -> None:
    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    template_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    old_state, old_label = _resolve_spec(old_spec, config, template_state)
    new_state, new_label = _resolve_spec(new_spec, config, template_state)

    logger.info("Constructing metadata for old state")
    old_metadata = _flatten_metadata(old_state)
    logger.info("Constructing metadata for new state")
    new_metadata = _flatten_metadata(new_state)

    _diff_metadata(old_metadata, new_metadata, old_label, new_label)
