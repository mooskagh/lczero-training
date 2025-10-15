"""Utility helpers for writing TensorBoard summaries during training."""

from dataclasses import dataclass
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import tree_util
from tensorboardX import SummaryWriter

from proto.data_loader_config_pb2 import DataLoaderConfig
from proto.training_config_pb2 import ScheduleConfig


@dataclass
class StepMetrics:
    """Metrics collected for a single optimization step."""

    step: int
    learning_rate: Optional[float]
    weighted_loss: float
    unweighted_losses: Mapping[str, float]
    gradient_norm: Optional[float]


class TensorboardLogger:
    """Writes training metrics and configuration to TensorBoard."""

    def __init__(
        self,
        logdir: Optional[str],
        *,
        data_loader_config: Optional[DataLoaderConfig] = None,
        schedule_config: Optional[ScheduleConfig] = None,
    ) -> None:
        if logdir:
            self._writer = SummaryWriter(logdir)
        else:
            self._writer = None

        self._batch_size = _extract_batch_size(data_loader_config)
        self._steps_per_network = (
            schedule_config.steps_per_network
            if schedule_config is not None
            else None
        )
        self._chunks_per_network = (
            schedule_config.chunks_per_network
            if schedule_config is not None
            else None
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def log_step(self, metrics: StepMetrics) -> None:
        if self._writer is None:
            return

        if metrics.learning_rate is not None:
            self._writer.add_scalar(
                "config/learning_rate", metrics.learning_rate, metrics.step
            )

        self._writer.add_scalar(
            "loss/weighted", metrics.weighted_loss, metrics.step
        )

        for name, value in sorted(metrics.unweighted_losses.items()):
            self._writer.add_scalar(
                f"loss/unweighted/{name}", value, metrics.step
            )

        if metrics.gradient_norm is not None:
            self._writer.add_scalar(
                "gradients/global_norm", metrics.gradient_norm, metrics.step
            )

        self._writer.flush()

    def log_epoch(self, step: int, model_state: nnx.State) -> None:
        if self._writer is None:
            return

        weights = _collect_weights(model_state)
        if weights is not None and weights.size:
            self._writer.add_histogram("weights/distribution", weights, step)
            self._writer.add_scalar(
                "weights/mean", float(np.mean(weights)), step
            )
            self._writer.add_scalar("weights/std", float(np.std(weights)), step)
            self._writer.add_scalar("weights/min", float(np.min(weights)), step)
            self._writer.add_scalar("weights/max", float(np.max(weights)), step)

        if self._batch_size is not None:
            self._writer.add_scalar("config/batch_size", self._batch_size, step)
        if self._steps_per_network is not None:
            self._writer.add_scalar(
                "config/steps_per_network", self._steps_per_network, step
            )
        if self._chunks_per_network is not None:
            self._writer.add_scalar(
                "config/chunks_per_network", self._chunks_per_network, step
            )

        self._writer.flush()


def _extract_batch_size(
    config: Optional[DataLoaderConfig],
) -> Optional[int]:
    if config is None:
        return None

    for stage in config.stage:
        if stage.HasField("tensor_generator"):
            generator = stage.tensor_generator
            if generator.HasField("batch_size"):
                return int(generator.batch_size)
    return None


def _collect_weights(model_state: nnx.State) -> Optional[np.ndarray]:
    leaves = tree_util.tree_leaves(
        model_state, is_leaf=lambda node: hasattr(node, "value")
    )
    arrays: list[np.ndarray] = []
    for leaf in leaves:
        array = _leaf_to_array(leaf)
        if array is None:
            continue
        flat = np.asarray(jnp.ravel(array))
        if flat.size:
            arrays.append(flat)

    if not arrays:
        return None
    return np.concatenate(arrays)


def _leaf_to_array(value: object) -> Optional[jax.Array]:
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    maybe_value = getattr(value, "value", None)
    if isinstance(maybe_value, (jax.Array, np.ndarray)):
        return jnp.asarray(maybe_value)
    return None
