"""Utilities for exercising the training data loader."""

import logging
import time
from collections.abc import Sequence
from contextlib import suppress
from typing import List, Optional, TextIO, Tuple

import numpy as np
from google.protobuf import text_format

from lczero_training.dataloader import DataLoader, make_dataloader
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)


_BIT_ORDER = (np.arange(64, dtype=np.uint64) ^ 7).reshape(1, 1, 64)


def _stop_loader(loader: DataLoader) -> None:
    with suppress(Exception):
        loader.stop()


def _extract_debug_chunk_info(
    batch: Sequence[np.ndarray],
) -> List[Tuple[int, int, int]]:
    """Decode debug chunk metadata from the first three planes of a batch."""

    if not batch:
        return []

    inputs = batch[0]
    if inputs.ndim < 4 or inputs.shape[1] < 3:
        return []

    planes = np.asarray(inputs[:, :3, :, :], dtype=np.float32)
    plane_bits = planes.reshape(planes.shape[0], 3, 64)
    bits = np.rint(plane_bits).astype(np.uint64) & 1
    weighted = np.left_shift(bits, _BIT_ORDER)
    decoded = np.asarray(np.sum(weighted, axis=-1, dtype=np.uint64))
    return [
        (int(sample[0]), int(sample[1]), int(sample[2])) for sample in decoded
    ]


def _maybe_write_debug_info(
    batch: Sequence[np.ndarray], debug_file: Optional[TextIO]
) -> None:
    if debug_file is None:
        return

    debug_info = _extract_debug_chunk_info(batch)
    debug_file.write(f"{debug_info}\n")


def probe_dataloader(
    config_filename: str, num_batches: int, debug_chunk_file: str | None = None
) -> None:
    """Measure latency and throughput for the configured data loader.

    Args:
        config_filename: Path to the root configuration proto file.
        num_batches: Total number of batches to fetch from the loader.
        debug_chunk_file: Optional path to write chunk metadata for each batch.
    """

    if num_batches < 1:
        raise ValueError("num_batches must be at least 1")

    config = RootConfig()
    logger.info("Reading configuration from proto file")
    with open(config_filename, "r") as config_file:
        text_format.Parse(config_file.read(), config)

    logger.info("Creating data loader")
    loader = make_dataloader(config.data_loader)

    debug_handle: Optional[TextIO] = None
    if debug_chunk_file:
        try:
            debug_handle = open(debug_chunk_file, "w")
        except Exception:
            _stop_loader(loader)
            raise

    first_batch_time = 0.0
    remaining_batches = num_batches - 1
    try:
        logger.info("Fetching first batch")
        start_time = time.perf_counter()
        first_batch = loader.get_next()
        first_batch_time = time.perf_counter() - start_time
        logger.info("Time to first batch: %.3f seconds", first_batch_time)
        _maybe_write_debug_info(first_batch, debug_handle)

        if remaining_batches <= 0:
            logger.info("Only fetched first batch; skipping throughput")
            return

        logger.info(
            "Fetching %d additional batches for throughput measurement",
            remaining_batches,
        )
        throughput_start = time.perf_counter()
        for _ in range(remaining_batches):
            batch = loader.get_next()
            _maybe_write_debug_info(batch, debug_handle)
        throughput_duration = time.perf_counter() - throughput_start

        if throughput_duration <= 0:
            logger.warning("Measured non-positive duration; skipping rate")
            return

        batches_per_second = remaining_batches / throughput_duration
        logger.info(
            "Throughput excluding first batch: %.2f batches/second",
            batches_per_second,
        )
        logger.info(
            "Total time excluding first batch: %.3f seconds",
            throughput_duration,
        )
    finally:
        if debug_handle is not None:
            debug_handle.close()
        _stop_loader(loader)
