"""Megatron-compatible weighted blending for Grain map datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from maxtext.input_pipeline import _mmap_index_utils


def build_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    weights: np.ndarray,
    num_datasets: int,
    size: int,
) -> None:
  """Populate blend indices using Megatron's greedy error minimization.

  This is a direct Python implementation of
  ``megatron/core/datasets/helpers.cpp::build_blending_indices``.  In
  particular, ``np.argmax`` keeps the first (lowest dataset ID) on ties.
  """
  if size < 0:
    raise ValueError(f"size must be non-negative, got {size}")
  if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
    raise ValueError("Blend index arrays must both have shape (size,)")
  if num_datasets <= 0:
    raise ValueError(f"num_datasets must be positive, got {num_datasets}")

  weights = np.asarray(weights, dtype=np.float64)
  if weights.shape != (num_datasets,) or np.any(weights <= 0):
    raise ValueError("weights must contain one positive value per dataset")
  if not np.isclose(weights.sum(), 1.0, rtol=1e-4, atol=1e-6):
    raise ValueError(f"weights must sum to 1, got {weights.tolist()}")

  current_samples = np.zeros(num_datasets, dtype=np.int64)
  for sample_id in range(size):
    errors = weights * max(float(sample_id), 1.0) - current_samples
    dataset_id = int(np.argmax(errors))
    dataset_index[sample_id] = dataset_id
    dataset_sample_index[sample_id] = current_samples[dataset_id]
    current_samples[dataset_id] += 1


def _normalize_datasets(map_datasets: Sequence, weights: Sequence[float]):
  if len(map_datasets) != len(weights) or not map_datasets:
    raise ValueError("At least one dataset and one corresponding weight are required")
  weights = np.asarray(weights, dtype=np.float64)
  if np.any(weights < 0) or weights.sum() <= 0:
    raise ValueError(f"weights must be non-negative with a positive total, got {weights.tolist()}")
  keep = weights > 0
  datasets = [dataset for dataset, keep_dataset in zip(map_datasets, keep) if keep_dataset]
  weights = weights[keep]
  return datasets, weights / weights.sum()


def _infer_size(weights: np.ndarray, lengths: Sequence[int]) -> int:
  if len(lengths) != len(weights) or any(length <= 0 for length in lengths):
    raise ValueError(f"Dataset lengths must be positive and match weights, got {lengths}")
  size = int(math.floor(min(length / weight for length, weight in zip(lengths, weights))))
  if size <= 0:
    raise ValueError("Blended dataset would be empty")
  return size


def _cache_paths(cache_dir: str, weights: np.ndarray, lengths: Sequence[int], size: int, split: str):
  payload = json.dumps(
      {"weights": weights.tolist(), "lengths": list(lengths), "size": size, "split": split},
      sort_keys=True,
      separators=(",", ":"),
  )
  key = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
  root = Path(cache_dir)
  return root / f"{key}-dataset_index.npy", root / f"{key}-dataset_sample_index.npy"


def _validate_indices(dataset_index, dataset_sample_index, lengths, size):
  if dataset_index.shape != (size,) or dataset_sample_index.shape != (size,):
    raise ValueError("Cached blend indices have an unexpected size")
  if np.any(dataset_index < 0) or np.any(dataset_index >= len(lengths)):
    raise ValueError("Cached blend dataset_index is out of range")
  for dataset_id, length in enumerate(lengths):
    used = dataset_sample_index[dataset_index == dataset_id]
    if used.size and (np.any(used < 0) or int(np.max(used)) >= length):
      raise ValueError(f"Cached blend indices exceed dataset {dataset_id} length")


class MegatronBlendedDataSource:
  """Random-access blend whose global order is identical to Megatron's."""

  def __init__(
      self,
      map_datasets: Sequence,
      weights: Sequence[float],
      size: int | None = None,
      dataset_lengths: Sequence[int] | None = None,
      cache_dir: str | None = None,
      blend_index_dir: str | None = None,
      split: str = "train",
  ):
    self._datasets, self._weights = _normalize_datasets(map_datasets, weights)
    self._lengths = list(dataset_lengths) if dataset_lengths is not None else [len(dataset) for dataset in self._datasets]
    if len(self._lengths) != len(self._datasets):
      raise ValueError("dataset_lengths must match the non-zero-weight datasets")
    self._size = _infer_size(self._weights, self._lengths) if size is None else int(size)
    if self._size <= 0:
      raise ValueError(f"size must be positive, got {self._size}")

    self._dataset_index, self._dataset_sample_index = self._load_or_build(
        cache_dir, blend_index_dir, split
    )

  def _load_or_build(self, cache_dir, blend_index_dir, split):
    candidates = []
    if blend_index_dir:
      root = Path(blend_index_dir)
      candidates.append((root / "dataset_index.npy", root / "dataset_sample_index.npy"))
    if cache_dir:
      candidates.append(_cache_paths(cache_dir, self._weights, self._lengths, self._size, split))

    for dataset_path, sample_path in candidates:
      try:
        dataset_index = np.load(dataset_path, allow_pickle=False)
        dataset_sample_index = np.load(sample_path, allow_pickle=False)
        _validate_indices(dataset_index, dataset_sample_index, self._lengths, self._size)
        return dataset_index, dataset_sample_index
      except (FileNotFoundError, OSError, ValueError):
        continue

    dataset_index = np.zeros(self._size, dtype=np.int16)
    dataset_sample_index = np.zeros(self._size, dtype=np.int64)
    build_blending_indices(
        dataset_index, dataset_sample_index, self._weights, len(self._datasets), self._size
    )
    _validate_indices(dataset_index, dataset_sample_index, self._lengths, self._size)

    if cache_dir and _mmap_index_utils.is_primary_process():
      dataset_path, sample_path = _cache_paths(cache_dir, self._weights, self._lengths, self._size, split)
      os.makedirs(cache_dir, exist_ok=True)
      _mmap_index_utils.save_npy_atomic(dataset_path, dataset_index)
      _mmap_index_utils.save_npy_atomic(sample_path, dataset_sample_index)
    return dataset_index, dataset_sample_index

  def __len__(self):
    return self._size

  def __getitem__(self, idx):
    if idx < 0:
      idx += self._size
    if idx < 0 or idx >= self._size:
      raise IndexError(f"Index {idx} out of range for blend of size {self._size}")
    return self._datasets[int(self._dataset_index[idx])][int(self._dataset_sample_index[idx])]
