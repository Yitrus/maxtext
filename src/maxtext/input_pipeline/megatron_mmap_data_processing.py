# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grain-backed iterator factories for ``dataset_type=megatron_mmap``."""

import functools

import grain.python as grain
import jax
import ml_collections

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import multihost_dataloading
from maxtext.input_pipeline._mmap_datasource import MegatronMMapDatasetConfig
from maxtext.input_pipeline._mmap_datasource import get_mmap_dataset
from maxtext.input_pipeline._mmap_datasource import get_mmap_npy_dataset
from maxtext.input_pipeline.grain_data_processing import _apply_mapdataset_transforms
from maxtext.utils import max_logging


def _build_dataset_config(config, *, num_samples, split_index):
  return MegatronMMapDatasetConfig(
      max_target_length=config.max_target_length,
      eod_id=config.mmap_eod_id,
      mmap_split_sentences=config.mmap_split_sentences,
      blend_cache_dir=config.blend_cache_dir,
      blend_index_dir=config.blend_index_dir,
      num_samples=num_samples,
      seed=config.data_shuffle_seed,
      split_ratio=config.mmap_npy_split or None,
      split_index=split_index,
  )


def _multiprocessing_options(dataset, config, worker_count, per_worker_buffer_size):
  if worker_count == -1:
    return grain.experimental.pick_performance_config(
        ds=dataset,
        ram_budget_mb=config.grain_ram_budget_mb,
        max_workers=None,
        max_buffer_size=None,
    ).multiprocessing_options
  return grain.MultiprocessingOptions(
      num_workers=worker_count,
      per_worker_buffer_size=per_worker_buffer_size,
  )


def _get_dataset(
    data_file_pattern,
    dataset_config,
    *,
    num_epoch,
    num_threads,
    prefetch_buffer_size,
    split,
    dataloading_host_index,
    dataloading_host_count,
    mode="mmap_npy",
    shuffle=False,
):
  """Build one host's Megatron mmap source without using Grain file types."""
  if mode == "mmap_npy":
    return get_mmap_npy_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        num_threads,
        prefetch_buffer_size,
        dataset_config.blend_cache_dir or None,
        dataset_config.blend_index_dir or None,
        split,
        apply_transforms=_apply_mapdataset_transforms,
        num_samples=dataset_config.num_samples,
        seed=dataset_config.seed,
        split=dataset_config.split_ratio,
        split_index=dataset_config.split_index,
    )
  if mode == "mmap":
    return get_mmap_dataset(
        data_file_pattern,
        dataset_config.mmap_split_sentences,
        dataset_config.max_target_length,
        dataset_config.eod_id,
        shuffle,
        dataset_config.seed,
        num_epoch,
        dataloading_host_index,
        dataloading_host_count,
        num_threads,
        prefetch_buffer_size,
        apply_transforms=_apply_mapdataset_transforms,
    )
  raise ValueError(f"Unsupported megatron_mmap_mode: {mode}")


def _preprocess(
    dataset,
    config,
    worker_count,
    per_worker_buffer_size,
    *,
    global_batch_size,
    is_train,
    mode="mmap_npy",
):
  """Apply Megatron sample semantics, Grain workers, and local batching."""
  eod_id = config.mmap_eod_id
  if mode == "mmap_npy":
    dataset = dataset.map(
        input_pipeline_utils.MegatronSplitInputsTargets(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
            min_segment_length=input_pipeline_utils.megatron_min_segment_length(config),
        )
    )
  elif mode == "mmap":
    dataset = dataset.map(input_pipeline_utils.Rekey({"inputs": "text", "targets": "text"}))
    dataset = dataset.map(
        input_pipeline_utils.GenerateDocSegmentIds(
            eod_id=eod_id,
            reset_attention_mask=config.reset_attention_mask,
            eod_mask_loss=config.eod_mask_loss,
            min_segment_length=input_pipeline_utils.megatron_min_segment_length(config),
        )
    )
  else:
    raise ValueError(f"Unsupported megatron_mmap_mode: {mode}")

  batch_size = global_batch_size // jax.process_count()
  if is_train and config.expansion_factor_real_data > 1:
    batch_size = int(batch_size // config.expansion_factor_real_data)
  batch_fn = functools.partial(
      grain.experimental.batch_and_pad,
      batch_size=batch_size,
      pad_value=eod_id,
  )
  if mode == "mmap_npy":
    dataset = dataset.mp_prefetch(_multiprocessing_options(dataset, config, worker_count, per_worker_buffer_size))
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)
  if mode == "mmap":
    if not config.eod_mask_loss:
      max_logging.warning(
          "WARNING: mmap mode with eod_mask_loss=False uses mmap_eod_id as both "
          "padding and EOD sentinel. ShiftData will zero targets_segmentation "
          "at all EOD positions, effectively masking EOD from loss regardless "
          "of eod_mask_loss. Use mmap_npy mode for correct eod_mask_loss=False behavior."
      )
    dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[eod_id], axis=1))
    dataset = dataset.mp_prefetch(_multiprocessing_options(dataset, config, worker_count, per_worker_buffer_size))
  return dataset


def _make_iterator(
    config,
    global_mesh,
    process_indices,
    get_ds_fn,
    preprocessing_fn,
    global_batch_size,
    generate_padding_batch,
    *,
    is_train,
):
  """Run this source through the Google branch's shared host lifecycle."""
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible by number of global devices."
  if config.grain_use_elastic_iterator:
    raise ValueError("grain_use_elastic_iterator is not supported by dataset_type=megatron_mmap.")

  if config.colocated_python_data_input:
    global_shape = (global_batch_size, config.max_target_length)
    return multihost_dataloading.RemoteIteratorWrapper(
        get_ds_fn,
        preprocessing_fn,
        global_mesh,
        global_shape,
        checkpoint_path=config.checkpoint_dir,
        elastic=False,
    )

  if is_train and 0 < config.expansion_factor_real_data < 1:
    num_dataloaders = int(1 / config.expansion_factor_real_data)
    host_count = len(process_indices) * num_dataloaders
    host_index = process_indices.index(jax.process_index())
    return [
        multihost_dataloading.MultiHostDataLoadIterator(
            preprocessing_fn(
                dataset=get_ds_fn(
                    dataloading_host_index=host_index + i * len(process_indices), dataloading_host_count=host_count
                )
            ),
            global_mesh,
            generate_padding_batch,
        )
        for i in range(num_dataloaders)
    ]

  dataset = get_ds_fn(
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
  )
  return multihost_dataloading.MultiHostDataLoadIterator(
      preprocessing_fn(dataset=dataset),
      global_mesh,
      generate_padding_batch,
      expansion_loading_factor_for_grain=config.expansion_factor_real_data if is_train else -1,
  )


def make_megatron_mmap_train_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices):
  """Create a training iterator for ``dataset_type=megatron_mmap``."""
  mode = config.megatron_mmap_mode
  num_samples = (
      config.steps * config.global_batch_size_to_load if mode == "mmap_npy" and getattr(config, "steps", 0) > 0 else None
  )
  dataset_config = _build_dataset_config(config, num_samples=num_samples, split_index=0)
  get_ds_fn = functools.partial(
      _get_dataset,
      config.megatron_train_files,
      dataset_config,
      num_epoch=config.num_epoch,
      num_threads=config.grain_num_threads,
      prefetch_buffer_size=config.grain_prefetch_buffer_size,
      split="train",
      mode=mode,
      shuffle=config.enable_data_shuffling,
  )
  preprocessing_fn = functools.partial(
      _preprocess,
      config=config,
      worker_count=config.grain_worker_count,
      per_worker_buffer_size=config.grain_per_worker_buffer_size,
      global_batch_size=config.global_batch_size_to_load,
      is_train=True,
      mode=mode,
  )
  return _make_iterator(
      config,
      global_mesh,
      process_indices,
      get_ds_fn,
      preprocessing_fn,
      config.global_batch_size_to_load,
      config.generate_padding_batch_train,
      is_train=True,
  )


def make_megatron_mmap_eval_iterator(config: ml_collections.ConfigDict, global_mesh, process_indices):
  """Create an evaluation iterator for ``dataset_type=megatron_mmap``."""
  mode = config.megatron_mmap_mode
  dataset_config = _build_dataset_config(config, num_samples=None, split_index=1)
  get_ds_fn = functools.partial(
      _get_dataset,
      config.megatron_eval_files,
      dataset_config,
      num_epoch=1,
      num_threads=config.grain_num_threads_eval,
      prefetch_buffer_size=config.grain_prefetch_buffer_size_eval,
      split="eval",
      mode=mode,
      shuffle=False,
  )
  preprocessing_fn = functools.partial(
      _preprocess,
      config=config,
      worker_count=config.grain_worker_count_eval,
      per_worker_buffer_size=config.grain_per_worker_buffer_size_eval,
      global_batch_size=config.global_batch_size_to_load_eval,
      is_train=False,
      mode=mode,
  )
  return _make_iterator(
      config,
      global_mesh,
      process_indices,
      get_ds_fn,
      preprocessing_fn,
      config.global_batch_size_to_load_eval,
      config.generate_padding_batch_eval,
      is_train=False,
  )
