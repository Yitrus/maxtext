"""Tests for the dedicated Megatron mmap dataset-type entry point."""

from types import SimpleNamespace

from maxtext.common.checkpointing import _is_grain_backed_dataset_type
from maxtext.configs.types import DatasetType
from maxtext.input_pipeline._mmap_datasource import MegatronMMapDatasetConfig
from maxtext.input_pipeline.megatron_mmap_data_processing import _get_dataset
from maxtext.input_pipeline.megatron_mmap_data_processing import _preprocess
from tests.unit.mmap_test_utils import create_mmap_test_data


def test_megatron_mmap_is_a_supported_dataset_type():
  assert DatasetType.MEGATRON_MMAP.value == "megatron_mmap"


def test_megatron_mmap_uses_grain_iterator_checkpointing():
  assert _is_grain_backed_dataset_type("grain")
  assert _is_grain_backed_dataset_type("megatron_mmap")
  assert not _is_grain_backed_dataset_type("hf")


def test_dedicated_entry_builds_mmap_npy_source(tmp_path):
  prefix = create_mmap_test_data(str(tmp_path / "data"), [[1, 2, 3, 4, 0, 5, 6]])
  index_dir = tmp_path / "indices"
  index_dir.mkdir()
  dataset_config = MegatronMMapDatasetConfig(
      max_target_length=4,
      eod_id=0,
      mmap_split_sentences=False,
      num_samples=1,
      seed=1234,
  )
  dataset = _get_dataset(
      f"{index_dir}|{prefix}",
      dataset_config,
      num_epoch=1,
      num_threads=1,
      prefetch_buffer_size=1,
      split="train",
      dataloading_host_index=0,
      dataloading_host_count=1,
  )
  sample = next(iter(dataset))
  assert sample["text"].shape == (5,)


def test_dedicated_entry_builds_direct_mmap_source(tmp_path):
  prefix = create_mmap_test_data(str(tmp_path / "data"), [[1, 2, 3, 4]])
  dataset_config = MegatronMMapDatasetConfig(
      max_target_length=4,
      eod_id=0,
      mmap_split_sentences=False,
      seed=1234,
  )
  dataset = _get_dataset(
      prefix,
      dataset_config,
      num_epoch=1,
      num_threads=1,
      prefetch_buffer_size=1,
      split="train",
      dataloading_host_index=0,
      dataloading_host_count=1,
      mode="mmap",
  )
  sample = next(iter(dataset))
  assert sample["text"].tolist() == [1, 2, 3, 4]


def test_eval_preprocessing_uses_eval_batch_size(tmp_path):
  """Eval batching must not inherit the training batch size or expansion."""
  prefix = create_mmap_test_data(
      str(tmp_path / "data"),
      [[1, 2, 3, 4], [5, 6, 7, 8]],
  )
  dataset = _get_dataset(
      prefix,
      MegatronMMapDatasetConfig(max_target_length=4, eod_id=0, mmap_split_sentences=False),
      num_epoch=1,
      num_threads=1,
      prefetch_buffer_size=1,
      split="eval",
      dataloading_host_index=0,
      dataloading_host_count=1,
      mode="mmap",
  )
  config = SimpleNamespace(
      mmap_eod_id=0,
      reset_attention_mask=False,
      eod_mask_loss=False,
      packing_max_segments_per_sample=25,
      expansion_factor_real_data=2,
      global_batch_size_to_load=4,
      grain_ram_budget_mb=256,
  )
  batch = next(
      iter(
          _preprocess(
              dataset,
              config,
              worker_count=0,
              per_worker_buffer_size=1,
              global_batch_size=2,
              is_train=False,
              mode="mmap",
          )
      )
  )
  assert batch["inputs"].shape[0] == 2
