"""Element-wise compatibility checks against Megatron-Core's GPTDataset.

These tests deliberately use Megatron-Core as the oracle.  They cover the
three persistent mmap_npy indices, the actual next-token samples returned by
the data source, and the blend scheduler.  They are skipped when the optional
Megatron-Core test dependency is unavailable.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("megatron.core")

from megatron.core.datasets import helpers as megatron_helpers
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    _build_document_index,
    _build_shuffle_index,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer
from megatron.core.datasets.utils import Split

from maxtext.input_pipeline._megatron_blending import build_blending_indices
from maxtext.input_pipeline._mmap_datasource import MegatronNpyDataSource, _discover_npy_indices
from maxtext.input_pipeline._mmap_index_utils import (
    build_document_index,
    build_sample_index,
    build_shuffle_index,
    convert,
    get_document_sizes,
)
from tests.unit.mmap_test_utils import create_mmap_test_data

pytestmark = [pytest.mark.cpu_only, pytest.mark.megatron_alignment]


class _StubTokenizer(MegatronLegacyTokenizer):
  """The smallest tokenizer surface required by ``GPTDataset``."""

  def __init__(self, eod):
    super().__init__(None)
    self._eod = eod

  @property
  def vocab_size(self):
    return 50000

  @property
  def vocab(self):
    raise NotImplementedError

  @property
  def inv_vocab(self):
    raise NotImplementedError

  @property
  def eod(self):
    return self._eod

  def tokenize(self, text):
    raise NotImplementedError

  def detokenize(self, ids):
    raise NotImplementedError


def _create_eod_dataset(tmp_dir, num_docs=10, eod_id=0, seed=123):
  """Create a valid Megatron indexed dataset with pre-appended EOD tokens."""
  rng = np.random.RandomState(seed)
  sequences = []
  for doc_id in range(num_docs):
    tokens = rng.randint(1, 10000, size=20 + doc_id * 3, dtype=np.int32)
    tokens[-1] = eod_id
    sequences.append(tokens)
  prefix = os.path.join(tmp_dir, "data")
  create_mmap_test_data(prefix, sequences, doc_boundaries=list(range(num_docs + 1)))
  return prefix


def _megatron_dataset(prefix, seq_length, seed, eod_id):
  indexed_dataset = IndexedDataset(prefix, multimodal=False, mmap=True)
  config = GPTDatasetConfig(
      random_seed=seed,
      sequence_length=seq_length,
      reset_position_ids=False,
      reset_attention_mask=False,
      eod_mask_loss=False,
      tokenizer=_StubTokenizer(eod_id),
  )
  return GPTDataset(
      indexed_dataset=indexed_dataset,
      dataset_path=prefix,
      indexed_indices=np.arange(indexed_dataset.document_indices.shape[0] - 1, dtype=np.int32),
      num_samples=None,
      index_split=Split.train,
      config=config,
  )


def _raw_megatron_tokens(dataset, sample_id):
  """Join GPTDataset's input and label views back into seq_length + 1 tokens."""
  sample = dataset[sample_id]
  tokens = sample["tokens"].numpy().astype(np.int32)
  labels = sample["labels"].numpy().astype(np.int32)
  return np.concatenate([tokens[:1], labels])


@pytest.mark.parametrize("num_docs,num_epochs,seed", [(10, 1, 42), (10, 3, 42), (50, 2, 1234)])
def test_document_index_matches_megatron(num_docs, num_epochs, seed):
  expected_rng = np.random.RandomState(seed)
  expected = _build_document_index(np.arange(num_docs, dtype=np.int32), num_epochs, expected_rng, False)
  actual = build_document_index(num_docs, num_epochs, seed)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "sizes,seq_length,num_epochs,seed",
    [([100], 8, 1, 42), ([50, 50], 16, 2, 1234), ([20, 30, 15], 8, 2, 42)],
)
def test_sample_index_matches_megatron_cpp(sizes, seq_length, num_epochs, seed):
  sizes = np.asarray(sizes, dtype=np.int32)
  rng = np.random.RandomState(seed)
  document_index = _build_document_index(np.arange(len(sizes), dtype=np.int32), num_epochs, rng, False)
  expected = megatron_helpers.build_sample_idx(
      sizes,
      document_index,
      seq_length,
      num_epochs=num_epochs,
      tokens_per_epoch=int(sizes.sum()),
      drop_last_partial_sequence=True,
      add_extra_token_to_sequence=True,
  )
  actual = build_sample_index(sizes.astype(np.int64), document_index, seq_length, add_extra_token=1)
  np.testing.assert_array_equal(actual, expected)


def test_persisted_shuffle_index_matches_megatron_rng_flow():
  """The production builder must consume one RNG exactly as Megatron does."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    prefix = _create_eod_dataset(tmp_dir)
    output_dir = os.path.join(tmp_dir, "indices")
    seq_length, seed, num_epochs = 8, 42, 2
    convert([prefix], output_dir, seq_length=seq_length, num_epochs=num_epochs, seed=seed)
    _, _, actual_path = _discover_npy_indices(output_dir)
    actual = np.load(actual_path)

    sizes = get_document_sizes([prefix]).astype(np.int32)
    rng = np.random.RandomState(seed)
    document_index = _build_document_index(np.arange(len(sizes), dtype=np.int32), num_epochs, rng, False)
    sample_index = megatron_helpers.build_sample_idx(
        sizes,
        document_index,
        seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=int(sizes.sum()),
        drop_last_partial_sequence=True,
        add_extra_token_to_sequence=True,
    )
    expected = _build_shuffle_index(sample_index.shape[0] - 1, sample_index.shape[0] - 1, rng)
    np.testing.assert_array_equal(actual, expected)


def test_all_mmap_npy_tokens_match_real_megatron_gpt_dataset():
  """Each mmap_npy sample must equal Megatron's actual GPTDataset output."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    eod_id, seq_length, seed = 0, 8, 42
    prefix = _create_eod_dataset(tmp_dir, eod_id=eod_id)
    output_dir = os.path.join(tmp_dir, "indices")
    convert([prefix], output_dir, seq_length=seq_length, num_epochs=1, seed=seed)

    actual = MegatronNpyDataSource(output_dir, prefix, eod_id=eod_id, seq_length=seq_length)
    expected = _megatron_dataset(prefix, seq_length, seed, eod_id)
    assert len(actual) == len(expected)
    for sample_id in range(len(actual)):
      np.testing.assert_array_equal(actual[sample_id]["text"], _raw_megatron_tokens(expected, sample_id))


@pytest.mark.parametrize("weights,size", [([0.7, 0.3], 50), ([0.5, 0.3, 0.2], 80)])
def test_blend_indices_match_megatron_cpp(weights, size):
  """The blend dispatcher must use exactly Megatron's greedy schedule."""
  normalized_weights = np.asarray(weights, dtype=np.float64)
  actual_dataset_indices = np.zeros(size, dtype=np.int16)
  actual_sample_indices = np.zeros(size, dtype=np.int64)
  build_blending_indices(
      actual_dataset_indices,
      actual_sample_indices,
      normalized_weights,
      len(normalized_weights),
      size,
  )

  expected_dataset_indices = np.zeros(size, dtype=np.int16)
  expected_sample_indices = np.zeros(size, dtype=np.int64)
  megatron_helpers.build_blending_indices(
      expected_dataset_indices,
      expected_sample_indices,
      normalized_weights,
      len(normalized_weights),
      size,
      False,
  )
  np.testing.assert_array_equal(actual_dataset_indices, expected_dataset_indices)
  np.testing.assert_array_equal(actual_sample_indices, expected_sample_indices)


def test_shuffle_index_helper_matches_megatron():
  """Keep a direct unit-level check independent of on-disk index creation."""
  seed, num_samples, total_size = 1234, 30, 50
  expected = _build_shuffle_index(num_samples, total_size, np.random.RandomState(seed))
  actual = build_shuffle_index(num_samples, total_size, seed)
  np.testing.assert_array_equal(actual, expected)
