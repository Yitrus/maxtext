<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Megatron indexed datasets

MaxText can read text datasets produced by the Megatron-LM preprocessing
pipeline. The source data is a pair of files with a common prefix:
`<prefix>.bin`, which contains token IDs, and `<prefix>.idx`, which contains
the sequence and document metadata.

Use `dataset_type=megatron_mmap` with `megatron_mmap_mode=mmap_npy` for the
Megatron-compatible GPT sample-order path. It builds or loads three NumPy
indices that define document order, fixed-length sample boundaries, and sample
order. The lower-level `mmap` mode can read the same data format, but does not
provide the full Megatron document/sample shuffle compatibility guarantee.

## Prepare the data

Generate the input using a Megatron-compatible preprocessing command and add
the end-of-document token during preprocessing. Do not add EOD tokens again in
MaxText: doing so changes token offsets and invalidates the generated indices.

The `mmap_eod_id` setting must equal the token ID written by preprocessing.

## Train with one dataset

For `mmap_npy`, `megatron_train_files` has the following form:

```text
<npy_index_dir>|<data_prefix>[:<another_data_prefix>...]
```

`npy_index_dir` stores the generated `document_index`, `sample_index`, and
`shuffle_index` files. The data prefix is the common path without the `.idx`
and `.bin` extensions; it can also be a directory containing multiple shards.

```sh
python3 -m maxtext.trainers.pre_train.train \
  dataset_type=megatron_mmap \
  megatron_mmap_mode=mmap_npy \
  megatron_train_files='/cache/wiki_indices|/data/wiki_text_document' \
  mmap_eod_id=2 \
  max_target_length=2048 \
  steps=1000
```

On a cache miss, every host deterministically builds the indices in memory;
host 0 persists them using atomic writes for later runs. This does not require
a cross-host barrier.

To split one source into train and evaluation documents, configure the same
source for both inputs and set a Megatron-style split ratio:

```sh
megatron_train_files='/cache/wiki_indices|/data/wiki_text_document' \
megatron_eval_files='/cache/wiki_indices|/data/wiki_text_document' \
mmap_npy_split='99,1'
```

## Blend datasets

Each component has the single-dataset form followed by a weight; separate
components with semicolons:

```text
megatron_train_files='/cache/wiki_indices|/data/wiki,0.7;/cache/code_indices|/data/code,0.3'
```

The blend is constructed in global sample order and then sharded across hosts.
Set `blend_cache_dir` to cache generated blend indices. Alternatively,
`blend_index_dir` can point to a directory containing a prebuilt
`dataset_index.npy` and `dataset_sample_index.npy` pair.

## Important limitations

- Multimodal Megatron indexed-dataset extensions are not supported.
- `grain_use_elastic_iterator=true` is not supported with
  `dataset_type=megatron_mmap`.
- `mmap_npy` requires the cache directory to be writable when indices are not
  already present.
- For correct `eod_mask_loss=false` behavior, use `mmap_npy`; the simpler
  `mmap` path uses EOD as the padding sentinel during shifting.
