# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Helper library for ALBERT fine-tuning.

This library can be used to construct ALBERT models for fine-tuning, either from
json config files or from TF-Hub modules.
"""

from albert import modeling
from albert import tokenization
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


def _create_model_from_hub(hub_module, is_training, input_ids, input_mask,
                           segment_ids):
  """Creates an ALBERT model from TF-Hub."""
  tags = set()
  if is_training:
    tags.add("train")
  albert_module = hub.Module(hub_module, tags=tags, trainable=True)
  albert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  albert_outputs = albert_module(
      inputs=albert_inputs,
      signature="tokens",
      as_dict=True)
  return (albert_outputs["pooled_output"], albert_outputs["sequence_output"])


def _create_model_from_scratch(albert_config, is_training, input_ids,
                               input_mask, segment_ids, use_one_hot_embeddings,
                               use_einsum):
  """Creates an ALBERT model from scratch/config."""
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_einsum=use_einsum)
  return (model.get_pooled_output(), model.get_sequence_output())


def create_albert(albert_config, is_training, input_ids, input_mask,
                  segment_ids, use_one_hot_embeddings, use_einsum, hub_module):
  """Creates an ALBERT, either from TF-Hub or from scratch."""
  if hub_module:
    tf.logging.info("creating model from hub_module: %s", hub_module)
    return _create_model_from_hub(hub_module, is_training, input_ids,
                                  input_mask, segment_ids)
  else:
    tf.logging.info("creating model from albert_config")
    return _create_model_from_scratch(albert_config, is_training, input_ids,
                                      input_mask, segment_ids,
                                      use_one_hot_embeddings, use_einsum)


def create_vocab(vocab_file, do_lower_case, spm_model_file, hub_module):
  """Creates a vocab, either from vocab file or from a TF-Hub module."""
  if hub_module:
    use_spm = True if spm_model_file else False
    return tokenization.FullTokenizer.from_hub_module(
        hub_module=hub_module, use_spm=use_spm)
  else:
    return tokenization.FullTokenizer.from_scratch(
        vocab_file=vocab_file, do_lower_case=do_lower_case,
        spm_model_file=spm_model_file)

