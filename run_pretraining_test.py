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
# Lint as: python2, python3
"""Tests for run_pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tempfile
from absl.testing import flagsaver
import modeling
import run_pretraining
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS


def _create_config_file(filename, max_seq_length, vocab_size):
  """Creates an AlbertConfig and saves it to file."""
  albert_config = modeling.AlbertConfig(
      vocab_size,
      embedding_size=5,
      hidden_size=14,
      num_hidden_layers=3,
      num_hidden_groups=1,
      num_attention_heads=2,
      intermediate_size=19,
      inner_group_num=1,
      down_scale_factor=1,
      hidden_act="gelu",
      hidden_dropout_prob=0,
      attention_probs_dropout_prob=0,
      max_position_embeddings=max_seq_length,
      type_vocab_size=2,
      initializer_range=0.02)
  with tf.gfile.Open(filename, "w") as outfile:
    outfile.write(albert_config.to_json_string())


def _create_record(max_predictions_per_seq, max_seq_length, vocab_size):
  """Returns a tf.train.Example containing random data."""
  example = tf.train.Example()
  example.features.feature["input_ids"].int64_list.value.extend(
      [random.randint(0, vocab_size - 1) for _ in range(max_seq_length)])
  example.features.feature["input_mask"].int64_list.value.extend(
      [random.randint(0, 1) for _ in range(max_seq_length)])
  example.features.feature["masked_lm_positions"].int64_list.value.extend([
      random.randint(0, max_seq_length - 1)
      for _ in range(max_predictions_per_seq)
  ])
  example.features.feature["masked_lm_ids"].int64_list.value.extend([
      random.randint(0, vocab_size - 1) for _ in range(max_predictions_per_seq)
  ])
  example.features.feature["masked_lm_weights"].float_list.value.extend(
      [1. for _ in range(max_predictions_per_seq)])
  example.features.feature["segment_ids"].int64_list.value.extend(
      [0 for _ in range(max_seq_length)])
  example.features.feature["next_sentence_labels"].int64_list.value.append(
      random.randint(0, 1))
  return example


def _create_input_file(filename,
                       max_predictions_per_seq,
                       max_seq_length,
                       vocab_size,
                       size=1000):
  """Creates an input TFRecord file of specified size."""
  with tf.io.TFRecordWriter(filename) as writer:
    for _ in range(size):
      ex = _create_record(max_predictions_per_seq, max_seq_length, vocab_size)
      writer.write(ex.SerializeToString())


class RunPretrainingTest(tf.test.TestCase):

  def _verify_output_file(self, basename):
    self.assertTrue(tf.gfile.Exists(os.path.join(FLAGS.output_dir, basename)))

  def _verify_checkpoint_files(self, name):
    self._verify_output_file(name + ".meta")
    self._verify_output_file(name + ".index")
    self._verify_output_file(name + ".data-00000-of-00001")

  @flagsaver.flagsaver
  def test_pretraining(self):
    # Set up required flags.
    vocab_size = 97
    FLAGS.max_predictions_per_seq = 7
    FLAGS.max_seq_length = 13
    FLAGS.output_dir = tempfile.mkdtemp("output_dir")
    FLAGS.albert_config_file = os.path.join(
        tempfile.mkdtemp("config_dir"), "albert_config.json")
    FLAGS.input_file = os.path.join(
        tempfile.mkdtemp("input_dir"), "input_data.tfrecord")
    FLAGS.do_train = True
    FLAGS.do_eval = True
    FLAGS.num_train_steps = 1
    FLAGS.save_checkpoints_steps = 1

    # Construct requisite input files.
    _create_config_file(FLAGS.albert_config_file, FLAGS.max_seq_length,
                        vocab_size)
    _create_input_file(FLAGS.input_file, FLAGS.max_predictions_per_seq,
                       FLAGS.max_seq_length, vocab_size)

    # Run the pretraining.
    run_pretraining.main(None)

    # Verify output.
    self._verify_checkpoint_files("model.ckpt-best")
    self._verify_checkpoint_files("model.ckpt-1")
    self._verify_output_file("eval_results.txt")
    self._verify_output_file("checkpoint")


if __name__ == "__main__":
  tf.test.main()
