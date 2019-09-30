"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy
import pdb

#
def self_attention_layer(from_tensor,
                    to_tensor,
                    self_adaptive=True,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = modeling.get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = modeling.get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = modeling.reshape_to_matrix(from_tensor)
  to_tensor_2d = modeling.reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  raw_query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=modeling.create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  raw_key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=modeling.create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  raw_value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=modeling.create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(raw_query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(raw_key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

  # self-interactive attention (alpha version)
  # [F, F_sm] x [F, T] x [T_sm, T] => [F, T]
  if self_adaptive:
    # `left_matrix` = [B, N, F, F_sm]
    left_matrix = tf.matmul(query_layer, query_layer, transpose_b=True)
    left_matrix = tf.nn.softmax(left_matrix)

    # `right_matrix` = [B, N, T_sm, T]
    right_matrix = tf.matmul(key_layer, key_layer, transpose_b=True)
    right_matrix = tf.nn.softmax(right_matrix)
    right_matrix = tf.transpose(right_matrix, [0,1,3,2])    

    # `left_product` = [B, N, F, F_sm] x [B, N, F, T]
    left_product = tf.matmul(left_matrix, attention_scores)

    # `attention_scores` = [B, N, F, T] x [B, N, T_sm, T]
    attention_scores = tf.matmul(left_product, right_matrix)

  attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = modeling.dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      raw_value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer #, raw_query_layer, raw_value_layer


#
def gather_indexes(sequence_tensor, positions):
  """
  Gathers the vectors at the specific positions over a minibatch.
  sequence_tensor: [batch, seq_length, width]
  positions: [batch, n]
  """
  sequence_shape  = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size      = sequence_shape[0]
  seq_length      = sequence_shape[1]
  width           = sequence_shape[2]

  flat_offsets    = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions  = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
  output_tensor   = tf.gather(flat_sequence_tensor, flat_positions)

  return output_tensor


#
def simple_attention(tensor_a, tensor_b, attention_mask, self_adaptive=False):
  '''
  input tensor_a/b: [B, L, H]

  '''
  sequence_shape  = modeling.get_shape_list(tensor_a, expected_rank=3)
  batch_size      = sequence_shape[0]
  seq_length      = sequence_shape[1]
  width           = sequence_shape[2]

  # self-interactive attention (alpha version)
  # [F, F_sm] x [F, T] x [T_sm, T] => [F, T]
  if self_adaptive:
    # `left_matrix` = [B, F, F_sm]
    left_matrix = tf.matmul(tensor_a, tensor_a, transpose_b=True)
    left_matrix = tf.nn.softmax(left_matrix)

    # `right_matrix` = [B, T_sm, T]
    right_matrix = tf.matmul(tensor_b, tensor_b, transpose_b=True)
    right_matrix = tf.nn.softmax(right_matrix)
    right_matrix = tf.transpose(right_matrix, [0, 2, 1])

    # `left_product` = [B, F, F_sm] x [B, F, T]
    attention_matrix = tf.matmul(tensor_a, tensor_b, transpose_b=True)
    left_product = tf.matmul(left_matrix, attention_matrix)

    # `attention_matrix` = [B, F, T] x [B, T_sm, T]
    attention_matrix = tf.matmul(left_product, right_matrix)
  else:
    # attention_matrix = [B, L, L] = [B, L, H] * [B, H, L]
    attention_matrix = tf.matmul(tensor_a, tensor_b, transpose_b=True)

  # attention_raw_value = [B, L]
  attention_raw_value = tf.reduce_mean(attention_matrix, axis=1)
  
  # apply mask
  adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
  attention_raw_value += adder

  # apply softmax
  # attention_scores = [B, L_sm]
  attention_scores = tf.nn.softmax(attention_raw_value)

  # get attended representation
  attended_repr = tf.matmul(tf.expand_dims(attention_scores, 1), tensor_a)
  attended_repr = tf.squeeze(attended_repr, 1)

  return attended_repr


#
def extract_span_tensor(bert_config, sequence_tensor, output_span_mask, 
                        start_positions, end_positions,   
                        scope=None):
  sequence_shape  = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  seq_length = sequence_shape[1]

  output_span_mask = tf.cast(output_span_mask, tf.float32)
  filtered_sequence_tensor = sequence_tensor * tf.expand_dims(output_span_mask, -1)

  with tf.variable_scope(scope, default_name="span_loss"):
    fw_output_tensor = gather_indexes(filtered_sequence_tensor, tf.expand_dims(start_positions, -1))
    bw_output_tensor = gather_indexes(filtered_sequence_tensor, tf.expand_dims(end_positions, -1))
    att_output_tensor = simple_attention(filtered_sequence_tensor, filtered_sequence_tensor, output_span_mask, self_adaptive=True)
    output_tensor = tf.concat([fw_output_tensor, bw_output_tensor, att_output_tensor], axis=-1)      

  return output_tensor


#
def attention_fusion_layer(bert_config,
                           input_tensor, input_ids, input_mask,
                           source_input_tensor, source_input_ids, source_input_mask, 
                           is_training=True, scope=None):
  '''
  Attention Fusion Layer for merging source representation and target representation.
  '''
  # universal shapes
  input_tensor_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_tensor_shape[0]
  seq_length = input_tensor_shape[1]
  hidden_size = input_tensor_shape[2]
  source_input_tensor_shape = modeling.get_shape_list(source_input_tensor, expected_rank=3)
  source_seq_length = source_input_tensor_shape[1]
  source_hidden_size = source_input_tensor_shape[2]

  # universal parameters
  UNIVERSAL_DROPOUT_RATE = 0.1
  if not is_training:
    UNIVERSAL_DROPOUT_RATE = 0  # we disable dropout when predicting
  UNIVERSAL_INIT_RANGE = bert_config.initializer_range
  NUM_ATTENTION_HEAD = bert_config.num_attention_heads

  # attention fusion module
  with tf.variable_scope(scope, default_name="attention_fusion"):
    ATTENTION_HEAD_SIZE = int(source_hidden_size / NUM_ATTENTION_HEAD)
    with tf.variable_scope("attention"):
      source_attended_repr = self_attention_layer(
        from_tensor=input_tensor,
        to_tensor=source_input_tensor,
        attention_mask=modeling.create_attention_mask_from_input_mask(input_ids, source_input_mask),
        num_attention_heads=NUM_ATTENTION_HEAD,
        size_per_head=ATTENTION_HEAD_SIZE,
        attention_probs_dropout_prob=UNIVERSAL_DROPOUT_RATE,
        initializer_range=UNIVERSAL_INIT_RANGE,
        do_return_2d_tensor=False,
        batch_size=batch_size,
        from_seq_length=seq_length,
        to_seq_length=source_seq_length,
        self_adaptive=True)

    with tf.variable_scope("transform"):
      source_attended_repr = tf.layers.dense(
                  source_attended_repr,
                  source_hidden_size,
                  kernel_initializer=modeling.create_initializer(UNIVERSAL_INIT_RANGE))
      source_attended_repr = modeling.dropout(source_attended_repr, UNIVERSAL_DROPOUT_RATE)
      source_attended_repr = modeling.layer_norm(source_attended_repr + source_input_tensor)

  final_output = tf.concat([input_tensor, source_attended_repr], axis=-1)
  
  return final_output


#
def span_output_layer(bert_config, input_tensor, input_span_mask=None, scope=None):
  input_tensor_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_tensor_shape[0]
  seq_length = input_tensor_shape[1]
  hidden_size = input_tensor_shape[2]

  # output layers
  with tf.variable_scope(scope, default_name="cls/squad"):
    output_weights = tf.get_variable("output_weights", [2, hidden_size], 
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [2], 
                                  initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(input_tensor, [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  if input_span_mask is not None:
    adder           = (1.0 - tf.cast(input_span_mask, tf.float32)) * -10000.0
    start_logits   += adder
    end_logits     += adder

  return (start_logits, end_logits)


#
def create_model(bert_config, is_training, 
                 input_ids, input_mask, segment_ids, input_span_mask, output_span_mask,
                 source_input_ids, source_input_mask, source_segment_ids, source_input_span_mask, source_output_span_mask,
                 start_positions, end_positions, source_start_positions, source_end_positions,
                 use_one_hot_embeddings):

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope='bert')

  source_model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=source_input_ids,
      input_mask=source_input_mask,
      token_type_ids=source_segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope='bert',
      reuse=True)

  # get BERT outputs
  target_final_hidden   = model.get_sequence_output()
  source_final_hidden    = source_model.get_sequence_output()

  # source BERT predictions
  (source_raw_start_logits, source_raw_end_logits) = span_output_layer(bert_config, 
                                                                       source_final_hidden, source_input_span_mask, 
                                                                       scope='source/cls/squad')

  # target BERT predictions
  source_attended_repr = attention_fusion_layer(bert_config,
                                                target_final_hidden, input_ids, input_mask,
                                                source_final_hidden, source_input_ids, source_input_mask, 
                                                is_training=is_training, scope='target/attention_fusion')  
  (target_start_logits, target_end_logits) = span_output_layer(bert_config, 
                                                               source_attended_repr, input_span_mask, 
                                                               scope='target/cls/squad')
  # 
  source_span_gt_tensor  = extract_span_tensor(bert_config, source_final_hidden, source_output_span_mask, 
                                              source_start_positions, source_end_positions,
                                              scope='source_gt')
  target_span_gt_tensor = extract_span_tensor(bert_config, target_final_hidden, output_span_mask, 
                                              start_positions, end_positions,
                                              scope='target_gt')  

  return (target_start_logits, target_end_logits, source_raw_start_logits, source_raw_end_logits, target_span_gt_tensor, source_span_gt_tensor)


#
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    input_span_mask = features["input_span_mask"]
    output_span_mask = features["output_span_mask"]

    source_input_ids = features["source_input_ids"]
    source_input_mask = features["source_input_mask"]
    source_segment_ids = features["source_segment_ids"]
    source_input_span_mask = features["source_input_span_mask"]
    source_output_span_mask = features["source_output_span_mask"]

    start_positions = features["start_positions"]
    end_positions   = features["end_positions"]
    source_start_positions = features["source_start_positions"]
    source_end_positions   = features["source_end_positions"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits, source_raw_start_logits, source_raw_end_logits, target_span_gt_tensor, source_span_gt_tensor) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        input_span_mask=input_span_mask,
        output_span_mask=output_span_mask,
        source_input_ids=source_input_ids,
        source_input_mask=source_input_mask,
        source_segment_ids=source_segment_ids,
        source_input_span_mask=source_input_span_mask,
        source_output_span_mask=source_output_span_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        source_start_positions=source_start_positions, 
        source_end_positions=source_end_positions,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # print info
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        on_hot_pos    = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
        log_probs     = tf.nn.log_softmax(logits, axis=-1)
        loss          = -tf.reduce_mean(tf.reduce_sum(on_hot_pos * log_probs, axis=-1))
        return loss

      def cosine_similarity(tensor1, tensor2):
        cosine_val = 1 - tf.losses.cosine_distance(tensor1, tensor2, axis=0)
        return cosine_val

      start_loss  = compute_loss(start_logits, start_positions)
      end_loss    = compute_loss(end_logits, end_positions)
      main_loss   = (start_loss + end_loss) / 2.0

      aux_lambda        = cosine_similarity(target_span_gt_tensor, source_span_gt_tensor)
      source_start_loss  = compute_loss(source_raw_start_logits, source_start_positions)
      source_end_loss    = compute_loss(source_raw_end_logits, source_end_positions)
      aux_loss          = tf.maximum(0.0, aux_lambda) * (source_start_loss + source_end_loss) / 2.0

      total_loss = main_loss + aux_loss

      train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      start_logits = tf.nn.log_softmax(start_logits, axis=-1)
      end_logits = tf.nn.log_softmax(end_logits, axis=-1)
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn

