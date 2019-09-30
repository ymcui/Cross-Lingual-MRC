# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

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
from layers import model_fn_builder

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")


## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "eval_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_integer("rand_seed", 12345, "set random seed")


# set random seed (i don't know whether it works or not)
if FLAGS.do_train:
  numpy.random.seed(int(FLAGS.rand_seed))
  tf.set_random_seed(int(FLAGS.rand_seed))
else:
  numpy.random.seed(12345)
  tf.set_random_seed(12345)

#
class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               source_question_text,
               source_doc_tokens,
               orig_answer_text=None,
               source_orig_answer_text=None,
               start_position=None,
               end_position=None,
               source_start_position=None,
               source_end_position=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.source_question_text = source_question_text
    self.source_doc_tokens = source_doc_tokens
    self.orig_answer_text = orig_answer_text
    self.source_orig_answer_text = source_orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.source_start_position = source_start_position
    self.source_end_position = source_end_position


  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               source_tokens,
               source_token_to_orig_map,
               source_token_is_max_context,
               source_input_ids,
               source_input_mask,
               source_segment_ids,
               input_span_mask,
               source_input_span_mask,
               start_position=None,
               end_position=None,
               source_start_position=None,
               source_end_position=None,
               output_span_mask=None,
               source_output_span_mask=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.source_tokens = source_tokens
    self.source_token_to_orig_map = source_token_to_orig_map
    self.source_token_is_max_context = source_token_is_max_context
    self.source_input_ids = source_input_ids
    self.source_input_mask = source_input_mask
    self.source_segment_ids = source_segment_ids
    self.input_span_mask = input_span_mask
    self.source_input_span_mask = source_input_span_mask
    self.start_position = start_position
    self.end_position = end_position
    self.source_start_position = source_start_position
    self.source_end_position = source_end_position
    self.output_span_mask = output_span_mask
    self.source_output_span_mask = source_output_span_mask


#
def customize_tokenizer(text, do_lower_case=False):
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
  temp_x = ""
  text = tokenization.convert_to_unicode(text)
  for c in text:
    if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(c) or tokenization._is_control(c):
      temp_x += " " + c + " "
    else:
      temp_x += c
  if do_lower_case:
    temp_x = temp_x.lower()
  return temp_x.split()

#
class ChineseFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False):
    self.vocab = tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    split_tokens = []
    for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return tokenization.convert_by_vocab(self.inv_vocab, ids)


#
def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of SquadExample."""
  is_training = True
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  #
  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      raw_doc_tokens = customize_tokenizer(paragraph_text, do_lower_case=FLAGS.do_lower_case)
      doc_tokens = []
      char_to_word_offset = []
      prev_is_whitespace = True

      k = 0
      temp_word = ""
      for c in paragraph_text:
        if tokenization._is_whitespace(c):
          char_to_word_offset.append(k-1)
          continue
        else:
          temp_word += c
          char_to_word_offset.append(k)
        
        if FLAGS.do_lower_case:
          temp_word = temp_word.lower()

        if temp_word == raw_doc_tokens[k]:
          doc_tokens.append(temp_word)
          temp_word = ""
          k += 1

      assert k==len(raw_doc_tokens)

      # process pivot example
      def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
          return True
        return False
      source_paragraph_text = paragraph["trans_context"]
      source_doc_tokens = []
      source_char_to_word_offset = []
      source_prev_is_whitespace = True
      for c in source_paragraph_text:
        if is_whitespace(c):
          source_prev_is_whitespace = True
        else:
          if source_prev_is_whitespace:
            source_doc_tokens.append(c)
          else:
            source_doc_tokens[-1] += c
          source_prev_is_whitespace = False
        source_char_to_word_offset.append(len(source_doc_tokens) - 1)


      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        source_question_text = qa["trans_question"]
        start_position = None
        end_position = None
        orig_answer_text = None

        source_start_position = None
        source_end_position = None
        source_orig_answer_text = None

        if is_training:
          answer = qa["answers"][0]
          orig_answer_text = answer["text"]

          if orig_answer_text not in paragraph_text:
            tf.logging.warning("Could not find answer")
            start_position = -1
            end_position = -1
            orig_answer_text = ""
          else:
            answer_offset = paragraph_text.index(orig_answer_text)
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = "".join(
                doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = "".join(
                tokenization.whitespace_tokenize(orig_answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
              pdb.set_trace()
              tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
              continue

          #
          source_answer = qa["answers"][0]
          source_orig_answer_text = source_answer["trans_aligned_text"]

          if source_orig_answer_text not in source_paragraph_text:
            tf.logging.warning("Could not find pivot answer: %s", source_orig_answer_text)
            source_start_position = -1
            source_end_position = -1
          else:
            source_answer_offset = source_paragraph_text.index(source_orig_answer_text)
            source_answer_length = len(source_orig_answer_text)
            source_start_position = source_char_to_word_offset[source_answer_offset]
            source_end_position = source_char_to_word_offset[source_answer_offset + source_answer_length - 1]

            #
            source_actual_text = "".join(
                source_doc_tokens[source_start_position:(source_end_position + 1)])
            source_cleaned_answer_text = "".join(
                tokenization.whitespace_tokenize(source_orig_answer_text))
            if source_actual_text.find(source_cleaned_answer_text) == -1:
              pdb.set_trace()
              tf.logging.warning("Could not find pivot answer: '%s' vs. '%s'", source_actual_text, source_cleaned_answer_text)
              continue

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            source_question_text=source_question_text,
            source_doc_tokens=source_doc_tokens,
            source_orig_answer_text=source_orig_answer_text,
            source_start_position=source_start_position,
            source_end_position=source_end_position)
        examples.append(example)
  
  tf.logging.info("**********read_squad_examples complete!**********")
  
  return examples


#
def convert_source_examples_to_features(tokenizer, example, is_training):
  """Loads a data file into a list of `InputBatch`s."""
  is_training = True
  query_tokens = tokenizer.tokenize(example.source_question_text)

  if len(query_tokens) > FLAGS.max_query_length:
    query_tokens = query_tokens[0: FLAGS.max_query_length]

  tok_to_orig_index = []
  orig_to_tok_index = []
  all_doc_tokens = []
  for (i, token) in enumerate(example.source_doc_tokens):
    orig_to_tok_index.append(len(all_doc_tokens))
    sub_tokens = tokenizer.tokenize(token)
    for sub_token in sub_tokens:
      tok_to_orig_index.append(i)
      all_doc_tokens.append(sub_token)

  if is_training:
    if example.source_start_position == -1 or example.source_end_position == -1:
      tok_start_position = -1
      tok_end_position = -1
    else:
      tok_start_position = orig_to_tok_index[example.source_start_position]
      if example.source_end_position < len(example.source_doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.source_end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.source_orig_answer_text)

  # The -3 accounts for [CLS], [SEP] and [SEP]
  max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 3

  # We can have documents that are longer than the maximum sequence length.
  # To deal with this we do a sliding window approach, where we take chunks
  # of the up to our max length with a stride of `doc_stride`.
  _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
      "DocSpan", ["start", "length"])
  doc_spans = []
  start_offset = 0
  while start_offset < len(all_doc_tokens):
    length = len(all_doc_tokens) - start_offset
    if length > max_tokens_for_doc:
      length = max_tokens_for_doc
    doc_spans.append(_DocSpan(start=start_offset, length=length))
    if start_offset + length == len(all_doc_tokens):
      break
    start_offset += min(length, FLAGS.doc_stride)

  #
  INPUT_IDS = []
  INPUT_MASK = []
  SEGMENT_IDS = []
  INPUT_SPAN_MASK = []
  OUTPUT_SPAN_MASK = []
  START_POSITION = []
  END_POSITION = []
  TOKENS = []
  TOKEN_TO_ORIG_MAP = []
  TOKEN_IS_MAX_CONTEXT = []

  for (doc_span_index, doc_span) in enumerate(doc_spans):
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    input_span_mask = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    input_span_mask.append(1)
    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
      input_span_mask.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_span_mask.append(0)

    for i in range(doc_span.length):
      split_token_index = doc_span.start + i
      token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

      is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                             split_token_index)
      token_is_max_context[len(tokens)] = is_max_context
      tokens.append(all_doc_tokens[split_token_index])
      segment_ids.append(1)
      input_span_mask.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    input_span_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < FLAGS.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      input_span_mask.append(0)

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length
    assert len(input_span_mask) == FLAGS.max_seq_length
    
    start_position = None
    end_position = None
    output_span_mask = None
    if is_training:
      # For training, if our document chunk does not contain an annotation
      # we throw it out, since there is nothing to predict.
      doc_start = doc_span.start
      doc_end = doc_span.start + doc_span.length - 1
      out_of_span = False
      if not (tok_start_position >= doc_start and
              tok_end_position <= doc_end):
        out_of_span = True
      if out_of_span:
        start_position = 0
        end_position = 0
      else:
        doc_offset = len(query_tokens) + 2
        start_position = tok_start_position - doc_start + doc_offset
        end_position = tok_end_position - doc_start + doc_offset

      output_span_mask = [0] * FLAGS.max_seq_length
      for osm_idx in range(start_position, end_position+1):
        output_span_mask[osm_idx] = 1


    INPUT_IDS.append(input_ids)
    INPUT_MASK.append(input_mask)
    SEGMENT_IDS.append(segment_ids)
    INPUT_SPAN_MASK.append(input_span_mask)
    OUTPUT_SPAN_MASK.append(output_span_mask)
    START_POSITION.append(start_position)
    END_POSITION.append(end_position)
    TOKENS.append(tokens)
    TOKEN_TO_ORIG_MAP.append(token_to_orig_map)
    TOKEN_IS_MAX_CONTEXT.append(token_is_max_context)

  ret_array = (INPUT_IDS, INPUT_MASK, SEGMENT_IDS, 
               INPUT_SPAN_MASK, OUTPUT_SPAN_MASK,
               START_POSITION, END_POSITION,
               TOKENS, TOKEN_TO_ORIG_MAP,TOKEN_IS_MAX_CONTEXT)

  return ret_array


#
def convert_token_to_ids(vocab, items):
  output = []
  for item in items:
    if item in vocab:
      output.append(vocab[item])
    else:
      output.append(vocab['[UNK]'])
  return output

#
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""
  is_training = True
  unique_id = 1000000000
  tokenizer = ChineseFullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  #source_tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.source_vocab_file, do_lower_case=FLAGS.source_do_lower_case)

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    # here we use a approximated span for pivot language
    (source_INPUT_IDS, source_INPUT_MASK, source_SEGMENT_IDS, source_INPUT_SPAN_MASK, source_OUTPUT_SPAN_MASK,  source_START_POSITION, source_END_POSITION, source_TOKENS, source_TOKEN_TO_ORIG_MAP, source_TOKEN_IS_MAX_CONTEXT) = convert_source_examples_to_features(tokenizer, example, is_training)
    source_DOC_SPAN_LEN = len(source_INPUT_IDS)

    # process target sample
    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      input_span_mask = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      input_span_mask.append(1)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
        input_span_mask.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)
      input_span_mask.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
        input_span_mask.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)
      input_span_mask.append(0)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        input_span_mask.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(input_span_mask) == max_seq_length

      start_position = None
      end_position = None
      output_span_mask = None
      if is_training:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

        output_span_mask = [0] * max_seq_length
        for osm_idx in range(start_position, end_position+1):
          output_span_mask[osm_idx] = 1

      # as target/source doc_span may differ, we use heuristic here to synchronize them
      if doc_span_index >= source_DOC_SPAN_LEN:
        source_tokens = source_TOKENS[-1]
        source_token_to_orig_map = source_TOKEN_TO_ORIG_MAP[-1]
        source_token_is_max_context = source_TOKEN_IS_MAX_CONTEXT[-1]
        source_input_ids = source_INPUT_IDS[-1]
        source_input_mask = source_INPUT_MASK[-1]
        source_segment_ids = source_SEGMENT_IDS[-1]
        source_input_span_mask = source_INPUT_SPAN_MASK[-1]
        source_output_span_mask = source_OUTPUT_SPAN_MASK[-1]
        source_start_position = source_START_POSITION[-1]
        source_end_position = source_END_POSITION[-1]
      else:
        source_tokens = source_TOKENS[doc_span_index]
        source_token_to_orig_map = source_TOKEN_TO_ORIG_MAP[doc_span_index]
        source_token_is_max_context = source_TOKEN_IS_MAX_CONTEXT[doc_span_index]
        source_input_ids = source_INPUT_IDS[doc_span_index]
        source_input_mask = source_INPUT_MASK[doc_span_index]
        source_segment_ids = source_SEGMENT_IDS[doc_span_index]
        source_input_span_mask = source_INPUT_SPAN_MASK[doc_span_index]
        source_output_span_mask = source_OUTPUT_SPAN_MASK[doc_span_index]
        source_start_position = source_START_POSITION[doc_span_index]
        source_end_position = source_END_POSITION[doc_span_index]        


      if example_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info(
          "input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
        tf.logging.info("source_input_ids: %s" % " ".join([str(x) for x in source_input_ids]))
        tf.logging.info(
            "source_input_mask: %s" % " ".join([str(x) for x in source_input_mask]))
        tf.logging.info(
            "source_segment_ids: %s" % " ".join([str(x) for x in source_segment_ids]))
        if is_training:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))

          source_answer_text = " ".join(source_tokens[source_start_position:(source_end_position + 1)])
          tf.logging.info("source_start_position: %d" % (source_start_position))
          tf.logging.info("source_end_position: %d" % (source_end_position))
          tf.logging.info("answer: %s" % (tokenization.printable_text(source_answer_text)))
          tf.logging.info("source_output_span_mask: %s" % ' '.join([str(x) for x in source_output_span_mask]))

      #else:
      #  if example_index % 100 ==0:
      #    tf.logging.info("%d processed", example_index)

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          input_span_mask=input_span_mask,
          start_position=start_position,
          end_position=end_position,
          source_tokens=source_tokens,
          source_token_to_orig_map=source_token_to_orig_map,
          source_token_is_max_context=source_token_is_max_context,
          source_input_ids=source_input_ids,
          source_input_mask=source_input_mask,
          source_segment_ids=source_segment_ids,
          source_input_span_mask=source_input_span_mask,
          source_start_position=source_start_position,
          source_end_position=source_end_position,
          output_span_mask=output_span_mask,
          source_output_span_mask=source_output_span_mask)

      # Run callback
      output_fn(feature)

      unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_span_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "source_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "source_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "source_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "source_input_span_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "output_span_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "source_output_span_mask": tf.FixedLenFeature([seq_length], tf.int64),
  }

  #if is_training:
  name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
  name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
  name_to_features["source_start_positions"] = tf.FixedLenFeature([], tf.int64)
  name_to_features["source_end_positions"] = tf.FixedLenFeature([], tf.int64)


  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]
    prelim_predictions = []
    
    for (feature_index, feature) in enumerate(features):  # multi-trunk
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        final_text = final_text.replace(' ','')
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit,
              start_index=pred.start_index,
              end_index=pred.end_index))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=-1, end_index=-1))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_index"] = entry.start_index
      output["end_index"] = entry.end_index
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    all_predictions[example.qas_id] = best_non_null_entry.text #nbest_json[0]["text"]
    all_nbest_json[example.qas_id] = nbest_json

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.gfile.GFile(output_prediction_file+"2", "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

  with tf.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["input_span_mask"] = create_int_feature(feature.input_span_mask)
    features["source_input_span_mask"] = create_int_feature(feature.source_input_span_mask)
    features["source_input_ids"] = create_int_feature(feature.source_input_ids)
    features["source_input_mask"] = create_int_feature(feature.source_input_mask)
    features["source_segment_ids"] = create_int_feature(feature.source_segment_ids)

    #if self.is_training:
    features["start_positions"] = create_int_feature([feature.start_position])
    features["end_positions"] = create_int_feature([feature.end_position])
    features["source_start_positions"] = create_int_feature([feature.source_start_position])
    features["source_end_positions"] = create_int_feature([feature.source_end_position])
    features["output_span_mask"] = create_int_feature(feature.output_span_mask)
    features["source_output_span_mask"] = create_int_feature(feature.source_output_span_mask)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
  
  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")
  if FLAGS.do_eval:
    if not FLAGS.eval_file:
      raise ValueError(
          "If `do_eval` is True, then `eval_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


############################
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=2,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = read_squad_examples(input_file=FLAGS.train_file, is_training=True)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(int(FLAGS.rand_seed))
    rng.shuffle(train_examples)

    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
        is_training=True)
    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()
    num_features = train_writer.num_features
    train_examples_len = len(train_examples)
    del train_examples

    num_train_steps = int(num_features / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", train_examples_len)
    tf.logging.info("  Num split examples = %d", num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # do training
  if FLAGS.do_train:
    train_writer_filename = train_writer.filename

    train_input_fn = input_fn_builder(
        input_file=train_writer_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  # do predictions
  if FLAGS.do_predict:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "predict.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))


    output_json_name = "dev_predictions.json"
    output_nbest_name = "dev_nbest_predictions.json"

    output_prediction_file = os.path.join(FLAGS.output_dir, output_json_name)
    output_nbest_file = os.path.join(FLAGS.output_dir, output_nbest_name)

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file)



if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()





