# Copyright 2024 Big Vision Authors.
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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import

import functools

from big_vision.evaluators import common
import big_vision.utils as u
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn, loss_name):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _eval_fn(train_state, batch, labels, mask):
    logits, out = predict_fn(train_state, batch)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    loss = getattr(u, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jnp.sum(loss * mask)

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(
        labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jnp.sum(top1_correct * mask)
    nseen = jnp.sum(mask)
    return ncorrect, loss, nseen, out["attn_distribution"], out["attn"], out["before_mlp"], out["query"], out["key"], out["value"]
  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, loss_name, label_key='labels', **kw):
    self.get_data_iter, self.steps = common.eval_input_pipeline(**kw)
    self.eval_fn = get_eval_fn(predict_fn, loss_name)
    self.label_key = label_key

  def run(self, train_state):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    # (224/16)^2 + (128/16)^2 + (64/16)^2 + (32/16)^2 + (16/16)^2 = 281
    attn_distribution, attn, before_mlp, query, key, value = jnp.empty((0, 281)), jnp.empty((0, 281)), \
      jnp.empty((0, 281)), jnp.empty((0, 281)), jnp.empty((0, 281)), jnp.empty((0, 281))
    for _, batch in zip(range(self.steps), self.get_data_iter()):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrect, batch_losses, batch_nseen, batch_attn_distribution, batch_attn, batch_before_mlp, batch_q, batch_k, batch_v = \
          multihost_utils.process_allgather(self.eval_fn(train_state, batch, labels, mask))
      ncorrect += batch_ncorrect
      loss += batch_losses
      nseen += batch_nseen

      attn_distribution = jnp.concatenate((attn_distribution, batch_attn_distribution), axis=0)
      attn = jnp.concatenate((attn, batch_attn), axis=0)
      before_mlp = jnp.concatenate((before_mlp, batch_before_mlp), axis=0)
      query = jnp.concatenate((query, batch_q), axis=0)
      key = jnp.concatenate((key, batch_k), axis=0)
      value = jnp.concatenate((value, batch_v), axis=0)

    # trim the values from empty input, direct output is ceil()ed to multiples of batch size
    attn_distribution = attn_distribution[0:int(nseen):]
    attn = attn[0:int(nseen):]
    before_mlp = before_mlp[0:int(nseen):]
    query = query[0:int(nseen):]
    key = key[0:int(nseen):]
    value = value[0:int(nseen):]

    yield ('prec@1', ncorrect / nseen)
    yield ('loss', loss / nseen)
    yield ('sample count', nseen)
    yield ('attn_distribution', attn_distribution)
    yield ('attn', attn)
    yield ('before_mlp', before_mlp)
    yield ('query', query)
    yield ('key', key)
    yield ('value', value)
