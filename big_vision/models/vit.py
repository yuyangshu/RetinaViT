# Copyright 2022 Big Vision Authors.
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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from math import ceil, log
from typing import Optional, Sequence, Union

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax.image as jmg
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)

  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)

  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)

  return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_1d(l, width, temperature=10_000., dtype=jnp.float32):
  """Follows attention is all you need"""
  def P(v):
    index = jnp.arange(width)
    y = v / temperature ** (2 * jnp.right_shift(index, 1) / width)
    return lax.select(jnp.bitwise_and(index, 1), jnp.cos(y), jnp.sin(y))

  x = jnp.mgrid[:l, :width][0]
  return jnp.asarray(jnp.apply_along_axis(P, 1, x), dtype)[None, :, :]


def posemb_mean_sincos_2d(input_size, patch_size, width, scales, temperature=10_000., dtype=jnp.float32):
  h = w = input_size // patch_size
  base_emb = posemb_sincos_2d(h, w, width, temperature, dtype)
  emb_2d = jnp.reshape(base_emb, [h, w, -1])

  # fill the embeddings into an image sized matrix
  f = lambda i, j, k: emb_2d[i // patch_size, j // patch_size, k]
  img = jnp.vectorize(f)(*jnp.indices((input_size, input_size, width)))

  # calculate weighted embedding for each scale
  mean_embs = []
  for scale in scales:
    h = scale // patch_size
    ratio = input_size / scale
    f = lambda x: jnp.asarray(jnp.split(x, h, axis=1))
    emb_at_scale = f(f(img))
    # mean_embs.append(emb_at_scale.mean(axis=(2, 3)))
    mean_embs.append(emb_at_scale.mean(axis=(2, 3)) * jnp.sqrt(ratio))
    # mean_embs.append(emb_at_scale.mean(axis=(2, 3)) * ratio)

  return jnp.concatenate([jnp.reshape(x, (-1, width)) for x in mean_embs + [base_emb]])


def posemb_conv(input_size, patch_size, width, scales, temperature=10_000., dtype=jnp.float32):
  h = w = input_size // patch_size
  base_emb = posemb_sincos_2d(h, w, width, temperature, dtype)
  emb_2d = jnp.reshape(base_emb, [h, w, -1])

  # fill the embeddings into an image sized matrix
  f = lambda i, j, k: emb_2d[i // patch_size, j // patch_size, k]
  img = jnp.vectorize(f)(*jnp.indices((input_size, input_size, width)))

  # calculate weighted embedding for each scale
  kernel_size, stride = 16, 8
  mean_embs = []
  for scale in scales + [input_size]:
    h = scale // kernel_size
    field_size = input_size // h
    step = field_size * stride // kernel_size
    logging.info(f"scale: {scale}, h: {h}, field_size: {field_size}, step: {step}")

    for i in range(0, input_size - field_size + 1, step):
      for j in range(0, input_size - field_size + 1, step):
        patch = lax.dynamic_slice(img, (i, j, 0), (field_size, field_size, width))
        mean_embs.append(patch.mean(axis=(0, 1)))

  logging.info(len(mean_embs))
  return jnp.asarray(mean_embs)

def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32, scales=None):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  elif typ == "sincos1d":
    return posemb_sincos_1d(seqshape[1], width, dtype=dtype)
  elif typ == "mean_sincos2d":
    return posemb_mean_sincos_2d(seqshape, self.patch_size[0], width, scales, dtype=dtype)
  elif typ == "conv_sincos":
    return posemb_conv(seqshape, self.patch_size[0], width, scales, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
    )(y, y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
    )(y, deterministic)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          mlp_dim=self.mlp_dim, num_heads=self.num_heads, dropout=self.dropout)
      x, out[f"block{lyr:02d}"] = block(x, deterministic)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    n, p, _, c = image.shape

    # get images of all scales
    initial, terminal = ceil(log(self.patch_size[0], 2)), ceil(log(p, 2))
    scales = map(lambda x : 2 ** x, range(initial, terminal))
    scales = [x for x in scales]
    # logging.info(f"scales: {[x for x in scales]}")
    images = [jmg.resize(image, (n, s, s, c), "bilinear") for s in scales] + [image]

    # Patch extraction
    kernel_size, stride = 16, 8
    x = out["stem"] = [nn.Conv(
      self.width, (kernel_size, kernel_size), strides=stride,
      padding="VALID", name=f"embedding_{image.shape[1]}")(image) for image in images]

    x = [jnp.reshape(p, (n, p.shape[1] ** 2, self.width)) for p in x]
    x = jnp.concatenate(x, axis=1)

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(self, self.posemb, p, self.width, "with_posemb", x.dtype, scales=scales)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, self.width), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x = nn.tanh(hid(x))

    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {"Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "H": 1280, "g": 1408, "G": 1664, "e": 1792}[v],
      "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "H": 32, "g": 40, "G": 48, "e": 56}[v],
      "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "H": 5120, "g": 6144, "G": 8192, "e": 15360}[v],
      "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "H": 16, "g": 16, "G": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)
  return jnp.array(grid)


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Original ViT paper variant had posemb in a module:
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # Widely used version before 2022 had posemb in Encoder:
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # Old vit.py used to first concat [cls] token, then add posemb.
  # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
  # so we changed to add posemb then concat [cls]. We can recover the old
  # checkpoint by manually summing [cls] token and its posemb entry.
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP-head variants during ViT-G development had it inlined:
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  del model_cfg

  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(None, init_file)

  restored_params = fix_old_checkpoints(restored_params)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load)

  # resample posemb if needed.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",
    # pylint: disable=line-too-long
    # pylint: enable=line-too-long
}
