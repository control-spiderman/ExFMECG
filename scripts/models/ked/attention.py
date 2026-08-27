"""Multi-head attention used by the ExFMECG query decoder."""

from collections import OrderedDict

import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, mode=""):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mode = mode
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = None
        self.attn_gradients = None
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict)

    @staticmethod
    def _pre_load_state_dict(
        state_dict: OrderedDict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        combined_key = prefix + "in_proj_weight"
        if combined_key not in state_dict:
            return
        weight = state_dict.pop(combined_key)
        bias = state_dict.pop(prefix + "in_proj_bias")
        size = weight.shape[1]
        for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
            state_dict[prefix + name + ".weight"] = weight[
                index * size:(index + 1) * size
            ]
            state_dict[prefix + name + ".bias"] = bias[
                index * size:(index + 1) * size
            ]

    def _save_attention_gradient(self, gradient):
        self.attn_gradients = gradient

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        del need_weights
        target_length, batch_size, embed_dim = query.shape
        source_length = key.shape[0]
        scale = self.head_dim ** -0.5

        query = self.q_proj(query) * scale
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.reshape(
            target_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        key = key.reshape(
            source_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)
        value = value.reshape(
            source_length, batch_size * self.num_heads, self.head_dim
        ).transpose(0, 1)

        scores = torch.bmm(query, key.transpose(1, 2))
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].expand(
                batch_size, self.num_heads, target_length, source_length
            )
            scores = scores.view(
                batch_size, self.num_heads, target_length, source_length
            )
            scores = scores.masked_fill(mask, float("-inf")).flatten(0, 1)

        attention = self.dropout(torch.softmax(scores, dim=-1))
        self.attn = attention
        if self.mode == "explain" and attention.requires_grad:
            attention.register_hook(self._save_attention_gradient)

        output = torch.bmm(attention, value)
        output = output.transpose(0, 1).reshape(
            target_length, batch_size, embed_dim
        )
        return self.out_proj(output)
