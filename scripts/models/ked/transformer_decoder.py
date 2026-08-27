"""Transformer decoder used for ECG-to-text-query prediction."""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from scripts.models.ked.attention import MultiheadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        target,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        output_type=None,
    ):
        del output_type
        output = target
        intermediates = []
        attention = None
        for layer in self.layers:
            output, attention = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediates.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediates)
        return output, attention


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        mode="",
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, mode=mode)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, mode=mode)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _activation(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def _with_positional_embedding(tensor, position):
        return tensor if position is None else tensor + position

    def _forward_pre(
        self,
        target,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        pos,
        query_pos,
    ):
        if target.shape[0] == 1:
            normalized = self.norm2(target)
            target = target + self.dropout2(
                self.multihead_attn(
                    self._with_positional_embedding(normalized, query_pos),
                    self._with_positional_embedding(memory, pos),
                    memory,
                    memory_key_padding_mask,
                    attn_mask=memory_mask,
                )
            )
            normalized = self.norm3(target)
            target = target + self.dropout3(
                self.linear2(self.dropout(self.activation(self.linear1(normalized))))
            )
            return target, None

        normalized = self.norm1(target)
        query = key = self._with_positional_embedding(normalized, query_pos)
        target = target + self.dropout1(
            self.self_attn(query, key, normalized, tgt_key_padding_mask, attn_mask=tgt_mask)
        )
        normalized = self.norm2(target)
        target = target + self.dropout2(
            self.multihead_attn(
                self._with_positional_embedding(normalized, query_pos),
                self._with_positional_embedding(memory, pos),
                memory,
                memory_key_padding_mask,
                attn_mask=memory_mask,
            )
        )
        normalized = self.norm3(target)
        target = target + self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(normalized))))
        )
        return target, None

    def _forward_post(
        self,
        target,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        pos,
        query_pos,
    ):
        query = key = self._with_positional_embedding(target, query_pos)
        attended = self.self_attn(query, key, target, tgt_key_padding_mask, attn_mask=tgt_mask)
        target = self.norm1(target + self.dropout1(attended))
        attended = self.multihead_attn(
            self._with_positional_embedding(target, query_pos),
            self._with_positional_embedding(memory, pos),
            memory,
            memory_key_padding_mask,
            attn_mask=memory_mask,
        )
        target = self.norm2(target + self.dropout2(attended))
        feedforward = self.linear2(self.dropout(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout3(feedforward)), None

    def forward(
        self,
        target,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        residual=True,
        output_type=None,
    ):
        del residual, output_type
        method = self._forward_pre if self.normalize_before else self._forward_post
        return method(
            target,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise ValueError(f"Unsupported activation: {name}")
