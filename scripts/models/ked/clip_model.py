"""Clinical-text encoder and text-query decoder used by ExFMECG."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, BertConfig

from scripts.models.ked.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)


class CLP_clinical(nn.Module):
    def __init__(
        self,
        bert_model_name,
        embed_dim=768,
        freeze_layers=None,
        unfreeze_layers=None,
    ):
        super().__init__()
        config = BertConfig.from_pretrained(
            bert_model_name,
            output_hidden_states=True,
        )
        self.bert_model = AutoModel.from_pretrained(bert_model_name, config=config)
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if unfreeze_layers:
            for parameter in self.bert_model.parameters():
                parameter.requires_grad = False
            for index in unfreeze_layers:
                for parameter in self.bert_model.encoder.layer[index].parameters():
                    parameter.requires_grad = True
        elif freeze_layers:
            for index in freeze_layers:
                for parameter in self.bert_model.encoder.layer[index].parameters():
                    parameter.requires_grad = False

    def encode_text(self, tokens):
        output = self.bert_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        return output.pooler_output

    def forward(self, first, second):
        first = F.normalize(self.encode_text(first), dim=-1)
        second = F.normalize(self.encode_text(second), dim=-1)
        return first, second, self.logit_scale.exp()


class TQNModel(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        class_num=2,
        num_layers=3,
        output_type="",
        mode="",
    ):
        del class_num
        super().__init__()
        self.d_model = embed_dim
        self.use_encoder = False
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        layer = TransformerDecoderLayer(
            embed_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            normalize_before=True,
            mode=mode,
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            layer,
            num_layers,
            self.decoder_norm,
            return_intermediate=False,
        )
        self.dropout_feas = nn.Dropout(0.1)
        self.output_type = output_type
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, ecg_features, text_features, return_atten=False, output_type=None):
        del output_type
        batch_size = ecg_features.shape[0]
        memory = self.decoder_norm(ecg_features.transpose(0, 1))
        queries = text_features.unsqueeze(1).repeat(1, batch_size, 1)
        queries = self.decoder_norm(queries)
        features, attention = self.decoder(queries, memory)
        features = self.dropout_feas(features).transpose(0, 1)
        return (features, attention) if return_atten else features
