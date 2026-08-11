from __future__ import annotations

import torch
from torch import nn


class MeanMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, features, padding_mask):
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (features * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.net(pooled)


class SPOTER208(nn.Module):
    def __init__(self, *, input_dim=208, d_model=256, nhead=8, encoder_layers=6,
                 decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="gelu", max_sequence_length=256, num_classes=300):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.position = nn.Parameter(torch.empty(1, max_sequence_length, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                         activation=activation, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, encoder_layers, norm=nn.LayerNorm(d_model))
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                         activation=activation, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, decoder_layers, norm=nn.LayerNorm(d_model))
        self.class_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.class_query, std=0.02)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features, padding_mask):
        length = features.shape[1]
        memory = self.input_norm(self.input_projection(features)) + self.position[:, :length]
        memory = self.encoder(memory, src_key_padding_mask=padding_mask)
        query = self.class_query.expand(features.shape[0], -1, -1)
        decoded = self.decoder(query, memory, memory_key_padding_mask=padding_mask)
        return self.classifier(self.output_norm(decoded[:, 0]))

