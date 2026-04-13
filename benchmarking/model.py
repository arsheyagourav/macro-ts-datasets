from __future__ import annotations

import math

import torch
from torch import nn


class ProbSparseSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.scale = self.head_dim**-0.5
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        sampled = max(1, min(seq_len, int(self.factor * math.log(seq_len + 1))))
        sparsity = q.pow(2).sum(dim=-1)
        top_idx = sparsity.topk(sampled, dim=-1).indices

        mean_context = v.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()
        selected_q = q.gather(
            2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        scores = torch.matmul(selected_q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(scores.softmax(dim=-1))
        selected_context = torch.matmul(attn, v)
        mean_context.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            selected_context,
        )

        context = mean_context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(context)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.attn = ProbSparseSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            factor=factor,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class InformerForecaster(nn.Module):
    def __init__(
        self,
        num_features: int,
        context_length: int,
        prediction_length: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        factor: int,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    factor=factor,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prediction_length * num_features),
        )
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embedding[:, : x.size(1)]
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        summary = torch.cat([hidden[:, -1], hidden.mean(dim=1)], dim=-1)
        forecast = self.head(summary)
        return forecast.view(-1, self.prediction_length, self.num_features)

