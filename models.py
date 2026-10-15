"""PyTorch language-model definitions."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class RecurrentLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        cell: str = "lstm",
        padding_idx: int = 0,
        embedding_weights: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if embedding_weights is not None:
            if tuple(embedding_weights.shape) != tuple(self.embedding.weight.shape):
                raise ValueError("embedding_weights 的形状与 embedding 表不一致")
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_weights)
        self.embedding.weight.requires_grad = not freeze_embedding
        recurrent_cls = nn.LSTM if cell == "lstm" else nn.RNN
        kwargs = dict(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        if cell == "rnn":
            kwargs["nonlinearity"] = "tanh"
        self.recurrent = recurrent_cls(**kwargs)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        states, _ = self.recurrent(embedded)
        return states

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode(input_ids))


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_length: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.input_projection = (
            nn.Identity() if embedding_dim == d_model else nn.Linear(embedding_dim, d_model)
        )
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.d_model = d_model

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # CHECK: 上三角位置为 True，使位置 i 无法注意到任何 j > i 的未来 token。
        # 这样训练可并行进行，同时严格保持自回归因果性。
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        if length > self.max_length:
            raise ValueError(f"序列长度 {length} 超过位置编码上限 {self.max_length}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.input_projection(self.embedding(input_ids)) * math.sqrt(self.d_model)
        hidden = hidden + self.position(positions)
        mask = self.causal_mask(length, input_ids.device)
        padding_mask = input_ids.eq(self.embedding.padding_idx)
        return self.encoder(hidden, mask=mask, src_key_padding_mask=padding_mask)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode(input_ids))


def build_model(
    name: str,
    vocab_size: int,
    pad_id: int,
    config: dict,
    embedding_weights: Optional[torch.Tensor] = None,
    freeze_embedding: bool = False,
) -> nn.Module:
    model_cfg = config["model"]
    if name in {"rnn", "lstm"}:
        recurrent = model_cfg["recurrent"]
        return RecurrentLanguageModel(
            vocab_size,
            model_cfg["embedding_dim"],
            recurrent["hidden_dim"],
            recurrent["num_layers"],
            recurrent["dropout"],
            cell=name,
            padding_idx=pad_id,
            embedding_weights=embedding_weights,
            freeze_embedding=freeze_embedding,
        )
    if name == "transformer":
        transformer = model_cfg["transformer"]
        return TransformerLanguageModel(
            vocab_size,
            model_cfg["embedding_dim"],
            transformer["d_model"],
            transformer["nhead"],
            transformer["num_layers"],
            transformer["dim_feedforward"],
            transformer["dropout"],
            config["data"]["context_length"],
            padding_idx=pad_id,
        )
    raise ValueError(f"未知模型：{name}")
