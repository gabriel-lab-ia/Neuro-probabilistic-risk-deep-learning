from __future__ import annotations

import torch
from torch import nn


def _mlp_block(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.LayerNorm(output_dim),
        nn.Dropout(dropout),
    )


class TabularBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for next_dim in hidden_dims:
            layers.append(_mlp_block(current_dim, next_dim, dropout))
            current_dim = next_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TemporalConvBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_channels: tuple[int, ...],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = input_channels
        for next_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(current_channels, next_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.GroupNorm(1, next_channels),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = next_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.output_dim = embedding_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        pooled = self.pool(encoded)
        return self.head(pooled)
