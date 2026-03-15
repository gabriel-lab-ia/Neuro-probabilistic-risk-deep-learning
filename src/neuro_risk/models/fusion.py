from __future__ import annotations

import torch
from torch import nn


class FusionModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        if not embeddings:
            raise ValueError("FusionModule requires at least one modality embedding.")
        if len(embeddings) == 1:
            fused = embeddings[0]
        else:
            fused = torch.cat(embeddings, dim=1)
        return self.network(fused)
