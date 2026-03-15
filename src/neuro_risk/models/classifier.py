from __future__ import annotations

import torch
from torch import nn

from neuro_risk.config import ModelConfig
from neuro_risk.models.backbones import TabularBackbone, TemporalConvBranch
from neuro_risk.models.fusion import FusionModule


class NeuroRiskClassifier(nn.Module):
    """Multimodal classifier with dropout-enabled branches for MC inference."""

    def __init__(
        self,
        num_tabular_features: int,
        temporal_channels: int,
        num_classes: int,
        config: ModelConfig,
        use_temporal_branch: bool = True,
    ) -> None:
        super().__init__()
        self.tabular_backbone = TabularBackbone(
            input_dim=num_tabular_features,
            hidden_dims=config.tabular_hidden_dims,
            dropout=config.dropout,
        )
        self.temporal_branch = (
            TemporalConvBranch(
                input_channels=temporal_channels,
                conv_channels=config.temporal_channels,
                embedding_dim=config.temporal_embedding_dim,
                dropout=config.dropout,
            )
            if use_temporal_branch
            else None
        )
        fusion_input_dim = self.tabular_backbone.output_dim
        if self.temporal_branch is not None:
            fusion_input_dim += self.temporal_branch.output_dim
        self.fusion = FusionModule(
            input_dim=fusion_input_dim,
            hidden_dim=config.fusion_hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(self.fusion.output_dim, num_classes)

    def forward(
        self,
        tabular_inputs: torch.Tensor | None = None,
        temporal_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        if tabular_inputs is not None:
            embeddings.append(self.tabular_backbone(tabular_inputs))
        if temporal_inputs is not None and self.temporal_branch is not None:
            embeddings.append(self.temporal_branch(temporal_inputs))
        fused = self.fusion(embeddings)
        return self.classifier(fused)
