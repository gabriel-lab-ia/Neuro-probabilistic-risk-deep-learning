from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def apply_temperature(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        safe_temperature = torch.clamp(temperature, min=1.0e-3)
    else:
        safe_temperature = max(float(temperature), 1.0e-3)
    return logits / safe_temperature


@dataclass(slots=True)
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return apply_temperature(logits, self.temperature)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 60) -> CalibrationResult:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)

        logits = logits.detach()
        labels = labels.detach()
        nll_before = float(criterion(logits, labels).item())

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=1.0e-3, max=10.0)
            nll_after = float(criterion(self.forward(logits), labels).item())

        return CalibrationResult(
            temperature=float(self.temperature.item()),
            nll_before=nll_before,
            nll_after=nll_after,
        )
