"""Control-conditioned expression networks used in the benchmark."""

from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "MLP_PRnet_controls"
sys.path.insert(0, str(BASE))

from models import ControlPRnet


class ExpressionMLP(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.is_residual = variant not in ["Direct", "DirectZeroInit"]
        width = {"ControlOnly": 978, "ResidualChemicalVAE": 1174}.get(variant, 2002)
        self.hidden = nn.Sequential(
            nn.Linear(width, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, 978)
        self.var = nn.Linear(256, 978)
        if variant not in ["Direct", "ResidualRandomInit"]:
            nn.init.zeros_(self.mean.weight)
            nn.init.zeros_(self.mean.bias)

    def forward(self, fingerprint, control, centered):
        features = (
            centered
            if self.variant == "ControlOnly"
            else torch.cat([fingerprint, centered], dim=1)
        )
        hidden = self.hidden(features)
        mean = self.mean(hidden)
        if self.is_residual:
            mean = mean + control
        return mean, nn.functional.softplus(self.var(hidden)) + 0.0001


def build(variant):
    return ControlPRnet() if variant == "PRnet" else ExpressionMLP(variant)


def predict(model, variant, fp, control, center, scale, noise):
    if variant == "PRnet":
        return model(fp, None, control, noise)
    return model(fp, control, (control - center) / scale)
