"""
DSK-only 3D HPE model for a controlled HPE-Li vs HPE-Li++ ablation.

The architecture matches HPE-Li-3D in channel widths, selective-kernel
branches, pooling, regression head, input shape, and output shape. The only
model-level ablation is the removal of ChannelTransformer refinement.
"""
import time

import torch
import torch.nn.functional as F
from torch import nn

from .utils import regression


DEFAULT_CONFIG = {
    "num_lay": 128,
    "hidden_reg": 32,
    "sk_m": 3,
    "sk_g": 32,
    "sk_r": 4,
    "sk_l": 32,
}

CHECKPOINT_FORMAT_VERSION = 1


def _normalize_config(config=None):
    """Merge model overrides into the ablation defaults and validate them."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        cfg.update(config)

    for key in cfg:
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive, got {cfg[key]}.")

    channels = cfg["num_lay"]
    groups = cfg["sk_g"]
    if channels % groups != 0 or (channels * 2) % groups != 0:
        raise ValueError(
            f"sk_g={groups} must divide num_lay={channels} and "
            f"2*num_lay={channels * 2}."
        )
    return cfg


def get_model_config_from_checkpoint(checkpoint):
    """Extract the DSK-only model configuration from a checkpoint."""
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return _normalize_config(checkpoint["model_config"])
    return _normalize_config()


def normalize_state_dict(state_dict):
    """Remove a DataParallel prefix without changing model parameter names."""
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class DSKConv(nn.Module):
    """Dual selective-kernel convolution without Transformer refinement.

    The block keeps the same HPE-Li++ feature paths:

    1. Parallel dilated grouped-convolution branches.
    2. Branch selection per feature channel.
    3. Branch selection per frequency row.
    4. Concatenation along the temporal width.
    5. Batch normalization and width pooling.

    Removing only the Transformer preserves the input/output shape of the
    corresponding HPE-Li++ block.
    """

    def __init__(
        self,
        features,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
    ):
        super().__init__()
        if features % groups != 0:
            raise ValueError(f"groups={groups} must divide features={features}.")

        bottleneck = max(features // reduction, min_bottleneck)
        self.features = int(features)
        self.m = int(m)

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + branch_idx,
                        dilation=1 + branch_idx,
                        groups=groups,
                        bias=False,
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                )
                for branch_idx in range(m)
            ]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList(
            [nn.Conv2d(bottleneck, features, kernel_size=1) for _ in range(m)]
        )
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(features)

    def forward(self, x):
        # (B, M, C, F, T)
        branch_features = torch.stack([conv(x) for conv in self.convs], dim=1)

        # CwSKA: select a kernel branch independently for every feature channel.
        merged = branch_features.sum(dim=1)
        descriptor = self.fc(self.gap(merged))
        channel_weights = self.softmax(
            torch.stack([fc(descriptor) for fc in self.fcs], dim=1)
        )
        channel_selected = (branch_features * channel_weights).sum(dim=1)

        # FwSKA: select a kernel branch independently for every frequency row.
        frequency_summary = branch_features.sum(dim=2)
        frequency_weights = self.softmax(
            F.adaptive_avg_pool2d(
                frequency_summary,
                (frequency_summary.size(2), 1),
            )
        )
        frequency_selected = (
            branch_features * frequency_weights.unsqueeze(2)
        ).sum(dim=1)

        # Keep the HPE-Li++ fusion geometry, but omit ChannelTransformer.
        fused = torch.cat([channel_selected, frequency_selected], dim=3)
        return F.avg_pool2d(self.norm(fused), kernel_size=(1, 2))


class SKUnit(nn.Module):
    """Project, downsample, apply DSKConv, and project to output channels."""

    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d((2, 2))
        self.dsk = DSKConv(
            mid_features,
            m=m,
            groups=groups,
            reduction=reduction,
            min_bottleneck=min_bottleneck,
            stride=stride,
        )
        self.norm = nn.BatchNorm2d(mid_features)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.dsk(x)
        return self.conv3(self.norm(x))


class DSKNetMMFI3D(nn.Module):
    """DSK-only 3D pose estimator used as the HPE-Li++ Transformer ablation."""

    def __init__(self, **model_config):
        super().__init__()
        self._cfg = _normalize_config(model_config)
        channels = self._cfg["num_lay"]
        hidden_reg = self._cfg["hidden_reg"]
        dsk_kwargs = {
            "m": self._cfg["sk_m"],
            "groups": self._cfg["sk_g"],
            "reduction": self._cfg["sk_r"],
            "min_bottleneck": self._cfg["sk_l"],
            "stride": 1,
        }

        self.skunit1 = SKUnit(3, channels, channels, **dsk_kwargs)
        self.bn = nn.BatchNorm2d(channels)
        self.skunit2 = SKUnit(
            channels,
            channels * 2,
            channels * 2,
            **dsk_kwargs,
        )
        self.final_pool = nn.AvgPool2d((2, 2))
        self.regression = regression(
            input_dim=3584,
            output_dim=51,
            hidden_dim=hidden_reg,
        )

    def _extract_features(self, x):
        x = self.skunit1(x)
        x = self.bn(x)
        x = self.skunit2(x)
        return self.final_pool(x)

    def forward(self, x):
        """Map CSI ``(B, 3, 114, 10)`` to pose ``(B, 17, 3)``."""
        start = time.time()
        features = self._extract_features(x)
        pose = self.regression(features).reshape(x.size(0), 17, 3)
        return pose, time.time() - start

    def get_model_config(self):
        return dict(self._cfg)
