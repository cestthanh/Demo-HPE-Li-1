"""DSKNet 3D without Transformer for MMFi Phase C.

This file ports the DSKConv logic from
``HPE-Li-ECCV2024/model/sknet_trans_mmfi.py`` to 3D pose output. The DSKConv
part keeps the author's dual selective-kernel structure:

1. stack multi-dilation convolution branches;
2. select branches by channel attention;
3. select branches by frequency attention;
4. concatenate both selected features along width;
5. batch-normalize and average-pool width back to the original size.

The ChannelTransformer found in the forked source is intentionally removed.
"""

import time

import torch
import torch.nn.functional as F
from torch import nn

from .utils import regression


MODEL_NAME = "DSKNetMMFI3D"
CHECKPOINT_FORMAT_VERSION = 2

DEFAULT_CONFIG = {
    "num_lay": 128,
    "hidden_reg": 32,
    "sk_m": 3,
    "sk_g": 32,
    "sk_r": 4,
    "sk_l": 32,
}


def _normalize_config(config=None):
    """Merge model overrides into DSKNet defaults and validate them."""
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
    """Extract the DSKNet model configuration from a checkpoint."""
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model_config"), dict
    ):
        return _normalize_config(checkpoint["model_config"])
    return _normalize_config()


def normalize_state_dict(state_dict):
    """Remove a DataParallel prefix without changing parameter names."""
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


class DSKConv(nn.Module):
    """Dual selective-kernel convolution without ChannelTransformer."""

    def __init__(
        self,
        features,
        img_size,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
    ):
        super().__init__()
        if features % groups != 0:
            raise ValueError(f"groups={groups} must divide features={features}.")
        if len(img_size) != 2:
            raise ValueError(f"img_size must be [height, width], got {img_size}.")

        bottleneck = max(int(features / reduction), min_bottleneck)
        self.features = int(features)
        self.img_size = [int(img_size[0]), int(img_size[1])]
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
        # [B, M, C, H, W]
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)

        # CwSKA: branch selection per feature channel.
        feats_u = feats.sum(dim=1)
        feats_s = self.gap(feats_u)
        feats_z = self.fc(feats_s)
        channel_attention = torch.stack(
            [fc(feats_z) for fc in self.fcs],
            dim=1,
        )
        channel_attention = self.softmax(channel_attention)
        feats_channel = (feats * channel_attention).sum(dim=1)

        # FwSKA: branch selection per frequency row.
        feats_frequency = feats.sum(dim=2)
        frequency_attention = F.adaptive_avg_pool2d(
            feats_frequency,
            (feats_frequency.size(2), 1),
        )
        frequency_attention = self.softmax(frequency_attention)
        feats_frequency = (feats * frequency_attention.unsqueeze(2)).sum(dim=1)

        fused = torch.cat([feats_channel, feats_frequency], dim=3)
        if list(fused.shape[2:4]) != self.img_size:
            raise RuntimeError(
                f"DSKConv expected fused spatial size {self.img_size}, "
                f"got {list(fused.shape[2:4])}."
            )

        fused = self.norm(fused)
        return F.avg_pool2d(fused, kernel_size=(1, 2))


class DSKUnit(nn.Module):
    """Projection, spatial pooling, DSKConv, normalization, and projection."""

    def __init__(
        self,
        in_features,
        mid_features,
        out_features,
        img_size,
        m=3,
        groups=32,
        reduction=4,
        min_bottleneck=32,
        stride=1,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )
        self.pooling = nn.AvgPool2d((2, 2))
        self.dsk = DSKConv(
            mid_features,
            img_size=img_size,
            m=m,
            groups=groups,
            reduction=reduction,
            min_bottleneck=min_bottleneck,
            stride=stride,
        )
        self.norm = nn.BatchNorm2d(mid_features)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.dsk(x)
        x = self.norm(x)
        return self.conv3(x)


class DSKNetMMFI3D(nn.Module):
    """3D DSKNet pose estimator without ChannelTransformer."""

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

        self.dskunit1 = DSKUnit(
            3,
            channels,
            channels,
            img_size=[57, 10],
            **dsk_kwargs,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.dskunit2 = DSKUnit(
            channels,
            channels * 2,
            channels * 2,
            img_size=[28, 4],
            **dsk_kwargs,
        )
        self.final_pool = nn.AvgPool2d((2, 2))
        self.regression = regression(
            input_dim=3584,
            output_dim=17 * 3,
            hidden_dim=hidden_reg,
        )

    def _extract_features(self, x):
        x = self.dskunit1(x)
        x = self.bn(x)
        x = self.dskunit2(x)
        return self.final_pool(x)

    def forward(self, x):
        """Map CSI ``(B, 3, 114, 10)`` to pose ``(B, 17, 3)``."""
        if tuple(x.shape[1:]) != (3, 114, 10):
            raise ValueError(
                "DSKNetMMFI3D expects input shape (B, 3, 114, 10), "
                f"got {tuple(x.shape)}."
            )
        start = time.time()
        features = self._extract_features(x)
        pose = self.regression(features).reshape(x.size(0), 17, 3)
        return pose, time.time() - start

    def get_model_config(self):
        return dict(self._cfg)
