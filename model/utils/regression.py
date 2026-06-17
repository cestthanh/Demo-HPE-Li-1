"""MLP regression head from flattened features to joint coordinates."""

import torch.nn as nn


class RegressionHead(nn.Module):
    """Three-layer regression head used by the original HPE-Li model."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.bn(self.fc2(x))))
        return self.fc3(x)


# Preserve the class name used by the original HPE-Li source.
regression = RegressionHead
