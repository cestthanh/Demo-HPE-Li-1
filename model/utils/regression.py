"""
model/utils/regression.py
MLP regression head: flattened features → 3D joint coordinates.
"""
import torch.nn as nn


class RegressionHead(nn.Module):
    """Three-layer MLP that maps a flattened feature vector to pose coordinates.

    Architecture:
        Linear(input_dim → hidden_dim) → ReLU → Dropout
        Linear(hidden_dim → hidden_dim*2) → BN → ReLU → Dropout
        Linear(hidden_dim*2 → output_dim)

    Args:
        input_dim (int):  Dimensionality of the flattened feature vector.
        output_dim (int): Number of output values (e.g. 17*3 = 51 for 3D pose).
        hidden_dim (int): Width of the first hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3     = nn.Linear(hidden_dim * 2, output_dim)
        self.bn      = nn.BatchNorm1d(hidden_dim * 2)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.bn(self.fc2(x))))
        return self.fc3(x)


# Keep the lowercase alias for compatibility with dsknet3d.py imports
regression = RegressionHead
