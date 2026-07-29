"""Minimal graph neural network modules for future GNN baselines.

These modules intentionally avoid PyTorch Geometric so the reproduction
environment remains lightweight. They are scaffolding for later experiments,
not evidence that a GNN method improves the paper baseline.
"""

from __future__ import annotations

import torch
from torch import nn


def aggregate_neighbors(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Aggregate node features for either one graph or a graph batch."""
    if x.dim() == 2:
        return adj @ x
    if x.dim() == 3:
        return torch.einsum("ij,bjf->bif", adj, x)
    raise ValueError(f"Expected 2D or 3D node tensor, got shape {tuple(x.shape)}")


def dense_normalized_adjacency(num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Build a row-normalized dense adjacency matrix from edge lists."""
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_weight.dtype, device=edge_weight.device)
    src, dst = edge_index[0].long(), edge_index[1].long()
    adj[dst, src] = edge_weight
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return adj / degree


def cached_normalized_adjacency(module: nn.Module, num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Return a cached dense normalized adjacency for static beam topology."""
    key = (
        num_nodes,
        edge_index.device.type,
        getattr(edge_index.device, "index", None),
        edge_weight.dtype,
        int(edge_index.shape[1]),
    )
    cached_key = getattr(module, "_cached_adj_key", None)
    cached_adj = getattr(module, "_cached_adj", None)
    if cached_adj is None or cached_key != key:
        module._cached_adj = dense_normalized_adjacency(num_nodes, edge_index, edge_weight)
        module._cached_adj_key = key
    return module._cached_adj


class GraphConv(nn.Module):
    """Simple weighted message-passing layer: H' = ReLU(A_norm H W)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(aggregate_neighbors(adj, x)))


class GraphSAGELayer(nn.Module):
    """Small GraphSAGE-style layer with separate self and neighbor transforms."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.self_linear(x) + self.neighbor_linear(aggregate_neighbors(adj, x)))


class GraphDQN(nn.Module):
    """Graph encoder plus Q-value head for channel-power actions.

    The head receives both graph-smoothed embeddings and the raw current-beam
    feature vector. The raw skip path is important because the action is
    channel-specific; message passing alone can blur per-channel occupancy and
    SINR details needed for channel selection.
    """

    def __init__(self, node_features: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_features = node_features
        self._cached_adj = None
        self._cached_adj_key = None
        self.conv1 = GraphConv(node_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + node_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, current_beam: torch.Tensor) -> torch.Tensor:
        """Return action Q-values.

        Args:
            x: `[batch, nodes, features]` node-feature tensor.
            edge_index: `[2, edges]` shared graph topology.
            edge_weight: `[edges]` shared edge weights.
            current_beam: `[batch]` integer beam ids for the currently arriving user.
        """
        batch, nodes, _ = x.shape
        adj = cached_normalized_adjacency(self, nodes, edge_index, edge_weight)
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        graph_pool = h.mean(dim=1)
        batch_idx = torch.arange(batch, device=x.device)
        beam_idx = current_beam.long()
        current = h[batch_idx, beam_idx]
        current_raw = x[batch_idx, beam_idx]
        return self.head(torch.cat([graph_pool, current, current_raw], dim=1))


class GraphSageDQN(nn.Module):
    """GraphSAGE encoder plus DQN head.

    This variant is useful when weighted GCN averaging is too smoothing for a
    channel-specific resource-allocation action.
    """

    def __init__(self, node_features: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_features = node_features
        self._cached_adj = None
        self._cached_adj_key = None
        self.conv1 = GraphSAGELayer(node_features, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + node_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, current_beam: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        adj = cached_normalized_adjacency(self, nodes, edge_index, edge_weight)
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        graph_pool = h.mean(dim=1)
        batch_idx = torch.arange(batch, device=x.device)
        beam_idx = current_beam.long()
        current = h[batch_idx, beam_idx]
        current_raw = x[batch_idx, beam_idx]
        return self.head(torch.cat([graph_pool, current, current_raw], dim=1))


class CandidateGraphSageDQN(nn.Module):
    """Candidate-aware GraphSAGE-DQN for channel-power scoring.

    Instead of treating actions as unrelated output indices, this model builds a
    small feature vector for each `(channel, power_level)` candidate. This makes
    the inductive bias closer to the actual resource-allocation problem and can
    expose power-cost trade-offs more directly.
    """

    def __init__(
        self,
        node_features: int,
        num_actions: int,
        num_channels: int = 16,
        num_power_levels: int = 5,
        hidden_dim: int = 64,
        power_penalty: float = 0.2,
    ) -> None:
        super().__init__()
        expected_actions = 1 + num_channels * num_power_levels
        if num_actions != expected_actions:
            raise ValueError(f"num_actions={num_actions} does not match 1 + channels*power_levels={expected_actions}")
        self.node_features = node_features
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_power_levels = num_power_levels
        self.power_penalty = float(power_penalty)
        self._cached_adj = None
        self._cached_adj_key = None
        self.conv1 = GraphSAGELayer(node_features, hidden_dim)
        self.conv2 = GraphSAGELayer(hidden_dim, hidden_dim)
        context_dim = hidden_dim * 2 + node_features
        self.reject_head = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        candidate_dim = context_dim + 5
        self.candidate_head = nn.Sequential(
            nn.Linear(candidate_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _current_channel_features(self, current_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = 9
        occ = current_raw[:, base : base + self.num_channels]
        power = current_raw[:, base + self.num_channels : base + 2 * self.num_channels]
        sinr = current_raw[:, base + 2 * self.num_channels : base + 3 * self.num_channels]
        return occ, power, sinr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, current_beam: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        adj = cached_normalized_adjacency(self, nodes, edge_index, edge_weight)
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        graph_pool = h.mean(dim=1)
        batch_idx = torch.arange(batch, device=x.device)
        beam_idx = current_beam.long()
        current = h[batch_idx, beam_idx]
        current_raw = x[batch_idx, beam_idx]
        context = torch.cat([graph_pool, current, current_raw], dim=1)

        reject_q = self.reject_head(context)
        occ, current_power, sinr = self._current_channel_features(current_raw)
        channels = torch.linspace(0.0, 1.0, self.num_channels, device=x.device, dtype=x.dtype)
        powers = torch.linspace(0.0, 1.0, self.num_power_levels, device=x.device, dtype=x.dtype)
        channel_grid = channels.view(1, self.num_channels, 1).expand(batch, -1, self.num_power_levels)
        power_grid = powers.view(1, 1, self.num_power_levels).expand(batch, self.num_channels, -1)
        occ_grid = occ.unsqueeze(2).expand(-1, -1, self.num_power_levels)
        current_power_grid = current_power.unsqueeze(2).expand(-1, -1, self.num_power_levels)
        sinr_grid = sinr.unsqueeze(2).expand(-1, -1, self.num_power_levels)
        candidate_small = torch.stack(
            [occ_grid, current_power_grid, sinr_grid, channel_grid, power_grid],
            dim=-1,
        )
        context_grid = context[:, None, None, :].expand(-1, self.num_channels, self.num_power_levels, -1)
        candidate_input = torch.cat([context_grid, candidate_small], dim=-1)
        candidate_q = self.candidate_head(candidate_input).squeeze(-1).reshape(batch, -1)
        candidate_q = candidate_q - self.power_penalty * power_grid.reshape(batch, -1)
        return torch.cat([reject_q, candidate_q], dim=1)
