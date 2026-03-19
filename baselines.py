"""
Baseline Models for Roundabout Trajectory Prediction

Four baselines for comparison with Graph-Enhanced Transformer:
  1. GRUBaseline: GRU encoder-decoder (no spatial interaction)
  2. GCNBaseline: GCN spatial only (no temporal modeling)
  3. GCNGRUBaseline: GCN spatial + GRU temporal
  4. TransformerBaseline: Transformer only (no spatial interaction)

Ablation matrix:
              Transformer    GRU    None
  With GCN  | GCN+Trans  | GCN+GRU | GCN-only
  No GCN    | Trans-only | GRU-only | -

All models accept the same interface:
  forward(obs_traj, neighbor_pos=None, neighbor_mask=None) -> pred_traj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_graph import SpatialGCN


# =============================================================================
# Baseline 1: GRU Encoder-Decoder (no spatial interaction)
# =============================================================================
class GRUBaseline(nn.Module):
    """
    GRU encoder-decoder baseline.
    No spatial interaction modeling - uses only ego trajectory.

    Encoder: 2-layer GRU processes observation sequence.
    Decoder: 2-layer GRU autoregressively predicts future positions.

    Purpose: Paired with GCN+GRU to isolate the GCN contribution under GRU.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.HIDDEN_DIM

        self.input_embedding = nn.Linear(config.INPUT_DIM, hidden_dim)

        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT,
        )

        self.decoder_input_proj = nn.Linear(config.INPUT_DIM, hidden_dim)
        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT,
        )

        self.output_projection = nn.Linear(hidden_dim, config.OUTPUT_DIM)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, obs_traj, neighbor_pos=None, neighbor_mask=None):
        """
        Args:
            obs_traj: [B, T_obs, 2]
            neighbor_pos: ignored
            neighbor_mask: ignored
        Returns:
            pred_traj: [B, T_pred, 2]
        """
        obs_traj = torch.clamp(obs_traj, -100, 100)
        pred_len = self.config.PREDICTION_LEN

        obs_embed = self.input_embedding(obs_traj)  # [B, T_obs, d]
        _, h = self.encoder(obs_embed)                # h: [2, B, d]

        dec_input = obs_traj[:, -1:, :]  # [B, 1, 2]
        predictions = []

        for t in range(pred_len):
            dec_embed = self.decoder_input_proj(dec_input)  # [B, 1, d]
            dec_out, h = self.decoder(dec_embed, h)          # [B, 1, d]
            next_pos = self.output_projection(dec_out)       # [B, 1, 2]
            next_pos = torch.clamp(next_pos, -100, 100)
            predictions.append(next_pos)
            dec_input = next_pos

        return torch.cat(predictions, dim=1)  # [B, pred_len, 2]


# =============================================================================
# Baseline 2: GCN-Only (Spatial, No Temporal)
# =============================================================================
class GCNBaseline(nn.Module):
    """
    GCN-only baseline.
    Uses star-graph GCN for spatial interaction at each observation timestep,
    then a simple MLP to predict future trajectory.
    No temporal sequence modeling (RNN/Transformer).

    Purpose: Show the necessity of temporal modeling.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.HIDDEN_DIM
        pred_len = config.PREDICTION_LEN

        self.spatial_gcn = SpatialGCN(
            input_dim=config.GCN_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=config.GCN_LAYERS,
        )

        self.ego_embedding = nn.Linear(config.INPUT_DIM, hidden_dim)

        # MLP predictor: flatten temporal GCN features -> predict all future steps
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * config.OBSERVATION_LEN, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim * 2, pred_len * config.OUTPUT_DIM),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'spatial_gcn' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, obs_traj, neighbor_pos=None, neighbor_mask=None):
        """
        Args:
            obs_traj: [B, T_obs, 2]
            neighbor_pos: [B, T_obs, K, 2]
            neighbor_mask: [B, T_obs, K]
        Returns:
            pred_traj: [B, T_pred, 2]
        """
        obs_traj = torch.clamp(obs_traj, -100, 100)
        B = obs_traj.size(0)
        pred_len = self.config.PREDICTION_LEN

        ego_embed = self.ego_embedding(obs_traj)  # [B, T, d]

        if neighbor_pos is not None and neighbor_mask is not None:
            neighbor_pos = torch.clamp(neighbor_pos, -100, 100)
            ego_gcn_input = torch.zeros_like(obs_traj)
            gcn_out = self.spatial_gcn(ego_gcn_input, neighbor_pos, neighbor_mask)
            fused = ego_embed + gcn_out  # [B, T, d]
        else:
            fused = ego_embed

        # Flatten time dimension and predict all future steps at once
        flat = fused.reshape(B, -1)  # [B, T_obs * d]
        pred_flat = self.predictor(flat)  # [B, pred_len * 2]
        pred_traj = pred_flat.reshape(B, pred_len, self.config.OUTPUT_DIM)

        return torch.clamp(pred_traj, -100, 100)


# =============================================================================
# Baseline 3: GCN + GRU
# =============================================================================
class GCNGRUBaseline(nn.Module):
    """
    GCN + GRU baseline.
    Same spatial GCN as the main model, but uses GRU instead of Transformer
    for temporal modeling.

    Purpose: Show the advantage of Transformer over GRU for temporal patterns.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.HIDDEN_DIM

        # Spatial GCN (same as main model)
        self.spatial_gcn = SpatialGCN(
            input_dim=config.GCN_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=config.GCN_LAYERS,
        )

        self.ego_embedding = nn.Linear(config.INPUT_DIM, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # GRU encoder (replaces Transformer encoder)
        self.gru_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT,
        )

        # GRU decoder (replaces Transformer decoder)
        self.decoder_input_proj = nn.Linear(config.INPUT_DIM, hidden_dim)
        self.gru_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT,
        )

        self.output_projection = nn.Linear(hidden_dim, config.OUTPUT_DIM)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'spatial_gcn' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, obs_traj, neighbor_pos=None, neighbor_mask=None):
        """
        Args:
            obs_traj: [B, T_obs, 2]
            neighbor_pos: [B, T_obs, K, 2]
            neighbor_mask: [B, T_obs, K]
        Returns:
            pred_traj: [B, T_pred, 2]
        """
        obs_traj = torch.clamp(obs_traj, -100, 100)
        pred_len = self.config.PREDICTION_LEN

        # Spatial features
        ego_embed = self.ego_embedding(obs_traj)  # [B, T, d]

        if neighbor_pos is not None and neighbor_mask is not None:
            neighbor_pos = torch.clamp(neighbor_pos, -100, 100)
            ego_gcn_input = torch.zeros_like(obs_traj)
            gcn_out = self.spatial_gcn(ego_gcn_input, neighbor_pos, neighbor_mask)
            fused = ego_embed + gcn_out
        else:
            fused = ego_embed

        fused = self.fusion_norm(fused)  # [B, T, d]

        # GRU encode
        _, h = self.gru_encoder(fused)  # h: [2, B, d]

        # Autoregressive GRU decode
        dec_input = obs_traj[:, -1:, :]  # [B, 1, 2]
        predictions = []

        for t in range(pred_len):
            dec_embed = self.decoder_input_proj(dec_input)  # [B, 1, d]
            dec_out, h = self.gru_decoder(dec_embed, h)      # [B, 1, d]
            next_pos = self.output_projection(dec_out)       # [B, 1, 2]
            next_pos = torch.clamp(next_pos, -100, 100)
            predictions.append(next_pos)
            dec_input = next_pos

        return torch.cat(predictions, dim=1)  # [B, pred_len, 2]


# =============================================================================
# Baseline 4: Transformer-Only (no GCN spatial interaction)
# =============================================================================
class TransformerBaseline(nn.Module):
    """
    Pure Transformer encoder + MLP decoder baseline.
    Same architecture as GraphEnhancedTransformer but WITHOUT the GCN module.
    Uses only ego trajectory — neighbor data is ignored.

    Purpose: Isolate the contribution of the GCN spatial component.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        from model_graph import PositionalEncoding

        hidden_dim = config.HIDDEN_DIM
        pred_len = config.PREDICTION_LEN

        self.input_embedding = nn.Linear(config.INPUT_DIM, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=config.N_HEADS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT, batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.N_LAYERS
        )

        # MLP decoder (non-autoregressive, same as main model)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * config.OBSERVATION_LEN, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim * 2, pred_len * config.OUTPUT_DIM),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

    def forward(self, obs_traj, neighbor_pos=None, neighbor_mask=None):
        """
        Args:
            obs_traj: [B, T_obs, 2]
            neighbor_pos: ignored
            neighbor_mask: ignored
        Returns:
            pred_traj: [B, T_pred, 2]
        """
        obs_traj = torch.clamp(obs_traj, -100, 100)
        B = obs_traj.size(0)
        pred_len = self.config.PREDICTION_LEN

        # Encode
        embedded = self.input_embedding(obs_traj)   # [B, T, d]
        embedded = self.input_norm(embedded)
        embedded = embedded.transpose(0, 1)          # [T, B, d]
        embedded = self.pos_encoder(embedded)
        memory = self.transformer_encoder(embedded)  # [T, B, d]

        # MLP decode (non-autoregressive)
        memory = memory.transpose(0, 1)              # [B, T, d]
        flat = memory.reshape(B, -1)                 # [B, T * d]
        pred_flat = self.predictor(flat)              # [B, pred_len * 2]
        pred_traj = pred_flat.reshape(B, pred_len, self.config.OUTPUT_DIM)

        return torch.clamp(pred_traj, -100, 100)


# =============================================================================
# Factory
# =============================================================================
MODEL_REGISTRY = {
    'gru': GRUBaseline,
    'gcn': GCNBaseline,
    'gcn_gru': GCNGRUBaseline,
    'transformer': TransformerBaseline,
}


def create_baseline_model(model_name, config):
    """
    Create a baseline model.

    Args:
        model_name: 'gru', 'gcn', 'gcn_gru', or 'transformer'
        config: Config object
    Returns:
        model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[model_name](config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{model_name.upper()} Baseline:")
    print(f"  Total parameters: {total_params:,}")

    return model


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from config import Config

    config = Config()
    B, T, K = 4, config.OBSERVATION_LEN, config.MAX_NEIGHBORS

    obs = torch.randn(B, T, 2)
    neigh = torch.randn(B, T, K, 2)
    mask = torch.ones(B, T, K)
    mask[:, :, 5:] = 0

    for name in MODEL_REGISTRY:
        print(f"\n{'='*40}")
        model = create_baseline_model(name, config)
        model.eval()
        with torch.no_grad():
            pred = model(obs, neigh, mask)
        print(f"  Output shape: {pred.shape} (expected [{B}, {config.PREDICTION_LEN}, 2])")
        print(f"  NaN: {torch.isnan(pred).any().item()}")

    print(f"\nAll baselines passed!")
