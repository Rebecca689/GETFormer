# GETFormer
Graph-Enhanced Transformer for roundabout trajectory prediction
# GETFormer: Graph-Enhanced Transformer for Roundabout Trajectory Prediction

A dual-branch architecture combining a star-graph GCN for spatial interaction 
modeling with a Transformer encoder for temporal dependency learning, evaluated 
on both simulated (CARLA-Round) and real-world (rounD) roundabout data.

## Architecture

- **Temporal branch**: Transformer encoder with sinusoidal positional encoding
- **Spatial branch**: Two-layer star-graph GCN 
- **Fusion**: Late gated fusion with learnable gate
- **Decoder**: Non-autoregressive MLP 

## Requirements
```bash
pip install torch numpy pandas
```

## Usage
```bash
# Train on CARLA-Round
python train_graph.py

# Train on rounD (in-domain)
python train_round_indomain.py

# Sim-to-real transfer evaluation
python transfer_test.py
```
