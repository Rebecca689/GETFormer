# GETFormer
Graph-Enhanced Transformer for roundabout trajectory prediction
# GETFormer: Graph-Enhanced Transformer for Roundabout Trajectory Prediction

A dual-branch architecture combining a star-graph GCN for spatial interaction 
modeling with a Transformer encoder for temporal dependency learning, evaluated 
on both simulated (CARLA-Round) and real-world (rounD) roundabout data.

## Architecture

- **Temporal branch**: Transformer encoder with sinusoidal positional encoding
- **Spatial branch**: Two-layer star-graph GCN (Kipf & Welling, 2017)
- **Fusion**: Late gated fusion with learnable gate
- **Decoder**: Non-autoregressive MLP (predicts all future steps at once)

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

## Dataset

The CARLA-Round simulation dataset is available at  
https://github.com/Rebecca689/CARLA-Round
