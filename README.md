# GETFormer: Graph-Enhanced Transformer for Roundabout Trajectory Prediction

Official implementation of **"GETFormer: Graph-Enhanced Transformer for Roundabout Trajectory Prediction"**.

## Overview

GETFormer integrates a **star-graph convolutional network (GCN)** for spatial interaction modeling with a **Transformer encoder** for temporal dependencies through a learned **gated fusion** mechanism and **non-autoregressive decoding**.

Key findings:
- The Transformer temporal component drives in-domain performance (21% improvement over GRU baselines)
- The GCN spatial branch grows in importance from simulation (0.5%) to real data (4.3%) to cross-domain transfer (10.7%)
- Non-autoregressive models maintain stable sim-to-real transfer, while autoregressive GRU models degrade catastrophically (8x error increase)
- Driving behavior dominates prediction difficulty (+255%), while road geometry has negligible impact

## Repository Structure

```
GETFormer/
├── config.py                    # Model and training hyperparameters
├── model_graph.py               # GETFormer architecture (GCN + Transformer + Gated Fusion)
├── dataset_graph.py             # CARLA-Round graph dataset loader
├── baselines.py                 # Baseline models (GRU-only, GCN+GRU, GCN-only, Trans-only)
├── utils.py                     # Coordinate transform, metrics, early stopping
│
├── train_graph.py               # Train GETFormer on CARLA-Round
├── train_and_test.py            # Train & evaluate all 5 model variants on CARLA-Round
├── transfer_test.py             # Sim-to-real transfer experiments (CARLA → rounD)
├── train_round_indomain.py      # rounD in-domain training per roundabout type
│
├── carla_config.py              # CARLA roundabout scenario configuration (5x5 factorial)
├── collect_carla_data.py        # CARLA-Round data collection script
├── clean_and_merge.py           # Data cleaning and merging pipeline
│
├── visualize.py                 # Training curves, trajectory plots, error analysis
├── collect_mean_traj.py         # Mean trajectory extraction for visualization
├── analyze_curvature_round.py   # Curvature analysis on rounD dataset
├── curvature_change_analysis.py # Curvature-change (stable vs. transition) analysis
│
├── requirements.txt
└── figures/
    └── framework_overall.png
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/GETFormer.git
cd GETFormer
pip install -r requirements.txt
```

For CARLA data collection only (optional):
```bash
# Install CARLA 0.9.x simulator
# See https://carla.readthedocs.io/en/latest/start_quickstart/
```

## Datasets

### CARLA-Round (Simulation)

A factorial simulation dataset with 5 weather conditions x 5 traffic density levels, collected from CARLA Town03 roundabout.

**To collect from scratch:**
```bash
# 1. Start CARLA server
# 2. Collect raw data (25 scenarios)
python collect_carla_data.py

# 3. Clean and merge
python clean_and_merge.py
```

**Expected data structure:**
```
data/
├── carla_round_all.csv      # Merged CARLA trajectories
├── train.csv                # 70% training split
├── val.csv                  # 15% validation split
└── test.csv                 # 15% test split
```

### rounD (Real-World)

Download the [rounD dataset](https://www.round-dataset.com/) and place it as:
```
data/rounD/
├── 00_tracks.csv
├── 00_tracksMeta.csv
├── 01_tracks.csv
├── ...
```

## Usage

### Train all models on CARLA-Round

```bash
python train_and_test.py --model all --data_dir data/
```

Train a specific model:
```bash
python train_and_test.py --model gcn_transformer --data_dir data/
```

Available models: `gcn_transformer` (GETFormer), `transformer`, `gru`, `gcn`, `gcn_gru`, `all`

### Evaluate on CARLA-Round (in-domain)

```bash
python train_and_test.py --model all --test_only --data_dir data/
```

### Sim-to-Real Transfer (CARLA → rounD)

```bash
python transfer_test.py --direction sim2real --round_dir data/rounD
```

### rounD In-Domain Training

```bash
# Train on specific roundabout types
python train_round_indomain.py --round_dir data/rounD --types 0 1 2 9

# Evaluate only
python train_round_indomain.py --round_dir data/rounD --test_only
```

### Curvature Analysis

```bash
python analyze_curvature_round.py --round_dir data/rounD --ckpt_dir checkpoints/
python curvature_change_analysis.py
```

### Visualization

```bash
python visualize.py --mode all --ckpt_dir checkpoints/ --data_dir data/
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Observation / Prediction | 25 frames (1s at 25 Hz) each |
| Hidden dimension | 64 |
| Transformer layers | 4, with 8 attention heads |
| GCN layers | 2, star-graph topology |
| Max neighbors | 8, within 30m radius |
| Fusion | Learned gating mechanism |
| Decoder | Non-autoregressive MLP (1600 → 256 → 128 → 50) |
| Optimizer | Adam (lr=5e-5, weight_decay=1e-4) |
| Training | Up to 50 epochs, early stopping (patience=10) |

## Results

### CARLA-Round In-Domain (Table III)

| Model | ADE | FDE |
|-------|-----|-----|
| GRU-only | 0.537 | 1.400 |
| GCN+GRU | 0.558 | 1.456 |
| GCN-only | 0.449 | 1.190 |
| Transformer-only | 0.422 | 1.126 |
| **GETFormer (Ours)** | **0.420** | **1.120** |

### rounD In-Domain Average (Table V)

| Model | ADE |
|-------|-----|
| GRU-only | 0.130 |
| GCN+GRU | 0.098 |
| GCN-only | 0.067 |
| Transformer-only | 0.069 |
| **GETFormer (Ours)** | **0.066** |

## Citation

```bibtex
@article{zhou2026getformer,
  title={GETFormer: Graph-Enhanced Transformer for Roundabout Trajectory Prediction},
  author={Zhou, Xiaotong and Yuan, Zhenhui and Xu, Tianhua and Song, Fei},
  journal={},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
