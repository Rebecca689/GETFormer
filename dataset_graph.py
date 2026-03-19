"""
Graph-Aware Dataset for CARLA Roundabout Trajectory Prediction

Extends the standard trajectory dataset to include neighbor vehicle information
for Graph Convolutional Network (GCN) spatial interaction modeling.

For each ego vehicle sliding window, this dataset also retrieves the positions
of nearby vehicles at each observation timestep, enabling the GCN to model
spatial interactions in the roundabout.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path


class CARLAGraphDataset(torch.utils.data.Dataset):
    """
    CARLA dataset with neighbor information for GCN.

    Each sample contains:
      - obs_traj: [obs_len, 2] ego observation trajectory (transformed)
      - pred_traj: [pred_len, 2] ego prediction trajectory (transformed)
      - neighbor_pos: [obs_len, max_neighbors, 2] neighbor relative positions (transformed)
      - neighbor_mask: [obs_len, max_neighbors] valid neighbor mask
    """

    def __init__(self, csv_file, transform, obs_len=25, pred_len=25,
                 neighbor_radius=30.0, max_neighbors=8):
        self.transform = transform
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors

        self.data = pd.read_csv(csv_file)

        # Build frame index for fast neighbor lookup
        self.frame_index = self._build_frame_index()

        # Prepare trajectory samples
        self.trajectories = self._prepare_trajectories()

        print(f"  Loaded {len(self.trajectories)} graph samples from {Path(csv_file).name}")
        print(f"  Neighbor radius: {neighbor_radius}m, max neighbors: {max_neighbors}")

    def _build_frame_index(self):
        """
        Build lookup table: (scenario_id, frame) -> array of [trackId, x, y, heading].

        Vehicles in different scenarios never interact, so we index by (scenario_id, frame).
        """
        frame_index = {}

        # Group by scenario_id and frame for efficient lookup
        grouped = self.data.groupby(['scenario_id', 'frame'])

        for (scenario_id, frame), group in grouped:
            frame_index[(scenario_id, frame)] = group[['trackId', 'x', 'y', 'heading']].values

        return frame_index

    def _prepare_trajectories(self):
        """
        Prepare sliding window trajectory samples, storing frame numbers
        and scenario_id for neighbor lookup.
        """
        trajectories = []
        skipped = 0

        for track_id in self.data['trackId'].unique():
            track_data = self.data[self.data['trackId'] == track_id].sort_values('frame')

            if len(track_data) < self.seq_len:
                skipped += 1
                continue

            scenario_id = track_data['scenario_id'].iloc[0]

            for i in range(len(track_data) - self.seq_len + 1):
                segment = track_data.iloc[i:i + self.seq_len]

                coords = segment[['x', 'y']].values.astype(np.float32)

                if np.isnan(coords).any() or np.isinf(coords).any():
                    continue
                if np.abs(coords).max() > 100:
                    continue

                frames = segment['frame'].values
                # Check frame consecutiveness
                if not np.all(np.diff(frames) == 1):
                    continue

                target_pos = coords[self.obs_len - 1]
                target_heading_rad = segment['heading'].iloc[self.obs_len - 1]
                target_heading_deg = float(np.rad2deg(target_heading_rad))

                trajectories.append({
                    'coords': coords,
                    'target_pos': target_pos,
                    'target_heading': target_heading_deg,
                    'frames': frames[:self.obs_len],  # only obs frames needed for neighbors
                    'scenario_id': scenario_id,
                    'track_id': track_id,
                })

        if skipped > 0:
            print(f"  Skipped {skipped} trajectories (too short)")

        return trajectories

    def _get_neighbors(self, track_id, scenario_id, frames, ego_coords,
                       target_pos, target_heading_deg):
        """
        Get neighbor positions for each observation frame.

        Args:
            track_id: ego vehicle track ID
            scenario_id: scenario identifier
            frames: [obs_len] frame numbers
            ego_coords: [obs_len, 2] ego positions in global frame
            target_pos: [2] target position for coordinate transform
            target_heading_deg: float, heading in degrees for coordinate transform

        Returns:
            neighbor_pos: [obs_len, max_neighbors, 2] transformed relative positions
            neighbor_mask: [obs_len, max_neighbors] validity mask
        """
        obs_len = len(frames)
        neighbor_pos = np.zeros((obs_len, self.max_neighbors, 2), dtype=np.float32)
        neighbor_mask = np.zeros((obs_len, self.max_neighbors), dtype=np.float32)

        for t in range(obs_len):
            frame = frames[t]
            ego_xy = ego_coords[t]

            frame_vehicles = self.frame_index.get((scenario_id, frame), None)
            if frame_vehicles is None or len(frame_vehicles) == 0:
                continue

            # Find neighbors within radius (exclude ego)
            candidates = []
            for row in frame_vehicles:
                tid, nx, ny, _ = row
                if int(tid) == track_id:
                    continue
                dist = np.sqrt((nx - ego_xy[0]) ** 2 + (ny - ego_xy[1]) ** 2)
                if dist <= self.neighbor_radius and dist > 0.1:  # exclude overlap
                    candidates.append((dist, nx, ny))

            # Sort by distance, take closest K
            candidates.sort(key=lambda c: c[0])

            for k, (_, nx, ny) in enumerate(candidates[:self.max_neighbors]):
                # Store global position (will be transformed below)
                neighbor_pos[t, k] = [nx, ny]
                neighbor_mask[t, k] = 1.0

        # Transform neighbor positions using the same coordinate transform as ego
        # This puts neighbors in the ego-centric, heading-aligned, normalized frame
        for t in range(obs_len):
            for k in range(self.max_neighbors):
                if neighbor_mask[t, k] > 0:
                    # Transform single neighbor position
                    neigh_global = neighbor_pos[t, k].reshape(1, 2)
                    try:
                        neigh_transformed = self.transform.transform(
                            neigh_global, target_pos, target_heading_deg
                        )
                        neighbor_pos[t, k] = neigh_transformed[0]
                    except Exception:
                        neighbor_mask[t, k] = 0.0
                        neighbor_pos[t, k] = [0.0, 0.0]

        return neighbor_pos, neighbor_mask

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        coords = traj['coords']
        target_pos = traj['target_pos']
        target_heading = traj['target_heading']
        frames = traj['frames']
        scenario_id = traj['scenario_id']
        track_id = traj['track_id']

        obs = coords[:self.obs_len]
        pred = coords[self.obs_len:]

        try:
            # Transform ego trajectories
            obs_transformed = self.transform.transform(obs, target_pos, target_heading)
            pred_transformed = self.transform.transform(pred, target_pos, target_heading)

            if np.isnan(obs_transformed).any() or np.isinf(obs_transformed).any():
                return self._zero_sample()
            if np.isnan(pred_transformed).any() or np.isinf(pred_transformed).any():
                return self._zero_sample()

            # Get neighbor data
            neighbor_pos, neighbor_mask = self._get_neighbors(
                track_id, scenario_id, frames, obs,
                target_pos, target_heading
            )

            # Safety check on neighbors
            if np.isnan(neighbor_pos).any() or np.isinf(neighbor_pos).any():
                neighbor_pos = np.nan_to_num(neighbor_pos, 0.0)

            return {
                'obs_traj': torch.FloatTensor(obs_transformed),
                'pred_traj': torch.FloatTensor(pred_transformed),
                'neighbor_pos': torch.FloatTensor(neighbor_pos),
                'neighbor_mask': torch.FloatTensor(neighbor_mask),
            }

        except Exception:
            return self._zero_sample()

    def _zero_sample(self):
        """Return zero-filled sample as fallback."""
        return {
            'obs_traj': torch.zeros(self.obs_len, 2),
            'pred_traj': torch.zeros(self.pred_len, 2),
            'neighbor_pos': torch.zeros(self.obs_len, self.max_neighbors, 2),
            'neighbor_mask': torch.zeros(self.obs_len, self.max_neighbors),
        }


def graph_collate_fn(batch):
    """
    Custom collate function for CARLAGraphDataset.
    Stacks dict-based samples into batched tensors.
    """
    return {
        'obs_traj': torch.stack([s['obs_traj'] for s in batch]),
        'pred_traj': torch.stack([s['pred_traj'] for s in batch]),
        'neighbor_pos': torch.stack([s['neighbor_pos'] for s in batch]),
        'neighbor_mask': torch.stack([s['neighbor_mask'] for s in batch]),
    }


if __name__ == '__main__':
    """Test the dataset with actual data."""
    import sys
    sys.path.append('.')
    from utils import CoordinateTransform

    data_file = '../../data/carla_round_all.csv'
    print(f"Testing CARLAGraphDataset with {data_file}")

    # Compute transform statistics (simplified)
    data = pd.read_csv(data_file)
    print(f"Total rows: {len(data):,}")
    print(f"Total tracks: {data['trackId'].nunique()}")

    # Fit transform on a small sample
    transform = CoordinateTransform()
    sample_coords = data[['x', 'y']].values[:10000].astype(np.float32)
    transform.fit(sample_coords)

    # Create dataset
    dataset = CARLAGraphDataset(
        data_file, transform,
        obs_len=25, pred_len=25,
        neighbor_radius=30.0, max_neighbors=8
    )

    print(f"\nDataset size: {len(dataset)}")

    # Check a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  obs_traj: {sample['obs_traj'].shape}")
        print(f"  pred_traj: {sample['pred_traj'].shape}")
        print(f"  neighbor_pos: {sample['neighbor_pos'].shape}")
        print(f"  neighbor_mask: {sample['neighbor_mask'].shape}")
        n_neighbors = sample['neighbor_mask'].sum(dim=-1)
        print(f"  neighbors per timestep: min={n_neighbors.min():.0f}, "
              f"max={n_neighbors.max():.0f}, mean={n_neighbors.mean():.1f}")

    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=graph_collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    print("\nDataset test passed!")
