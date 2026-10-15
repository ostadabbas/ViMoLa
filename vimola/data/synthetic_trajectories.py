"""
Generate synthetic trajectory data for testing the velocity tokenizer.
"""
from typing import Tuple, Optional
import torch
import numpy as np


def generate_random_walk_trajectory(
    num_frames: int = 30,
    step_std: float = 0.02,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random walk trajectory.
    
    Args:
        num_frames: Number of timesteps
        step_std: Standard deviation of step size
        seed: Random seed for reproducibility
    
    Returns:
        positions: (T, 2) positions at each timestep
        velocities: (T, 2) velocities at each timestep
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Random velocities (small steps)
    velocities = torch.randn(num_frames, 2) * step_std
    
    # Random initial position in [-1, 1]^2
    initial_position = torch.rand(2) * 2 - 1
    
    # Cumsum to get positions
    positions = initial_position.unsqueeze(0) + torch.cumsum(velocities, dim=0)
    
    return positions, velocities, initial_position


def generate_smooth_trajectory(
    num_frames: int = 30,
    frequency_range: Tuple[float, float] = (0.1, 0.5),
    amplitude_range: Tuple[float, float] = (0.1, 0.5),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a smooth trajectory using sinusoidal functions.
    Useful for testing that the tokenizer can capture smooth motion patterns.
    
    Args:
        num_frames: Number of timesteps
        frequency_range: Range of frequency for sine waves
        amplitude_range: Range of amplitude for motion
        seed: Random seed
    
    Returns:
        positions: (T, 2) positions
        velocities: (T, 2) velocities
        initial_position: (2,) starting point
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    t = torch.linspace(0, 2 * np.pi, num_frames)
    
    # Random frequencies and amplitudes for each dimension
    freq_x = np.random.uniform(*frequency_range)
    freq_y = np.random.uniform(*frequency_range)
    amp_x = np.random.uniform(*amplitude_range)
    amp_y = np.random.uniform(*amplitude_range)
    phase_x = np.random.uniform(0, 2 * np.pi)
    phase_y = np.random.uniform(0, 2 * np.pi)
    
    # Smooth motion
    positions_x = amp_x * torch.sin(freq_x * t + phase_x)
    positions_y = amp_y * torch.sin(freq_y * t + phase_y)
    positions = torch.stack([positions_x, positions_y], dim=1)
    
    # Initial position offset
    initial_position = torch.rand(2) * 0.5 - 0.25
    positions = positions + initial_position.unsqueeze(0)
    
    # Compute velocities by differentiation
    velocities = positions[1:] - positions[:-1]
    # Pad last velocity to maintain shape
    velocities = torch.cat([velocities, velocities[-1:]], dim=0)
    
    return positions, velocities, initial_position


def generate_bouncing_trajectory(
    num_frames: int = 30,
    gravity: float = 0.05,
    restitution: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a bouncing ball trajectory with physics.
    Tests if tokenizer can capture ballistic motion.
    
    Args:
        num_frames: Number of timesteps
        gravity: Gravity constant (positive = down)
        restitution: Bounciness (0-1)
        seed: Random seed
    
    Returns:
        positions: (T, 2) positions
        velocities: (T, 2) velocities
        initial_position: (2,) starting point
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Random initial position and velocity
    position = torch.rand(2) * 1.6 - 0.8  # Start in [-0.8, 0.8]^2
    velocity = torch.randn(2) * 0.2
    velocity[1] = abs(velocity[1])  # Upward initial y velocity
    
    positions = []
    velocities = []
    initial_position = position.clone()
    
    for _ in range(num_frames):
        positions.append(position.clone())
        velocities.append(velocity.clone())
        
        # Update
        position = position + velocity
        velocity[1] = velocity[1] - gravity  # Gravity on y
        
        # Bounce off floor (y = -1)
        if position[1] < -1:
            position[1] = -1 + (-1 - position[1])  # Reflect position
            velocity[1] = -velocity[1] * restitution
        
        # Bounce off walls
        if position[0] > 1 or position[0] < -1:
            velocity[0] = -velocity[0]
            position[0] = torch.clamp(position[0], -1, 1)
    
    positions = torch.stack(positions)
    velocities = torch.stack(velocities)
    
    return positions, velocities, initial_position


def generate_synthetic_dataset(
    num_samples: int = 1000,
    num_frames: int = 30,
    trajectory_type: str = "mixed",
    seed: int = 42,
) -> torch.utils.data.Dataset:
    """
    Generate a synthetic dataset of trajectories.
    
    Args:
        num_samples: Number of trajectories
        num_frames: Frames per trajectory
        trajectory_type: "random", "smooth", "bouncing", or "mixed"
        seed: Random seed
    
    Returns:
        dataset: PyTorch dataset with (velocities, positions, initial_position) items
    """
    from torch.utils.data import TensorDataset
    
    velocities_list = []
    positions_list = []
    initial_pos_list = []
    
    for i in range(num_samples):
        sample_seed = seed + i
        
        if trajectory_type == "random":
            pos, vel, init = generate_random_walk_trajectory(num_frames, seed=sample_seed)
        elif trajectory_type == "smooth":
            pos, vel, init = generate_smooth_trajectory(num_frames, seed=sample_seed)
        elif trajectory_type == "bouncing":
            pos, vel, init = generate_bouncing_trajectory(num_frames, seed=sample_seed)
        elif trajectory_type == "mixed":
            choice = i % 3
            if choice == 0:
                pos, vel, init = generate_random_walk_trajectory(num_frames, seed=sample_seed)
            elif choice == 1:
                pos, vel, init = generate_smooth_trajectory(num_frames, seed=sample_seed)
            else:
                pos, vel, init = generate_bouncing_trajectory(num_frames, seed=sample_seed)
        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")
        
        velocities_list.append(vel)
        positions_list.append(pos)
        initial_pos_list.append(init)
    
    velocities = torch.stack(velocities_list)
    positions = torch.stack(positions_list)
    initial_positions = torch.stack(initial_pos_list)
    
    return TensorDataset(velocities, positions, initial_positions)


def test_generation():
    """Test synthetic trajectory generation."""
    print("Testing random walk:")
    pos, vel, init = generate_random_walk_trajectory(num_frames=30, seed=42)
    print(f"  Positions: {pos.shape}, range: [{pos.min():.3f}, {pos.max():.3f}]")
    print(f"  Velocities: {vel.shape}, mean: {vel.mean():.4f}, std: {vel.std():.4f}")
    print(f"  Initial: {init}")
    
    print("\nTesting smooth trajectory:")
    pos, vel, init = generate_smooth_trajectory(num_frames=30, seed=42)
    print(f"  Positions: {pos.shape}, range: [{pos.min():.3f}, {pos.max():.3f}]")
    
    print("\nTesting bouncing trajectory:")
    pos, vel, init = generate_bouncing_trajectory(num_frames=30, seed=42)
    print(f"  Positions: {pos.shape}, y-range: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")
    
    print("\nTesting dataset generation:")
    dataset = generate_synthetic_dataset(num_samples=100, num_frames=30, trajectory_type="mixed", seed=42)
    print(f"  Dataset size: {len(dataset)}")
    vel, pos, init = dataset[0]
    print(f"  Sample shapes: vel={vel.shape}, pos={pos.shape}, init={init.shape}")
    print("✓ All tests passed")


if __name__ == "__main__":
    from typing import Optional
    test_generation()
