#!/usr/bin/env python3
"""
Visualization script for Dampened Wave Equation Simulation

Reads binary output files and creates MP4/GIF animations showing
the wave propagation on the rectangular drumhead.
"""

import os
import glob
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from pathlib import Path


def read_binary_file(filepath: str) -> tuple[np.ndarray, float]:
    """
    Read a binary wave simulation output file.

    File format:
    - int32: nx (grid points in x)
    - int32: ny (grid points in y)
    - float64: time
    - float64[nx*ny]: data (row-major order)

    Returns:
        tuple: (2D numpy array of wave data, simulation time)
    """
    with open(filepath, 'rb') as f:
        nx = struct.unpack('i', f.read(4))[0]
        ny = struct.unpack('i', f.read(4))[0]
        time = struct.unpack('d', f.read(8))[0]
        data = np.frombuffer(f.read(nx * ny * 8), dtype=np.float64)
        data = data.reshape((ny, nx))
    return data, time


def load_all_frames(output_dir: str) -> tuple[list[np.ndarray], list[float]]:
    """Load all binary files from the output directory."""
    files = sorted(glob.glob(os.path.join(output_dir, 'wave_*.bin')))

    if not files:
        raise FileNotFoundError(f"No wave_*.bin files found in {output_dir}")

    print(f"Found {len(files)} output files")

    frames = []
    times = []

    for i, filepath in enumerate(files):
        data, time = read_binary_file(filepath)
        frames.append(data)
        times.append(time)

        if (i + 1) % 50 == 0:
            print(f"Loaded {i + 1}/{len(files)} files...")

    print(f"Loaded all {len(files)} frames")
    return frames, times


def create_animation(
    frames: list[np.ndarray],
    times: list[float],
    output_path: str,
    fps: int = 30,
    dpi: int = 150,
    colormap: str = 'viridis',
    title: str = 'Dampened Wave Equation - Drumhead Simulation'
) -> None:
    """
    Create an MP4 or GIF animation from simulation frames.

    Args:
        frames: List of 2D numpy arrays (wave data for each time step)
        times: List of simulation times
        output_path: Output file path (.mp4 or .gif)
        fps: Frames per second
        dpi: Resolution (dots per inch)
        colormap: Matplotlib colormap name
        title: Title for the animation
    """
    # Determine global min/max for consistent color scaling
    all_data = np.concatenate([f.flatten() for f in frames])
    vmin, vmax = np.percentile(all_data, [1, 99])

    # Make symmetric around zero for better visualization
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Initial plot
    im = ax.imshow(
        frames[0],
        cmap=colormap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        origin='lower',
        extent=[0, 1, 0, 1],
        interpolation='bilinear'
    )

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{title}\nt = {times[0]:.4f} s', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Displacement')

    # Add grid lines for reference
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    def update(frame_num: int):
        """Update function for animation."""
        im.set_array(frames[frame_num])
        ax.set_title(f'{title}\nt = {times[frame_num]:.4f} s', fontsize=14)
        return [im]

    # Create animation
    print(f"Creating animation with {len(frames)} frames at {fps} fps...")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True
    )

    # Save animation
    output_ext = Path(output_path).suffix.lower()

    if output_ext == '.gif':
        print(f"Saving GIF to {output_path}...")
        ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    elif output_ext == '.mp4':
        print(f"Saving MP4 to {output_path}...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)
    else:
        # Default to MP4
        output_path = str(Path(output_path).with_suffix('.mp4'))
        print(f"Saving MP4 to {output_path}...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
        ani.save(output_path, writer=writer, dpi=dpi)

    plt.close()
    print(f"Animation saved to {output_path}")


def create_snapshot_grid(
    frames: list[np.ndarray],
    times: list[float],
    output_path: str,
    num_snapshots: int = 9,
    colormap: str = 'viridis',
    dpi: int = 150
) -> None:
    """
    Create a grid of snapshots at different time points.

    Args:
        frames: List of 2D numpy arrays
        times: List of simulation times
        output_path: Output file path (.png)
        num_snapshots: Number of snapshots to include
        colormap: Matplotlib colormap name
        dpi: Resolution
    """
    # Select frames to display
    indices = np.linspace(0, len(frames) - 1, num_snapshots, dtype=int)

    # Determine grid layout
    ncols = int(np.ceil(np.sqrt(num_snapshots)))
    nrows = int(np.ceil(num_snapshots / ncols))

    # Determine global min/max
    all_data = np.concatenate([f.flatten() for f in frames])
    vmin, vmax = np.percentile(all_data, [1, 99])
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < num_snapshots:
            idx = indices[i]
            im = ax.imshow(
                frames[idx],
                cmap=colormap,
                norm=Normalize(vmin=vmin, vmax=vmax),
                origin='lower',
                extent=[0, 1, 0, 1],
                interpolation='bilinear'
            )
            ax.set_title(f't = {times[idx]:.3f} s', fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            ax.axis('off')

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label='Displacement', shrink=0.8)

    plt.suptitle('Dampened Wave Equation - Time Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Snapshot grid saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize dampened wave equation simulation results'
    )
    parser.add_argument(
        'output_dir',
        help='Directory containing wave_*.bin output files'
    )
    parser.add_argument(
        '-o', '--output',
        default='wave_animation.mp4',
        help='Output file path (default: wave_animation.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Resolution in DPI (default: 150)'
    )
    parser.add_argument(
        '--colormap',
        default='viridis',
        help='Matplotlib colormap (default: viridis)'
    )
    parser.add_argument(
        '--snapshots',
        action='store_true',
        help='Also create a grid of snapshots'
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=1,
        help='Use every Nth frame (default: 1, use all)'
    )

    args = parser.parse_args()

    # Load frames
    frames, times = load_all_frames(args.output_dir)

    # Apply frame skipping if requested
    if args.skip > 1:
        frames = frames[::args.skip]
        times = times[::args.skip]
        print(f"Using every {args.skip}th frame: {len(frames)} frames total")

    # Create animation
    create_animation(
        frames,
        times,
        args.output,
        fps=args.fps,
        dpi=args.dpi,
        colormap=args.colormap
    )

    # Create snapshots if requested
    if args.snapshots:
        snapshot_path = str(Path(args.output).with_suffix('')) + '_snapshots.png'
        create_snapshot_grid(
            frames,
            times,
            snapshot_path,
            colormap=args.colormap,
            dpi=args.dpi
        )


if __name__ == '__main__':
    main()
