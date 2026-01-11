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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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

    # Set up the figure with dimensions divisible by 2 (required for h264)
    # 8x8 inches at default dpi gives clean pixel dimensions
    fig, ax = plt.subplots(figsize=(8, 8))

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
        saved = False

        # Try different codec configurations in order of preference
        codec_configs = [
            {'codec': 'libx264', 'extra_args': ['-pix_fmt', 'yuv420p']},
            {'codec': 'mpeg4', 'extra_args': []},
            {'codec': 'h264', 'extra_args': ['-pix_fmt', 'yuv420p']},
            {'codec': None, 'extra_args': []},  # Let FFmpeg choose
        ]

        for config in codec_configs:
            try:
                writer = animation.FFMpegWriter(
                    fps=fps,
                    codec=config['codec'],
                    extra_args=config['extra_args'] if config['extra_args'] else None
                )
                ani.save(output_path, writer=writer, dpi=dpi)
                saved = True
                break
            except Exception as e:
                codec_name = config['codec'] or 'default'
                print(f"Codec {codec_name} failed: {e}")
                continue

        if not saved:
            print("All FFmpeg codecs failed. Falling back to GIF format...")
            output_path = str(Path(output_path).with_suffix('.gif'))
            ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    else:
        # Default to GIF (more compatible)
        output_path = str(Path(output_path).with_suffix('.gif'))
        print(f"Saving GIF to {output_path}...")
        ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)

    plt.close()
    print(f"Animation saved to {output_path}")


def create_3d_animation(
    frames: list[np.ndarray],
    times: list[float],
    output_path: str,
    fps: int = 30,
    dpi: int = 100,
    colormap: str = 'viridis',
    title: str = 'Dampened Wave Equation - 3D Drumhead',
    downsample: int = 4,
    elevation: float = 30,
    azimuth: float = 45,
    rotate: bool = False
) -> None:
    """
    Create a 3D surface animation of the wave simulation.

    Args:
        frames: List of 2D numpy arrays (wave data for each time step)
        times: List of simulation times
        output_path: Output file path (.mp4 or .gif)
        fps: Frames per second
        dpi: Resolution (dots per inch)
        colormap: Matplotlib colormap name
        title: Title for the animation
        downsample: Factor to reduce grid resolution (for performance)
        elevation: Initial viewing elevation angle
        azimuth: Initial viewing azimuth angle
        rotate: If True, slowly rotate the view during animation
    """
    # Downsample frames for performance
    ds_frames = [f[::downsample, ::downsample] for f in frames]
    ny, nx = ds_frames[0].shape

    # Create meshgrid for surface plot
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Determine global min/max for consistent color and z scaling
    all_data = np.concatenate([f.flatten() for f in ds_frames])
    vmin, vmax = np.percentile(all_data, [1, 99])
    vabs = max(abs(vmin), abs(vmax))
    zmin, zmax = -vabs * 1.2, vabs * 1.2  # Add some headroom

    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get colormap
    cmap = plt.colormaps.get_cmap(colormap)
    norm = Normalize(vmin=-vabs, vmax=vabs)

    # Initial surface plot
    surf = ax.plot_surface(
        X, Y, ds_frames[0],
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        rcount=ny,
        ccount=nx
    )

    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Displacement', fontsize=12)
    ax.set_title(f'{title}\nt = {times[0]:.4f} s', fontsize=14)

    # Set fixed axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(zmin, zmax)

    # Set initial view angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Displacement')

    plt.tight_layout()

    def update(frame_num: int):
        """Update function for animation."""
        ax.clear()

        # Replot surface
        surf = ax.plot_surface(
            X, Y, ds_frames[frame_num],
            cmap=cmap,
            norm=norm,
            linewidth=0,
            antialiased=True,
            rcount=ny,
            ccount=nx
        )

        # Reset labels and limits
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Displacement', fontsize=12)
        ax.set_title(f'{title}\nt = {times[frame_num]:.4f} s', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(zmin, zmax)

        # Rotate view if enabled
        if rotate:
            ax.view_init(elev=elevation, azim=azimuth + frame_num * 0.5)
        else:
            ax.view_init(elev=elevation, azim=azimuth)

        return [surf]

    # Create animation
    print(f"Creating 3D animation with {len(ds_frames)} frames at {fps} fps...")
    print(f"Grid downsampled from {frames[0].shape} to {ds_frames[0].shape}")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(ds_frames),
        interval=1000 / fps,
        blit=False  # 3D plots don't support blitting
    )

    # Save animation
    output_ext = Path(output_path).suffix.lower()

    if output_ext == '.gif':
        print(f"Saving GIF to {output_path}...")
        ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    elif output_ext == '.mp4':
        print(f"Saving MP4 to {output_path}...")
        saved = False

        # Try different codec configurations in order of preference
        codec_configs = [
            {'codec': 'libx264', 'extra_args': ['-pix_fmt', 'yuv420p']},
            {'codec': 'mpeg4', 'extra_args': []},
            {'codec': 'h264', 'extra_args': ['-pix_fmt', 'yuv420p']},
            {'codec': None, 'extra_args': []},  # Let FFmpeg choose
        ]

        for config in codec_configs:
            try:
                writer = animation.FFMpegWriter(
                    fps=fps,
                    codec=config['codec'],
                    extra_args=config['extra_args'] if config['extra_args'] else None
                )
                ani.save(output_path, writer=writer, dpi=dpi)
                saved = True
                break
            except Exception as e:
                codec_name = config['codec'] or 'default'
                print(f"Codec {codec_name} failed: {e}")
                continue

        if not saved:
            print("All FFmpeg codecs failed. Falling back to GIF format...")
            output_path = str(Path(output_path).with_suffix('.gif'))
            ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    else:
        output_path = str(Path(output_path).with_suffix('.gif'))
        print(f"Saving GIF to {output_path}...")
        ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)

    plt.close()
    print(f"3D animation saved to {output_path}")


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
    parser.add_argument(
        '--3d',
        dest='three_d',
        action='store_true',
        help='Create 3D surface animation instead of 2D heatmap'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=4,
        help='Downsample factor for 3D mode (default: 4, use 1 for full resolution)'
    )
    parser.add_argument(
        '--elevation',
        type=float,
        default=30,
        help='3D view elevation angle in degrees (default: 30)'
    )
    parser.add_argument(
        '--azimuth',
        type=float,
        default=45,
        help='3D view azimuth angle in degrees (default: 45)'
    )
    parser.add_argument(
        '--rotate',
        action='store_true',
        help='Slowly rotate the 3D view during animation'
    )

    args = parser.parse_args()

    # Load frames
    frames, times = load_all_frames(args.output_dir)

    # Apply frame skipping if requested
    if args.skip > 1:
        frames = frames[::args.skip]
        times = times[::args.skip]
        print(f"Using every {args.skip}th frame: {len(frames)} frames total")

    # Create animation (3D or 2D)
    if args.three_d:
        create_3d_animation(
            frames,
            times,
            args.output,
            fps=args.fps,
            dpi=args.dpi,
            colormap=args.colormap,
            downsample=args.downsample,
            elevation=args.elevation,
            azimuth=args.azimuth,
            rotate=args.rotate
        )
    else:
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
