"""Build comparison videos from per-condition rollouts via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None


def hstack_conditions_for_seed(
    output_dir: Path,
    condition_videos: dict[str, Path],
    seed: int,
    env_name: str,
    target_duration_s: float = 12.0,
    fps: int = 20,
    panel_w: int = 768,
    panel_h: int = 384,
    band_top: int = 70,
    band_bot: int = 60,
) -> Path | None:
    """hstack one rollout per condition for the same seed/env into a single mp4.

    Each panel is captioned with the condition name; a master title is added.
    """
    if not _ffmpeg_ok() or not condition_videos:
        return None
    out_dir = output_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{env_name}  ·  seed={seed}  ·  conditions side-by-side"
    cap_dir = out_dir / "_captions"
    cap_dir.mkdir(exist_ok=True)
    title_file = cap_dir / f"title_seed{seed}.txt"
    title_file.write_text(title)

    cap_files: dict[str, Path] = {}
    for cond_name in condition_videos:
        f = cap_dir / f"cap_{cond_name}_seed{seed}.txt"
        f.write_text(cond_name)
        cap_files[cond_name] = f

    # Build filter graph
    inputs: list[str] = []
    chain: list[str] = []
    labels = []
    for i, (cond_name, vid) in enumerate(condition_videos.items()):
        inputs.extend(["-i", str(vid)])
        chain.append(
            f"[{i}:v]scale={panel_w}:{panel_h}:flags=lanczos,"
            f"tpad=stop_mode=clone:stop_duration={target_duration_s},"
            f"trim=duration={target_duration_s},setpts=PTS-STARTPTS,"
            f"pad={panel_w}:{panel_h + band_top + band_bot}:0:{band_top}:black,"
            f"drawtext=fontfile={_FONT}:textfile={cap_files[cond_name]}:"
            f"x=(w-tw)/2:y=22:fontsize=22:fontcolor=white"
            f"[P{i}]"
        )
        labels.append(f"[P{i}]")
    chain.append("".join(labels) + f"hstack=inputs={len(condition_videos)}[stacked]")
    chain.append(
        f"[stacked]pad=iw:ih+60:0:60:black,"
        f"drawtext=fontfile={_FONT}:textfile={title_file}:"
        f"x=(w-tw)/2:y=20:fontsize=24:fontcolor=0xFFD700[final]"
    )
    filter_complex = ";".join(chain)
    out_path = out_dir / f"{env_name.replace('/', '__')}_seed{seed}_grid.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[final]",
        "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
