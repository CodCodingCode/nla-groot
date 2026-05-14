#!/usr/bin/env python
"""Offline AV text overlay on LeRobot episode video.

Prerequisites
-------------
1. **Matching data**: ``--activations-root`` must come from
   ``scripts/extraction/run_extract.py`` (or the same storage layout) using the
   **same** LeRobot dataset path / episodes as ``--dataset-root``. If you mix a
   dump from dataset A with MP4s from dataset B, frames and activations will
   not correspond.

2. **SFT checkpoint**: ``--av-dir`` must be the ``av/`` subdirectory written by
   ``nla.training.sft.run_sft`` (contains ``adapter_config.json``,
   ``av_config.json``, ``act_proj.pt``, …)—not the parent ``output_dir``.

3. **Records**: Rows in ``index.jsonl`` need ``episode_index`` and ``step_index``.
   We decode the video frame at ``step_index`` (zero-based), which matches
   typical LeRobot alignment with policy steps for the primary camera.

Example::

    PYTHONPATH=src python scripts/eval/overlay_av_video.py \\
        --activations-root data/activations/my_run \\
        --dataset-root     third_party/Isaac-GR00T/demo_data/simplerenv_bridge_sample \\
        --av-dir           data/sft/my_run/av \\
        --out              /tmp/av_overlay.mp4 \\
        --max-steps 120 --greedy

Output: an MP4 with semi-transparent text strip at the bottom showing the AV
generation for one sampled token position per timestep (see ``--position-type``).

Frames are upscaled (default target height 720p) so output is viewable full-screen;
text is wrapped to **pixel width** (not a fixed character count) so lines are not
clipped. Use ``--encode-h264`` to mux a Mac/QuickTime-friendly H.264 file (requires
``ffmpeg`` on PATH).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def _line_metrics(font_scale: float, thickness: int) -> tuple[int, int]:
    """Returns (line_height_px, baseline_to_bottom_px) for one text row."""
    (_, h), baseline = cv2.getTextSize("Ag_jy", FONT_FACE, font_scale, thickness)
    return int(h * 1.35 + baseline), h + baseline


def _wrap_lines_to_pixels(
    text: str,
    max_pixel_width: int,
    *,
    font_scale: float,
    thickness: int,
) -> list[str]:
    """Word-wrap and hard-break long tokens so no line exceeds ``max_pixel_width``."""
    if max_pixel_width < 32:
        max_pixel_width = 32

    def fits(t: str) -> bool:
        w, _ = cv2.getTextSize(t, FONT_FACE, font_scale, thickness)[0]
        return w <= max_pixel_width

    out_lines: list[str] = []
    for para in text.split("\n"):
        para = para.rstrip()
        if not para:
            out_lines.append("")
            continue
        words = para.split()
        cur: list[str] = []
        for word in words:
            trial = " ".join(cur + [word]) if cur else word
            if fits(trial):
                cur.append(word)
                continue
            if cur:
                out_lines.append(" ".join(cur))
                cur = []
            # Single word may still overflow
            if fits(word):
                cur = [word]
                continue
            chunk = ""
            for ch in word:
                t2 = chunk + ch
                if fits(t2):
                    chunk = t2
                else:
                    if chunk:
                        out_lines.append(chunk)
                    chunk = ch
            if chunk:
                cur = [chunk]
        if cur:
            out_lines.append(" ".join(cur))
    return out_lines or [""]


def _upscale_bgr(bgr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize so height == ``target_h`` (LANCZOS), preserving aspect ratio."""
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0 or target_h <= 0:
        return bgr
    if h == target_h:
        return bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = target_h
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _draw_text_overlay(
    bgr: np.ndarray,
    lines: list[str],
    *,
    margin: int,
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.62,
) -> np.ndarray:
    """Bottom panel: semi-transparent fill plus outlined text (readable on video)."""
    h, w = bgr.shape[:2]
    line_h, _ = _line_metrics(font_scale, thickness)
    n = len(lines)
    pad_top = margin
    pad_bottom = margin
    box_h = min(h - margin, pad_top + pad_bottom + n * line_h)
    y0 = max(0, h - box_h)

    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (24, 24, 24), -1)
    out = cv2.addWeighted(overlay, bg_alpha, bgr, 1.0 - bg_alpha, 0)

    x0 = margin
    y = y0 + pad_top + line_h
    for line in lines:
        s = line if line else " "
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            cv2.putText(
                out, s, (x0 + dx, y + dy), FONT_FACE, font_scale, outline_color, thickness + 1, cv2.LINE_AA
            )
        cv2.putText(out, s, (x0, y), FONT_FACE, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h
    return out


def _post_encode_h264(src_path: Path, dst_path: Path, *, crf: int) -> None:
    """Mux H.264 + yuv420p + faststart for QuickTime / Safari."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to use --encode-h264")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        "-an",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--activations-root", required=True, help="Directory with index.jsonl + shard_*")
    p.add_argument("--dataset-root", required=True, help="LeRobot v2.1 dataset root (MP4s)")
    p.add_argument("--av-dir", required=True, help="Path to SFT av/ checkpoint directory")
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument(
        "--episode",
        type=int,
        default=None,
        help="If set, only examples with this episode_index",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Cap number of frames written")
    p.add_argument(
        "--video-key",
        default=None,
        help="Modality short key for video (default: first key in meta/modality.json)",
    )
    p.add_argument(
        "--position-type",
        choices=["last_text", "image_patch", "anchor", "mixture"],
        default="last_text",
        help="Token position for the verbalizer (mixture = training-style random mix)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for mixture / image_patch sampling")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decode (do_sample=False); stable captions",
    )
    p.add_argument(
        "--output-height",
        type=int,
        default=720,
        help="Upscale each frame so height matches this (0 = keep native resolution)",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=None,
        help="OpenCV font scale (default: auto from output height)",
    )
    p.add_argument(
        "--no-h264",
        action="store_true",
        help="Do not run ffmpeg libx264 pass (OpenCV mp4v only; worse QuickTime support)",
    )
    p.add_argument(
        "--h264-crf",
        type=int,
        default=18,
        help="CRF for libx264 when encoding (lower = larger file, better quality)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    from nla.extraction.storage import ActivationShardReader
    from nla.labeling.frames import DatasetInfo, EpisodeFrameLoader
    from nla.training.checkpoint import load_av_from_sft
    from nla.training.sampling import TokenPositionSampler

    root_act = Path(args.activations_root)
    av_dir = Path(args.av_dir)
    ds_root = Path(args.dataset_root)
    if not root_act.is_dir():
        logger.error("activations-root is not a directory: %s", root_act)
        return 2
    if not av_dir.is_dir():
        logger.error("av-dir is not a directory: %s", av_dir)
        return 2
    av_cfg_path = av_dir / "av_config.json"
    if not av_cfg_path.is_file():
        logger.error(
            "Missing %s — no SFT verbalizer checkpoint. Run "
            "``scripts/training/run_sft.py`` with this directory as "
            "``--output-dir`` …/av, or restore the av/ folder from backup.",
            av_cfg_path,
        )
        return 2
    if not ds_root.is_dir():
        logger.error("dataset-root is not a directory: %s", ds_root)
        return 2

    info = DatasetInfo.from_root(ds_root)
    video_key = args.video_key or (info.video_keys[0] if info.video_keys else None)
    if not video_key:
        logger.error("No video keys in dataset modality.json")
        return 2

    reader = ActivationShardReader(root_act)
    force_type: str | None = None if args.position_type == "mixture" else args.position_type
    sampler = TokenPositionSampler(seed=args.seed)

    av = load_av_from_sft(args.av_dir, device=args.device, freeze=True)

    records = []
    for rec in reader.records:
        if args.episode is not None and rec.episode_index != args.episode:
            continue
        if rec.episode_index is None or rec.step_index is None:
            logger.warning("Skipping %s: missing episode_index or step_index", rec.example_id)
            continue
        records.append(rec)
        if args.max_steps is not None and len(records) >= args.max_steps:
            break

    if not records:
        logger.error("No examples to render (check filters and index.jsonl).")
        return 2

    fps = float(info.fps) if info.fps > 0 else 15.0
    use_h264 = (not args.no_h264) and bool(shutil.which("ffmpeg"))
    if not use_h264 and not args.no_h264:
        logger.warning("ffmpeg not on PATH; writing OpenCV mp4v only. Install ffmpeg for H.264 / QuickTime.")

    tmp_path: Path | None = None
    if use_h264:
        fd, tmp_name = tempfile.mkstemp(suffix=".mp4", prefix="nla_overlay_")
        os.close(fd)
        tmp_path = Path(tmp_name)
    capture_path = tmp_path if tmp_path is not None else Path(args.out)

    writer: cv2.VideoWriter | None = None
    loader: EpisodeFrameLoader | None = None
    current_ep: int | None = None
    frames_written = 0

    try:
        for rec in records:
            if current_ep != rec.episode_index:
                if loader is not None:
                    loader.close()
                loader = EpisodeFrameLoader(ds_root, int(rec.episode_index))
                current_ep = int(rec.episode_index)

            blob = reader[reader._by_id[rec.example_id]]
            feat = blob["features"]
            attn = blob["attention_mask"].reshape(-1)
            imgm = blob["image_mask"].reshape(-1)
            if feat.dim() == 3 and feat.shape[0] == 1:
                feat = feat[0]
            if feat.dim() != 2:
                logger.error("Expected features [T,H], got %s", tuple(feat.shape))
                return 2

            ptype, tok_idx = sampler.sample(attn.cpu(), imgm.cpu(), force_type=force_type)
            vec = feat[tok_idx : tok_idx + 1].to(args.device)

            gen = av.generate(
                vec,
                [ptype],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=not args.greedy,
            )
            text = gen["text"][0].strip()
            header = f"ep={rec.episode_index} step={rec.step_index} pos={ptype} idx={tok_idx}"
            body = text if text else "(empty)"
            full_text = header + "\n" + body

            try:
                rgb = loader.frame(video_key, int(rec.step_index))
            except (FileNotFoundError, IndexError, ValueError, OSError) as e:
                logger.warning("Frame decode failed for %s: %s", rec.example_id, e)
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.output_height > 0:
                bgr = _upscale_bgr(bgr, args.output_height)

            h2, w2 = bgr.shape[:2]
            margin = max(14, h2 // 48)
            if args.font_scale is not None:
                font_scale = float(args.font_scale)
            else:
                font_scale = max(0.48, min(0.92, (h2 / 720.0) * 0.68))
            thickness = max(1, min(3, int(round(font_scale * 2.0))))

            max_tw = w2 - 2 * margin
            lines = _wrap_lines_to_pixels(
                full_text, max_tw, font_scale=font_scale, thickness=thickness
            )
            frame = _draw_text_overlay(
                bgr,
                lines,
                margin=margin,
                font_scale=font_scale,
                thickness=thickness,
            )

            if writer is None:
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(capture_path), fourcc, fps, (fw, fh))
                if not writer.isOpened():
                    logger.error("Could not open VideoWriter for %s", capture_path)
                    return 2

            writer.write(frame)
            frames_written += 1

        if writer is None or frames_written == 0:
            logger.error("No frames written.")
            return 2
        logger.info("Wrote %d frames to %s", frames_written, capture_path)
    finally:
        if writer is not None:
            writer.release()
        if loader is not None:
            loader.close()

    if use_h264 and tmp_path is not None and frames_written > 0:
        out_p = Path(args.out)
        try:
            _post_encode_h264(tmp_path, out_p, crf=args.h264_crf)
            logger.info("H.264 output (QuickTime-friendly): %s", out_p)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed: %s\n%s", e, e.stderr or "")
            logger.info("Leaving intermediate OpenCV file at %s", tmp_path)
            return 2
        else:
            tmp_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
