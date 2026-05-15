#!/usr/bin/env python
"""Write an MP4 with **real LeRobot frames** + overlay in the same style as
``nla_steer_overlay_video.py``, but **without loading GR00T** (no Cosmos token).

Use this to sanity-check decoding + overlay layout when ``nla_steer_overlay_video.py``
cannot run. Overlay text clearly states **POLICY NOT RUN**.

Example::

    PYTHONPATH=src .venv/bin/python scripts/eval/nla_steer_overlay_visual_demo.py \\
        --dataset-path third_party/Isaac-GR00T/demo_data/droid_sample \\
        --out nla_steer_overlay_VISUAL_DEMO.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def _line_metrics(font_scale: float, thickness: int) -> tuple[int, int]:
    (_, h), baseline = cv2.getTextSize("Ag_jy", FONT_FACE, font_scale, thickness)
    return int(h * 1.35 + baseline), h + baseline


def _wrap_lines_to_pixels(text: str, max_pixel_width: int, *, font_scale: float, thickness: int) -> list[str]:
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
    h, w = bgr.shape[:2]
    if target_h <= 0 or h <= 0:
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
    h, w = bgr.shape[:2]
    line_h, _ = _line_metrics(font_scale, thickness)
    n = len(lines)
    pad_top = pad_bottom = margin
    box_h = min(h - margin, pad_top + pad_bottom + n * line_h)
    y0 = max(0, h - box_h)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (24, 24, 24), -1)
    out = cv2.addWeighted(overlay, bg_alpha, bgr, 1.0 - bg_alpha, 0)
    x0 = margin
    y = y0 + pad_top + line_h
    for line in lines:
        s = line if line else " "
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(
                out, s, (x0 + dx, y + dy), FONT_FACE, font_scale, outline_color, thickness + 1, cv2.LINE_AA
            )
        cv2.putText(out, s, (x0, y), FONT_FACE, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h
    return out


def _banner(out: np.ndarray, text: str) -> np.ndarray:
    """Red translucent strip at top."""
    h, w = out.shape[:2]
    overlay = out.copy()
    bh = max(36, h // 22)
    cv2.rectangle(overlay, (0, 0), (w, bh), (0, 0, 180), -1)
    o2 = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    fs = max(0.45, min(1.0, w / 920.0))
    cv2.putText(o2, text, (12, int(bh * 0.72)), FONT_FACE, fs, (255, 255, 255), max(1, int(round(fs * 2))), cv2.LINE_AA)
    return o2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--traj-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--step-stride", type=int, default=2)
    p.add_argument("--video-key", default=None)
    p.add_argument("--out", default="nla_steer_overlay_VISUAL_DEMO.mp4")
    p.add_argument("--output-height", type=int, default=720)
    args = p.parse_args()

    from nla.labeling.frames import DatasetInfo, EpisodeFrameLoader

    root = Path(args.dataset_path)
    info = DatasetInfo.from_root(root)
    vk = args.video_key or (info.video_keys[0] if info.video_keys else None)
    if not vk:
        raise SystemExit("No video keys")
    fps = float(info.fps) if info.fps > 0 else 15.0

    loader = EpisodeFrameLoader(root, int(args.traj_id))
    writer: cv2.VideoWriter | None = None
    out_path = Path(args.out)

    try:
        for step in range(0, min(args.max_steps, 500), max(1, args.step_stride)):
            rgb = loader.frame(vk, step)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.output_height > 0:
                bgr = _upscale_bgr(bgr, args.output_height)
            h2, w2 = bgr.shape[:2]

            # Illustrative numbers only (no policy).
            fake_base = np.sin(step * 0.11 + np.arange(5)) * 0.02
            fake_steer = fake_base + np.sin(step * 0.17 + np.arange(5)) * 0.015
            fake_delta = float(np.max(np.abs(fake_steer - fake_base)))

            block = (
                "VISUAL DEMO — GR00T/Cosmos NOT LOADED (set HF_TOKEN for real steer video)\n"
                f"Illustrative overlay layout only | traj={args.traj_id} step={step}\n"
                f"(fake) global max|dA| = {fake_delta:.6f}\n"
                f"(fake) preview: base={np.array2string(fake_base, precision=3)} | steer={np.array2string(fake_steer, precision=3)}\n"
                "Real script: scripts/eval/nla_steer_overlay_video.py"
            )
            margin = max(14, h2 // 48)
            font_scale = max(0.45, min(0.82, (h2 / 720.0) * 0.62))
            thickness = max(1, min(3, int(round(font_scale * 2.0))))
            lines = _wrap_lines_to_pixels(block, w2 - 2 * margin, font_scale=font_scale, thickness=thickness)
            frame = _draw_text_overlay(bgr, lines, margin=margin, font_scale=font_scale, thickness=thickness)
            frame = _banner(frame, "DEMO ONLY — NO POLICY")

            if writer is None:
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
                if not writer.isOpened():
                    raise SystemExit(f"Could not open VideoWriter for {out_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
        loader.close()

    print(f"Wrote {out_path.resolve()}  ({fps} fps, ~{args.max_steps // max(1, args.step_stride)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
