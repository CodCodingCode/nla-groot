#!/usr/bin/env python
"""MP4: dataset frames + **baseline vs NLA-steered** GR00T actions (numeric overlay).

For each timestep we run ``policy.get_action`` **twice** on the same observation:
once normal, once with :func:`nla.steering.attach_backbone_steer` using
``AR(text) → ĥ``. The video shows the camera frame and a text panel with
``max|Δaction|`` and a short preview of one action vector (so you can *see*
whether injection moves the policy output on real pixels).

Requires the same Isaac-GR00T + Hugging Face access as ``nla_steer_groot_action.py``
(Cosmos processor gate, GR00T weights, GPU).

Output: an MP4 with the **camera image on top** and a **dedicated text panel below**
(so lines are never clipped). Frames are upscaled (default **1080p** height) for a
sharper picture; when ``ffmpeg`` is available, a **libx264** pass is used by default
(avoid ``--no-h264`` unless you must — OpenCV ``mp4v`` is often blocky).

Example::

    PYTHONPATH=src .venv/bin/python scripts/eval/nla_steer_overlay_video.py \\
        --model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path third_party/Isaac-GR00T/demo_data/droid_sample \\
        --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \\
        --ar-dir data/sft/droid_100ep_v2_nce/ar \\
        --traj-id 0 --max-steps 32 \\
        --placement anchor --blend 1.0 \\
        --text-file steer_bullets.txt \\
        --out /tmp/nla_steer_compare.mp4

**Stochastic policies:** diffusion sampling can change both runs. Use ``--seed``
to fix PyTorch RNG before each pair (helps; does not guarantee bit-identical
baselines across steps if the policy reseeds internally).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("nla.steer_overlay_vid")

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX


def _imports():
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from nla.extraction._compat import apply_all as apply_groot_compat

    apply_groot_compat()

    return dict(
        LeRobotEpisodeLoader=LeRobotEpisodeLoader,
        extract_step_data=extract_step_data,
        EmbodimentTag=EmbodimentTag,
        Gr00tPolicy=Gr00tPolicy,
    )


def _policy_get_action(policy: Any, observation: dict[str, Any]) -> dict[str, Any]:
    fn = getattr(policy, "get_action", None)
    if fn is None:
        raise RuntimeError("Gr00tPolicy.get_action missing")
    out = fn(observation)
    if isinstance(out, tuple) and len(out) >= 1:
        out = out[0]
    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected get_action return: {type(out)}")
    if any(isinstance(v, dict) for v in out.values()):
        flat: dict[str, Any] = {}
        for k, v in out.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        return flat
    return out


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if hasattr(x, "detach"):
        return np.asarray(x.detach().cpu().float().numpy())
    return np.asarray(x, dtype=np.float64)


def _action_delta_global(a0: dict[str, Any], a1: dict[str, Any]) -> tuple[float, dict[str, float]]:
    keys = sorted(set(a0.keys()) | set(a1.keys()))
    per: dict[str, float] = {}
    gmax = 0.0
    for k in keys:
        u = _to_numpy(a0.get(k)).ravel()
        v = _to_numpy(a1.get(k)).ravel()
        if u.shape != v.shape:
            per[k] = float("nan")
            continue
        if u.size == 0:
            continue
        d = np.abs(v.astype(np.float64) - u.astype(np.float64))
        mx = float(d.max())
        per[k] = mx
        gmax = max(gmax, mx)
    return gmax, per


def _pick_preview_key(a: dict[str, Any]) -> str | None:
    for k in sorted(a.keys()):
        if _to_numpy(a[k]).size > 0:
            return k
    return None


def _line_metrics(font_scale: float, thickness: int) -> tuple[int, int]:
    (_, h), baseline = cv2.getTextSize("Ag_jy", FONT_FACE, font_scale, thickness)
    return int(h * 1.35 + baseline), h + baseline


def _wrap_lines_to_pixels(
    text: str,
    max_pixel_width: int,
    *,
    font_scale: float,
    thickness: int,
) -> list[str]:
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
    if h <= 0 or w <= 0 or target_h <= 0:
        return bgr
    if h == target_h:
        return bgr
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = target_h
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def _draw_text_panel_below_frame(
    bgr: np.ndarray,
    lines: list[str],
    *,
    margin: int,
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.72,
) -> np.ndarray:
    """Paste image on top; draw **all** lines in a new panel below (nothing clipped)."""
    h, w = bgr.shape[:2]
    line_h, _ = _line_metrics(font_scale, thickness)
    n = len(lines)
    panel_h = max(line_h, margin + margin + n * line_h)
    out_h = h + int(panel_h)
    canvas = np.zeros((out_h, w, 3), dtype=np.uint8)
    canvas[:h, :w] = bgr

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h), (w, out_h), (18, 18, 22), -1)
    out = cv2.addWeighted(overlay, bg_alpha, canvas, 1.0 - bg_alpha, 0)

    x0 = margin
    y = h + margin + line_h
    for line in lines:
        s = line if line else " "
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            cv2.putText(
                out, s, (x0 + dx, y + dy), FONT_FACE, font_scale, outline_color, thickness + 1, cv2.LINE_AA
            )
        cv2.putText(out, s, (x0, y), FONT_FACE, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h
    return out


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
    """Legacy: overlay on **same** canvas (text can clip if panel is too tall). Prefer
    :func:`_draw_text_panel_below_frame` for steer videos.
    """
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
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            cv2.putText(
                out, s, (x0 + dx, y + dy), FONT_FACE, font_scale, outline_color, thickness + 1, cv2.LINE_AA
            )
        cv2.putText(out, s, (x0, y), FONT_FACE, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_h
    return out


def _post_encode_h264(src_path: Path, dst_path: Path, *, crf: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not on PATH")
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
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ar-dir", required=True)
    p.add_argument("--traj-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--video-key", default=None, help="Default: first key in meta/modality.json")
    p.add_argument("--text", default=None)
    p.add_argument("--text-file", default=None)
    p.add_argument(
        "--placement",
        default="anchor",
        choices=["last_text", "image_patch", "anchor", "image_patch_all", "fixed"],
    )
    p.add_argument("--blend", type=float, default=1.0)
    p.add_argument("--fixed-token-index", type=int, default=None)
    p.add_argument("--image-patch-seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output MP4 path")
    p.add_argument(
        "--output-height",
        type=int,
        default=1080,
        help="Upscale frame height before compositing (0 = native; 1080 recommended)",
    )
    p.add_argument(
        "--panel-max-lines",
        type=int,
        default=26,
        help="Fixed wrapped-line count so panel height is constant for VideoWriter",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=None,
        help="OpenCV font scale (default: auto from output height)",
    )
    p.add_argument("--seed", type=int, default=0, help="Torch seed before each baseline/steer pair")
    p.add_argument("--preview-dims", type=int, default=5, help="How many leading dims to print for preview key")
    p.add_argument("--no-h264", action="store_true", help="OpenCV mp4v only (often blocky); prefer default H.264 when ffmpeg exists")
    p.add_argument("--h264-crf", type=int, default=18, help="libx264 CRF when re-muxing (lower = larger / sharper)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    import torch

    from nla.labeling.frames import DatasetInfo, EpisodeFrameLoader
    from nla.steering import SteerSpec, attach_backbone_steer, ar_text_to_backbone_vec
    from nla.steering.groot_obs import build_observation_for_step
    from nla.training.checkpoint import load_ar_from_sft

    if args.text and args.text_file:
        raise SystemExit("Use only one of --text or --text-file")
    if args.placement == "fixed" and args.fixed_token_index is None:
        raise SystemExit("--fixed-token-index required when --placement=fixed")
    steer_text = Path(args.text_file).read_text() if args.text_file else args.text
    if not steer_text:
        raise SystemExit("Provide --text or --text-file")

    mods = _imports()
    embodiment_tag = mods["EmbodimentTag"].resolve(args.embodiment_tag)

    ds_info = DatasetInfo.from_root(Path(args.dataset_path))
    video_key = args.video_key or (ds_info.video_keys[0] if ds_info.video_keys else None)
    if not video_key:
        raise SystemExit("No video keys in dataset modality.json")

    logger.info("Loading policy…")
    policy = mods["Gr00tPolicy"](
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    # Gr00tPolicy is not an nn.Module; the underlying VLA model is.
    policy.model.eval()

    modality_configs = deepcopy(policy.modality_configs)
    modality_configs.pop("action", None)
    loader = mods["LeRobotEpisodeLoader"](
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_configs,
        video_backend=args.video_backend,
    )
    language_keys = list(policy.modality_configs["language"].modality_keys)
    traj = loader[args.traj_id]

    ar = load_ar_from_sft(Path(args.ar_dir), device=args.device, freeze=True)
    steer_vec = ar_text_to_backbone_vec(ar, steer_text).to(args.device)
    spec = SteerSpec(
        placement=args.placement,  # type: ignore[arg-type]
        blend=float(args.blend),
        fixed_token_index=args.fixed_token_index,
        image_patch_seed=int(args.image_patch_seed),
    )

    fps = float(ds_info.fps) if ds_info.fps > 0 else 15.0
    use_h264 = (not args.no_h264) and bool(shutil.which("ffmpeg"))
    if not use_h264 and not args.no_h264:
        logger.warning("ffmpeg not on PATH; writing OpenCV mp4v only.")

    tmp_path: Path | None = None
    if use_h264:
        fd, tmp_name = tempfile.mkstemp(suffix=".mp4", prefix="nla_steer_ov_")
        os.close(fd)
        tmp_path = Path(tmp_name)
    capture_path = tmp_path if tmp_path is not None else Path(args.out)

    frame_loader = EpisodeFrameLoader(Path(args.dataset_path), int(args.traj_id))
    writer: cv2.VideoWriter | None = None
    n_steps = len(traj)
    limit = min(n_steps, args.max_steps)
    stride = max(1, int(args.step_stride))

    backbone = policy.model.backbone
    preview_n = max(1, int(args.preview_dims))

    try:
        with torch.inference_mode():
            for step_idx in range(0, limit, stride):
                obs = build_observation_for_step(
                    traj,
                    step_idx,
                    modality_configs,
                    embodiment_tag,
                    language_keys,
                    mods["extract_step_data"],
                )

                torch.manual_seed(int(args.seed) + step_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(args.seed) + step_idx)

                base = _policy_get_action(policy, obs)
                with attach_backbone_steer(backbone, steer_vec, spec):
                    steered = _policy_get_action(policy, obs)

                gmax, per = _action_delta_global(base, steered)
                pkey = _pick_preview_key(base) or _pick_preview_key(steered)
                prev_line = ""
                if pkey is not None:
                    b0 = _to_numpy(base.get(pkey)).ravel()[:preview_n]
                    b1 = _to_numpy(steered.get(pkey)).ravel()[:preview_n]
                    prev_line = (
                        f"preview {pkey} base: {np.array2string(b0, precision=3, floatmode='fixed')}\n"
                        f"preview {pkey} steer: {np.array2string(b1, precision=3, floatmode='fixed')}"
                    )

                top_keys = sorted(per.keys())[:4]
                per_bits = " ".join(f"{k}:{per[k]:.4f}" for k in top_keys if np.isfinite(per[k]))

                steer_bullets = [ln.strip() for ln in steer_text.strip().split("\n") if ln.strip()]
                steer_block = "\n".join(f"  - {b}" for b in steer_bullets[:10])

                block = (
                    f"NLA STEER | traj={args.traj_id} step={step_idx}\n"
                    f"{args.placement} | blend={args.blend}\n"
                    f"global max|dA| = {gmax:.6f}\n"
                    + (f"per-key max: {per_bits}\n" if per_bits else "")
                    + (prev_line + "\n" if prev_line else "")
                    + "steer bullets:\n"
                    + steer_block
                )

                try:
                    rgb = frame_loader.frame(video_key, step_idx)
                except Exception as e:
                    logger.warning("Frame decode failed step %s: %s", step_idx, e)
                    continue
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if bgr.dtype != np.uint8:
                    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
                if args.output_height > 0:
                    bgr = _upscale_bgr(bgr, args.output_height)
                h2, w2 = bgr.shape[:2]
                margin = max(16, h2 // 42)
                if args.font_scale is not None:
                    font_scale = float(args.font_scale)
                else:
                    font_scale = max(0.40, min(0.74, (h2 / 1080.0) * 0.56))
                thickness = max(1, min(3, int(round(font_scale * 2.0))))
                lines = _wrap_lines_to_pixels(
                    block, w2 - 2 * margin, font_scale=font_scale, thickness=thickness
                )
                cap = max(8, int(args.panel_max_lines))
                if len(lines) > cap:
                    lines = lines[: cap - 1] + ["… (truncated; increase --panel-max-lines)"]
                while len(lines) < cap:
                    lines.append("")

                frame = _draw_text_panel_below_frame(
                    bgr, lines, margin=margin, font_scale=font_scale, thickness=thickness
                )

                if writer is None:
                    fh, fw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(capture_path), fourcc, fps, (fw, fh))
                    if not writer.isOpened():
                        logger.error("VideoWriter failed for %s", capture_path)
                        return 2
                else:
                    if frame.shape[0] != fh or frame.shape[1] != fw:
                        logger.error(
                            "Frame size mismatch %s vs (%d,%d); keep --panel-max-lines fixed",
                            frame.shape[:2],
                            fh,
                            fw,
                        )
                        return 2
                writer.write(frame)

        if writer is None:
            logger.error("No frames written.")
            return 2
        logger.info("Wrote MP4 scaffold to %s", capture_path)
    finally:
        if writer is not None:
            writer.release()
        frame_loader.close()

    if use_h264 and tmp_path is not None:
        out_p = Path(args.out)
        try:
            _post_encode_h264(tmp_path, out_p, crf=args.h264_crf)
            logger.info("H.264 mux: %s", out_p)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed: %s", e.stderr or e)
            return 2
        finally:
            tmp_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
