#!/usr/bin/env python
"""Per-step AV captioning during a closed-loop LIBERO rollout.

Drives ``gym.make("libero_sim/<task>")`` end-to-end against a local
``Gr00tPolicy`` (no ZMQ server needed) and on every step registers a
forward hook on the backbone that captures ``backbone_features`` at the
same token position the NLA steer hook would target. After each episode
we batch-decode the captured ``h`` vectors through the SFT-trained AV and
write::

    captions.jsonl   {"episode": int, "step": int, "position": str, "caption": str, "instruction": str}
    rollout_<ep>.mp4 video of agent-view frames with each step's AV caption overlayed

The script is a "what does the model think mid-rollout" interpretability
tool. Pair it with ``scripts/eval/closed_loop_sim_ab.py``: that one is the
quantitative success-rate A/B, this one is the qualitative caption trace.

Optional steering: when ``--ar-dir`` plus a ``--steer-text-file`` are
provided, the same backbone hook that captures ``h`` is *also* in the
modify path -- so the captured ``h`` reflects what the steered policy
actually computed at that step, which is what you want for "did the
steering shift what the AV would say".

Usage::

    PYTHONPATH=src python scripts/eval/closed_loop_av_capture.py \\
        --ckpt-dir         data/sft/libero_4suite_v3 \\
        --groot-model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --env-name         libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \\
        --n-episodes       2 \\
        --max-steps        200 \\
        --position-type    last_text \\
        --out-dir          data/sft/libero_4suite_v3/av_capture/goal_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Hook that captures backbone_features per step
# ---------------------------------------------------------------------------

@dataclass
class _StepCapture:
    """One step's captured h plus context for AV.generate."""
    episode: int
    step: int
    position_type: str
    h: np.ndarray            # (D,) raw activation
    instruction: str
    success: bool


def _install_capture_hook(
    backbone: Any,
    spec: Any,                # SteerSpec
    state: dict,
    *,
    batch_index: int = 0,
):
    """Register a forward hook that records the same indices the steer hook
    would target, and stash h into ``state['rows']`` for post-rollout decode.

    Returns the hook handle (caller must remove on shutdown).
    """
    import torch
    from nla.steering.backbone_steer import resolve_steer_indices

    def _hook(module, inputs, output):
        del module, inputs
        feats = output["backbone_features"]
        attn = output["backbone_attention_mask"]
        img_m = output["image_mask"]
        try:
            idxs = resolve_steer_indices(attn, img_m, spec, batch_index=batch_index)
        except Exception:
            return
        if not idxs:
            return
        # First resolved index is the canonical anchor for this position type;
        # if multiple (e.g. image_patch_all), we average them so the captured
        # h is a single vector per step.
        bi = batch_index
        slab = feats[bi, idxs]    # (k, D)
        h = slab.mean(dim=0).detach().to("cpu").float().numpy()
        state["rows"].append((h, idxs[0]))
    return backbone.register_forward_hook(_hook)


# ---------------------------------------------------------------------------
# Frame capture + overlay helpers
# ---------------------------------------------------------------------------

def _save_video_with_captions(
    frames: list[np.ndarray],         # HWC uint8 BGR or RGB
    captions: list[str],              # parallel to frames
    out_path: Path,
    *,
    fps: float = 20.0,
    target_height: int = 480,
) -> None:
    """Compose a single MP4 with each frame's caption overlayed at the bottom.
    Uses cv2.VideoWriter via mp4v; for QuickTime-friendly H.264 re-encode
    with ``scripts/eval/overlay_av_video.py``'s ``--encode-h264`` post-pass."""
    import cv2

    if not frames:
        return
    # Standardize frames: BGR uint8 + scaled to target height
    h0, w0 = frames[0].shape[:2]
    if h0 < target_height:
        scale = target_height / h0
        w_t = int(round(w0 * scale))
        h_t = target_height
    else:
        scale = 1.0
        w_t, h_t = w0, h0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_t, h_t))
    try:
        for idx, frame in enumerate(frames):
            if scale != 1.0:
                frame = cv2.resize(frame, (w_t, h_t), interpolation=cv2.INTER_CUBIC)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Frames from LiberoEnv come in RGB; convert for VideoWriter
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
            cap = captions[idx] if idx < len(captions) else ""
            frame_bgr = _draw_caption(frame_bgr, cap)
            writer.write(frame_bgr)
    finally:
        writer.release()


def _draw_caption(bgr: np.ndarray, caption: str) -> np.ndarray:
    """Bottom semi-transparent box + outlined white text, line-wrapped."""
    import cv2
    h, w = bgr.shape[:2]
    font_scale = 0.45
    thickness = 1
    margin = 6
    lines = _wrap_to_pixels(caption, w - 2 * margin, font_scale, thickness)[:6]
    line_h = 18
    box_h = min(h, margin * 2 + line_h * len(lines))
    y0 = h - box_h if lines else h
    if lines:
        overlay = bgr.copy()
        cv2.rectangle(overlay, (0, y0), (w, h), (16, 16, 16), -1)
        out = cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0)
        y = y0 + margin + 12
        for ln in lines:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cv2.putText(out, ln, (margin + dx, y + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                            thickness + 1, cv2.LINE_AA)
            cv2.putText(out, ln, (margin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            y += line_h
        return out
    return bgr


def _wrap_to_pixels(text: str, max_px: int, font_scale: float, thickness: int) -> list[str]:
    import cv2
    if not text:
        return []
    def fits(t: str) -> bool:
        (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return w <= max_px
    out: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        cur = ""
        for w in words:
            cand = (cur + " " + w).strip()
            if fits(cand):
                cur = cand
            else:
                if cur:
                    out.append(cur)
                cur = w
        if cur:
            out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------

def _make_env(env_name: str):
    """Register LIBERO envs (idempotent) and gym.make the requested one."""
    import gymnasium as gym
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs
    register_libero_envs()
    return gym.make(env_name)


def _load_groot_policy(model_path: str, embodiment_tag: str, *, device: str = "cuda"):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from nla.extraction._compat import apply_all as apply_groot_compat
    apply_groot_compat()
    tag = EmbodimentTag.resolve(embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=tag,
        model_path=model_path,
        device=device,
        strict=True,
    )
    policy.model.eval()
    return Gr00tSimPolicyWrapper(policy, strict=True)


def _build_steer_spec(args):
    from nla.steering.backbone_steer import SteerSpec
    return SteerSpec(
        placement=args.position_type,    # capture at the same position the steer hook would
        blend=float(args.blend),
        fixed_token_index=args.fixed_token_index,
        image_patch_seed=int(args.image_patch_seed),
    )


def _maybe_apply_steer(policy, args):
    """If --ar-dir + --steer-text-file are provided, return a context manager
    that applies the steer hook. Otherwise return a no-op."""
    import contextlib
    if not args.ar_dir or not args.steer_text_file:
        @contextlib.contextmanager
        def _noop():
            yield None
        return _noop()
    import torch
    from nla.steering import (
        SteerSpec, ar_text_to_backbone_vec, attach_backbone_steer,
    )
    from nla.training.checkpoint import load_ar_from_sft
    from nla.steering.sim_policy_wrapper import _resolve_inner_backbone
    backbone = _resolve_inner_backbone(policy)
    ar = load_ar_from_sft(Path(args.ar_dir), device=args.device, freeze=True)
    steer_vec = ar_text_to_backbone_vec(ar, Path(args.steer_text_file).read_text())
    spec = SteerSpec(
        placement=args.position_type, blend=float(args.blend),
        fixed_token_index=args.fixed_token_index,
        image_patch_seed=int(args.image_patch_seed),
    )
    return attach_backbone_steer(backbone, steer_vec, spec)


def _decode_captions(av, h_list: list[np.ndarray], position_types: list[str], *, device: str,
                     temperature: float, max_new_tokens: int, batch_size: int = 4) -> list[str]:
    """Batch-decode captions for a list of captured h vectors."""
    import torch
    out: list[str] = []
    do_sample = float(temperature) > 0.0
    for start in range(0, len(h_list), batch_size):
        H = torch.tensor(np.stack(h_list[start:start + batch_size]), device=device)
        pts = position_types[start:start + batch_size]
        with torch.no_grad():
            gen = av.generate(
                activations=H, position_types=pts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else 1.0,
            )
        out.extend(t.strip() for t in gen["text"])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt-dir", required=True, help="SFT run dir with av/ subdir")
    p.add_argument("--groot-model-path", required=True)
    p.add_argument("--env-name", required=True,
                   help="Gym env id, e.g. libero_sim/KITCHEN_SCENE3_<task>")
    p.add_argument("--ar-dir", default=None,
                   help="If provided with --steer-text-file, applies NLA steer alongside capture.")
    p.add_argument("--steer-text-file", default=None)
    p.add_argument("--position-type", default="last_text",
                   choices=["last_text", "image_patch", "anchor"],
                   help="Token slot to capture h at on every step.")
    p.add_argument("--blend", type=float, default=1.0,
                   help="Only used when steering is on.")
    p.add_argument("--fixed-token-index", type=int, default=None)
    p.add_argument("--image-patch-seed", type=int, default=0)
    p.add_argument("--n-episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--av-batch-size", type=int, default=4)
    p.add_argument("--out-dir", required=True,
                   help="Will write captions.jsonl + rollout_<ep>.mp4 here.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AV from {args.ckpt_dir}/av/ ...", flush=True)
    from nla.training.checkpoint import load_av_from_sft
    av = load_av_from_sft(Path(args.ckpt_dir) / "av",
                         device=args.device, freeze=True)

    print(f"Loading GR00T policy from {args.groot_model_path} ...", flush=True)
    policy = _load_groot_policy(
        args.groot_model_path, args.embodiment_tag, device=args.device,
    )

    from nla.steering.sim_policy_wrapper import _resolve_inner_backbone
    backbone = _resolve_inner_backbone(policy)

    # Build the capture hook spec (same placement we'd steer, even when not
    # steering, so the captured h is semantically aligned with what the SFT
    # AV was trained to verbalize).
    spec = _build_steer_spec(args)

    env = _make_env(args.env_name)
    captions_path = out_dir / "captions.jsonl"
    with captions_path.open("w") as cap_f:
        for ep in range(args.n_episodes):
            print(f"\n==== episode {ep+1}/{args.n_episodes} ====", flush=True)
            t0 = time.time()
            obs, info = env.reset(seed=args.seed + ep)
            instr = obs.get("annotation.human.action.task_description", "")
            frames: list[np.ndarray] = []
            captured: list[np.ndarray] = []
            captured_position: list[str] = []
            success = False
            steps_done = 0

            steer_cm = _maybe_apply_steer(policy, args)
            with steer_cm:
                state: dict = {"rows": []}
                handle = _install_capture_hook(backbone, spec, state)
                try:
                    for step in range(args.max_steps):
                        state["rows"] = []
                        action = policy.get_action(obs)
                        # The hook fires when policy invokes backbone. Pull captured h.
                        if state["rows"]:
                            h, _idx = state["rows"][-1]
                            captured.append(h)
                            captured_position.append(args.position_type)
                        else:
                            # Hook didn't fire (unexpected). Skip caption for this step.
                            captured.append(np.zeros(2048, dtype=np.float32))
                            captured_position.append(args.position_type)

                        frame = obs.get("video.image")
                        if frame is not None:
                            frames.append(np.asarray(frame))
                        obs, reward, terminated, truncated, info = env.step(action)
                        steps_done = step + 1
                        if info.get("success"):
                            success = True
                            break
                        if terminated or truncated:
                            break
                finally:
                    handle.remove()

            elapsed = time.time() - t0
            print(f"  steps={steps_done} success={success} elapsed={elapsed:.1f}s", flush=True)
            if not captured:
                print("  WARN: no h captured this episode; skipping AV decode.", flush=True)
                continue

            print(f"  decoding {len(captured)} captions ...", flush=True)
            captions = _decode_captions(
                av, captured, captured_position,
                device=args.device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.av_batch_size,
            )
            for step, cap in enumerate(captions):
                cap_f.write(json.dumps({
                    "episode": ep, "step": step,
                    "position_type": captured_position[step],
                    "instruction": instr,
                    "success": success,
                    "caption": cap,
                }) + "\n")
            cap_f.flush()

            if frames:
                vid_path = out_dir / f"rollout_ep{ep:02d}.mp4"
                # Keep one caption line on screen per step so the viewer can
                # follow it. Render the first 4-5 wrapped lines only.
                shown = [
                    "\n".join(c.splitlines()[:4]) if c else "" for c in captions
                ]
                _save_video_with_captions(frames, shown, vid_path)
                print(f"  -> wrote {vid_path}")
    env.close()
    print(f"\nDone. captions.jsonl + per-episode mp4s in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
