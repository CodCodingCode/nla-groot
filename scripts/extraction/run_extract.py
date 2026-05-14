#!/usr/bin/env python
"""Run GR00T over a LeRobot dataset, capturing per-token Qwen3-VL activations.

Example::

    python scripts/extraction/run_extract.py \
        --model-path  $HOME/nla-groot/.hf_cache/models--nvidia--GR00T-N1.7-3B/snapshots/* \
        --dataset-path third_party/Isaac-GR00T/demo_data/simplerenv_bridge_sample \
        --embodiment-tag BRIDGE_V2 \
        --out-root data/activations/bridge_pilot \
        --traj-ids 0 1 2 \
        --steps-per-traj 8 \
        --max-examples-per-shard 256

What gets saved::

    <out_root>/manifest.json
    <out_root>/index.jsonl
    <out_root>/shard_NNNNNN/activations.safetensors  # act_*, attn_*, img_*, ids_*
    <out_root>/shard_NNNNNN/meta.jsonl
    <out_root>/stats.json                            # α (P75 norm) and friends

We deliberately bypass the action head: we run only ``model.backbone(...)`` so
extraction cost is just one VLM forward per step rather than 4+ DiT denoising
passes.  The hook fires on the wrapper Qwen3Backbone, which returns the
``backbone_features``, ``backbone_attention_mask``, and ``image_mask`` we need.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import tree

logger = logging.getLogger("nla.extract")


# Note: we deliberately do NOT redirect HF_HOME. The HF auth token used for the
# gated Cosmos-Reason2-2B processor download lives in the user's default cache
# (~/.cache/huggingface/stored_tokens); overriding HF_HOME hides it.


def _preflight_processor_access(model_path: str) -> None:
    """Fail fast with a clear message if Cosmos-Reason2-2B isn't fetchable.

    The GR00T processor wraps ``Qwen3VLProcessor.from_pretrained("nvidia/Cosmos-Reason2-2B")``,
    which is gated.  Detect this and surface an actionable error instead of a
    deep transformers stacktrace.
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return  # Trust the user knows what they're doing.
    try:
        from huggingface_hub import HfApi  # type: ignore
        api = HfApi()
        api.model_info("nvidia/Cosmos-Reason2-2B")
    except Exception as e:  # pragma: no cover - depends on network/auth
        logger.warning(
            "Could not verify access to nvidia/Cosmos-Reason2-2B (%s). "
            "If the run fails with a GatedRepoError, request access at "
            "https://huggingface.co/nvidia/Cosmos-Reason2-2B and set HF_TOKEN.",
            type(e).__name__,
        )


def _imports():
    """Lazy import so --help works without a working GR00T env."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    # Apply runtime patches for known checkpoint/code drift in vendored GR00T.
    from nla.extraction._compat import apply_all as _apply_groot_compat
    _apply_groot_compat()

    from nla.extraction import (
        ActivationShardWriter,
        BackboneFeatureHook,
        RunManifest,
        attach_hooks,
        compute_stats,
        save_stats,
    )
    from nla.extraction.storage import ActivationShardReader
    from nla.layer_spec import (
        BACKBONE_EMBEDDING_DIM,
        GROOT_HF_REPO,
        TARGET_BACKBONE_FEATURES,
    )

    return dict(
        LeRobotEpisodeLoader=LeRobotEpisodeLoader,
        extract_step_data=extract_step_data,
        EmbodimentTag=EmbodimentTag,
        MessageType=MessageType,
        Gr00tPolicy=Gr00tPolicy,
        ActivationShardWriter=ActivationShardWriter,
        BackboneFeatureHook=BackboneFeatureHook,
        RunManifest=RunManifest,
        attach_hooks=attach_hooks,
        compute_stats=compute_stats,
        save_stats=save_stats,
        ActivationShardReader=ActivationShardReader,
        BACKBONE_EMBEDDING_DIM=BACKBONE_EMBEDDING_DIM,
        GROOT_HF_REPO=GROOT_HF_REPO,
        TARGET_BACKBONE_FEATURES=TARGET_BACKBONE_FEATURES,
    )


def _parse_observation(obs_flat, modality_configs):
    """Flat observation dict -> nested {video, state, language}.

    Mirrors ``parse_observation_gr00t`` from the standalone inference script but
    avoids importing it (which would drag in matplotlib).
    """
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs_flat[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def _prepare_step_obs(traj, step_count, modality_configs, embodiment_tag, language_keys, mods):
    """Build a model-ready (batched) observation for one trajectory step."""
    data_point = mods["extract_step_data"](
        traj, step_count, modality_configs, embodiment_tag, allow_padding=True
    )
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in language_keys:
        obs[language_key] = data_point.text
    return _parse_observation(obs, modality_configs)


def _run_backbone_forward(policy, observation, mods):
    """Drive only the backbone forward (no DiT, no action sampling)."""
    # Step 1: replicate Gr00tPolicy._get_action up to the model call.
    unbatched_observations = policy._unbatch_observation(observation)
    processed_inputs = []
    for obs in unbatched_observations:
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": mods["MessageType"].EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))
    collated_inputs = policy.collate_fn(processed_inputs)

    # Cast floats to model dtype (bf16 by default for inference).
    def _to_dtype(x):
        if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
            return x.to(dtype=policy.model.dtype)
        return x

    collated_inputs = tree.map_structure(_to_dtype, collated_inputs)

    # The Gr00tN1d7DataCollator wraps its output as ``{"inputs": batch}``; the
    # policy normally unpacks via ``model.get_action(**collated_inputs)``.
    if "inputs" in collated_inputs and "input_ids" not in collated_inputs:
        inner = collated_inputs["inputs"]
    else:
        inner = collated_inputs

    # Step 2: prepare_input + backbone only.
    with torch.inference_mode():
        backbone_inputs, _action_inputs = policy.model.prepare_input(inner)
        _ = policy.model.backbone(backbone_inputs)
    return backbone_inputs


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-path", required=True, help="Path or HF repo for GR00T checkpoint")
    p.add_argument("--dataset-path", required=True, help="LeRobot-format dataset root")
    p.add_argument("--embodiment-tag", required=True, help="e.g. BRIDGE_V2, OXE_DROID_*, LIBERO_PANDA")
    p.add_argument("--out-root", required=True, help="Where to write shards + manifest")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--traj-ids", type=int, nargs="*", default=None,
                   help="Trajectory IDs (default: all)")
    p.add_argument("--steps-per-traj", type=int, default=-1,
                   help="Max steps per trajectory (-1 = all)")
    p.add_argument("--step-stride", type=int, default=1,
                   help="Process every Nth step (1 = every step)")
    p.add_argument("--max-examples-per-shard", type=int, default=512)
    p.add_argument("--video-backend", default="torchcodec",
                   choices=["decord", "torchvision_av", "torchcodec"])
    p.add_argument("--store-input-ids", action="store_true",
                   help="Also store input_ids per example (useful for labeling)")
    p.add_argument("--compute-stats", action="store_true",
                   help="After extraction, compute α (P75 norm) and save stats.json")
    p.add_argument("--stats-max-positions", type=int, default=2_000_000,
                   help="Cap on positions used for percentile computation")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    mods = _imports()

    # Resolve embodiment tag (accepts string).
    embodiment_tag = mods["EmbodimentTag"].resolve(args.embodiment_tag)
    _preflight_processor_access(args.model_path)
    logger.info("Loading policy from %s on %s", args.model_path, args.device)
    policy = mods["Gr00tPolicy"](
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    logger.info("Model dtype: %s, device: %s", policy.model.dtype, policy.model.device)

    loader = mods["LeRobotEpisodeLoader"](
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_configs,
        video_backend=args.video_backend,
    )
    language_keys = list(policy.modality_configs["language"].modality_keys)

    # Build an index->task mapping so we can resolve task text for datasets
    # that only carry task_index in the parquet (LeRobot v2.1 default).
    task_index_to_text: dict[int, str] = {}
    tasks_jsonl = Path(args.dataset_path) / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        with tasks_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ti = obj.get("task_index")
                text = obj.get("task") or ""
                if ti is not None and text:
                    task_index_to_text[int(ti)] = str(text)

    manifest = mods["RunManifest"](
        schema_version=1,
        model_repo=str(args.model_path),
        layer_module_path=mods["TARGET_BACKBONE_FEATURES"].module_path,
        hidden_size=mods["BACKBONE_EMBEDDING_DIM"],
        activation_dtype="float32",  # CapturedActivation default
        embodiment_tag=embodiment_tag.value,
        extra={
            "dataset_path": str(args.dataset_path),
            "device": args.device,
            "step_stride": args.step_stride,
        },
    )

    out_root = Path(args.out_root)
    writer = mods["ActivationShardWriter"](
        out_root,
        manifest,
        max_examples_per_shard=args.max_examples_per_shard,
    )
    hook = mods["BackboneFeatureHook"](to_cpu=True, store_dtype=torch.float32)

    # Iterate trajectories.
    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(loader)))
    logger.info("Extracting over %d trajectories.", len(traj_ids))

    # Modality configs without action (we only need observations).
    modality_configs = deepcopy(policy.modality_configs)
    if "action" in modality_configs:
        modality_configs.pop("action")

    start = time.time()
    n_written = 0
    with mods["attach_hooks"](policy.model.backbone, hook):
        for traj_id in traj_ids:
            traj = loader[traj_id]
            n_steps = len(traj)
            limit = n_steps if args.steps_per_traj < 0 else min(args.steps_per_traj, n_steps)
            step_iter = range(0, limit, max(1, args.step_stride))
            logger.info("  traj %d: %d steps (taking %d)", traj_id, n_steps, len(list(step_iter)))
            for step_idx in range(0, limit, max(1, args.step_stride)):
                try:
                    observation = _prepare_step_obs(
                        traj, step_idx, modality_configs, embodiment_tag, language_keys, mods
                    )
                    backbone_inputs = _run_backbone_forward(policy, observation, mods)
                except Exception as e:
                    logger.warning("  traj %d step %d failed: %s", traj_id, step_idx, e)
                    continue

                captured = hook.last
                assert captured is not None, "Hook did not fire (backbone module mismatch?)"
                assert captured.batch_size == 1, (
                    f"Extraction assumes batch_size=1 per example; got {captured.batch_size}"
                )
                features = captured.features[0]
                attn = captured.attention_mask[0]
                img = captured.image_mask[0]

                input_ids = None
                if args.store_input_ids:
                    raw_ids = backbone_inputs.get("input_ids")
                    if raw_ids is not None:
                        input_ids = raw_ids[0].detach().cpu().to(torch.int64).contiguous()

                example_id = f"traj{traj_id:06d}_step{step_idx:06d}"
                task_text = traj["task"].iloc[step_idx] if "task" in traj.columns else None
                # task_text may be a list; flatten in that case.
                if isinstance(task_text, (list, tuple)) and task_text:
                    task_text = task_text[0]
                # LeRobot v2.1 typically stores only `task_index` in the parquet
                # and the index->text mapping lives in `meta/tasks.jsonl`. Fall
                # back to that lookup if the inline `task` column is missing.
                if not task_text and "task_index" in traj.columns:
                    ti = int(traj["task_index"].iloc[step_idx])
                    task_text = task_index_to_text.get(ti)
                writer.write(
                    example_id=example_id,
                    features=features,
                    attention_mask=attn,
                    image_mask=img,
                    input_ids=input_ids,
                    task_index=int(traj["task_index"].iloc[step_idx])
                        if "task_index" in traj.columns else None,
                    task_text=str(task_text) if task_text is not None else None,
                    episode_index=int(traj_id),
                    step_index=int(step_idx),
                    embodiment_tag=embodiment_tag.value,
                )
                n_written += 1
                if n_written % 32 == 0:
                    elapsed = time.time() - start
                    logger.info(
                        "  wrote %d examples (%.2fs, %.1f ex/s)",
                        n_written, elapsed, n_written / max(elapsed, 1e-6),
                    )

    writer.close()
    elapsed = time.time() - start
    logger.info("Extraction done: %d examples in %.1fs", n_written, elapsed)

    if args.compute_stats and n_written > 0:
        reader = mods["ActivationShardReader"](out_root)
        stats = mods["compute_stats"](reader, max_positions=args.stats_max_positions)
        mods["save_stats"](stats, out_root / "stats.json")
        logger.info(
            "α (P75 norm) = %.4f  [P50=%.3f P90=%.3f P99=%.3f mean=%.3f]  over %d positions",
            stats.p75_norm, stats.p50_norm, stats.p90_norm, stats.p99_norm,
            stats.mean_norm, stats.n_positions,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
