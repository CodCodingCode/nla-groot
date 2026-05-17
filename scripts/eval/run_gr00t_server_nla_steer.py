#!/usr/bin/env python
"""Launch a GR00T policy server with an NLA backbone steer applied per step.

Mirrors the minimal startup path of
``third_party/Isaac-GR00T/gr00t/eval/run_gr00t_server.py`` (same
``ServerConfig``-style ``tyro`` CLI, same ``PolicyServer`` wiring) and inserts
:class:`nla.steering.NlaSteerGr00tPolicy` between the ZMQ endpoint and the
``Gr00tPolicy`` whenever ``--ar-dir`` plus a steer prompt are provided. Without
those flags the server runs as the bare upstream policy, so this script is a
drop-in replacement for the vanilla launcher.

Run it from the GR00T ``.venv`` (Python 3.10 with ``isaac-gr00t`` installed),
using the same ``HF_TOKEN`` you normally need for ``nvidia/Cosmos-Reason2-2B``::

    PYTHONPATH=src python scripts/eval/run_gr00t_server_nla_steer.py \\
        --model-path     checkpoints/GR00T-N1.7-LIBERO/libero_goal \\
        --embodiment-tag LIBERO_PANDA \\
        --ar-dir         data/sft/libero_goal_pilot_v3/ar \\
        --steer-text-file my_steer_bullets.txt \\
        --placement image_patch \\
        --blend 1.0 \\
        --port 5555

Then point any Isaac-GR00T sim client (LIBERO, SimplerEnv, …) at the same host
and port; the server will transparently apply the steer on every ``get_action``.
See ``docs/evals/sim_steer_rollout.md`` for the two-terminal runbook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for the NLA-steered GR00T inference server."""

    model_path: str | None = None
    """Path or HF repo id of the GR00T checkpoint to serve."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Must match the checkpoint and sim."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Host address for the ZMQ server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """TCP port for the ZMQ server."""

    strict: bool = True
    """Whether to enforce strict input/output validation in the policy."""

    use_sim_policy_wrapper: bool = False
    """Wrap the policy with Gr00tSimPolicyWrapper (flat-key sim observations)."""

    ar_dir: str | None = None
    """SFT ``ar/`` checkpoint directory. If unset, the server runs without steering."""

    steer_text: str | None = None
    """Inline AR bullet prompt. Mutually exclusive with --steer-text-file."""

    steer_text_file: str | None = None
    """UTF-8 file with the AR bullet prompt (preferred for multi-line input)."""

    placement: Literal[
        "last_text", "image_patch", "anchor", "image_patch_all", "fixed"
    ] = "image_patch"
    """Where in the backbone token sequence to inject the steer vector."""

    blend: float = 1.0
    """Blend factor (1.0 = hard replace; 0.0 = no-op passthrough)."""

    fixed_token_index: int | None = None
    """Required when --placement=fixed."""

    image_patch_seed: int = 0
    """RNG seed for selecting a single image token when --placement=image_patch."""

    steer_off: bool = False
    """Build the wrapper but start with steering disabled (A/B passthrough)."""


def _load_steer_text(config: ServerConfig) -> str:
    if config.steer_text and config.steer_text_file:
        raise SystemExit("Use only one of --steer-text or --steer-text-file")
    if config.steer_text_file:
        return Path(config.steer_text_file).read_text()
    if config.steer_text:
        return config.steer_text
    raise SystemExit(
        "Provide --steer-text or --steer-text-file when --ar-dir is set "
        "(or omit --ar-dir to run without steering)."
    )


def main(config: ServerConfig) -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    from nla.extraction._compat import apply_all as apply_groot_compat

    apply_groot_compat()

    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting NLA-steered GR00T inference server...")
    print(f"  Embodiment tag: {embodiment_tag}")
    print(f"  Model path:     {config.model_path}")
    print(f"  Device:         {config.device}")
    print(f"  Host:           {config.host}")
    print(f"  Port:           {config.port}")

    if config.model_path is None:
        raise SystemExit("--model-path is required")
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")
    if config.placement == "fixed" and config.fixed_token_index is None:
        raise SystemExit("--fixed-token-index is required when --placement=fixed")

    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    policy.model.eval()

    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy, strict=config.strict)

    if config.ar_dir is not None:
        from nla.steering import (
            NlaSteerGr00tPolicy,
            SteerSpec,
            ar_text_to_backbone_vec,
        )
        from nla.training.checkpoint import load_ar_from_sft

        steer_text = _load_steer_text(config)
        print(f"  AR dir:         {config.ar_dir}")
        print(f"  Placement:      {config.placement}  blend={config.blend}")
        print(
            "  Steer text:     "
            f"{steer_text.strip().splitlines()[0][:80] if steer_text.strip() else '<empty>'}…"
        )

        ar = load_ar_from_sft(Path(config.ar_dir), device=config.device, freeze=True)
        steer_vec = ar_text_to_backbone_vec(ar, steer_text)
        spec = SteerSpec(
            placement=config.placement,
            blend=float(config.blend),
            fixed_token_index=config.fixed_token_index,
            image_patch_seed=int(config.image_patch_seed),
        )
        policy = NlaSteerGr00tPolicy(
            policy,
            steer_vec=steer_vec,
            spec=spec,
            enabled=not config.steer_off,
            strict=config.strict,
        )
        print(f"  Steering:       {'OFF (passthrough)' if config.steer_off else 'ON'}")
    else:
        print("  Steering:       disabled (no --ar-dir provided)")

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    print(f"\nServer ready — listening on {config.host}:{config.port}\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
