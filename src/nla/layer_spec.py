"""Canonical architecture constants for GR00T N1.7.

These were verified by reading the published checkpoint at
nvidia/GR00T-N1.7-3B and the source at Isaac-GR00T@3df8b38:
  - config.json of the HF checkpoint (authoritative)
  - gr00t/model/modules/qwen3_backbone.py (Qwen3Backbone.forward)
  - gr00t/model/gr00t_n1d7/gr00t_n1d7.py (Gr00tN1d7ActionHead.process_backbone_output)

IMPORTANT: the published checkpoint differs from the source-code defaults.
  Source default            | Published checkpoint
  --------------------------+---------------------
  select_layer = 12         | select_layer = 16
  diffusion num_layers = 16 | diffusion num_layers = 32
The plan's "first 12 layers" framing was based on the source default;
the actual posttrained checkpoint uses the first 16. This affects layer
indexing for hooks and the layer-sweep choices.

GR00T physically truncates Cosmos-Reason2-2B (Qwen3-VL) to the first
`SELECT_LAYER` decoder layers via:
    while len(self.model.language_model.layers) > select_layer:
        self.model.language_model.layers.pop(-1)
so the "final layer of System 2" is layer (SELECT_LAYER - 1) in 0-indexed terms,
i.e. the 16th layer counting from 1.

The hook target `backbone_features` is the output of that last surviving layer,
BEFORE `vlln` (LayerNorm) and BEFORE `vl_self_attention`. It is the cleanest
representation of "what the VLM has committed to" before any action-head specific
post-processing. The post-vlln tensor `vl_embeds` is an ablation alternative.
"""

from __future__ import annotations

from dataclasses import dataclass


GROOT_HF_REPO = "nvidia/GR00T-N1.7-3B"
COSMOS_HF_REPO = "nvidia/Cosmos-Reason2-2B"     # gated; requires HF license accept
QWEN3_VL_BASE_REPO = "Qwen/Qwen3-VL-2B-Instruct"  # ungated base of Cosmos-Reason2

# Bridge-specific posttrained variant (used for Bridge experiments). If absent,
# we fall back to the base 3B and load Bridge weights via finetune config.
GROOT_BRIDGE_HF_REPO = "nvidia/GR00T-N1.7-Bridge"

# Authoritative values from the published checkpoint's config.json.
SELECT_LAYER: int = 16                  # last surviving Qwen3-VL decoder layer
BACKBONE_EMBEDDING_DIM: int = 2048      # hidden_size of Cosmos-Reason2-2B
DIT_NUM_LAYERS: int = 32                # AlternateVLDiT layers (checkpoint)
DIT_HIDDEN_SIZE: int = 1024
ACTION_HORIZON: int = 40
MAX_ACTION_DIM: int = 132

# Layer sweep choices for the warm-start pilot (per plan §5.3).
# Plan suggested (8, 10, 12); we shift to (8, 12, 16) to match the actual
# 16-layer truncation in the checkpoint - "middle" becomes layer 8, "late-middle"
# layer 12, "final" layer 16. Caller may override at extraction time.
LAYER_SWEEP: tuple[int, ...] = (8, 12, 16)

# Pooling decision: per-token training (not mean-pooled).
# Probabilities for the three token-position types used in §5.2 of the plan.
POSITION_MIX = {
    "last_text": 0.40,   # last non-image, non-pad text token
    "image_patch": 0.40, # uniform random over image-token positions
    "anchor": 0.20,      # designated query/EOS token
}


@dataclass(frozen=True)
class HookTarget:
    """One choice of where to hook in GR00T's forward pass."""

    name: str                # human-readable
    module_path: str         # dotted path from the loaded Gr00tN1d7 module
    pre_vlln: bool           # True if we hook before LayerNorm/SelfAttn
    notes: str = ""

    @property
    def dim(self) -> int:
        if "dit" in self.module_path:
            return DIT_HIDDEN_SIZE
        return BACKBONE_EMBEDDING_DIM


# Primary target: backbone_features (pre-vlln). This is the canonical NLA hook.
TARGET_BACKBONE_FEATURES = HookTarget(
    name="backbone_features_pre_vlln",
    module_path="backbone.model.language_model.layers",  # last index in this ModuleList
    pre_vlln=True,
    notes="Output of Qwen3-VL decoder layer (SELECT_LAYER-1), shape [B,T,2048]. "
          "Equivalent to `outputs.hidden_states[-1]` returned by Qwen3Backbone.forward.",
)

# Ablation target: post-vlln (i.e. what the DiT actually cross-attends to).
TARGET_VL_EMBEDS = HookTarget(
    name="vl_embeds_post_vlln",
    module_path="action_head.vlln",
    pre_vlln=False,
    notes="backbone_features after vlln + (optional) vl_self_attention.",
)

# DiT-pathway ablation (Plan §7.7): middle DiT block, encoder-stream
# (the stream that sees the noisy actions, not the VL stream).
TARGET_DIT_MID = HookTarget(
    name="dit_mid_block",
    module_path="action_head.model.transformer_blocks",  # index ~ N//2
    pre_vlln=False,
    notes="DiT layer 8 of 16. Encodes motor program rather than goal semantics.",
)


def describe() -> str:
    return (
        f"GR00T N1.7 hook spec\n"
        f"  HF repo:        {GROOT_HF_REPO}\n"
        f"  VLM backbone:   {COSMOS_HF_REPO}  (Qwen3-VL architecture)\n"
        f"  select_layer:   {SELECT_LAYER}   -> first {SELECT_LAYER} decoder layers kept\n"
        f"  backbone_dim:   {BACKBONE_EMBEDDING_DIM}\n"
        f"  DiT layers:     {DIT_NUM_LAYERS}, hidden {DIT_HIDDEN_SIZE}\n"
        f"  action horizon: {ACTION_HORIZON}, action dim: {MAX_ACTION_DIM}\n"
        f"  primary hook:   {TARGET_BACKBONE_FEATURES.module_path}[-1]  (output, pre-vlln)\n"
        f"  layer sweep:    {LAYER_SWEEP}\n"
        f"  position mix:   {POSITION_MIX}\n"
    )


if __name__ == "__main__":
    print(describe())
