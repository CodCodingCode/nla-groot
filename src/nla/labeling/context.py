"""Build per-position labeling inputs from an extraction dump.

Pipeline::

    ActivationShardReader  +  dataset path  +  Qwen3-VL tokenizer
      |
      v
    iterate examples -> sample one position per example
      |
      v
    build PositionLabelInput  (instruction, frames, decoded text, position)
      |
      v
    OpenAI client -> bullet description

This module is the bridge.  It does not call OpenAI itself (that's
``openai_client``).  It also does not load GR00T's full processor — only the
Qwen3-VL tokenizer, which is small and fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Iterable

import numpy as np
import torch

from nla.extraction.sampler import (
    PositionType as _SamplerPositionType,
    SampledPosition,
    _anchor_index,
    _last_text_index,
    sample_position,
    sample_positions,
)
from nla.extraction.storage import ActivationShardReader, ExampleRecord
from nla.labeling.frames import EpisodeFrameLoader, save_jpeg
from nla.labeling.prompts import PositionLabelInput, PositionType

logger = logging.getLogger(__name__)


# Maximum text-context length (in *decoded characters*) we pass to the labeler.
# Keeps token cost bounded and is well above what's needed for short
# instructions and a single image's worth of vision tokens.
DEFAULT_CONTEXT_CHAR_BUDGET = 2_000


# ---------------------------------------------------------------------------
# Tokenizer loading (cached)
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, object] = {}

# We default to the public Qwen3-VL-2B-Instruct because nvidia/Cosmos-Reason2-2B
# is gated (NVIDIA-approved access only) and Cosmos was fine-tuned *from*
# Qwen3-VL-2B-Instruct — the tokenizer vocabulary and special tokens are
# identical for the text channel.  Pass an explicit ``tokenizer_repo`` to
# override if you have Cosmos access.
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-VL-2B-Instruct"


def load_qwen3_vl_tokenizer(model_name: str = DEFAULT_TOKENIZER_REPO):
    """Load (and cache) the Qwen3-VL tokenizer used by GR00T's backbone.

    Falls back from a gated repo to the public base if the gated download
    fails with an access error.
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except (OSError, Exception) as e:
        msg = str(e)
        if "gated" in msg.lower() or "authorized list" in msg.lower() or "401" in msg or "403" in msg:
            if model_name != DEFAULT_TOKENIZER_REPO:
                logger.warning(
                    "Could not load gated tokenizer %s (%s); falling back to %s.",
                    model_name, type(e).__name__, DEFAULT_TOKENIZER_REPO,
                )
                tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_REPO)
            else:
                raise
        else:
            raise
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def _image_token_id(tokenizer) -> int | None:
    """Best-effort lookup of the image-token id used by Qwen3-VL."""
    for name in ("<|vision_pad|>", "<|image_pad|>", "<|image|>"):
        try:
            ids = tokenizer.encode(name, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        except Exception:
            continue
    # Fallback: try the convert_tokens_to_ids path.
    for name in ("<|vision_pad|>", "<|image_pad|>", "<|image|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", -1):
                return int(tid)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Text-context rendering
# ---------------------------------------------------------------------------

def decode_text_context(
    input_ids: torch.Tensor,
    image_mask: torch.Tensor,
    tokenizer,
    *,
    char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
) -> str:
    """Render input_ids as readable text with image regions collapsed.

    Contiguous runs of image tokens become ``<image: N patches>`` placeholders.
    Pad/special tokens that decode to whitespace are kept as-is.  If the
    rendered text exceeds ``char_budget``, we keep a head and tail and elide
    the middle.
    """
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    mask = image_mask.tolist() if isinstance(image_mask, torch.Tensor) else list(image_mask)
    assert len(ids) == len(mask), "input_ids and image_mask must align"

    out_parts: list[str] = []
    i = 0
    n = len(ids)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out_parts.append(f"<image: {j - i} patches>")
            i = j
        else:
            j = i
            while j < n and not mask[j]:
                j += 1
            text_chunk = tokenizer.decode(ids[i:j], skip_special_tokens=False)
            out_parts.append(text_chunk)
            i = j

    rendered = "".join(out_parts)
    if len(rendered) <= char_budget:
        return rendered
    head_n = char_budget // 2
    tail_n = char_budget - head_n - 32
    return rendered[:head_n] + "\n...[elided]...\n" + rendered[-tail_n:]


# ---------------------------------------------------------------------------
# Position metadata for image-patch positions
# ---------------------------------------------------------------------------

def image_patch_meta(
    image_mask: torch.Tensor, position_index: int
) -> tuple[int, int] | None:
    """If ``position_index`` is an image-patch token, return ``(k, n)``.

    k = 0-indexed image-patch number within the full image-token stretch.
    n = total number of image-patch tokens in this example.
    """
    mask = image_mask.bool()
    if not mask[position_index]:
        return None
    n = int(mask.sum().item())
    k = int(mask[:position_index].sum().item())
    return k, n


# ---------------------------------------------------------------------------
# Frame-loader cache keyed by (dataset_root, episode_index)
# ---------------------------------------------------------------------------

class FrameLoaderPool:
    """Reuses ``EpisodeFrameLoader`` across calls for the same episode.

    Keeps at most ``max_open`` containers open; LRU-evicts the rest.
    """

    def __init__(self, max_open: int = 4) -> None:
        self.max_open = int(max_open)
        self._cache: dict[tuple[str, int], EpisodeFrameLoader] = {}
        self._order: list[tuple[str, int]] = []

    def get(self, dataset_root: str | Path, episode_index: int) -> EpisodeFrameLoader:
        key = (str(Path(dataset_root).resolve()), int(episode_index))
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        loader = EpisodeFrameLoader(dataset_root, episode_index)
        self._cache[key] = loader
        self._order.append(key)
        if len(self._order) > self.max_open:
            oldest = self._order.pop(0)
            self._cache.pop(oldest).close()
        return loader

    def close_all(self) -> None:
        for k in list(self._cache.keys()):
            self._cache.pop(k).close()
        self._order.clear()


# ---------------------------------------------------------------------------
# Position-input builder
# ---------------------------------------------------------------------------

@dataclass
class SampledExample:
    """One sampled (example, position) pair before frames are loaded."""

    record: ExampleRecord
    position_index: int
    position_type: PositionType
    decoded_text_context: str
    image_patch_meta: tuple[int, int] | None


def sample_one_position_per_example(
    reader: ActivationShardReader,
    tokenizer,
    *,
    seed: int = 0,
    require_input_ids: bool = True,
    record_filter=None,
) -> Iterator[SampledExample]:
    """Backwards-compatible: one position per example."""
    yield from sample_positions_per_example(
        reader, tokenizer,
        n_per_example=1,
        seed=seed,
        require_input_ids=require_input_ids,
        record_filter=record_filter,
    )


def _draw_positions_for_example(
    attention_mask: torch.Tensor,
    image_mask: torch.Tensor,
    n: int,
    *,
    rng: np.random.Generator,
    guarantee_strata: bool,
) -> list[SampledPosition]:
    """Pick ``n`` distinct positions for one example.

    Default behavior mirrors ``sample_positions``: every slot is drawn from
    ``POSITION_MIX``.  When ``guarantee_strata`` is set (and ``n >= 2``) the
    first slots are reserved for ``last_text`` and ``anchor`` whenever those
    indices exist, and the remainder are filled from ``POSITION_MIX``
    excluding indices already chosen.  This avoids the
    ``image_patch``-dominated mix that ``sample_positions`` produces when
    ``n=4`` against sequences with ~256 image-patch tokens but only one
    ``last_text`` and one ``anchor`` (see ``docs/sft_plan/01_data_audit.md``).
    """
    if not guarantee_strata or n < 2:
        return sample_positions(attention_mask, image_mask, n=n, rng=rng)

    chosen: list[SampledPosition] = []
    used: set[int] = set()

    last_idx = _last_text_index(attention_mask, image_mask)
    if last_idx is not None and last_idx not in used:
        chosen.append(SampledPosition(last_idx, _SamplerPositionType.LAST_TEXT))
        used.add(last_idx)

    anchor_idx = _anchor_index(attention_mask)
    if anchor_idx is not None and anchor_idx not in used:
        chosen.append(SampledPosition(anchor_idx, _SamplerPositionType.ANCHOR))
        used.add(anchor_idx)

    remaining = n - len(chosen)
    if remaining <= 0:
        return chosen[:n]

    max_tries = max(8, 2 * remaining)
    for _ in range(remaining):
        for _attempt in range(max_tries):
            sp = sample_position(attention_mask, image_mask, rng=rng)
            if sp.index not in used:
                chosen.append(sp)
                used.add(sp.index)
                break
        else:
            sp = sample_position(attention_mask, image_mask, rng=rng)
            chosen.append(sp)
            used.add(sp.index)
    return chosen


def sample_positions_per_example(
    reader: ActivationShardReader,
    tokenizer,
    *,
    n_per_example: int = 1,
    seed: int = 0,
    require_input_ids: bool = True,
    record_filter=None,
    guarantee_strata: bool = False,
) -> Iterator[SampledExample]:
    """Iterate (example, sampled-position) pairs; ``n_per_example`` per example.

    Each example contributes up to ``n_per_example`` SampledExample rows, with
    distinct positions drawn from ``POSITION_MIX`` (no replacement within the
    example).  Set ``n_per_example=1`` to recover the original one-per-example
    behavior.

    When ``guarantee_strata=True`` and ``n_per_example >= 2``, the sampler
    always allocates one slot to ``last_text`` and one to ``anchor`` (when
    those indices exist for the example), then fills the remaining slots with
    the usual ``POSITION_MIX`` draw.  Use this when relabeling or running new
    label campaigns that want a more even strata distribution than the natural
    ~75% ``image_patch`` mix.
    """
    rng = np.random.default_rng(seed)
    for item in reader.iter_examples(record_filter=record_filter):
        rec: ExampleRecord = item["_record"]
        attn = item["attention_mask"]
        img = item["image_mask"]
        ids = item.get("input_ids")
        if require_input_ids and ids is None:
            logger.warning(
                "Example %s has no input_ids; skipping (re-run extraction with "
                "--store-input-ids).", rec.example_id,
            )
            continue
        text_ctx = (
            decode_text_context(ids, img, tokenizer)
            if ids is not None
            else "(input_ids unavailable; text context not rendered)"
        )
        sps = _draw_positions_for_example(
            attn, img, n_per_example, rng=rng, guarantee_strata=guarantee_strata,
        )
        for sp in sps:
            meta = image_patch_meta(img, sp.index)
            yield SampledExample(
                record=rec,
                position_index=sp.index,
                position_type=sp.type.value,  # type: ignore[arg-type]
                decoded_text_context=text_ctx,
                image_patch_meta=meta,
            )


def build_position_inputs(
    sampled: Iterable[SampledExample],
    *,
    dataset_root: str | Path,
    frame_cache_dir: str | Path,
    video_keys: list[str] | None = None,
    state_name: str | None = None,
    pool: FrameLoaderPool | None = None,
) -> Iterator[PositionLabelInput]:
    """Convert sampled positions into PositionLabelInput objects with images.

    Frames are extracted from the LeRobot dataset and saved as JPEGs under
    ``frame_cache_dir``; their paths are attached to the returned input.
    """
    dataset_root = Path(dataset_root)
    frame_cache_dir = Path(frame_cache_dir)
    pool = pool or FrameLoaderPool(max_open=4)

    from nla.labeling.frames import DatasetInfo
    di = DatasetInfo.from_root(dataset_root)
    if video_keys is None:
        video_keys = di.video_keys

    for s in sampled:
        rec = s.record
        if rec.episode_index is None or rec.step_index is None:
            logger.warning(
                "Example %s missing episode/step index; cannot fetch frames.",
                rec.example_id,
            )
            continue
        loader = pool.get(dataset_root, rec.episode_index)

        image_paths: list[str] = []
        for vk in video_keys:
            try:
                frame = loader.frame(vk, rec.step_index)
            except (FileNotFoundError, IndexError) as e:
                logger.warning(
                    "Skipping %s: could not load %s frame %d (%s)",
                    rec.example_id, vk, rec.step_index, e,
                )
                image_paths = []
                break
            out = frame_cache_dir / f"{rec.example_id}__{vk}.jpg"
            save_jpeg(frame, out)
            image_paths.append(str(out))
        if not image_paths:
            continue

        # Prefer the task text stored with the activation; fall back to the
        # dataset's ``meta/episodes.jsonl`` mapping if extraction couldn't
        # resolve it (e.g. LeRobot v2.1 stores it by task_index only).
        instruction = rec.task_text or ""
        if not instruction and rec.episode_index is not None:
            instruction = di.episode_to_task.get(int(rec.episode_index), "")

        yield PositionLabelInput(
            example_id=f"{rec.example_id}@p{s.position_index:03d}_{s.position_type}",
            instruction=instruction,
            decoded_text_context=s.decoded_text_context,
            position_index=s.position_index,
            position_type=s.position_type,  # type: ignore[arg-type]
            sequence_length=rec.seq_len,
            image_paths=image_paths,
            image_patch_meta=s.image_patch_meta,
            state=None,
            state_name=state_name,
            episode_index=rec.episode_index,
            step_index=rec.step_index,
            extra={"source_example_id": rec.example_id},
        )
