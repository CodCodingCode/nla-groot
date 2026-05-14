"""Pull individual frames out of a LeRobot v2.1 dataset for labeling.

Why a dedicated module: at extraction time we *don't* store frames in the
shard files (too big). At labeling time we need them to attach to the OpenAI
multimodal call. So given an extraction record's ``(episode_index,
step_index)`` plus the dataset root, we read the matching MP4 and decode just
the target frame.

Implementation uses PyAV (``av``) only — no decord/torchcodec required.  The
loader caches an open container per video so successive seeks within the same
episode are fast.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    fps: float
    video_keys: list[str]                  # short keys like "exterior_1_left"
    video_path_template: str               # e.g. "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    chunks_size: int
    video_key_to_original: dict[str, str]  # short -> dataset video-key (e.g. "exterior_1_left" -> "observation.images.exterior_1_left")
    episode_to_task: dict[int, str]        # episode_index -> task text (from meta/episodes.jsonl + tasks.jsonl)

    @classmethod
    def from_root(cls, root: str | Path) -> "DatasetInfo":
        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text())
        modality = json.loads((root / "meta" / "modality.json").read_text())
        video_block = modality.get("video", {})
        video_keys = list(video_block.keys())
        # ``original_key`` maps the short modality key (used by gr00t configs)
        # back to the on-disk video directory name (often prefixed with
        # ``observation.images.`` in LeRobot exports).
        key_to_original: dict[str, str] = {}
        for short, spec in video_block.items():
            key_to_original[short] = (spec or {}).get("original_key", short)

        episode_to_task = _load_episode_to_task(root)

        return cls(
            fps=float(info.get("fps", 15)),
            video_keys=video_keys,
            video_path_template=info["video_path"],
            chunks_size=int(info.get("chunks_size", 1000)),
            video_key_to_original=key_to_original,
            episode_to_task=episode_to_task,
        )


def _load_episode_to_task(root: Path) -> dict[int, str]:
    """Build ``{episode_index: task_text}`` from ``meta/episodes.jsonl``.

    LeRobot v2.1 stores per-step ``task_index`` in the parquet and the
    ``index -> text`` mapping in ``meta/tasks.jsonl``.  ``meta/episodes.jsonl``
    additionally lists the human-readable task strings per episode under
    ``tasks``, which is the easiest single source of truth.
    """
    ep_path = root / "meta" / "episodes.jsonl"
    if not ep_path.exists():
        return {}
    mapping: dict[int, str] = {}
    with ep_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ep_idx = obj.get("episode_index")
            tasks = obj.get("tasks") or []
            if ep_idx is None:
                continue
            if isinstance(tasks, list) and tasks:
                text = tasks[0]
            else:
                text = ""
            if text:
                mapping[int(ep_idx)] = str(text)
    return mapping


class EpisodeFrameLoader:
    """Random-frame access for a single LeRobot v2.1 episode.

    Caches the PyAV container so multiple step accesses don't reopen the file.
    Use one instance per (dataset_root, episode_index) tuple. Close with
    ``__exit__`` or ``close()``.

    Decoding strategy: PyAV doesn't support exact O(1) frame indexing for all
    codecs, so we seek to the nearest keyframe and decode forward until we
    reach the target frame.  This is fast for typical demo episodes (a few
    hundred frames at 5-15 fps).
    """

    def __init__(self, dataset_root: str | Path, episode_index: int) -> None:
        self.root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.info = DatasetInfo.from_root(self.root)
        self._containers: dict[str, av.container.InputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        self._last_frame_index: dict[str, int] = {}

    def _video_path(self, video_key: str) -> Path:
        ep_chunk = self.episode_index // self.info.chunks_size
        # Translate the short key (matching modality.json) to whatever
        # ``original_key`` says lives on disk; preserve the literal otherwise.
        on_disk_key = self.info.video_key_to_original.get(video_key, video_key)
        rel = self.info.video_path_template.format(
            episode_chunk=ep_chunk,
            video_key=on_disk_key,
            episode_index=self.episode_index,
        )
        return self.root / rel

    def _open(self, video_key: str) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
        if video_key in self._containers:
            return self._containers[video_key], self._streams[video_key]
        path = self._video_path(video_key)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing video for {video_key} at {path}. "
                f"Available keys in dataset: {self.info.video_keys}"
            )
        container = av.open(str(path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        self._containers[video_key] = container
        self._streams[video_key] = stream
        self._last_frame_index[video_key] = -1
        return container, stream

    def frame(self, video_key: str, frame_index: int) -> np.ndarray:
        """Return the RGB frame at ``frame_index`` (zero-based) as uint8 ndarray.

        Shape: ``(H, W, 3)``.
        """
        container, stream = self._open(video_key)
        # Convert frame index -> presentation timestamp (PTS) -> seek.
        time_base = stream.time_base
        if time_base is None or stream.average_rate is None:
            # Fall back to linear scan from start.
            target_pts = None
        else:
            target_seconds = frame_index / float(stream.average_rate)
            target_pts = int(target_seconds / time_base)

        # Seek to nearest keyframe at-or-before target.
        if target_pts is not None:
            container.seek(target_pts, any_frame=False, backward=True, stream=stream)
        else:
            container.seek(0)

        last_frame = None
        for i, frame in enumerate(container.decode(stream)):
            current_idx = int(frame.pts * time_base * float(stream.average_rate)) if (
                frame.pts is not None and time_base is not None and stream.average_rate is not None
            ) else i
            last_frame = frame
            if current_idx >= frame_index:
                break
        if last_frame is None:
            raise IndexError(
                f"Could not decode frame {frame_index} from {video_key} "
                f"(episode {self.episode_index})."
            )
        # ndarray in RGB uint8.
        return last_frame.to_ndarray(format="rgb24")

    def frames(self, video_keys: Iterable[str], frame_index: int) -> dict[str, np.ndarray]:
        """Return ``{video_key: rgb_array}`` for one frame_index across keys."""
        return {k: self.frame(k, frame_index) for k in video_keys}

    def close(self) -> None:
        for c in self._containers.values():
            c.close()
        self._containers.clear()
        self._streams.clear()
        self._last_frame_index.clear()

    def __enter__(self) -> "EpisodeFrameLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# IO helpers for the OpenAI image_url payload
# ---------------------------------------------------------------------------

def save_jpeg(frame: np.ndarray, path: str | Path, quality: int = 85) -> Path:
    """Save an RGB ndarray to JPEG; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path, format="JPEG", quality=quality)
    return path


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode an RGB ndarray to in-memory JPEG bytes (for base64 attaching)."""
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
