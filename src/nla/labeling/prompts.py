"""Warm-start labeling prompts for GR00T NLA.

Two prompt families:

1. ``build_position_prompt`` — *per-token-position* labeling, used as the SFT
   target for the AV.  This is the NLA paper's recipe adapted for a VLA: ask
   what features the model is tracking *at this token position* in order to
   choose the next action (or next token).  The labeler sees the camera
   frame(s), the text instruction, the decoded text portion of the context,
   robot state, and the position type (last_text / image_patch / anchor).

2. ``build_step_prompt`` — legacy *per-step* "scene description" labeling,
   useful as a fast/cheap baseline.  Kept for back-compat with earlier
   scaffolding; not used by the NLA pipeline.

Both prompts ask for the same fixed bullet style so the AV inherits a
predictable output format across SFT (per the NLA paper's observation that
"the AV inherits the format of warm-start data ... this style persists
through training").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Style spec (shared)
# ---------------------------------------------------------------------------

BULLET_CATEGORIES: tuple[str, ...] = (
    "scene",          # what's in the camera view, surfaces, lighting
    "target",         # the object/region the task is about
    "distractor",     # nearby objects that are NOT the target
    "spatial",        # relative positions, distances, frames of reference
    "plan",           # high-level sub-goal / next action class
    "motion",         # current and imminent end-effector motion direction
    "gripper",        # gripper state and grasp readiness
    "language",       # what the instruction tells the model to do
    "image_region",   # what region of the image the model attends to
)

_STYLE_CLAUSE = (
    "Output exactly 4-5 short bullets, each on its own line, in this fixed style:\n"
    '    "- <category>: <concrete content>."\n'
    f"  where <category> is drawn from: {', '.join(BULLET_CATEGORIES)}.\n"
    "- Be specific: reference concrete objects, colors, surfaces, relative positions.\n"
    "- Use declarative tone (no hedging like 'seems' or 'might').\n"
    "- No preamble, no conclusion, no non-bullet text."
)


# ---------------------------------------------------------------------------
# Per-position prompt (the NLA target)
# ---------------------------------------------------------------------------

PositionType = Literal["last_text", "image_patch", "anchor", "fallback"]


@dataclass
class PositionLabelInput:
    """A single per-token-position labeling request.

    Attributes
    ----------
    example_id:
        Stable id including the position, e.g. ``"traj000001_step000017@p042_image_patch"``.
    instruction:
        Natural-language task text (may be empty if the dataset lacked one).
    decoded_text_context:
        The text portion of the model's tokenized input, with image regions
        collapsed to ``<image: N patches>`` placeholders.  Truncated if very
        long.
    position_index:
        Token position the AV will see (0-indexed, in the original sequence).
    position_type:
        One of ``last_text``, ``image_patch``, ``anchor``, ``fallback``.
    sequence_length:
        T of the original sequence (used for "position 42 of 277").
    image_patch_meta:
        Optional ``(k_within_image, n_image_tokens)`` for image-patch positions
        so the labeler can be told "image patch 42 of 248".
    image_paths:
        Local paths to the camera frames at this step.  All attached to the
        OpenAI call; ordering preserved.
    state:
        Optional robot state vector (free-form list of floats, with a name
        hint in ``state_name`` for the prompt header).
    state_name:
        Human-readable name for the state vector, e.g.
        ``"x, y, z, roll, pitch, yaw, pad, gripper"``.
    episode_index, step_index:
        Provenance only; do not affect the prompt.
    """

    example_id: str
    instruction: str
    decoded_text_context: str
    position_index: int
    position_type: PositionType
    sequence_length: int
    image_paths: list[str]
    image_patch_meta: tuple[int, int] | None = None
    state: list[float] | None = None
    state_name: str | None = None
    episode_index: int | None = None
    step_index: int | None = None
    extra: dict = field(default_factory=dict)
    # SA2 hook: optional dataset-suite tag (e.g. ``"libero_spatial"``).  When
    # set, V4 builder appends the suite-specific addendum.  Left as the last
    # field so existing callers that pass positional args are unaffected.
    suite: str | None = None


# Known LIBERO suite names recognized by ``infer_suite_from_example_id``.  Kept
# as a module-level tuple so audit / tooling code can reuse the canonical list
# without re-deriving from prose.
_LIBERO_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_goal",
    "libero_object",
    "libero_10",
)


def infer_suite_from_example_id(
    example_id: str | None,
    *,
    extra: dict | None = None,
) -> str | None:
    """Best-effort suite inference for a position-labeling row.

    The eval-style ``example_id`` used by the judge prefixes the suite
    (``"libero_spatial::traj000017_step000014@p151_anchor"``).  Production
    labeling rows omit that prefix, so callers that already have the suite in
    row metadata can pass ``extra={"suite": ...}`` and we will honor that.

    Returns one of ``"libero_spatial" / "libero_goal" / "libero_object" /
    "libero_10"`` if recognizable, else ``None``.  Unknown / missing values
    are a silent no-op so SA5's pipeline wiring can call this on every row
    without guarding.
    """
    if isinstance(extra, dict):
        s = extra.get("suite")
        if isinstance(s, str) and s in _LIBERO_SUITES:
            return s
    if isinstance(example_id, str):
        for suite in _LIBERO_SUITES:
            if example_id.startswith(f"{suite}::"):
                return suite
    return None


# Rules that apply when the highlighted token is an IMAGE-PATCH token.
#
# We have observed (see ``docs/sft_plan/01_data_audit.md`` §3.2 "Confabulated
# image_region content") that labelers will guess a quadrant or screen
# location from the ``(k, n)`` patch index alone — e.g. "lower-left of the
# image patch" derived purely from k=89/256.  The labeler is NOT shown which
# patch is k; only the full camera frame(s).  Treating those guesses as gold
# trains the AV to confidently invent spatial layout.
#
# These rules forbid that pattern.  They are appended to ``_POSITION_SYSTEM``
# so they are inherited by ``_STRICT_POSITION_SYSTEM`` (and any future
# variant) without duplication; they only "kick in" when the user prompt
# marks the highlighted position as IMAGE-PATCH.
# Rules for IMAGE-PATCH labels.  We have observed (see
# ``docs/sft_plan/01_data_audit.md`` §3.2 "Confabulated image_region content")
# that labelers will guess a quadrant or screen location from the ``(k, n)``
# patch index alone — e.g. "lower-left of the image patch" derived purely
# from k=89/256.  The labeler is NOT shown which patch is k; only the full
# camera frame(s).  Treating those guesses as gold trains the AV to
# confidently invent spatial layout.  We also outright steer labelers away
# from the ``image_region`` category for these positions (the LIBERO pilot
# audit, May-2026, found 41% of image_patch rows still emitted
# ``image_region`` bullets despite the earlier guard, with measurable
# downstream noise).
_IMAGE_PATCH_RULES = (
    "\nRules for IMAGE-PATCH positions:\n"
    "- The patch index '(k of n)' is metadata about the model's token layout. "
    "Do NOT use it to guess screen quadrant, side, or pixel coordinates "
    "(no 'upper-left', 'lower-right', 'top quadrant', etc. inferred from k).\n"
    "- Spatial language is allowed only when it is directly visible in the "
    "attached camera frame(s) and refers to objects/regions in the frame — "
    "not to the patch grid.\n"
    "- Prefer the 'target', 'scene', or 'spatial' categories for the last "
    "bullet of an IMAGE-PATCH label.  Avoid the 'image_region' category for "
    "IMAGE-PATCH labels: it invites position-guessing from the patch index, "
    "which we have observed labelers do unreliably.\n"
    "- If no clear target object is visible at all, prefer "
    "'- target: none in this patch.' over inventing one.\n"
)

# Rules against anthropomorphic / decision-attributing wording.  The C-axis
# grader (gpt-5.1) consistently rejects bullets that ascribe internal
# cognitive state to the policy.  The LIBERO pilot audit (May-2026) found
# that the bulk of C-fails came from the ``last_text`` bullet emitting
# "instruction has been read", "goal committed", or "has been read and
# committed to" — phrasing the prompt itself was nudging via "what the model
# is committing to at that moment".  This clause forbids those patterns.
_FORBIDDEN_PHRASING = (
    "\nForbidden phrasings (any one of these makes the label unusable):\n"
    "- Do NOT ascribe internal cognitive state to the policy: avoid "
    "'committing to', 'committed to', 'decides', 'intends', 'wants', "
    "'hopes', 'believes', 'feels'. Describe what is represented at this "
    "position, not what the model 'wants' to do.\n"
    "- Do NOT write 'instruction has been read', 'goal committed', "
    "'has been read and committed to', or any close paraphrase. Describe "
    "the parsed task or the active plan phase in neutral, perceptual "
    "terms (e.g., 'plan: pick-and-place phase; reach over the bowl, then "
    "place on the plate.').\n"
)

_POSITION_SYSTEM = f"""You are an interpretability annotator for the GR00T N1.7 \
vision-language-action robot model.

You are shown one step of robot input: an instruction, robot state, the text \
tokens the model has processed, and one or more camera frames.  Your job is to \
predict 4-5 features that the model is internally tracking *at the highlighted \
token position*, in order to choose its next action (or generate its next token).

{_STYLE_CLAUSE}

Crucially, the LAST bullet must specifically describe what feature this token \
position carries forward into the action head — i.e., the perceptual, spatial, \
or plan-level content tied to this slot — not just the overall scene.  Examples \
of last-bullet content by position type:
- last_text: "plan: pick-and-place phase active; reach over the bowl, then place on the plate."
- image_patch: "target: bowl rim and gripper visible at this position; the specific patch within the frame is not localized."
- anchor: "plan: arm staged at the start of the reaching trajectory."
{_IMAGE_PATCH_RULES}{_FORBIDDEN_PHRASING}"""


def _format_state(state: list[float] | None, name: str | None) -> str:
    if not state:
        return ""
    body = ", ".join(f"{x:.3f}" for x in state)
    if name:
        return f"\nRobot state ({name}): [{body}]\n"
    return f"\nRobot state: [{body}]\n"


def _format_position_clause(inp: PositionLabelInput) -> str:
    pos_idx = inp.position_index
    seq_len = inp.sequence_length
    base = (
        f"\nThe model has processed the input up to and including token "
        f"position {pos_idx} (out of {seq_len}). "
    )
    if inp.position_type == "image_patch":
        if inp.image_patch_meta is not None:
            k, n = inp.image_patch_meta
            base += (
                f"This position is an IMAGE-PATCH token (image patch {k} of {n} "
                "across the attached camera frame(s))."
            )
        else:
            base += "This position is an IMAGE-PATCH token."
    elif inp.position_type == "last_text":
        base += "This position is the LAST TEXT token before the action head reads the context."
    elif inp.position_type == "anchor":
        base += "This position is the ANCHOR token at the very end of the prompt."
    else:
        base += "This position is a generic context token (fallback)."
    return base + "\n"


def build_position_prompt(inp: PositionLabelInput) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)``; images attached separately."""
    instruction = inp.instruction.strip() or "(no instruction provided)"
    user = (
        f'Task instruction: "{instruction}"\n'
        f"{_format_state(inp.state, inp.state_name)}"
        f"\nText portion of the model's tokenized input "
        "(image regions collapsed to placeholders):\n"
        f"<context>\n{inp.decoded_text_context}\n</context>\n"
        f"{_format_position_clause(inp)}"
        "\nProduce the 4-5 bullet description now."
    )
    return _POSITION_SYSTEM, user


_STRICT_EXTRA = (
    "\nAdditional rules (strict):\n"
    "- Both 'scene' and 'target' bullets are REQUIRED in every response. "
    "If no clear target object is visible at this position (especially for "
    "image_patch tokens that fall on background pixels), write exactly: "
    "'- target: none in this patch.'\n"
    "- Use ONLY these bullet categories: "
    f"{', '.join(BULLET_CATEGORIES)}. Do not invent new categories "
    "(no 'tool', 'object', 'secondary_target', etc.).\n"
)

_STRICT_POSITION_SYSTEM = _POSITION_SYSTEM + _STRICT_EXTRA


def build_strict_position_prompt(inp: PositionLabelInput) -> tuple[str, str]:
    """Like ``build_position_prompt`` but enforces required bullets and category set.

    Used for re-labeling rows that were missing ``scene:`` / ``target:`` bullets
    or that invented non-canonical categories. The user prompt is unchanged; the
    system prompt has two extra rules appended.
    """
    _, user = build_position_prompt(inp)
    return _STRICT_POSITION_SYSTEM, user


# ---------------------------------------------------------------------------
# V4 prompt — Phase-1 repair of the V3 LIBERO corpus
# ---------------------------------------------------------------------------
#
# Provenance: synthesized from five V3-audit findings (May-2026):
#   * Multimodal judge: libero_spatial B-pass ~73% (relations vs frame).
#   * Prompt regression: ~10k non-canonical ``gripper:`` / ``motion:`` headers
#     and motor imperatives in ``plan`` bullets.
#   * Diversity audit: prompt-scaffold leakage ("action head", "this patch
#     carries") and ``plan`` boilerplate.
#   * Bullet-structure audit: ``language`` bullet present on only ~20% of rows
#     and very high Jaccard between ``image_patch`` and ``last_text`` captions
#     (the labeler ignores the highlighted position).
#   * Hard-neg audit: ``last_text`` retrieval saturated; the model is solving
#     contrastive by caption template alone.
#
# V4 layers the following onto ``_POSITION_SYSTEM`` (so the image-patch and
# anthropomorphic-phrasing rules are inherited verbatim):
#
#   1. Strict required-bullets + canonical-category enforcement (like
#      ``_STRICT_EXTRA``) but with a *narrower* allowed category set that
#      explicitly drops ``gripper``, ``motion``, and ``image_region`` as
#      bullet headers.
#   2. ``_V4_EXTRA_RULES``: scaffold-leakage ban, motor-imperative ban, plan
#      phase enumeration, ``language:`` optionality.
#   3. ``_LAST_BULLET_BY_POSITION_TYPE``: per-position-type rule for what the
#      last bullet must contain (drives down ``image_patch``↔``last_text``
#      Jaccard).
#   4. Optional ``_V4_SUITE_ADDENDA[suite]``: per-suite addendum (SA2 fills
#      in ``libero_spatial``).
#
# Layout decisions (also relevant to SA2-SA5 extenders):
#   * Forbidden-phrase lists are module-level tuples so SA3's audit regex can
#     import them directly without re-deriving from prose.
#   * The position-type-conditional clause is built from a dict, NOT an
#     ``if/elif`` chain, so SA4 can override one ptype without forking the
#     function.
#   * V4 bullet categories are a strict subset of the legacy
#     ``BULLET_CATEGORIES`` tuple — V3 tests / audits keep using
#     ``BULLET_CATEGORIES`` and remain green.

V4_BULLET_CATEGORIES: tuple[str, ...] = (
    "scene",
    "target",
    "distractor",
    "spatial",
    "plan",
    "language",
)
"""Allowed bullet headers for V4 labels (narrower than ``BULLET_CATEGORIES``)."""

V4_FORBIDDEN_HEADERS: tuple[str, ...] = ("gripper", "motion", "image_region")
"""Legacy headers explicitly dropped in V4 (kept here for audit-side reuse)."""

V4_SCAFFOLD_FORBIDDEN_PHRASES: tuple[str, ...] = (
    "action head",
    "this patch carries",
    "token carries",
    "the patch carries",
    "carries the",
    "transformer",
    "embedding",
    "hidden state",
    "residual stream",
)
"""Substrings echoed from the system prompt that V3 labels regurgitated."""

V4_MOTOR_IMPERATIVE_PHRASES: tuple[str, ...] = (
    "grasp the",
    "reach toward",
    "reach over",
    "align the gripper",
    "lower the gripper",
    "raise the gripper",
    "move toward",
    "approach the",
    "carry it",
    "place it",
    "release the",
    "open the gripper",
    "close the gripper",
)
"""Second-person imperatives addressed at the robot, forbidden in V4."""

V4_PLAN_PHASES: tuple[str, ...] = (
    "approach",
    "pickup",
    "lift",
    "carry",
    "place",
    "release",
    "retract",
    "idle",
    "reorient",
    "align",
    "insert",
    "open-drawer",
    "close-drawer",
    "pour",
)
"""Allowed verbs for the ``plan:`` bullet's named phase."""


def _format_phrase_list(phrases: tuple[str, ...]) -> str:
    return ", ".join(f'"{p}"' for p in phrases)


_V4_STRICT_BLOCK = (
    "\nAdditional rules (V4 strict):\n"
    "- Both 'scene' and 'target' bullets are REQUIRED in every response. "
    "If no clear target object is visible at this position (especially for "
    "image_patch tokens that fall on background pixels), write exactly: "
    "'- target: none in this patch.'\n"
    "- Use ONLY these bullet categories: "
    f"{', '.join(V4_BULLET_CATEGORIES)}. Do not invent new categories "
    "(no 'tool', 'object', 'secondary_target', etc.).\n"
    "- The following V3 bullet headers are FORBIDDEN in V4 — do not emit them "
    f"under any circumstance: {', '.join(V4_FORBIDDEN_HEADERS)}. Gripper state "
    "and end-effector motion, when relevant, must be folded into the 'plan', "
    "'target', or 'spatial' bullets in descriptive third-person form (e.g., "
    "'plan: pickup; gripper closing on the bowl').\n"
    "- The 'language' bullet is OPTIONAL. It MAY be included for `last_text` "
    "positions (where binding the parsed instruction is informative); for "
    "`image_patch` and `anchor` positions it is OPTIONAL and may be omitted. "
    "A 4-bullet response without 'language' is canonical for image_patch and "
    "anchor and MUST NOT be treated as a conformance failure.\n"
)


_V4_EXTRA_RULES = (
    "\nV4 additional rules (do not violate any):\n"
    "\n"
    "Scaffold-leakage ban:\n"
    "- Do NOT echo any prompt-scaffolding vocabulary into the caption. The "
    f"following substrings are FORBIDDEN: {_format_phrase_list(V4_SCAFFOLD_FORBIDDEN_PHRASES)}. "
    "Describe what is in the frame and what the robot is doing; do NOT "
    "describe what the token carries into the action head, what the patch "
    "carries forward, or anything about the model's internals.\n"
    "\n"
    "Motor-imperative ban:\n"
    "- Do NOT address the robot in second-person imperative form. The "
    "following phrasings are FORBIDDEN as imperatives in any bullet: "
    f"{_format_phrase_list(V4_MOTOR_IMPERATIVE_PHRASES)}. Use descriptive "
    "third-person observation instead. Bad: '- plan: grasp the bowl.' "
    "Good: '- plan: pickup; gripper closing on the bowl.' Bad: "
    "'- plan: reach over the plate and place the bowl.' Good: "
    "'- plan: approach; gripper above the plate, bowl still held.'\n"
    "\n"
    "Plan-bullet diversity:\n"
    "- When you emit a 'plan' bullet, it MUST name a specific phase from this "
    f"allowed list: {', '.join(V4_PLAN_PHASES)}. The named phase MUST match "
    "the visible robot/scene state at this moment: do NOT write 'place' if "
    "the gripper is still above the table empty-handed; do NOT write "
    "'release' if the object is not yet over its target. Prefer the format "
    "'plan: <phase>; <one-sentence neutral description of what is visible>.'\n"
    "\n"
    "Rule V4-LEAK-1 — Position-type discipline (anti-cross-leak):\n"
    "- Do NOT write image_patch-style perceptual bullets ('visible in this "
    "frame: ...', 'in this frame: ...', '<object> upright on the tabletop "
    "next to the gripper') on last_text or anchor rows, and do NOT write "
    "last_text-style temporal-plan bullets ('over the next 3 timesteps: "
    "...', 'over the next ~3 timesteps: ...', 'before releasing ...', "
    "'before placing ...') on image_patch rows. The position_type clause "
    "at the bottom of the user prompt specifies which style applies to "
    "this row; obey it strictly. The same task instruction and the same "
    "frame must produce DIFFERENT last bullets for image_patch vs "
    "last_text vs anchor.\n"
)


_LAST_BULLET_BY_POSITION_TYPE: dict[str, str] = {
    "image_patch": (
        "Position-conditional last bullet (IMAGE-PATCH):\n"
        "- For IMAGE-PATCH positions, the LAST bullet MUST be a 'target:' or "
        "'scene:' bullet that (a) names a specific object or scene region "
        "currently visible in the attached camera frame, (b) Do NOT restate "
        "the task instruction, paraphrase it, or quote any of its content "
        "words; the last bullet is a perceptual description of THIS frame, "
        "not a rephrasing of the instruction, (c) does NOT use "
        "temporal/predictive phrasing such as 'is about to', 'will then', "
        "'next step', 'about to', 'before placing', 'over the next ... "
        "timesteps', or any allowed plan-phase verb (approach, pickup, "
        "lift, carry, place, release, retract, idle, reorient, align, "
        "insert, open-drawer, close-drawer, pour) as the main predicate, "
        "and (d) MUST use past- or present-tense perceptual phrasing of "
        "the form 'visible in this frame: <object> <observable state>'. "
        "Concrete example (canonical): "
        "'- target: black wine bottle upright on the wooden tabletop next "
        "to the gripper.' Another canonical example: "
        "'- scene: visible in this frame: wooden tabletop with a blue mat, "
        "a silver gripper held just above the bowl rim.' Do NOT write things "
        "like '- target: bowl that the robot will pick up next' or "
        "'- target: pickup phase: bowl about to be grasped.'\n"
    ),
    "last_text": (
        "Position-conditional last bullet (LAST-TEXT):\n"
        "- For LAST-TEXT positions, the LAST bullet MUST be a 'plan:' bullet "
        "that (a) names a specific phase from the allowed plan-phase list "
        f"({', '.join(V4_PLAN_PHASES)}), (b) includes an explicit temporal "
        "connector binding the parsed instruction to the upcoming motion — "
        "use one of 'over the next 3 timesteps', 'over the next ~3 "
        "timesteps', 'before releasing', 'before placing', 'before "
        "retracting', or 'until the gripper closes', (c) does NOT use "
        "pixel/region/patch vocabulary ('upper-left', 'lower-right', "
        "'patch', 'region', 'visible in this frame', 'this frame shows', "
        "'in the camera frame at this exact moment'), and (d) references "
        "the parsed instruction verbatim or near-verbatim (quote the object "
        "and target words from the task text). Concrete example "
        "(canonical): '- plan: pickup phase over the next 3 timesteps: "
        "gripper closes on the wine bottle, then lifts before placing on "
        "the rack.' Another canonical example: "
        "'- plan: place phase over the next ~3 timesteps: bowl held above "
        "the plate, gripper opens before releasing the bowl onto the plate.' "
        "Do NOT write things like '- plan: bowl on the wooden tabletop next "
        "to the gripper' (that is image_patch phrasing).\n"
    ),
    "anchor": (
        "Position-conditional last bullet (ANCHOR):\n"
        "- For ANCHOR positions, the LAST bullet MUST be a 'plan:' bullet "
        "describing the overall trajectory phase from the allowed plan-phase "
        f"list ({', '.join(V4_PLAN_PHASES)}) AND summarising the remaining "
        "steps of the trajectory (NOT just the immediate next motion, and "
        "NOT just the current frame). Use phrasings like 'overall "
        "trajectory: <phase>; remaining steps <short summary>' or "
        "'<phase> trajectory still in progress; remaining: <short "
        "summary>'. Do NOT describe just the next single step (that's "
        "last_text's job) and do NOT describe just the current frame "
        "(that's image_patch's job). Concrete example (canonical): "
        "'- plan: approach trajectory; arm staged above the table, "
        "remaining steps: reach over the bowl, close on the wine bottle, "
        "lift, and place on the rack.'\n"
    ),
    "fallback": (
        "Position-conditional last bullet (FALLBACK):\n"
        "- For FALLBACK (generic context) positions, the LAST bullet MUST be "
        "either a 'plan:' bullet (preferred, naming a phase from the allowed "
        f"list: {', '.join(V4_PLAN_PHASES)}) or a 'target:' bullet, "
        "describing whichever is more informative at this position. If "
        "'plan:' is chosen, follow the LAST-TEXT temporal-connector "
        "convention; if 'target:' is chosen, follow the IMAGE-PATCH "
        "perceptual convention.\n"
    ),
}


_V4_LIBERO_SPATIAL_ADDENDUM = (
    "\nV4 LIBERO-SPATIAL addendum (this suite has elevated spatial-hallucination "
    "risk; obey every rule):\n"
    "\n"
    "Rule SP-1 — In-frame verification of spatial relations:\n"
    "- Spatial relations (\"left of\", \"right of\", \"behind\", \"in front of\", "
    "\"next to\", \"above\", \"below\", \"between\") may ONLY be stated when "
    "BOTH the anchor and reference object are visible in the attached camera "
    "frame at this timestep. If either side of the relation is not visible, "
    "OMIT the relation entirely. Do NOT fall back to the task instruction or "
    "prior knowledge of typical LIBERO scenes.\n"
    "\n"
    "Rule SP-2 — Frame-of-reference must be explicit:\n"
    "- Spatial language must name the frame of reference. Prefer phrasings "
    "like \"in the camera frame: <X> is left of <Y>\" or \"from the robot's "
    "POV: <X> sits behind <Y>\". Bare \"the X is left of the Y\" is FORBIDDEN "
    "because it is frame-ambiguous (camera left vs robot left vs world left).\n"
    "\n"
    "Rule SP-3 — No invented relations between commonly co-located objects:\n"
    "- The labeler has been observed inventing relations between LIBERO objects "
    "that are commonly co-located in the suite but not actually visible "
    "together in this particular frame. Known confabulation pairs include: "
    "bowl <-> plate, mug <-> shelf, cube <-> tray, wine bottle <-> rack. "
    "If both members of such a pair are not BOTH visible in the attached "
    "frame, say nothing about the relation — do NOT infer it from the "
    "instruction or from suite priors.\n"
    "\n"
    "Rule SP-4 — Spatial bullet must include a visually verifiable landmark:\n"
    "- For the 'spatial:' bullet specifically, include at least one concrete, "
    "visually verifiable landmark — e.g. \"on the wooden tabletop\", \"near "
    "the silver gripper\", \"against the dark cabinet wall\", \"beside the "
    "blue mat\". Generic positional language without a visible landmark (\"to "
    "the side\", \"in the middle\", \"near the edge\") is FORBIDDEN.\n"
    "\n"
    "Rule SP-5 — Occlusion must be named, not invented:\n"
    "- If the target or a referenced object is partially hidden in this frame "
    "(by the gripper, by another object, by the cabinet edge, by frame "
    "cropping, etc.), the 'spatial:' bullet MUST explicitly name the "
    "occlusion (e.g. \"the bowl is partially occluded by the gripper from the "
    "camera frame\"). Do NOT describe the hidden side, the hidden contents, "
    "or relations involving the hidden portion.\n"
    "\n"
    "Rule SP-6 — Do NOT anchor object identity on the instruction text:\n"
    "- The task instruction often names an object with a specific color or "
    "modifier (e.g. \"pick up the BLACK bowl\", \"the WHITE plate\"). The "
    "instruction is a goal description, not ground truth about the current "
    "frame. You MUST visually verify every object color, material, and "
    "identity from the attached camera frame(s) BEFORE writing it in any "
    "bullet. If the instruction says \"black bowl\" but the bowls visible in "
    "the frame are metallic / gray / silver / unpainted, write the bowls' "
    "ACTUAL visible color (e.g. \"metallic gray bowl\", \"silver bowl\") and "
    "do NOT call them black. The same applies to plate color (\"white plate\" "
    "vs visible red-rimmed plate), drawer state (\"top drawer\" vs visible "
    "closed cabinet), etc. It is acceptable to write the 'language:' bullet "
    "as the instruction text verbatim — but every other bullet must describe "
    "what is VISIBLE, not what the instruction claims.\n"
    "\n"
    "Rule SP-7 — Object color / material must come from the pixels:\n"
    "- Every color or material modifier (\"black\", \"white\", \"metallic\", "
    "\"silver\", \"gray\", \"red-rimmed\", \"wooden\", \"plastic\", "
    "\"patterned\") attached to an object name in 'scene', 'target', "
    "'distractor', or 'spatial' bullets MUST be the color/material you "
    "actually see in the attached frame at this timestep. If you are not "
    "sure, OMIT the modifier (write \"bowl\" instead of guessing "
    "\"black bowl\") — an unmodified noun is safer than an invented color.\n"
)
"""Per-suite system-prompt block for libero_spatial. Built as a module-level
constant so audits / tests can import it without re-deriving from prose."""


_V4_SUITE_ADDENDA: dict[str, str] = {
    "libero_spatial": _V4_LIBERO_SPATIAL_ADDENDUM,
    # Other suites (libero_goal, libero_object, libero_10) are healthy at V3
    # (B-pass >= 91% per Agent-1 audit); leave them as no-ops for V4.
}
"""Per-suite system-prompt addenda for V4. Keys are suite names (e.g.
``"libero_spatial"``, ``"libero_goal"``); values are the extra block of
prose to append.  Missing keys are treated as a no-op."""


def _v4_last_bullet_clause(position_type: str) -> str:
    return _LAST_BULLET_BY_POSITION_TYPE.get(
        position_type, _LAST_BULLET_BY_POSITION_TYPE["fallback"]
    )


_V4_POSITION_SYSTEM = (
    _POSITION_SYSTEM
    + _V4_STRICT_BLOCK
    + _V4_EXTRA_RULES
)
"""Base V4 system prompt (position-type-agnostic).

Position-type-conditional last-bullet rules and any suite-specific addendum
are appended by :func:`build_v4_position_prompt` because they depend on the
input."""


def build_v4_position_prompt(
    inp: PositionLabelInput,
    suite: str | None = None,
) -> tuple[str, str]:
    """V4 per-position labeling prompt.

    Compared to :func:`build_strict_position_prompt`, V4 additionally:

    * Drops ``gripper``, ``motion``, and ``image_region`` from the allowed
      bullet headers (folded into ``plan`` / ``target`` / ``spatial``).
    * Bans prompt-scaffold echoes (``"action head"``, ``"this patch carries"``,
      ``"transformer"``, etc.).
    * Bans second-person motor imperatives addressed at the robot
      (``"grasp the"``, ``"align the gripper"``, ``"reach toward"``, etc.).
    * Requires the ``plan`` bullet, when present, to name a phase from a
      fixed enumerated list (``approach``, ``pickup``, ``lift``, ...) and to
      match the visible state.
    * Switches the LAST-bullet requirement based on the position type
      (``image_patch`` -> perceptual ``target`` / ``scene``; ``last_text``
      and ``anchor`` -> ``plan``).
    * Makes the ``language`` bullet OPTIONAL for ``image_patch`` and
      ``anchor`` positions (4-bullet image_patch responses are canonical).

    Parameters
    ----------
    inp:
        The same :class:`PositionLabelInput` accepted by every other prompt
        builder.  The ``position_type`` field selects the last-bullet rule.
    suite:
        Optional dataset-suite hook (e.g. ``"libero_spatial"``).  When
        provided and present in :data:`_V4_SUITE_ADDENDA`, the registered
        addendum block is appended to the system prompt.  Unknown / missing
        suites are a no-op (no error).  When ``suite`` is ``None`` (the
        default), we attempt to auto-infer it from
        ``inp.example_id`` / ``inp.extra["suite"]`` /  ``inp.suite`` via
        :func:`infer_suite_from_example_id`; explicit ``suite=...`` from the
        caller always wins.
    """
    _, user = build_position_prompt(inp)
    if suite is None:
        suite = inp.suite or infer_suite_from_example_id(
            inp.example_id, extra=inp.extra,
        )
    parts = [
        _V4_POSITION_SYSTEM,
        "\n" + _v4_last_bullet_clause(inp.position_type),
    ]
    if suite:
        addendum = _V4_SUITE_ADDENDA.get(suite, "")
        if addendum:
            parts.append("\n" + addendum)
    return "".join(parts), user


# ---------------------------------------------------------------------------
# Per-step prompt (legacy / scene-level baseline, kept for completeness)
# ---------------------------------------------------------------------------

@dataclass
class LabelInput:
    """Legacy per-episode-step labeling input (one label per step, not per token)."""

    example_id: str
    instruction: str
    image_path: str
    state: list[float] | None = None
    state_name: str | None = None
    episode_id: str | None = None
    timestep: int | None = None


_STEP_SYSTEM = f"""You are an interpretability annotator for the GR00T N1.7 \
vision-language-action robot model.  For each scene, produce a short \
structured description of what features the model is likely tracking in its \
internal representation at this step.

{_STYLE_CLAUSE}
"""


def build_step_prompt(inp: LabelInput) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for per-step labeling."""
    instruction = inp.instruction.strip() or "(no instruction provided)"
    user = (
        f'Task instruction: "{instruction}"\n'
        f"{_format_state(inp.state, inp.state_name)}"
        "\nProduce the 4-5 bullet description now."
    )
    return _STEP_SYSTEM, user


# ---------------------------------------------------------------------------
# Back-compat shim (some earlier scaffolding imported ``build_label_prompt``)
# ---------------------------------------------------------------------------

def build_label_prompt(inp) -> tuple[str, str]:
    """Dispatch by input type so old call sites keep working."""
    if isinstance(inp, PositionLabelInput):
        return build_position_prompt(inp)
    return build_step_prompt(inp)
