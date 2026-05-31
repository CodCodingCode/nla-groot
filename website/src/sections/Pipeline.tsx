export default function Pipeline() {
  return (
    <section id="pipeline">
      <h2>Pipeline: extract, label, train, eval, steer</h2>

      <p>
        Every component runs from the open-source repo against a fresh
        GR00T-N1.7 checkpoint and a LIBERO-style suite. No closed teacher
        model is required for evaluation; gold captions are used only for
        SFT training.
      </p>

      <pre>
        <code>
{`GR00T-N1.7 forward hook (layer-16 backbone hidden state)
    → per-token activations + attention/image masks per trajectory
    → multimodal teacher labels (5-6 bullet intent-conditioned captions)
    → SFT: AV(h, intent → caption) + AR(caption → ĥ)
        - K=128 image-patch slots + 1 last_text slot, packed per row
        - intent-conditioned multi-slot prompt
        - decomposed reconstruction loss (cosine + log-magnitude)
        - action-consistency auxiliary (frozen policy forward every 2 steps)
        - best-checkpoint snapshot tracked by closed_greedy/cosine
    → AR(text) injection at K=128 image-patch positions in the live policy
    → counterfactual sim evaluation: matched vs mismatched intent under
       eval_protocol=language_swap, cached no-steer arms for efficiency
    → three-axis scorecard: codec quality, intent specificity, causal steering`}
        </code>
      </pre>

      <h3>Components</h3>
      <ul>
        <li>
          <strong>Hook.</strong> A forward hook on{" "}
          <code>backbone_features</code> records{" "}
          <code>h ∈ ℝ²⁰⁴⁸</code> per token at layer 16. Position roles:{" "}
          <code>last_text</code>, <code>image_patch</code>,{" "}
          <code>anchor</code>. The image-patch position pool is{" "}
          strided-multi: 128 evenly-spaced patch tokens out of the live
          backbone&apos;s patch grid.
        </li>
        <li>
          <strong>AV (verbalizer).</strong> LoRA fine-tune of
          Qwen3-4B-Instruct. The prompt contains 128 reserved{" "}
          <code>&lt;|act_slot_i|&gt;</code> tokens for the image-patch grid
          plus one{" "}
          <code>&lt;|act_slot_last_text|&gt;</code> token for the language
          channel. Each slot embedding is overwritten with{" "}
          <code>α · normalize(W_p · h_slot)</code> where{" "}
          <code>α</code> is the 75th percentile activation norm from
          extraction stats.
        </li>
        <li>
          <strong>AR (reconstructor).</strong> LoRA fine-tune of the same
          base, full-depth (all 36 transformer layers). Reads the caption
          and emits one 2048-dimensional vector per image-patch position
          via a spatial K=128 head. Loss is the decomposed cosine +
          log-magnitude formulation; an InfoNCE term with mined hard
          negatives penalizes captions that AR can invert to many activations.
        </li>
        <li>
          <strong>Steering.</strong> At inference, a user provides a target
          intent <code>y</code>. AV generates a caption{" "}
          <code>C = AV(h, y)</code>;{" "}
          <code>ĥ = AR(C)</code> emits a (128, 2048) grid which the wrapper
          injects at the 128 image-patch positions of the live policy on
          every <code>get_action</code> call.
        </li>
      </ul>

      <h3>Read vs. write &mdash; two distinct claims</h3>
      <p>
        <strong>Reading</strong>{" "}
        goes activation-to-text:{" "}
        <code>h → AV → caption</code>. Verified by: closed-loop
        reconstruction quality (cosine, MSE, FVE), caption character
        overlap across intent variants on the same activation.
      </p>
      <p>
        <strong>Writing</strong>{" "}
        goes text-to-activation-to-behavior:{" "}
        <code>caption → AR → ĥ → policy → action</code>. Verified by: the
        counterfactual sim eval that compares matched intent with codec
        injection against three other arms (no codec, mismatched intent,
        mismatched without codec) to isolate the codec&apos;s causal
        contribution from the policy&apos;s built-in language channel.
      </p>
    </section>
  );
}
