import PipelineDiagram from "../components/PipelineDiagram";

export default function Pipeline() {
  return (
    <section id="pipeline">
      <h2>Pipeline: extract, label, train, audit, steer</h2>
      <p>
        Solid arrows below are training signals; dashed arrows are inference
        and steering. Activations are real (real frames pushed through real
        GR00T forward passes). Captions are synthetic (multimodal teacher).
      </p>

      <PipelineDiagram />

      <h3>Components</h3>
      <ul>
        <li>
          <strong>Hook.</strong> A forward hook on{" "}
          <code>backbone_features</code> records{" "}
          <code>h ∈ ℝ²⁰⁴⁸</code> per token at layer 16, before the post-VL
          LayerNorm. Position types: <code>last_text</code>,{" "}
          <code>image_patch</code>, <code>anchor</code>.
        </li>
        <li>
          <strong>AV (verbalizer).</strong> LoRA fine-tune of
          Qwen3-4B-Instruct. The prompt has a reserved{" "}
          <code>&lt;|act_slot|&gt;</code> token whose embedding is overwritten
          with <code>α · normalize(W_p · h)</code>. <code>α</code> is the 75th
          percentile of <code>‖h‖</code> from extraction stats.
        </li>
        <li>
          <strong>AR (reconstructor).</strong> LoRA fine-tune of the same base,
          truncated to 16 layers. Reads &ldquo;Summary of the following text:&rdquo;
          and predicts <code>ĥ / α</code>. Loss is MSE in
          <code> α</code>-scaled space plus an InfoNCE term with mined hard
          negatives.
        </li>
        <li>
          <strong>Steering.</strong> At inference, a user provides text{" "}
          <code>y</code>; we compute <code>ĥ = AR(y)</code> and blend it into
          backbone tokens (one image patch or all image patches) on every{" "}
          <code>get_action</code> call in the official LIBERO server.
        </li>
      </ul>

      <h3>Audit vs. steer (different things)</h3>
      <p>
        <strong>Auditing</strong> goes left-to-right: sample <code>h</code>,
        run AV, read the caption, optionally test it against pixels or against
        counterfactual <code>h</code> edits. <strong>Steering</strong> goes
        right-to-left: write text, get <code>ĥ</code>, inject. Both share
        AR/AV but they are <em>different</em> claims and need different gates.
        Hence the three-axis protocol below.
      </p>
    </section>
  );
}
