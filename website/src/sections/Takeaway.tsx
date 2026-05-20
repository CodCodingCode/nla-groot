export default function Takeaway() {
  return (
    <section id="takeaway">
      <h2>Use this before you trust or steer</h2>

      <h3>For auditing</h3>
      <ul>
        <li>
          Do not treat <code>AV(h)</code> as ground truth without Axis 2 on
          held-out frames; report gold-vs-AV gaps, not AV alone.
        </li>
        <li>
          When making a specific claim about <code>h</code>, run a
          counterfactual panel: pre-registered hypothesis, <code>h</code>{" "}
          edit, matched-magnitude random control, judge or auto-metric on
          baseline / edited / control text.
        </li>
        <li>
          Stratify by position type (<code>last_text</code>,{" "}
          <code>image_patch</code>, <code>anchor</code>). Aggregate
          reconstruction can hide visual-slot underfitting.
        </li>
      </ul>

      <h3>For steering</h3>
      <ul>
        <li>
          Do not deploy <code>AR(y)</code> backbone steers because
          reconstruction is high. Require Δ<sub>cw</sub> &gt; 0 on a held-out
          set of tasks before believing in semantic control.
        </li>
        <li>
          Calibrate dose. Effects in this repo flip from &ldquo;no
          effect&rdquo; at low blend to &ldquo;break the policy&rdquo; at
          high blend, with little semantic separation in between.
        </li>
        <li>
          Treat AV.generate output as out-of-distribution for AR. Steer
          prompts written by humans in the same format as training labels
          are a safer starting point than free-form AV captions.
        </li>
      </ul>

      <h3>For training</h3>
      <ul>
        <li>
          Teacher-only SFT lacks an explicit pixel-alignment term. Reconstruction
          rewards happily pay off short, generic captions that AR can invert.
        </li>
        <li>
          Mitigations partially explored in this repo: hard-negative InfoNCE,
          AR–AV scheduled mixing, label quality grading, judge-blended GRPO.
          None close the gap on its own.
        </li>
        <li>
          The cheap tell is gold vs. AV on the <em>same</em> rows of the
          grounding judge. If gold pass rate drops below ~60%, the bug is in
          your labels, not your model.
        </li>
      </ul>
    </section>
  );
}
