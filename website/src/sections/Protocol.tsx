export default function Protocol() {
  return (
    <section id="protocol">
      <h2>The three-axis evaluation protocol</h2>
      <p>
        A single composite score hides too much. We score each checkpoint
        on three independent axes; overall PASS requires every required
        metric to pass against pre-registered thresholds.
      </p>

      <table>
        <thead>
          <tr>
            <th>Axis</th>
            <th>What it measures</th>
            <th>What it doesn&apos;t</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <strong>1. Reconstruction &amp; retrieval</strong>
            </td>
            <td>
              Closed-loop cosine <code>cos(h, AR(AV(h)))</code> and matched
              vs. cross-pair retrieval margin. Verifies the codec.
            </td>
            <td>
              Whether captions describe the actual frame or are reusable
              templates. High retrieval margin can coexist with template
              collapse.
            </td>
          </tr>
          <tr>
            <td>
              <strong>2. Vision-grounded judge</strong>
            </td>
            <td>
              Multimodal LLM judge scores AV and gold captions against{" "}
              <em>cached frames</em>: grounding (specific to the image),
              appropriateness (right type), template-distinguishability.
              Gold sets the ceiling.
            </td>
            <td>
              Whether the policy can be controlled by language. A grounded
              caption may still not steer the action head.
            </td>
          </tr>
          <tr>
            <td>
              <strong>3. Closed-loop steer A/B</strong>
            </td>
            <td>
              Match vs. mismatch sweep on a LIBERO task: baseline,{" "}
              <em>correct</em> steer text, <em>wrong</em> steer text. Primary
              metric: Δ<sub>cw</sub> = succ<sub>correct</sub> −
              succ<sub>wrong</sub>. Δ<sub>cw</sub> &gt; 0 is the
              non-trivial part.
            </td>
            <td>
              Whether the change is semantically tied to <em>that</em> prompt.
              Steering can break a task without redirecting it.
            </td>
          </tr>
          <tr>
            <td>
              <strong>(Optional) Counterfactual panel</strong>
            </td>
            <td>
              Hypothesis-driven <code>h</code> edits + matched-magnitude
              random control + pre-registered expected direction. Tests
              <em>faithfulness</em>.
            </td>
            <td>
              Population-level grounding (use Axis 2).
            </td>
          </tr>
        </tbody>
      </table>

      <p style={{ marginTop: "0.8rem" }}>
        Rule of thumb encoded in <code>build_v3_scorecard.py</code>:{" "}
        <span className="badge pass">PASS</span> only when{" "}
        <em>retrieval margin</em>, <em>judge grounding</em>, and{" "}
        <em>judge anti-template specificity</em> all pass and Δ<sub>cw</sub>{" "}
        ≥ 5pp where sim data is present.
      </p>
    </section>
  );
}
