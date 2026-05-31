export default function Protocol() {
  return (
    <section id="protocol">
      <h2>Three-axis evaluation: codec, intent, steering</h2>
      <p>
        A single composite score hides too much. We test the
        natural-language bridge claim on three independent axes; the
        bridge is supported only when every axis lands above zero
        on its respective scale.
      </p>

      <table>
        <thead>
          <tr>
            <th>Axis</th>
            <th>What it measures</th>
            <th>Sufficient gate</th>
            <th>Key metric</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <strong>1. Codec quality</strong>
            </td>
            <td>
              Does <code>AR(AV(h)) ≈ h</code> for held-out activations?
              Tests whether the activation manifold admits a
              natural-language encoding at all.
            </td>
            <td>
              Closed-greedy cosine substantially above zero (random); MSE
              within a small fraction of the activation norm.
            </td>
            <td>
              <code>closed_greedy/cosine</code>,{" "}
              <code>closed_greedy/mse</code>,{" "}
              <code>closed_greedy/fve</code>
            </td>
          </tr>
          <tr>
            <td>
              <strong>2. Intent specificity</strong>
            </td>
            <td>
              Do AV captions differ when the target intent changes on the
              same activation? Tests whether the encoding actually
              conditions on the steering target rather than producing a
              canonical description per activation.
            </td>
            <td>
              Character-level overlap substantially below 1.0; bullet
              difference count materially above 0.
            </td>
            <td>
              <code>mean_char_overlap</code>,{" "}
              <code>mean_bullet_diff_count</code>
            </td>
          </tr>
          <tr>
            <td>
              <strong>3. Causal steering</strong>
            </td>
            <td>
              Does injecting <code>ĥ = AR(text)</code> at the policy&apos;s
              image-patch positions move behavior toward the captioned
              task? Tests the activation channel&apos;s causal
              load-bearing-ness.
            </td>
            <td>
              <code>steer_lift &gt; 0</code> with{" "}
              <code>t &gt; 2</code>;{" "}
              <code>codec_above_lang &gt; 0</code> isolates the
              activation-channel signal from the policy&apos;s built-in
              language input.
            </td>
            <td>
              <code>steer_lift</code>,{" "}
              <code>sem_gap</code>,{" "}
              <code>codec_above_lang</code>
            </td>
          </tr>
        </tbody>
      </table>

      <h3>The counterfactual eval design (Axis 3)</h3>
      <p>
        The CF eval samples 32 held-out activations with paired{" "}
        <code>(matched_intent, mismatched_intent)</code> labels. For each
        sample we run four short LIBERO rollouts that fully cross the
        codec and the language channels:
      </p>

      <table>
        <thead>
          <tr>
            <th>Arm</th>
            <th>Policy obs.language</th>
            <th>Steer hook</th>
            <th>Question it answers</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>matched / semantic</strong></td>
            <td>matched intent</td>
            <td><code>ĥ = AR(AV(h, matched intent))</code></td>
            <td>The model performs the matched task with the codec helping.</td>
          </tr>
          <tr>
            <td><strong>matched / no_steer</strong></td>
            <td>matched intent</td>
            <td>disabled</td>
            <td>Baseline: how well the policy does on its own.</td>
          </tr>
          <tr>
            <td><strong>mismatched / semantic</strong></td>
            <td>mismatched intent</td>
            <td><code>ĥ = AR(AV(h, mismatched intent))</code></td>
            <td>Codec encoding of a different task injected into a different language context.</td>
          </tr>
          <tr>
            <td><strong>mismatched / no_steer</strong></td>
            <td>mismatched intent</td>
            <td>disabled</td>
            <td>Pure language-channel effect with no activation-channel signal.</td>
          </tr>
        </tbody>
      </table>

      <p style={{ marginTop: "0.8rem" }}>
        From these four arms we derive four numbers per sample:
      </p>

      <ul>
        <li>
          <code>steer_lift</code> = <em>matched/sem</em> &minus;{" "}
          <em>matched/no_steer</em>. The codec&apos;s effect with the
          language obs held to the matched intent.
        </li>
        <li>
          <code>sem_gap</code> = <em>matched/sem</em> &minus;{" "}
          <em>mismatched/sem</em>. How much better the matched arm does
          than the mismatched arm when both have the codec injected.
        </li>
        <li>
          <code>lang_swap</code> = <em>matched/no_steer</em> &minus;{" "}
          <em>mismatched/no_steer</em>. How much the policy&apos;s
          obs.language alone differentiates between matched and mismatched
          intents.
        </li>
        <li>
          <code>codec_above_lang</code> ={" "}
          <code>sem_gap</code> &minus; <code>lang_swap</code>. The
          incremental signal the codec contributes <em>beyond</em> the
          policy&apos;s built-in language conditioning. This is the
          critical &ldquo;is the activation-channel injection doing
          independent work&rdquo; metric.
        </li>
      </ul>
    </section>
  );
}
