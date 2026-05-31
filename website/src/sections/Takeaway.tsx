export default function Takeaway() {
  return (
    <section id="takeaway">
      <h2>What the codec is, and what it is not</h2>

      <h3>What you can do with a codec like this</h3>
      <ul>
        <li>
          <strong>Causally-verified behavioral steering.</strong> Write
          text, get a vector, inject it &mdash; the policy executes the
          captioned task. No retraining, no loss-function modification, no
          prompt engineering on the policy&apos;s native language slot.
          You modify the policy&apos;s internal state directly via a
          natural-language interface.
        </li>
        <li>
          <strong>Reading what the policy is &ldquo;thinking.&rdquo;</strong>{" "}
          Run AV at any timestep and get a 5-6 bullet English description
          of what the model is internally tracking right now. A
          thought-bubble for the robot.
        </li>
        <li>
          <strong>Counterfactual evaluation at scale.</strong> For each
          held-out rollout, ask: &ldquo;if the policy had been told to do
          task A instead of B at this exact mental state, would it have
          done A?&rdquo; The matched-vs-mismatched intent eval is the
          methodology. Currently impossible to do cleanly for a VLA.
        </li>
        <li>
          <strong>Composition in language space.</strong>{" "}
          <code>policy = base + steer(text)</code>. Compose multiple
          descriptions to produce composite behaviors. Maps onto
          Word2Vec-style activation arithmetic but for robot intents.
        </li>
        <li>
          <strong>Cross-architecture interpretation.</strong> The codec
          architecture is general &mdash; same recipe could be applied to
          RT-2, OpenVLA, π0, Octo. Compare internal representations across
          VLAs via their codec outputs.
        </li>
      </ul>

      <h3>What this work does not establish</h3>
      <ul>
        <li>
          <strong>Optimality.</strong> A working codec at the reported
          fidelity exists. Tighter codecs may be achievable.
        </li>
        <li>
          <strong>Real-world transfer.</strong> All evaluations are in
          LIBERO simulation. Real-robot performance and OOD generalization
          remain open.
        </li>
        <li>
          <strong>Steering precision.</strong>{" "}
          <code>steer_lift &gt; 0</code> shows the codec moves the policy
          toward the captioned task. It does not measure how finely we
          can control specific motion parameters &mdash; e.g.
          &ldquo;approach at 30° rather than 45°&rdquo; may not be
          controllable through this interface.
        </li>
        <li>
          <strong>Beating the per-dim mean baseline on FVE.</strong>{" "}
          Closed-greedy fraction-of-variance-explained is near zero but
          slightly negative. The codec is well above random by cosine and
          tight by magnitude in relative-norm terms, but does not yet
          outperform a trivial &ldquo;always predict the per-dimension
          batch mean&rdquo; baseline on variance explained. The cosine
          and relative-norm claims do not depend on FVE.
        </li>
        <li>
          <strong>Decomposability.</strong> This is not a circuit-level
          interpretation. We do not decompose the codec or the activation
          manifold into atomic interpretable features. Sparse-autoencoder
          and dictionary-learning tools remain orthogonal &mdash; the
          codec sits at a different level of granularity.
        </li>
        <li>
          <strong>Full task completion.</strong> The headline steering
          metric is task progress (<code>r_sim</code>) within a 100-sim-step
          horizon. Predicate-firing rates (full task completion) remain at
          0% across all arms for the evaluated horizon. We demonstrate
          better task progress, not full task success.
        </li>
      </ul>

      <h3>For researchers who want to extend this</h3>
      <ul>
        <li>
          <strong>For steering precision:</strong> add fine-grained intent
          descriptions to the labels (e.g. quantitative spatial
          specifications) and re-train. Test if the codec inherits the
          extra control.
        </li>
        <li>
          <strong>For FVE &gt; 0:</strong> push{" "}
          <code>--ar-scale-weight</code> higher on the decomposed loss to
          attack the magnitude axis more aggressively. The diminishing
          returns curve is open.
        </li>
        <li>
          <strong>For real-robot transfer:</strong> bring up the steer
          server against a real GR00T deployment and run the same CF
          methodology on real hardware. The injection point is the same
          (image-patch positions of the backbone forward).
        </li>
        <li>
          <strong>For cross-VLA comparison:</strong> use the same recipe
          (combined-mode input, intent-conditioned prompt, decomposed
          loss, K=128 spatial AR head) on RT-2 / OpenVLA / Octo. The
          activation extraction hook changes; everything else carries
          over.
        </li>
      </ul>
    </section>
  );
}
