export default function Problem() {
  return (
    <section id="problem">
      <h2>The bridge we built, and why it matters</h2>
      <p>
        GR00T-N1.7 couples a Cosmos-Reason2-2B / Qwen3-VL vision-language
        backbone to a diffusion action head. Only the first{" "}
        <code>SELECT_LAYER = 16</code> decoder layers are kept at deployment,
        so the policy reads a 2048-dimensional hidden state <code>h</code> at
        layer 16 before the action head consumes it. That state is causally
        upstream of every action the robot takes.
      </p>

      <p>
        Traditional interpretability tools tell you what is{" "}
        <em>inside</em> activations &mdash; probes, sparse autoencoders,
        attention rollouts. None of them prove the description is causally
        complete. A probe can say &ldquo;activation A encodes reaching for
        the bowl,&rdquo; and you have no way to verify that is the whole
        story versus a correlated feature.
      </p>

      <p>
        This work tests the harder claim:{" "}
        <strong>
          natural language is a sufficient code for a vision-language-action
          model&apos;s internal state.
        </strong>
        {" "}The test is concrete:
      </p>

      <ol>
        <li>
          Take activation <code>h</code> from a working robot policy at one
          token position.
        </li>
        <li>
          Train AV to map <code>h</code> + an intended target task to a
          natural-language caption <code>C</code> describing the scene, the
          target object, the gripper state, the spatial relations, and the
          task.
        </li>
        <li>
          Train AR to map <code>C</code> back to a reconstructed activation
          vector <code>ĥ</code>.
        </li>
        <li>
          Inject <code>ĥ</code> at the policy&apos;s image-patch token
          positions on a fresh rollout in LIBERO simulation.
        </li>
        <li>
          If the rollout does what <code>C</code> describes,{" "}
          <code>C</code> is a causally-faithful summary of the activation.
          The English description captures the load-bearing content of the
          state.
        </li>
      </ol>

      <p className="callout">
        <strong>Why this matters.</strong>{" "}
        A causally-verified language interface to a VLA opens five practical
        capabilities: read what the policy is &ldquo;thinking,&rdquo;
        rewrite its internal state from English, run counterfactual
        evaluations on novel intents at scale, compose multi-intent steers
        via caption concatenation, and compare internal representations
        across VLA architectures.
      </p>

      <h3>Architecture choices that make this work</h3>
      <p>
        Three SFT-time choices were necessary for the codec to develop
        intent-conditional, magnitude-calibrated reconstructions:
      </p>
      <ul>
        <li>
          <strong>Combined-mode input.</strong> Every training row packs all
          128 image-patch activations alongside the last-text token
          activation into a single 129-slot prompt. AV sees both
          vision-grounded and language-grounded context simultaneously; the
          codec generalises to either channel without a stratified
          per-token-role architecture.
        </li>
        <li>
          <strong>Intent-conditioned prompting.</strong> Every prompt
          contains a <code>Target task: &lt;intent&gt;</code> line, so AV
          learns to generate captions specifically conditional on the
          steering target, not just descriptive of the activation.
        </li>
        <li>
          <strong>Decomposed reconstruction loss.</strong> AR&apos;s loss
          splits direction (cosine error) from magnitude (log-norm error)
          so the optimizer can attack each independently. This produces
          reconstructions with both correct direction and correct scale,
          measurable on the FVE metric, not just cosine.
        </li>
      </ul>
    </section>
  );
}
