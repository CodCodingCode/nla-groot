export default function Problem() {
  return (
    <section id="problem">
      <h2>Problem: VLA black box, language as a tempting interface</h2>
      <p>
        GR00T couples a Cosmos-Reason2-2B / Qwen3-VL backbone to a
        diffusion action head. Only the first <code>SELECT_LAYER = 16</code>{" "}
        decoder layers are kept at deployment, so the policy reads a
        2048-dimensional hidden state <code>h</code> at layer 16 before
        the action head consumes it.
      </p>
      <p>
        That state is causally upstream of every action the robot takes,
        but there is no standard way to <em>read</em> it as language or to{" "}
        <em>write</em> into it from language. NLA proposes a clean answer:
        train two small models — a verbalizer{" "}
        <code>AV: h → text</code> and a reconstructor{" "}
        <code>AR: text → ĥ</code>. The pair is a natural-language autoencoder
        on activations: it gives an interpretability artifact (the caption)
        and a causal handle (the reconstructed vector).
      </p>

      <h3>The teacher confound (read this once)</h3>
      <p>
        Supervision in this stack is <strong>synthetic</strong>. A multimodal
        teacher (a GPT-class model) sees the raw camera frame, the
        instruction, and token-position metadata, and writes the gold caption.
        It <em>does not</em> see <code>h</code> floats. SFT therefore
        optimizes <em>P(teacher text | h)</em>, not &ldquo;truth about
        what <code>h</code> encodes.&rdquo; The teacher can describe scene
        attributes that are only weakly present in <code>h</code>; AV can
        learn to mimic the teacher without the activation actually carrying
        the information.
      </p>
      <p className="callout">
        <strong>Why this matters.</strong> If the only thing you measure is
        whether <code>AR(AV(h)) ≈ h</code>, you are scoring the codec, not
        the explanation. AV captions can collapse into invertible{" "}
        <em>templates</em> that AR happily reconstructs while saying almost
        nothing about the actual frame.
      </p>
    </section>
  );
}
