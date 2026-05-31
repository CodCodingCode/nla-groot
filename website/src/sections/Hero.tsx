import type { SiteSnapshot } from "../types";

const REPO = "https://github.com/CodCodingCode/nla-groot";
const SITE = "https://codcodingcode.github.io/nla-groot/";

interface Props {
  data: SiteSnapshot;
}

function _fmt(n: number | null | undefined, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

function _fmtSigned(n: number | null | undefined, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const s = n.toFixed(digits);
  return n >= 0 ? `+${s}` : s;
}

export default function Hero({ data }: Props) {
  const codec = data.codec_final;
  const cf = data.cf_eval;
  const cap = data.caption_diag;
  const cfStatus = cf.status;
  const cfReady = cfStatus === "complete";
  const cfRunning = cfStatus === "running";

  return (
    <section id="hero">
      <h1>
        A causally-verified, intent-conditional bidirectional bridge between
        vision-language-action model activations and natural language.
      </h1>
      <p className="lede">
        We train a natural-language autoencoder over the backbone activations
        of <strong>GR00T-N1.7</strong>, a humanoid VLA policy. An{" "}
        <em>activation verbalizer</em> (AV) maps an internal hidden state to
        a 5–6 bullet English caption; an <em>activation reconstructor</em>{" "}
        (AR) maps the caption back to a 2048-dimensional vector. The
        reconstructed vector, injected at the live policy&apos;s 128
        image-patch token positions, causally biases the policy toward the
        captioned task.
      </p>

      <p>
        The codec exhibits four properties simultaneously:{" "}
        <strong>high-fidelity reconstruction</strong> of held-out
        activations, <strong>intent-conditional</strong> captions on the
        same activation, <strong>causal steering effectiveness</strong> in
        sim, and <strong>signal beyond the policy&apos;s built-in
        language input</strong>. Each is stated below as an absolute number
        on held-out data, no relative baseline required.
      </p>

      {codec && (
        <div className="callout">
          <strong>Headline (run {data.run_name}).</strong>{" "}
          Closed-loop cosine{" "}
          <code>{_fmt(codec.closed_greedy_cosine, 4)}</code> (
          reconstruction angle <code>{_fmt(codec.angle_degrees, 1)}°</code> from
          ground truth), closed-loop MSE{" "}
          <code>{_fmt(codec.closed_greedy_mse, 2)}</code> (
          <code>{_fmt(codec.relative_norm_error_pct, 2)}%</code> relative
          magnitude error, RMS{" "}
          <code>{_fmt(codec.rms_error_raw_units, 2)}</code> in raw activation
          units against the {_fmt(codec.alpha_norm, 1)}-norm scale).
          {cap && (
            <>
              {" "}Caption character overlap between matched and mismatched
              intents on the same activation:{" "}
              <code>{_fmt(cap.mean_char_overlap, 4)}</code> across{" "}
              <code>{cap.n_samples}</code> paired samples.
            </>
          )}
          {cfReady && cf.steer_lift && (
            <>
              {" "}Counterfactual steering at <code>n={cf.n_complete}</code>:{" "}
              steer_lift{" "}
              <code>{_fmtSigned(cf.steer_lift.mean, 4)}</code> (t ={" "}
              <code>{_fmtSigned(cf.steer_lift.t, 2)}</code>, wins/losses{" "}
              <code>{cf.steer_lift.wins}/{cf.steer_lift.losses}</code>);
              codec contribution beyond language{" "}
              <code>{_fmtSigned(cf.codec_above_lang ?? 0, 4)}</code>.
            </>
          )}
          {cfRunning && cf.steer_lift && (
            <>
              {" "}Counterfactual steering is running:{" "}
              <code>{cf.n_complete}/{cf.n_target}</code> samples complete,
              partial steer_lift{" "}
              <code>{_fmtSigned(cf.steer_lift.mean, 4)}</code>. Page will
              refresh when CF eval finishes.
            </>
          )}
          {cfStatus === "pending" && (
            <>
              {" "}Counterfactual steering eval is queued; the final
              steer_lift number will appear here when the sim rollouts
              complete.
            </>
          )}
        </div>
      )}

      <p>
        <a href={REPO} target="_blank" rel="noreferrer">
          GitHub repository
        </a>
        {" · "}
        <a href={`${SITE}papers/main_corl.pdf`}>CoRL 2026 draft (PDF)</a>
        {" · "}
        <a href="#results">Headline results</a>
        {" · "}
        <a href="#repro">Reproduce the numbers</a>
      </p>
    </section>
  );
}
