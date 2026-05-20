import MetricBadge from "../components/MetricBadge";
import type { Scorecard } from "../types";

interface Props {
  scorecard: Scorecard | null;
}

export default function Hero({ scorecard }: Props) {
  return (
    <section id="hero">
      <h1>
        Natural-language autoencoders on a vision-language-action policy
        backbone
      </h1>
      <p className="lede">
        We port the NLA recipe from large language models to GR00T,
        a humanoid VLA policy. We extract per-token backbone activations,
        train a verbalizer (AV) and a reconstructor (AR), and inject AR&apos;s
        output back into the policy as a live steering vector.
      </p>
      <p>
        We then run a three-axis evaluation: offline reconstruction,
        a multimodal grounding judge on cached frames, and a closed-loop
        steerability A/B in LIBERO. <strong>Reconstruction passes; the
        other two axes do not.</strong> This site is a guided tour of the
        pipeline and the evidence behind that claim, with charts pulled
        directly from the run artifacts in the repo.
      </p>

      {scorecard && (
        <div className="callout">
          <strong>Headline.</strong> Overall verdict on the V3 checkpoint
          (<code>{scorecard.checkpoint}</code>):{" "}
          <MetricBadge verdict={scorecard.overall} />. Required metrics that
          gate this verdict are <em>retrieval margin</em>,{" "}
          <em>judge grounding</em>, and <em>judge anti-template specificity</em>.
        </div>
      )}

      <p>
        <a href="https://anonymous.4open.science/" target="_blank" rel="noreferrer">
          Anonymous code
        </a>
        {" · "}
        <a href="../paper/main.pdf">Workshop paper (PDF)</a>
        {" · "}
        <a href="#repro">Reproduce the numbers</a>
      </p>
    </section>
  );
}
