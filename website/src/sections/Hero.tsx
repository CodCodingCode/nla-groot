import MetricBadge from "../components/MetricBadge";
import type { Scorecard } from "../types";

const REPO = "https://github.com/CodCodingCode/nla-groot";
const SITE = "https://codcodingcode.github.io/nla-groot/";

interface Props {
  scorecard: Scorecard | null;
}

export default function Hero({ scorecard }: Props) {
  return (
    <section id="hero">
      <h1>
        When reconstruction passes: NLAs on a VLA backbone still fail grounding
        and semantic steering
      </h1>
      <p className="lede">
        We port the NLA recipe from large language models to GR00T, a humanoid
        VLA policy. We extract per-token backbone activations, train a
        verbalizer (AV) and a reconstructor (AR), and inject AR&apos;s output
        back into the policy as a live steering vector in LIBERO simulation.
      </p>
      <p>
        We release <strong>nla-groot</strong> and a three-axis evaluation:
        offline reconstruction, a multimodal grounding judge on cached frames,
        and closed-loop steer A/B (matched vs. mismatched language).{" "}
        <strong>Reconstruction passes; the other two axes do not.</strong>{" "}
        Aggregate metrics hide collapse on <code>image_patch</code> tokens.
        This site walks through the pipeline and the evidence—with charts pulled
        from real run artifacts.
      </p>

      {scorecard && (
        <div className="callout">
          <strong>Headline (V3 checkpoint).</strong> Overall verdict on{" "}
          <code>{scorecard.checkpoint}</code>:{" "}
          <MetricBadge verdict={scorecard.overall} />. Required gates:{" "}
          <em>retrieval margin</em>, <em>judge grounding</em>, and{" "}
          <em>judge anti-template specificity</em>.
        </div>
      )}

      <p>
        <a href={REPO} target="_blank" rel="noreferrer">
          GitHub repository
        </a>
        {" · "}
        <a href={`${SITE}papers/main_corl.pdf`}>CoRL 2026 draft (PDF)</a>
        {" · "}
        <a href={`${SITE}papers/main.pdf`}>Workshop paper (PDF)</a>
        {" · "}
        <a href="#repro">Reproduce the numbers</a>
      </p>
    </section>
  );
}
