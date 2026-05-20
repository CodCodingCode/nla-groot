import JudgeChart from "../components/JudgeChart";
import ScorecardChart from "../components/ScorecardChart";
import SimSteerChart from "../components/SimSteerChart";
import TrainingChart from "../components/TrainingChart";
import type { SiteSnapshot } from "../types";

interface Props {
  data: SiteSnapshot;
}

export default function Results({ data }: Props) {
  return (
    <section id="results">
      <h2>Results: metrics pass, semantics fail</h2>
      <p>
        All numbers below are pulled from the actual run artifacts shipped
        with the repo (paths cited under each chart). No placeholder values.
      </p>

      <h3>Axis 1 — the codec works</h3>
      {data.scorecard && <ScorecardChart scorecard={data.scorecard} />}
      <p>
        Retrieval margin <strong>passes</strong> (0.124 vs. 0.05 threshold)
        and judge appropriateness is 100%. By those two metrics alone, this
        is a successful run. The other required metrics tell a different
        story.
      </p>

      {data.training.length > 0 && <TrainingChart points={data.training} />}

      <h3>Axis 2 — captions don&apos;t describe frames</h3>
      <JudgeChart rows={data.judge} />
      <p>
        On the 4-suite holdout, gold (teacher) captions ground the frame
        75% of the time and pass anti-template specificity 50% of the time.
        AV captions drop to <strong>41.7%</strong> grounding and just{" "}
        <strong>8.3%</strong> anti-template specificity — captions look
        valid (100% appropriateness) but reuse a small set of scene
        templates across genuinely different frames. This is the
        <em> shorthand template collapse</em> failure mode.
      </p>

      <h3>Axis 3 — steering is symmetric</h3>
      {data.sim.length > 0 && <SimSteerChart rows={data.sim} />}
      <p>
        Baseline solves the task on every seed. Every steered arm fails to
        complete the task — including the <em>matching</em> prompt that
        describes the correct goal. Δ<sub>cw</sub> = 0. The intervention
        has a clear behavioral effect (bowl displacement drops from
        0.145 m at baseline to ~0.07 m), but the effect is the same whether
        we use the right text or the wrong text. <strong>Steerability is
        not yet semantic faithfulness.</strong>
      </p>

      <div className="grid-2">
        <figure className="figure">
          <img
            src="figures/per_object_displacement.png"
            alt="Per-object displacement bar chart from steerability eval"
          />
          <figcaption>
            Per-object displacement across conditions. Baseline (top group)
            is the only condition where the bowl moves a task-typical
            distance. Source:{" "}
            <code>data/eval/steerability_v1_vs_v3/figures/</code>.
          </figcaption>
        </figure>
        <figure className="figure">
          <img
            src="figures/per_object_min_ee_distance.png"
            alt="Per-object minimum end-effector distance bar chart from steerability eval"
          />
          <figcaption>
            Minimum end-effector distance per object — every steered arm
            keeps the gripper farther from the bowl, regardless of which
            object the steer text named.
          </figcaption>
        </figure>
      </div>

      <div className="callout">
        <strong>One-line negative result.</strong> Strong activation
        reconstruction can coexist with vision-ungrounded captions and
        non-semantic steering. Reporting Axis 1 alone would have led to an
        overclaim.
      </div>
    </section>
  );
}
