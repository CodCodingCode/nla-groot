import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  LabelList,
} from "recharts";
import type { Scorecard } from "../types";

interface Props {
  scorecard: Scorecard;
}

const VERDICT_COLOR: Record<string, string> = {
  PASS: "#1f7a3d",
  WARN: "#b07a14",
  FAIL: "#a32424",
  NA: "#9a9a9a",
};

// Friendly labels for the long metric keys.
const LABELS: Record<string, string> = {
  retrieval_margin: "Retrieval margin",
  retrieval_at_1: "Retrieval @ 1",
  retrieval_at_5: "Retrieval @ 5",
  judge_grounding_specific_pct: "Judge: grounding",
  judge_appropriateness_pct: "Judge: appropriate",
  judge_anti_template_specific_pct: "Judge: anti-template",
  closed_greedy_cosine: "Closed-loop cosine",
  sim_correct_success: "Sim: correct success",
  sim_correct_minus_baseline_floor: "Sim: correct − baseline",
  sim_wrong_minus_baseline: "Sim: wrong − baseline",
  sim_correct_minus_wrong: "Sim: correct − wrong (Δcw)",
};

export default function ScorecardChart({ scorecard }: Props) {
  // Show required metrics first, then a few of the most informative.
  const ordered = [...scorecard.metrics].sort(
    (a, b) =>
      Number(b.required_for_overall) - Number(a.required_for_overall) ||
      a.name.localeCompare(b.name)
  );

  const data = ordered.map((m) => ({
    metric: LABELS[m.name] ?? m.name,
    rawValue: m.value ?? 0,
    // Always plot magnitudes so negative bars (sim deltas) don't render below axis.
    value: m.value ?? 0,
    threshold: m.threshold_pass,
    verdict: m.verdict,
    required: m.required_for_overall,
  }));

  return (
    <div className="card">
      <p className="chart-title">
        V3 scorecard — overall verdict:{" "}
        <span className={`badge ${scorecard.overall.toLowerCase()}`}>
          {scorecard.overall}
        </span>
      </p>
      <p className="chart-sub">
        Bars: metric value. Dashed line at zero. PASS / WARN / FAIL color
        follows pre-registered thresholds in <code>build_v3_scorecard.py</code>.
        Required-for-overall metrics are marked with an asterisk.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(280, 28 * data.length)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 36, bottom: 16, left: 8 }}
        >
          <XAxis
            type="number"
            domain={[-1.05, 1.05]}
            tick={{ fontSize: 11 }}
            label={{
              value: "Value",
              position: "insideBottom",
              offset: -4,
              fontSize: 11,
            }}
          />
          <YAxis
            type="category"
            dataKey="metric"
            width={170}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: string, i: number) => {
              const required = data[i]?.required ? " *" : "";
              return v + required;
            }}
          />
          <ReferenceLine x={0} stroke="#888" strokeDasharray="3 3" />
          <Tooltip
            formatter={(_: number, _name, item: { payload?: Record<string, unknown> }) => {
              const p = item.payload || {};
              return [
                `${p.rawValue} (thr ${p.threshold}) — ${p.verdict}`,
                "value",
              ];
            }}
          />
          <Bar dataKey="value" isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={VERDICT_COLOR[d.verdict] || "#888"}
                fillOpacity={d.required ? 1 : 0.65}
              />
            ))}
            <LabelList
              dataKey="rawValue"
              position="right"
              fontSize={10}
              formatter={(v: number) => v.toFixed(2)}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="source">
        * required-for-overall. Source:{" "}
        <code>data/sft/libero_4suite_v3/v3_scorecard.json</code>.
      </p>
    </div>
  );
}
