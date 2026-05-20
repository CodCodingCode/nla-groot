import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { JudgeRow } from "../types";

interface Props {
  rows: JudgeRow[];
  /**
   * Optional filter: only show one (checkpoint, holdout) pair. We default to
   * the in-distribution pair that appears in the paper's headline table.
   */
  preferred?: { checkpoint: string; holdout: string };
}

const DEFAULT_PREFERRED = {
  checkpoint: "libero_4suite_v3",
  holdout: "libero_4suite_holdout",
};

export default function JudgeChart({ rows, preferred = DEFAULT_PREFERRED }: Props) {
  // Filter to one pair so bars stay legible. Fall back to first available.
  let filtered = rows.filter(
    (r) => r.checkpoint === preferred.checkpoint && r.holdout === preferred.holdout
  );
  let pair = preferred;
  if (filtered.length === 0 && rows.length > 0) {
    pair = { checkpoint: rows[0].checkpoint, holdout: rows[0].holdout };
    filtered = rows.filter(
      (r) => r.checkpoint === pair.checkpoint && r.holdout === pair.holdout
    );
  }

  // Reshape to one row per axis with gold/av_pred columns.
  const gold = filtered.find((r) => r.source === "gold");
  const av = filtered.find((r) => r.source === "av_pred");
  const data = [
    {
      axis: "Grounding",
      gold: gold?.grounding ?? 0,
      av: av?.grounding ?? 0,
    },
    {
      axis: "Appropriateness",
      gold: gold?.appropriateness ?? 0,
      av: av?.appropriateness ?? 0,
    },
    {
      axis: "Anti-template",
      gold: gold?.template_distinguishable ?? 0,
      av: av?.template_distinguishable ?? 0,
    },
  ];

  const nRows = gold?.n_rows ?? av?.n_rows ?? 0;

  return (
    <div className="card">
      <p className="chart-title">
        Multimodal judge: gold (teacher) vs. AV captions
      </p>
      <p className="chart-sub">
        Pass rate on a held-out sample. Captions look <em>appropriate</em> but
        are not <em>vision-grounded</em> and collapse onto reusable templates.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 12, right: 12, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="axis"
            tick={{ fontSize: 12 }}
            label={{ value: "Judge axis", position: "insideBottom", offset: -8, fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            label={{
              value: "Pass rate",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip
            formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
          />
          <Legend />
          <Bar dataKey="gold" name="gold (teacher)" fill="#1f3a8a" />
          <Bar dataKey="av" name="AV (greedy)" fill="#a32424" />
        </BarChart>
      </ResponsiveContainer>
      <p className="source">
        Checkpoint: <code>{pair.checkpoint}</code> · holdout:{" "}
        <code>{pair.holdout}</code> · n={nRows}. Source:{" "}
        <code>data/eval/steerability_v1_vs_v3/av_metrics.json</code>.
      </p>
    </div>
  );
}
