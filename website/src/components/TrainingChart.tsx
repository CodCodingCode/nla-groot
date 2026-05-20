import {
  CartesianGrid,
  Line,
  LineChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainingPoint } from "../types";

interface Props {
  points: TrainingPoint[];
}

export default function TrainingChart({ points }: Props) {
  if (!points.length) return null;

  const data = points
    .filter((p) => p.step !== null && p.step !== undefined)
    .map((p) => ({
      step: p.step,
      fve: p.fve ?? null,
      closed_greedy_fve: p.closed_greedy_fve ?? null,
    }));

  return (
    <div className="card">
      <p className="chart-title">SFT validation curves</p>
      <p className="chart-sub">
        Teacher-forced FVE versus closed-loop greedy FVE during the
        <code> libero_4suite_v3</code> run. Closed-loop tracks teacher-forced
        closely — a strong autoencoder signal that, on its own, does not
        predict caption grounding or behavioral steering.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 12, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 11 }}
            label={{
              value: "training step",
              position: "insideBottom",
              offset: -8,
              fontSize: 11,
            }}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(2)}
            label={{
              value: "FVE",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip formatter={(v: number) => (v == null ? "—" : v.toFixed(3))} />
          <Legend />
          <Line
            type="monotone"
            dataKey="fve"
            name="teacher-forced FVE"
            stroke="#1f3a8a"
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="closed_greedy_fve"
            name="closed-loop greedy FVE"
            stroke="#b07a14"
            strokeWidth={1.6}
            strokeDasharray="4 3"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="source">
        Source: <code>data/sft/libero_4suite_v3/metrics.jsonl</code> (val and
        final phases, subsampled).
      </p>
    </div>
  );
}
