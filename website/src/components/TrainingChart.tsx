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
      val_cosine: p.val_cosine ?? null,
      closed_greedy_cosine: p.closed_greedy_cosine ?? null,
    }));

  return (
    <div className="card">
      <p className="chart-title">SFT closed-loop reconstruction during training</p>
      <p className="chart-sub">
        Held-out closed-greedy and teacher-forced cosine over training
        steps. Closed-loop cosine climbing past 0.8 indicates the codec is
        producing direction-correct reconstructions even from its own
        decoded captions, not just teacher-forced text.
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
              value: "cosine (higher = better)",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip formatter={(v: number) => (v == null ? "—" : v.toFixed(3))} />
          <Legend />
          <Line
            type="monotone"
            dataKey="val_cosine"
            name="teacher-forced cosine"
            stroke="#1f3a8a"
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="closed_greedy_cosine"
            name="closed-loop greedy cosine"
            stroke="#b07a14"
            strokeWidth={1.6}
            strokeDasharray="4 3"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="source">
        Source: parsed from <code>data/sft/&lt;run&gt;_launch.log</code>{" "}
        <code>[step N] val</code> and <code>[final] val</code> lines
        (subsampled).
      </p>
    </div>
  );
}
