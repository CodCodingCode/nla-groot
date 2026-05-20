import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimRow } from "../types";

interface Props {
  rows: SimRow[];
}

// The matching prompt is steer_bowl_plate (the baseline task is to put the
// bowl on the plate). Everything else is a contradictory or off-target prompt.
const MATCHING = new Set(["steer_bowl_plate", "steer_bowl_plate_v3"]);

function shortLabel(cond: string): string {
  if (cond === "baseline") return "baseline (no steer)";
  return cond.replace(/^steer_/, "").replace(/_/g, " ");
}

export default function SimSteerChart({ rows }: Props) {
  // Two charts share a single x-axis: success_any and bowl_disp_m.
  const data = rows.map((r) => ({
    cond: shortLabel(r.condition),
    success: r.success_any ?? 0,
    bowl_disp: r.bowl_disp_m ?? 0,
    isBaseline: r.condition === "baseline",
    isMatching: MATCHING.has(r.condition),
  }));

  return (
    <div className="card">
      <p className="chart-title">
        Closed-loop steering on <code>put_the_bowl_on_the_plate</code>
      </p>
      <p className="chart-sub">
        Three seeds per condition. Success rate (top) collapses to zero for
        every steered arm — matching <strong>and</strong> contradictory prompts
        alike. Bowl displacement (bottom) shows the symptom: every steer dampens
        motion uniformly rather than redirecting it. Δ<sub>correct−wrong</sub> = 0.
      </p>

      <h4 style={{ margin: "0.6rem 0 0.2rem", fontSize: "0.95rem" }}>
        Success rate (success_any)
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="cond"
            tick={{ fontSize: 10 }}
            interval={0}
            angle={-12}
            textAnchor="end"
            height={50}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            label={{
              value: "success_any",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Legend />
          <Bar dataKey="success" name="Success rate" fill="#1f3a8a">
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={
                  d.isBaseline ? "#1f7a3d" : d.isMatching ? "#1f3a8a" : "#a32424"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <h4 style={{ margin: "1rem 0 0.2rem", fontSize: "0.95rem" }}>
        Bowl displacement (m, mean over seeds)
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 28, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="cond"
            tick={{ fontSize: 10 }}
            interval={0}
            angle={-12}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            label={{
              value: "displacement (m)",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip formatter={(v: number) => v.toFixed(3) + " m"} />
          <Bar dataKey="bowl_disp" name="bowl displacement (m)" fill="#5a5a5a">
            {data.map((d, i) => (
              <Cell key={i} fill={d.isBaseline ? "#1f7a3d" : "#5a5a5a"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="source">
        Color legend: green = baseline, blue = matching prompt
        (<code>steer_bowl_plate*</code>), red = contradictory prompt. Source:{" "}
        <code>data/eval/steerability_v1_vs_v3/metrics.json</code>.
      </p>
    </div>
  );
}
