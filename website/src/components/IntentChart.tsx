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
import type { CaptionDiagnostic } from "../types";

interface Props {
  caption: CaptionDiagnostic;
}

const ZONES = [
  { upper: 0.5, label: "strongly intent-conditional", colour: "#1f7a3b" },
  { upper: 0.7, label: "moderately intent-conditional", colour: "#8fa31a" },
  { upper: 0.9, label: "weakly intent-conditional", colour: "#b07a14" },
  { upper: 1.01, label: "not intent-conditional", colour: "#a93232" },
];

function _colourFor(overlap: number) {
  for (const z of ZONES) {
    if (overlap <= z.upper) return z.colour;
  }
  return ZONES[ZONES.length - 1].colour;
}

export default function IntentChart({ caption }: Props) {
  const data = caption.paired_samples.map((s, i) => ({
    idx: `s${i}`,
    overlap: Number(s.char_overlap.toFixed(3)),
    bullet_diff: s.bullet_diff,
    source_id: s.source_id,
    matched_intent: s.matched_intent,
    mismatched_intent: s.mismatched_intent,
  }));

  return (
    <div className="card">
      <p className="chart-title">Paired-caption character overlap per sample</p>
      <p className="chart-sub">
        For each held-out activation, AV generates two captions under
        different target intents. Lower bars = more intent-conditional
        codec. Mean overlap{" "}
        <code>{caption.mean_char_overlap.toFixed(4)}</code>; mean
        bullet-diff{" "}
        <code>{caption.mean_bullet_diff_count.toFixed(2)}</code>.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={data}
          margin={{ top: 12, right: 16, bottom: 24, left: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="idx"
            tick={{ fontSize: 11 }}
            label={{
              value: "sample",
              position: "insideBottom",
              offset: -8,
              fontSize: 11,
            }}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            domain={[0, 1]}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "char overlap (lower = better)",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
            }}
          />
          <Tooltip
            formatter={(v: number) => (v == null ? "—" : v.toFixed(3))}
            labelFormatter={(idx: string, payload) => {
              const d = payload?.[0]?.payload;
              if (!d) return idx;
              return `${idx}: ${d.matched_intent} ↔ ${d.mismatched_intent}`;
            }}
          />
          <Legend />
          <Bar dataKey="overlap" name="char_overlap">
            {data.map((d, i) => (
              <Cell key={i} fill={_colourFor(d.overlap)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="source">
        Source:{" "}
        <code>data/eval/&lt;run&gt;_paired_captions.json</code>{" "}
        produced by <code>av_caption_intent_diff.py</code>.
      </p>
    </div>
  );
}
