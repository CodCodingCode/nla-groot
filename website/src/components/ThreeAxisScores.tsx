/**
 * Three-axis evaluation scores. A faithful causal bridge must clear all three:
 *   1. Reconstruction quality   (codec round-trips the activation)
 *   2. Intent specificity       (captions encode the target, not boilerplate)
 *   3. Causal steering          (injecting the reconstruction bends behavior)
 *
 * Numbers are the supervised-only (no-RL) v9 checkpoint on the held-out n=100
 * counterfactual set. Inline SVG to match the codec diagram.
 */

import { CSSProperties } from "react";

const W = 720;
const H = 300;
const BASE = 250; // baseline y
const TOP = 70; // plot top y
const PLOT = BASE - TOP; // 180

const svgStyle: CSSProperties = {
  width: "100%",
  height: "auto",
  maxWidth: 720,
  display: "block",
  margin: "0 auto",
};

function Bar({
  x,
  w,
  frac,
  value,
  emphasis,
}: {
  x: number;
  w: number;
  frac: number;
  value: string;
  emphasis?: boolean;
}) {
  const h = Math.max(2, frac * PLOT);
  const y = BASE - h;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        fill={emphasis ? "#1a1a1a" : "#cfd6cf"}
        stroke="#1a1a1a"
        strokeWidth={1}
      />
      <text x={x + w / 2} y={y - 6} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1a1a1a">
        {value}
      </text>
    </g>
  );
}

function PanelTitle({ cx, n, title }: { cx: number; n: string; title: string }) {
  return (
    <>
      <text x={cx} y={28} textAnchor="middle" fontSize={12} fontWeight={700} fill="#5a5a5a">
        {n}
      </text>
      <text x={cx} y={46} textAnchor="middle" fontSize={15} fontWeight={700} fill="#1a1a1a">
        {title}
      </text>
    </>
  );
}

export default function ThreeAxisScores() {
  // panel centers
  const p1 = 120;
  const p2 = 360;
  const p3 = 600;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={svgStyle}
      role="img"
      aria-label="Three-axis evaluation: reconstruction cosine 0.85, intent specificity 78% distinct captions, and causal steering where matched intent (0.39) beats no-steer (0.35) beats mismatched (0.24)."
    >
      {/* baselines */}
      <line x1={20} y1={BASE} x2={230} y2={BASE} stroke="#1a1a1a" strokeWidth={1} />
      <line x1={250} y1={BASE} x2={470} y2={BASE} stroke="#1a1a1a" strokeWidth={1} />
      <line x1={490} y1={BASE} x2={700} y2={BASE} stroke="#1a1a1a" strokeWidth={1} />

      {/* Axis 1 — reconstruction */}
      <PanelTitle cx={p1} n="AXIS 1" title="Reconstruction" />
      <Bar x={p1 - 45} w={90} frac={0.85} value="0.85" emphasis />
      <text x={p1} y={272} textAnchor="middle" fontSize={11} fill="#5a5a5a">
        cosine similarity
      </text>

      {/* Axis 2 — intent specificity */}
      <PanelTitle cx={p2} n="AXIS 2" title="Intent specificity" />
      <Bar x={p2 - 45} w={90} frac={0.78} value="78%" emphasis />
      <text x={p2} y={272} textAnchor="middle" fontSize={11} fill="#5a5a5a">
        distinct captions (22% overlap)
      </text>

      {/* Axis 3 — causal steering */}
      <PanelTitle cx={p3} n="AXIS 3" title="Causal steering" />
      <Bar x={p3 - 75} w={42} frac={0.388 / 0.5} value="0.39" emphasis />
      <Bar x={p3 - 21} w={42} frac={0.345 / 0.5} value="0.35" />
      <Bar x={p3 + 33} w={42} frac={0.239 / 0.5} value="0.24" />
      <text x={p3 - 54} y={266} textAnchor="middle" fontSize={9.5} fill="#5a5a5a">match</text>
      <text x={p3} y={266} textAnchor="middle" fontSize={9.5} fill="#5a5a5a">no-steer</text>
      <text x={p3 + 54} y={266} textAnchor="middle" fontSize={9.5} fill="#5a5a5a">mismatch</text>
      <text x={p3} y={284} textAnchor="middle" fontSize={11} fill="#5a5a5a">
        r_sim (task progress)
      </text>
    </svg>
  );
}
