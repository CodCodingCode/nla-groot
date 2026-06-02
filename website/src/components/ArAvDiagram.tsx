/**
 * Focused AV/AR diagram: the bidirectional natural-language codec.
 *
 *   GR00T policy ──read──▶ h ──AV──▶ caption ──AR──▶ ĥ ──inject──▶ GR00T policy
 *
 * AV (activation → English) and AR (English → activation) are the two
 * directions of the codec; the bottleneck in the middle is plain English.
 * Authored as inline SVG so it stays responsive and theme-agnostic.
 */

import { CSSProperties } from "react";

const W = 720;
const H = 600;
const CX = W / 2;
const BW = 300;

const svgStyle: CSSProperties = {
  width: "100%",
  height: "auto",
  maxWidth: 720,
  display: "block",
  margin: "0 auto",
};

interface Node {
  y: number;
  h: number;
  label: string;
  subs: string[];
  emphasis?: boolean;
}

function Box({ y, h, label, subs, emphasis }: Node) {
  const x = CX - BW / 2;
  const labelY = emphasis ? y + 32 : y + 28;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={BW}
        height={h}
        rx={8}
        ry={8}
        fill={emphasis ? "#eef1ee" : "#ffffff"}
        stroke="#1a1a1a"
        strokeWidth={1.4}
      />
      <text
        x={CX}
        y={labelY}
        textAnchor="middle"
        fontSize={emphasis ? 23 : 17}
        fontWeight={700}
        fill="#1a1a1a"
      >
        {label}
      </text>
      {subs.map((s, i) => (
        <text
          key={i}
          x={CX}
          y={labelY + 20 + i * 15}
          textAnchor="middle"
          fontSize={11.5}
          fill="#5a5a5a"
        >
          {s}
        </text>
      ))}
    </g>
  );
}

interface EdgeProps {
  d: string;
  dashed?: boolean;
  op: string;
  caption: string;
  lx: number;
  ly: number;
}

function Edge({ d, dashed, op, caption, lx, ly }: EdgeProps) {
  return (
    <g>
      <path
        d={d}
        fill="none"
        stroke="#1a1a1a"
        strokeWidth={2}
        strokeDasharray={dashed ? "6 5" : undefined}
        markerEnd="url(#ah)"
      />
      <text x={lx} y={ly} textAnchor="middle" fontSize={17} fontWeight={700} fill="#1a1a1a">
        {op}
      </text>
      <text x={lx} y={ly + 16} textAnchor="middle" fontSize={11.5} fill="#5a5a5a">
        {caption}
      </text>
    </g>
  );
}

const POLICY: Node = { y: 20, h: 60, label: "GR00T policy", subs: ["VLA backbone · frozen"] };
const H_NODE: Node = {
  y: 160,
  h: 88,
  label: "h",
  subs: ["activation · backbone layer 16", "129 slots: 128 image-patch + 1 last-text"],
  emphasis: true,
};
const CAP: Node = {
  y: 328,
  h: 88,
  label: "caption",
  subs: ["plain English (the bottleneck)", "scene · target · gripper · plan"],
};
const HHAT: Node = {
  y: 496,
  h: 88,
  label: "ĥ",
  subs: ["reconstruction", "128-patch grid · steerable channel"],
  emphasis: true,
};

export default function ArAvDiagram() {
  const left = CX - BW / 2;
  return (
    <svg
      viewBox={`0 0 ${W} ${H + 8}`}
      style={svgStyle}
      role="img"
      aria-label="Bidirectional natural-language codec: AV maps a GR00T activation to an English caption; AR maps the caption back to a reconstructed vector that is injected into the policy's image-patch tokens to steer behavior."
    >
      <defs>
        <marker
          id="ah"
          viewBox="0 0 10 10"
          refX={9}
          refY={5}
          markerWidth={8}
          markerHeight={8}
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a" />
        </marker>
      </defs>

      <Box {...POLICY} />
      <Box {...H_NODE} />
      <Box {...CAP} />
      <Box {...HHAT} />

      {/* read: policy -> h */}
      <Edge d={`M ${CX} 80 L ${CX} 160`} op="read" caption="extract hidden state" lx={CX + 110} ly={112} />
      {/* AV: h -> caption */}
      <Edge d={`M ${CX} 248 L ${CX} 328`} op="AV" caption="activation → English" lx={CX + 110} ly={280} />
      {/* AR: caption -> hhat */}
      <Edge d={`M ${CX} 416 L ${CX} 496`} op="AR" caption="English → activation" lx={CX + 110} ly={448} />
      {/* inject: hhat -> policy (loop back on the left, dashed) */}
      <Edge
        d={`M ${left} 540 Q 60 540 60 50 Q 60 50 ${left} 50`}
        dashed
        op="inject"
        caption="write into image-patch tokens → steer"
        lx={155}
        ly={300}
      />
    </svg>
  );
}
