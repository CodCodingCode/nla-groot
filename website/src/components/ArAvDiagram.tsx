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
const H = 720;

const svgStyle: CSSProperties = {
  width: "100%",
  height: "auto",
  maxWidth: 720,
  display: "block",
  margin: "0 auto",
};

interface BoxProps {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string;
  emphasis?: boolean;
}

function Box({ x, y, w, h, label, sub, emphasis }: BoxProps) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        ry={8}
        fill={emphasis ? "#eef1ee" : "#ffffff"}
        stroke="#1a1a1a"
        strokeWidth={1.4}
      />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 5 : y + h / 2 + 6}
        textAnchor="middle"
        fontSize={emphasis ? 22 : 17}
        fontWeight={700}
        fill="#1a1a1a"
      >
        {label}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 16}
          textAnchor="middle"
          fontSize={12}
          fill="#5a5a5a"
        >
          {sub}
        </text>
      )}
    </g>
  );
}

interface EdgeProps {
  d: string;
  dashed?: boolean;
  op: string; // "AV" / "AR" / "read" / "inject"
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

export default function ArAvDiagram() {
  const cx = W / 2;
  const bw = 260;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={svgStyle}
      role="img"
      aria-label="Bidirectional natural-language codec: AV maps an activation to a caption, AR maps it back to a reconstructed vector that is injected into the policy."
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

      {/* nodes */}
      <Box x={cx - bw / 2} y={20} w={bw} h={62} label="GR00T policy" sub="VLA backbone · frozen" />
      <Box x={cx - bw / 2} y={170} w={bw} h={70} label="h" sub="activation · layer 16 · [128 × 2048]" emphasis />
      <Box x={cx - bw / 2} y={330} w={bw} h={70} label="caption" sub="English: scene · target · gripper · plan" />
      <Box x={cx - bw / 2} y={490} w={bw} h={70} label="ĥ" sub="reconstruction · [128 × 2048]" emphasis />

      {/* read: policy -> h */}
      <Edge d={`M ${cx} 82 L ${cx} 170`} op="read" caption="extract hidden state" lx={cx + 96} ly={120} />
      {/* AV: h -> caption */}
      <Edge d={`M ${cx} 240 L ${cx} 330`} op="AV" caption="activation → English" lx={cx + 96} ly={278} />
      {/* AR: caption -> hhat */}
      <Edge d={`M ${cx} 400 L ${cx} 490`} op="AR" caption="English → activation" lx={cx + 96} ly={438} />
      {/* inject: hhat -> policy (loop back, dashed) */}
      <Edge
        d={`M ${cx - bw / 2} 525 Q 70 525 70 51 Q 70 51 ${cx - bw / 2} 51`}
        dashed
        op="inject"
        caption="write into image-patch tokens → steer"
        lx={150}
        ly={300}
      />
    </svg>
  );
}
