/**
 * Full process diagram for GR00T-NLA (static).
 *
 *   Cosmos Reason2 VLM -> Layer 16 (143 tokens)
 *     keep 129 (128 image patches + 1 last text) -> AV -> verbal text -> AR -> (1,128,2048)
 *     drop 14 (template / instruction / vision markers) + last text bypass the codec
 *   recombine to 143 -> inject -> steer
 *
 * Inline SVG, theme-agnostic.
 */

import { CSSProperties } from "react";

const W = 620;
const H = 900;
const CX = 240; // codec column centre

const svgStyle: CSSProperties = {
  width: "100%",
  height: "auto",
  maxWidth: 620,
  display: "block",
  margin: "0 auto",
};

interface NodeProps {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  subs?: string[];
  emphasis?: boolean;
}

function Box({ x, y, w, h, label, subs = [], emphasis }: NodeProps) {
  const labelY = subs.length ? y + h / 2 - 6 * (subs.length - 1) - 2 : y + h / 2 + 5;
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
      <text x={x + w / 2} y={labelY} textAnchor="middle" fontSize={emphasis ? 16 : 14} fontWeight={700} fill="#1a1a1a">
        {label}
      </text>
      {subs.map((s, i) => (
        <text key={i} x={x + w / 2} y={labelY + 16 + i * 14} textAnchor="middle" fontSize={11} fill="#5a5a5a">
          {s}
        </text>
      ))}
    </g>
  );
}

function Arrow({ d, dashed }: { d: string; dashed?: boolean }) {
  return (
    <path
      d={d}
      fill="none"
      stroke="#1a1a1a"
      strokeWidth={1.8}
      strokeDasharray={dashed ? "6 5" : undefined}
      markerEnd="url(#ah)"
    />
  );
}

export default function ArAvDiagram() {
  const colX = 95;
  const colW = 290;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={svgStyle} role="img"
      aria-label="GR00T-NLA full process: Cosmos Reason2 VLM, Layer 16 with 143 tokens, keep 129 into AV, verbal text, AR to a 128 by 2048 vector, recombine and inject to steer.">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a" />
        </marker>
      </defs>

      {/* nodes */}
      <Box x={55} y={20} w={510} h={58} label="Cosmos Reason2 VLM · 2B  (Qwen3-VL backbone)"
        subs={["28 decoder layers · hidden 2048 · 16 heads (8 KV) · SwiGLU 6144 · MRoPE"]} />
      <Box x={55} y={112} w={510} h={58} label="Layer 16"
        subs={["VLM to DiT action head · 143 tokens at this layer"]} />

      <Box x={colX} y={210} w={colW} h={66} emphasis label="h · 128 image patches + 1 last text"
        subs={["(1, 129, 2048)"]} />
      <Box x={colX} y={318} w={colW} h={54} emphasis label="AV" subs={["activation verbalizer"]} />
      <Box x={colX} y={414} w={colW} h={66} label="verbal text" subs={["scene · target · spatial · plan · task"]} />
      <Box x={colX} y={522} w={colW} h={54} emphasis label="AR" subs={["activation reconstructor"]} />
      <Box x={colX} y={618} w={colW} h={66} emphasis label="ĥ · (1, 128, 2048)" subs={["image patch grid only"]} />

      <Box x={55} y={726} w={510} h={66} label="inject into policy  →  steer"
        subs={["128 from AR  +  14 dropped  +  1 last text   recombine = 143"]} />

      {/* main flow arrows */}
      <Arrow d={`M ${CX} 78 L ${CX} 112`} />
      <Arrow d={`M ${CX} 170 L ${CX} 210`} />
      <Arrow d={`M ${CX} 276 L ${CX} 318`} />
      <Arrow d={`M ${CX} 372 L ${CX} 414`} />
      <Arrow d={`M ${CX} 480 L ${CX} 522`} />
      <Arrow d={`M ${CX} 576 L ${CX} 618`} />
      <Arrow d={`M ${CX} 684 L ${CX} 726`} />

      {/* bypass lane: dropped 14 + last text skip the codec and rejoin at inject */}
      <path d={`M 470 158 L 540 158 L 540 716 L 470 716`} fill="none" stroke="#9a9a9a" strokeWidth={1.6} strokeDasharray="6 5" markerEnd="url(#ah)" />
      <g transform="translate(548, 360)">
        <text x={0} y={0} fontSize={11} fontWeight={700} fill="#5a5a5a" transform="rotate(90)">
          14 dropped + last text bypass
        </text>
        <text x={0} y={14} fontSize={10} fill="#9a9a9a" transform="rotate(90)">
          template · instruction · vision markers
        </text>
      </g>

    </svg>
  );
}
