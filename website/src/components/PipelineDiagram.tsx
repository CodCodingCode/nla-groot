/**
 * Pipeline diagram for nla-groot.
 *
 * Solid arrows  = supervised training signal (h, gold caption -> CE / MSE).
 * Dashed arrows = inference / steering path (AR(text) injected back).
 *
 * Mirrors the LaTeX/TikZ figure in paper/figures/system_overview.tex but is
 * authored as inline SVG so it remains responsive and theme-aware.
 */

import { CSSProperties } from "react";

const W = 880;
const H = 460;

const style: CSSProperties = {
  width: "100%",
  height: "auto",
  maxWidth: 880,
  display: "block",
  margin: "0.5rem auto",
};

interface BoxProps {
  x: number;
  y: number;
  w?: number;
  h?: number;
  label: string;
  sub?: string;
  emphasis?: boolean;
}

function Box({ x, y, w = 130, h = 56, label, sub, emphasis }: BoxProps) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={6}
        ry={6}
        fill={emphasis ? "#f0f0ec" : "#ffffff"}
        stroke="#1a1a1a"
        strokeWidth={1}
      />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 4 : y + h / 2 + 5}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill="#1a1a1a"
      >
        {label}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 14}
          textAnchor="middle"
          fontSize={11}
          fill="#5a5a5a"
        >
          {sub}
        </text>
      )}
    </g>
  );
}

interface ArrowProps {
  d: string;
  dashed?: boolean;
  label?: string;
  labelX?: number;
  labelY?: number;
}

function Arrow({ d, dashed, label, labelX, labelY }: ArrowProps) {
  return (
    <g>
      <path
        d={d}
        fill="none"
        stroke="#1a1a1a"
        strokeWidth={1.6}
        strokeDasharray={dashed ? "5 4" : undefined}
        markerEnd="url(#arrowhead)"
      />
      {label && labelX !== undefined && labelY !== undefined && (
        <g>
          <rect
            x={labelX - label.length * 3.6}
            y={labelY - 9}
            width={label.length * 7.2}
            height={14}
            fill="#fbfbfa"
          />
          <text
            x={labelX}
            y={labelY + 1}
            fontSize={11}
            textAnchor="middle"
            fill="#1a1a1a"
          >
            {label}
          </text>
        </g>
      )}
    </g>
  );
}

export default function PipelineDiagram() {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={style}
      role="img"
      aria-label="Pipeline diagram showing extraction, training, and steering paths."
    >
      <defs>
        <marker
          id="arrowhead"
          viewBox="0 0 10 10"
          refX={9}
          refY={5}
          markerWidth={7}
          markerHeight={7}
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#1a1a1a" />
        </marker>
      </defs>

      {/* Top row: extraction */}
      <Box x={20} y={28} label="Camera frame" sub="+ instruction" />
      <Box x={210} y={28} label="GR00T" sub="backbone" />
      <Box x={400} y={20} w={150} h={70} label="h ∈ ℝ²⁰⁴⁸" sub="layer 16" emphasis />

      {/* Teacher row */}
      <Box x={210} y={120} label="Multimodal teacher" sub="(GPT)" />
      <Box x={400} y={120} w={150} label="Gold caption" sub="(bullets)" />

      {/* Middle row: AV / AR */}
      <Box x={120} y={250} w={150} h={70} label="AV" sub="LoRA Qwen3-4B" emphasis />
      <Box x={330} y={258} w={110} label="y (text)" />
      <Box x={490} y={250} w={150} h={70} label="AR" sub="LoRA Qwen3-4B" emphasis />
      <Box x={690} y={258} w={150} h={56} label="ĥ" sub="reconstruction" emphasis />

      {/* Bottom row: deployment */}
      <Box x={490} y={380} w={150} label="GR00T policy server" />
      <Box x={690} y={380} w={150} label="LIBERO sim" />

      {/* --- Extraction edges (solid) --- */}
      <Arrow d="M 150 56 L 210 56" />
      <Arrow d="M 340 56 L 400 56" />
      {/* frame -> teacher */}
      <Arrow d="M 80 56 Q 80 145 210 145" />
      <Arrow d="M 340 145 L 400 145" />

      {/* h -> AV (CE supervision uses gold caption coming from the right) */}
      <Arrow
        d="M 460 90 Q 460 180 195 250"
        label="inject h"
        labelX={300}
        labelY={195}
      />
      {/* gold -> AV (CE on y_star) */}
      <Arrow
        d="M 475 165 Q 360 200 240 250"
        label="CE on y★"
        labelX={365}
        labelY={235}
      />

      {/* gold -> AR (training) */}
      <Arrow
        d="M 550 145 Q 600 200 565 250"
        label="text → AR"
        labelX={605}
        labelY={195}
      />
      {/* AR -> hhat */}
      <Arrow d="M 640 285 L 690 285" />
      {/* hhat -> h MSE (back up to h) */}
      <Arrow
        d="M 765 250 Q 765 140 555 60"
        label="MSE in h/α"
        labelX={770}
        labelY={170}
      />

      {/* --- Inference / steering (dashed) --- */}
      {/* AV -> y -> AR */}
      <Arrow d="M 270 290 L 330 290" dashed />
      <Arrow d="M 440 290 L 490 290" dashed />
      {/* AR -> server -> sim */}
      <Arrow
        d="M 565 320 L 565 380"
        dashed
        label="inject in backbone"
        labelX={620}
        labelY={355}
      />
      <Arrow d="M 640 408 L 690 408" dashed />

      {/* Legend */}
      <g transform={`translate(${W - 240}, 20)`}>
        <rect
          x={0}
          y={0}
          width={228}
          height={56}
          fill="#ffffff"
          stroke="#e5e5e2"
        />
        <line x1={12} y1={20} x2={48} y2={20} stroke="#1a1a1a" strokeWidth={1.6} />
        <text x={56} y={24} fontSize={11} fill="#1a1a1a">
          training (CE, MSE)
        </text>
        <line
          x1={12}
          y1={42}
          x2={48}
          y2={42}
          stroke="#1a1a1a"
          strokeWidth={1.6}
          strokeDasharray="5 4"
        />
        <text x={56} y={46} fontSize={11} fill="#1a1a1a">
          inference / steering
        </text>
      </g>
    </svg>
  );
}
