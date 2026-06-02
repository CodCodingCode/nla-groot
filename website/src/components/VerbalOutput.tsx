/**
 * The verbal output of AV: the structured caption schema (v5/v6), as a chart.
 * What each field means, plus a real example, plus which channel carries what.
 */

import { CSSProperties } from "react";

const card: CSSProperties = {
  border: "1.4px solid #1a1a1a",
  borderRadius: 10,
  padding: "1.1rem 1.25rem",
  maxWidth: 620,
  margin: "0 auto",
  background: "#ffffff",
};
const head: CSSProperties = { fontSize: 13, fontWeight: 700, color: "#5a5a5a", margin: "0 0 0.6rem" };
const row: CSSProperties = { display: "flex", gap: "0.75rem", padding: "0.32rem 0", borderTop: "1px solid #ececea" };
const field: CSSProperties = { flex: "0 0 84px", fontWeight: 700, fontFamily: "ui-monospace, monospace", fontSize: 14 };
const desc: CSSProperties = { color: "#3a3a3a", fontSize: 14 };
const example: CSSProperties = {
  marginTop: "0.9rem",
  background: "#f6f7f6",
  borderRadius: 8,
  padding: "0.8rem 0.9rem",
  fontFamily: "ui-monospace, monospace",
  fontSize: 12.5,
  lineHeight: 1.55,
  color: "#1a1a1a",
  whiteSpace: "pre-wrap",
};
const noteWrap: CSSProperties = { display: "flex", gap: "0.75rem", marginTop: "0.9rem", fontSize: 12.5, color: "#5a5a5a" };
const chip: CSSProperties = { flex: 1, border: "1px solid #ececea", borderRadius: 8, padding: "0.55rem 0.7rem" };

const FIELDS: [string, string][] = [
  ["scene", "what is visible at this token: surfaces, objects, lighting"],
  ["target", "the object or region this token attends to (patch local, not the whole task)"],
  ["spatial", "relative layout in the frame, or NA"],
  ["plan", "ONE imminent motion as “phase: detail”, or NA"],
  ["task", "the full language instruction (v6 addition)"],
];

const EXAMPLE =
  "scene: Wood tabletop under a robotic arm; a black bowl with a patterned\n" +
  "       interior sits near a box and a plate on the left.\n" +
  "target: the black bowl rim and interior pattern in the close up patch\n" +
  "spatial: the bowl fills the left of the patch; the box edge is lower right\n" +
  "plan: NA\n" +
  "task: pick up the black bowl next to the cookie box and place it on the plate";

export default function VerbalOutput() {
  return (
    <div style={card}>
      <p style={head}>WHAT AV WRITES (verbal output schema)</p>
      {FIELDS.map(([f, d]) => (
        <div key={f} style={row}>
          <div style={field}>{f}</div>
          <div style={desc}>{d}</div>
        </div>
      ))}

      <div style={example}>{EXAMPLE}</div>

      <div style={noteWrap}>
        <div style={chip}>
          <strong>image patch tokens</strong> describe what the policy sees, so plan is NA
          (a vision token has no motion).
        </div>
        <div style={chip}>
          <strong>last text token</strong> carries the intent, so plan becomes a real
          “phase: detail” (approach, reach, grasp, lift, place, ...).
        </div>
      </div>
    </div>
  );
}
