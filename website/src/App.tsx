import { CSSProperties } from "react";
import ArAvDiagram from "./components/ArAvDiagram";
import ThreeAxisScores from "./components/ThreeAxisScores";

const NLA_PAPER = "https://transformer-circuits.pub/2026/nla/index.html#introduction";
const GITHUB = "https://github.com/CodCodingCode/nla-groot";

const page: CSSProperties = {
  maxWidth: 760,
  margin: "0 auto",
  padding: "3rem 1.5rem 4rem",
  lineHeight: 1.65,
  fontSize: 17,
  color: "#1a1a1a",
};
const h1: CSSProperties = { fontSize: 30, fontWeight: 800, margin: "0 0 1.75rem" };
const ul: CSSProperties = { margin: "0.5rem 0 1.1rem 1.25rem", padding: 0 };
const fig: CSSProperties = { margin: "2.25rem 0 0.5rem" };
const cap: CSSProperties = { fontSize: 13, color: "#5a5a5a", textAlign: "center", margin: "0 0 1rem" };
const foot: CSSProperties = { color: "#5a5a5a", fontSize: 16, marginTop: "2.5rem" };

export default function App() {
  return (
    <main style={page}>
      <h1 style={h1}>
        GR00T-NLA: a bidirectional natural-language bridge for VLA activations
      </h1>

      <p>
        VLA systems can drive robot policies to perform a range of actions, but they remain
        less interpretable than we'd like. Previous interpretability work, like linear probes
        and SAEs, can decode layers, but can't write activations back to prove the findings
        are right. GR00T-NLA changes this by using{" "}
        <a href={NLA_PAPER} target="_blank" rel="noreferrer">
          natural language autoencoders
        </a>{" "}
        to reconstruct an activation from its English caption and inject it back into the live
        policy. By providing bidirectionality, we can verbalize an activation and prove the
        interpretation is correct.
      </p>

      <p>
        Two small models form the bridge. An <strong>activation verbalizer (AV)</strong> reads
        the final-layer hidden state of GR00T-N1.7 and renders it as a structured English
        caption containing:
      </p>
      <ul style={ul}>
        <li>what the policy is attending to</li>
        <li>what it intends to do</li>
      </ul>
      <p>
        An <strong>activation reconstructor (AR)</strong> inverts that caption back into a
        vector in the backbone's activation space. Together they let anyone look inside the
        policy's "thinking," put it into words, and turn those words back into a usable
        activation that can be injected to steer the task.
      </p>

      <figure style={fig}>
        <ArAvDiagram />
        <figcaption style={cap}>
          The codec. AV maps an activation to English; AR maps English back to a vector; the
          reconstruction is injected at the policy's image-patch tokens to steer behavior.
        </figcaption>
      </figure>

      <p>
        With supervised training alone, the round-trip reaches a cosine similarity of{" "}
        <strong>0.85</strong>. We evaluate the model along three axes to ensure quality:
      </p>
      <ul style={ul}>
        <li>
          <strong>Reconstruction quality</strong>: if we run the round trip, how does ĥ
          compare to h?
        </li>
        <li>
          <strong>Intent specificity</strong>: how different are the generated captions for
          intent A versus intent B?
        </li>
        <li>
          <strong>Causal steering</strong>: if we inject an activation into the policy
          mid-rollout, does its behavior change to complete the steered task?
        </li>
      </ul>

      <figure style={fig}>
        <ThreeAxisScores />
        <figcaption style={cap}>
          Three-axis evaluation (supervised-only checkpoint, held-out n=100). Causal steering:
          matched &gt; no-steer &gt; mismatched.
        </figcaption>
      </figure>

      <p style={foot}>
        <a href={NLA_PAPER} target="_blank" rel="noreferrer">
          NLA paper (Transformer Circuits, 2026)
        </a>{" "}
        ·{" "}
        <a href={GITHUB} target="_blank" rel="noreferrer">
          Source on GitHub
        </a>
      </p>
    </main>
  );
}
