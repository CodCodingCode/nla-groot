import { CSSProperties } from "react";
import ArAvDiagram from "./components/ArAvDiagram";
import ThreeAxisScores from "./components/ThreeAxisScores";

const page: CSSProperties = {
  maxWidth: 760,
  margin: "0 auto",
  padding: "3rem 1.5rem 4rem",
  lineHeight: 1.65,
  fontSize: 17,
  color: "#1a1a1a",
};

const h1: CSSProperties = { fontSize: 30, fontWeight: 800, margin: "0 0 0.25rem" };
const lede: CSSProperties = { color: "#5a5a5a", fontSize: 16, margin: "0 0 2.5rem" };
const h2: CSSProperties = {
  fontSize: 14,
  fontWeight: 700,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "#5a5a5a",
  margin: "2.5rem 0 0.5rem",
};
const fig: CSSProperties = { margin: "2rem 0 0.5rem" };
const cap: CSSProperties = { fontSize: 13, color: "#5a5a5a", textAlign: "center", margin: "0 0 1rem" };

export default function App() {
  return (
    <main style={page}>
      <h1 style={h1}>A bidirectional natural-language bridge for VLA activations</h1>
      <p style={lede}>
        Reading — and writing — GR00T-N1.7's internal state in plain English.
      </p>

      <h2 style={h2}>Abstract</h2>
      <p>
        Vision-language-action (VLA) models now drive capable robot policies, yet
        they remain black boxes: we can watch what they do but not inspect the
        internal state that produces it. Prior interpretability work — linear
        probes, sparse autoencoders, dictionary learning — can{" "}
        <em>decode</em> what a layer represents, but only in one direction: it
        reads activations out and never writes them back. We take a different
        route. Instead of merely describing the hidden state, we build a{" "}
        <em>bidirectional</em> codec whose bottleneck is natural language, then
        close the loop — we reconstruct an activation from its English caption and
        inject it back into the live policy. This lets us ask a question read-only
        methods cannot: is the language we recover <em>causally sufficient</em> to
        steer behavior? We run it on a VLA policy and verify the bridge causally,
        not just descriptively.
      </p>

      <h2 style={h2}>Summary</h2>
      <p>
        Two small models form the bridge. An{" "}
        <strong>activation verbalizer (AV)</strong> reads the final-layer hidden
        state of GR00T-N1.7 and renders it as a structured English caption — what
        the policy is attending to and what it intends to do. An{" "}
        <strong>activation reconstructor (AR)</strong> inverts that caption back
        into a vector in the backbone's activation space. Together they let us
        look inside the policy's "thinking," put it into words, and turn those
        words back into a usable activation that can be injected to steer the task.
      </p>

      <figure style={fig}>
        <ArAvDiagram />
        <figcaption style={cap}>
          The codec. AV maps an activation to English; AR maps English back to a
          vector; the reconstruction is injected at the policy's image-patch
          tokens to steer behavior.
        </figcaption>
      </figure>

      <h2 style={h2}>Current findings</h2>
      <p>
        With supervised training alone — no reinforcement learning — the codec
        round-trips an activation at a cosine similarity of{" "}
        <strong>0.85</strong>. We evaluate it along three independent axes, and a
        codec must clear all three to count as a faithful, causal bridge:
        reconstruction quality, intent specificity, and causal steering. The
        causal axis is the decisive one — injecting the <em>matched</em> intent
        raises task-progress reward above the unsteered baseline, while a{" "}
        <em>mismatched</em> intent drops it below, showing the recovered vector
        carries specific, behaviorally-effective meaning rather than a generic
        prior.
      </p>

      <figure style={fig}>
        <ThreeAxisScores />
        <figcaption style={cap}>
          Three-axis evaluation (supervised-only checkpoint, held-out n=100).
          Causal steering: matched &gt; no-steer &gt; mismatched.
        </figcaption>
      </figure>

      <p style={{ ...lede, marginTop: "2.5rem", marginBottom: 0 }}>
        <a href="https://github.com/CodCodingCode/nla-groot" target="_blank" rel="noreferrer">
          Source on GitHub
        </a>
      </p>
    </main>
  );
}
