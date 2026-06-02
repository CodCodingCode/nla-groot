import { CSSProperties } from "react";
import ArAvDiagram from "./components/ArAvDiagram";
import VerbalOutput from "./components/VerbalOutput";
import ThreeAxisScores from "./components/ThreeAxisScores";

const NLA_PAPER = "https://transformer-circuits.pub/2026/nla/index.html#introduction";
const GITHUB = "https://github.com/CodCodingCode/nla-groot";
const CONFIG_L16 =
  "https://github.com/CodCodingCode/nla-groot/blob/main/checkpoints/GR00T-N1.7-LIBERO/libero_goal/config.json#L63";

const page: CSSProperties = {
  maxWidth: 760,
  margin: "0 auto",
  padding: "3rem 1.5rem 4rem",
  lineHeight: 1.65,
  fontSize: 17,
  color: "#1a1a1a",
};
const h1: CSSProperties = { fontSize: 30, fontWeight: 800, margin: "0 0 1.75rem" };
const list: CSSProperties = { margin: "0.4rem 0 1.2rem 1.4rem", padding: 0 };
const fig: CSSProperties = { margin: "2.25rem 0 0.5rem" };
const cap: CSSProperties = { fontSize: 13, color: "#5a5a5a", textAlign: "center", margin: "0 0 1rem" };
const axis: CSSProperties = {
  borderLeft: "3px solid #1a1a1a",
  padding: "0.15rem 0 0.15rem 0.85rem",
  margin: "0.9rem 0",
};
const axisName: CSSProperties = { fontWeight: 700 };
const foot: CSSProperties = { color: "#5a5a5a", fontSize: 16, marginTop: "2.5rem" };

export default function App() {
  return (
    <main style={page}>
      <h1 style={h1}>
        GR00T-NLA: a bidirectional natural language bridge for VLA activations
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
        GR00T-N1.7's vision language backbone is Cosmos
        Reason2, a 2B VLM trained from Qwen3-VL 2B. The VLM backbone has 28 decoder layers, a
        hidden size 2048, and 16 attention heads. We read{" "}
        <strong>Layer 16</strong>, the layer whose output is handed to the action head (the VLM
        to DiT encoder connection;{" "}
        <a href={CONFIG_L16} target="_blank" rel="noreferrer">
          config.json L63
        </a>
        ).
      </p>

      <p>At Layer 16 the sequence is 143 tokens. We keep 129 of them:</p>
      <ul style={list}>
        <li>the 128 image patch tokens</li>
        <li>the 1 last text token</li>
      </ul>
      <p>We drop the other 14:</p>
      <ul style={list}>
        <li>the system and chat template</li>
        <li>the language instruction</li>
        <li>the vision markers</li>
      </ul>

      <p>
        The activation verbalizer
        (AV) reads those 129 vectors, shape (1, 129, 2048), and writes a structured English
        caption. The activation reconstructor (AR) reads that caption and outputs (1, 128,
        2048), which goes back only into the image patch slots. The 14 dropped tokens and the 1
        last text token pass through unchanged and recombine with AR's output into the full 143
        token sequence the policy runs forward.
      </p>

      <figure style={fig}>
        <ArAvDiagram />
        <figcaption style={cap}>
          The full process. Keep 129 into AV, verbalize, reconstruct to (1, 128, 2048),
          recombine with the bypassed tokens, inject, and steer.
        </figcaption>
      </figure>

      <figure style={fig}>
        <VerbalOutput />
        <figcaption style={cap}>
          What AV writes: a structured caption. Image patch tokens describe what is seen; the
          last text token carries the intended motion.
        </figcaption>
      </figure>

      <p>
        With supervised training alone the round trip reaches a
        cosine similarity of <strong>0.85</strong>. We check the codec on three axes, and it
        has to pass all three.
      </p>

      <div style={axis}>
        <span style={axisName}>1. Reconstruction quality:</span> Run the full round trip,
        activation to caption to vector, and ask whether the rebuilt vector ĥ matches the
        original h. We score it with cosine similarity, currently 0.85.
      </div>
      <div style={axis}>
        <span style={axisName}>2. Intent specificity:</span> If the scene is the same, but the
        caption is changed, do we get two different captions and two different vectors, or the
        same verbal text? This tests the generalizability of AV as a verbal reconstructor.
        Currently, captions for different goals share only 22 percent of their content, so they
        are genuinely specific.
      </div>
      <div style={axis}>
        <span style={axisName}>3. Causal steering:</span> If we inject an activation into the
        policy mid rollout, does its behavior change to complete the steered task? If we inject
        the wrong goal does it do nothing? This tests the full AV and AR reconstruction
        process, seeing if the end to end sequence can be steered using cached vectors.
      </div>

      <figure style={fig}>
        <ThreeAxisScores />
        <figcaption style={cap}>
          Three axis evaluation (supervised only checkpoint, held out n=100). Causal steering:
          matched &gt; no steer &gt; mismatched.
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
