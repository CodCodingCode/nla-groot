export default function Repro() {
  return (
    <section id="repro">
      <h2>Reproduce the numbers</h2>
      <p>
        Every chart on this page is generated from a JSON snapshot built
        from the actual run artifacts under <code>data/</code>. To rebuild
        the snapshot after re-running evals:
      </p>
      <pre>
        <code>
{`# 1. Refresh evals (see paper/repro/canonical_commands.sh)
PYTHONPATH=src python scripts/eval/build_v3_scorecard.py \\
  --ckpt-dir data/sft/libero_4suite_v3 \\
  --out-json data/sft/libero_4suite_v3/v3_scorecard.json

# 2. Refresh the website data snapshot
python scripts/website/export_site_data.py

# 3. Rebuild the static site
cd website && npm install && npm run build`}
        </code>
      </pre>

      <p>
        Build output goes to <code>website/dist/</code>; the bundle is
        relocatable (Vite <code>base: &quot;./&quot;</code>) so you can
        serve it from any path, including a GitHub Pages subpath. See{" "}
        <code>website/README.md</code> for the longer runbook.
      </p>

      <h3>Where the numbers come from</h3>
      <ul>
        <li>
          <code>data/sft/libero_4suite_v3/v3_scorecard.json</code> — V3
          scorecard, produced by{" "}
          <code>scripts/eval/build_v3_scorecard.py</code>.
        </li>
        <li>
          <code>data/eval/steerability_v1_vs_v3/av_metrics.json</code> —
          gold vs. AV judge means, produced by{" "}
          <code>scripts/eval/steerability_eval.py</code> (AV fidelity arm).
        </li>
        <li>
          <code>data/eval/steerability_v1_vs_v3/metrics.json</code> — sim
          rollout metrics, produced by the same script.
        </li>
        <li>
          <code>data/sft/libero_4suite_v3/metrics.jsonl</code> — SFT
          training/validation metrics from <code>scripts/training/run_sft.py</code>.
        </li>
      </ul>
    </section>
  );
}
