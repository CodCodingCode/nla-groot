export default function Repro() {
  return (
    <section id="repro">
      <h2>Reproduce the numbers</h2>
      <p>
        Every chart on this page is generated from a JSON snapshot built
        from the actual run artifacts under <code>data/</code>. The full
        recipe is in the repo&apos;s README; the abbreviated commands here
        produce the exact metrics shown on this page.
      </p>

      <h3>1. SFT (combined-mode + intent-conditioned + decomposed loss)</h3>
      <pre>
        <code>
{`PYTHONPATH=src python scripts/training/run_sft.py \\
  --recipe v7 \\
  --activations-root data/activations/libero_4suite_v4_combined \\
  --labels-jsonl     data/labels/libero_4suite_v6_with_task/labels.jsonl \\
  --output-dir       data/sft/v9_combined_12k \\
  --stats-json       data/activations/libero_4suite_v4_combined/stats.json \\
  --ar-nce-hard-negative-index-path \\
      data/activations/libero_4suite_v4_combined/hard_negatives_v5.jsonl \\
  --ar-spatial-n-positions 128 \\
  --image-patch-pooling-strided-k 128 \\
  --av-num-image-slots 128 \\
  --combine-positions \\
  --av-intent-conditioned \\
  --ar-layers 0 \\
  --ar-loss-mode decomposed \\
  --ar-scale-weight 0.1 \\
  --num-workers 8 \\
  --action-consistency-every-n-steps 2 \\
  --total-steps 12000 \\
  --eval-every 600 \\
  --save-every 1200 \\
  --max-val-items 512 \\
  --wandb-project nla-groot \\
  --wandb-run-name v9_combined_12k`}
        </code>
      </pre>
      <p>
        Closed-loop eval at every <code>eval_every</code> writes one row
        per metric to <code>metrics.jsonl</code> and a parallel{" "}
        <code>best_av/</code> + <code>best_ar/</code> snapshot whenever the
        configured peak metric (default <code>closed_greedy/cosine</code>)
        is reached.
      </p>

      <h3>2. Steer server</h3>
      <pre>
        <code>
{`scripts/eval/launch_steer_server.sh \\
  --sft-dir data/sft/v9_combined_12k \\
  -- \\
  --embodiment-tag LIBERO_PANDA \\
  --steer-text-file scripts/eval/default_steer_boot.txt \\
  --placement image_patch_all`}
        </code>
      </pre>

      <h3>3. Caption diagnostic (Axis 2)</h3>
      <pre>
        <code>
{`PYTHONPATH=src python scripts/eval/av_caption_intent_diff.py \\
  --sft-dir data/sft/v9_combined_12k \\
  --activations-root data/activations/libero_4suite_v4_combined \\
  --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
  --n-samples 10 \\
  --out-json data/eval/v9_combined_12k_paired_captions.json`}
        </code>
      </pre>

      <h3>4. Counterfactual sim eval (Axis 3)</h3>
      <pre>
        <code>
{`PYTHONPATH=src python scripts/eval/compare_cf_steer_checkpoints.py \\
  --sft-dir data/sft/v9_combined_12k \\
  --grpo-av-dir data/sft/v9_combined_12k/av \\
  --pairs-path data/grpo/libero_goal_counterfactual_pairs_cfonly.jsonl \\
  --activations-root data/activations/libero_4suite_v4_combined \\
  --n-samples 32 \\
  --conditions sft_av \\
  --intent-arms matched,mismatched_source \\
  --causal-arms semantic,no_steer \\
  --sim-placement image_patch_strided --strided-k 128 \\
  --eval-protocol language_swap \\
  --sim-cache-path data/eval/sim_rollout_cache.jsonl \\
  --out-json data/eval/v9_combined_12k_cf_strided_cached.json`}
        </code>
      </pre>

      <h3>5. Re-export this site&apos;s data snapshot</h3>
      <pre>
        <code>
{`# Re-read SFT log + caption diag + CF JSON, regenerate snapshot.json
python scripts/website/export_site_data.py \\
  --run-name v9_combined_12k \\
  --sft-log data/sft/v9_combined_12k_launch.log \\
  --captions-json data/eval/v9_combined_12k_paired_captions.json \\
  --cf-json data/eval/v9_combined_12k_cf_strided_cached.json

# Rebuild the static site
cd website && npm install && npm run build`}
        </code>
      </pre>

      <p>
        The full auto-pipeline (caption diag → steer server → CF eval →
        W&amp;B consolidation → snapshot regen) is wrapped by{" "}
        <code>scripts/eval/auto_cf_eval_after_sft.sh</code>. Launch it
        detached after kicking off SFT and the eval will fire automatically
        when <code>SFT done</code> appears in the training log.
      </p>

      <h3>Where the numbers come from</h3>
      <ul>
        <li>
          Codec metrics: parsed from{" "}
          <code>data/sft/v9_combined_12k_launch.log</code>{" "}
          <code>[step N] val ...</code> and <code>[final] val ...</code>{" "}
          lines.
        </li>
        <li>
          Caption diagnostic:{" "}
          <code>data/eval/v9_combined_12k_paired_captions.json</code>{" "}
          produced by <code>av_caption_intent_diff.py</code>.
        </li>
        <li>
          CF eval summary:{" "}
          <code>data/eval/v9_combined_12k_cf_strided_cached.json</code>{" "}
          produced by <code>compare_cf_steer_checkpoints.py</code>.
        </li>
        <li>
          W&amp;B run for live monitoring: project <code>nla-groot</code>{" "}
          on Weights &amp; Biases.
        </li>
      </ul>
    </section>
  );
}
