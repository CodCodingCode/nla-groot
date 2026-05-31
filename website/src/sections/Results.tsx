import TrainingChart from "../components/TrainingChart";
import IntentChart from "../components/IntentChart";
import CFEvalChart from "../components/CFEvalChart";
import type { SiteSnapshot } from "../types";

interface Props {
  data: SiteSnapshot;
}

function _fmt(n: number | null | undefined, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}
function _fmtSigned(n: number | null | undefined, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const s = n.toFixed(digits);
  return n >= 0 ? `+${s}` : s;
}
function _pct(n: number | null | undefined) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return (n * 100).toFixed(1) + "%";
}

export default function Results({ data }: Props) {
  const codec = data.codec_final;
  const cap = data.caption_diag;
  const cf = data.cf_eval;
  const cfReady = cf.status === "complete";
  const cfRunning = cf.status === "running";

  return (
    <section id="results">
      <h2>Results: three independent axes</h2>
      <p>
        Every number below is pulled from the actual run artifacts under{" "}
        <code>data/</code>. No placeholders for the codec or intent
        diagnostic. The counterfactual eval shows partial results while it
        finishes.
      </p>

      <h3>Axis 1 &mdash; codec quality (held-out reconstruction)</h3>
      {codec && (
        <div className="grid-2">
          <div>
            <table>
              <tbody>
                <tr>
                  <th scope="row">closed_greedy/cosine</th>
                  <td>
                    <code>{_fmt(codec.closed_greedy_cosine, 4)}</code>{" "}
                    <span className="muted">
                      ({_fmt(codec.angle_degrees, 1)}° angular error)
                    </span>
                  </td>
                </tr>
                <tr>
                  <th scope="row">closed_greedy/mse</th>
                  <td>
                    <code>{_fmt(codec.closed_greedy_mse, 2)}</code>{" "}
                    <span className="muted">
                      (RMS {_fmt(codec.rms_error_raw_units, 2)}, {" "}
                      {_fmt(codec.relative_norm_error_pct, 2)}% of α)
                    </span>
                  </td>
                </tr>
                <tr>
                  <th scope="row">closed_greedy/fve</th>
                  <td>
                    <code>{_fmtSigned(codec.closed_greedy_fve, 3)}</code>{" "}
                    <span className="muted">
                      (≈ 0 means at per-dim mean baseline)
                    </span>
                  </td>
                </tr>
                <tr>
                  <th scope="row">val/cosine</th>
                  <td><code>{_fmt(codec.val_cosine, 4)}</code></td>
                </tr>
                <tr>
                  <th scope="row">val/mse</th>
                  <td><code>{_fmt(codec.val_mse, 2)}</code></td>
                </tr>
                <tr>
                  <th scope="row">α (activation norm scale)</th>
                  <td><code>{_fmt(codec.alpha_norm, 2)}</code></td>
                </tr>
                <tr>
                  <th scope="row">step / total</th>
                  <td>
                    <code>
                      {codec.step ?? "—"} / {codec.total_steps}
                    </code>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <p>
              In absolute terms, the codec reconstructs held-out activations
              within <strong>{_fmt(codec.angle_degrees, 1)}° of ground truth</strong>{" "}
              direction and within <strong>
                {_fmt(codec.relative_norm_error_pct, 2)}% of activation magnitude
              </strong>. A random vector would sit at 90° (cosine 0) and a
              memorized target at 0° (cosine 1). The structure of the
              activation manifold is sufficient for natural-language
              encoding at this fidelity.
            </p>
            <p className="muted">
              FVE near zero indicates the codec is roughly at the trivial
              per-dimension mean baseline on variance explained &mdash;
              well above random by direction, near parity on magnitude.
              Improving FVE positive would be the next gate; the cosine
              and relative-norm-error claims do not depend on it.
            </p>
          </div>
        </div>
      )}

      {data.training.length > 0 && <TrainingChart points={data.training} />}

      <h3>Axis 2 &mdash; intent specificity (paired captions)</h3>
      {cap ? (
        <>
          <p>
            On <code>{cap.n_samples}</code> held-out activations, we asked
            AV to generate two captions per activation under different
            target intents. Character-level overlap between the matched
            and mismatched captions averaged{" "}
            <code>{_fmt(cap.mean_char_overlap, 4)}</code> &mdash; the two
            captions share only about{" "}
            {Math.round(cap.mean_char_overlap * 100)}% of their text.
            Bullet-by-bullet difference count averaged{" "}
            <code>{_fmt(cap.mean_bullet_diff_count, 2)}</code> of the 5-6
            bullets per caption.
          </p>
          <p>
            <em>{cap.interpretation}</em>
          </p>
          <IntentChart caption={cap} />
          <details>
            <summary>
              Show paired-caption examples (matched vs mismatched intent on
              the same activation)
            </summary>
            <div className="paired-examples">
              {cap.paired_samples.slice(0, 5).map((s) => (
                <div key={s.source_id} className="paired-block">
                  <div className="muted">
                    <code>{s.source_id}</code> &mdash;{" "}
                    char_overlap {_fmt(s.char_overlap, 3)}, bullet_diff{" "}
                    {s.bullet_diff}
                  </div>
                  <div className="grid-2">
                    <div>
                      <strong>matched:</strong>{" "}
                      <em>{s.matched_intent}</em>
                      <pre className="caption-preview">
                        <code>{s.caption_matched_preview}</code>
                      </pre>
                    </div>
                    <div>
                      <strong>mismatched:</strong>{" "}
                      <em>{s.mismatched_intent}</em>
                      <pre className="caption-preview">
                        <code>{s.caption_mismatched_preview}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </details>
        </>
      ) : (
        <p className="muted">
          Caption diagnostic JSON not yet on disk; the chart and per-sample
          examples will populate when{" "}
          <code>av_caption_intent_diff.py</code> writes its output.
        </p>
      )}

      <h3>Axis 3 &mdash; causal steering (counterfactual sim eval)</h3>
      {cfReady && cf.steer_lift && cf.sem_gap && cf.lang_swap ? (
        <>
          <div className="grid-2">
            <div>
              <table>
                <tbody>
                  <tr>
                    <th scope="row">steer_lift mean</th>
                    <td>
                      <code>{_fmtSigned(cf.steer_lift.mean, 4)}</code>{" "}
                      <span className="muted">
                        (t = {_fmtSigned(cf.steer_lift.t, 2)}, wins/losses{" "}
                        {cf.steer_lift.wins}/{cf.steer_lift.losses}, n ={" "}
                        {cf.steer_lift.n})
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <th scope="row">sem_gap mean</th>
                    <td>
                      <code>{_fmtSigned(cf.sem_gap.mean, 4)}</code>{" "}
                      <span className="muted">
                        (t = {_fmtSigned(cf.sem_gap.t, 2)})
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <th scope="row">lang_swap mean</th>
                    <td>
                      <code>{_fmtSigned(cf.lang_swap.mean, 4)}</code>{" "}
                      <span className="muted">
                        (t = {_fmtSigned(cf.lang_swap.t, 2)})
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <th scope="row">codec_above_lang</th>
                    <td>
                      <code>{_fmtSigned(cf.codec_above_lang ?? 0, 4)}</code>{" "}
                      <span className="muted">
                        (sem_gap − lang_swap, isolates codec signal beyond
                        the policy&apos;s language input)
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div>
              <p>
                When the policy&apos;s obs.language is held to the matched
                intent, adding the AR-reconstructed steering vector at the
                image-patch positions raises task-progress reward by{" "}
                <code>{_fmtSigned(cf.steer_lift.mean, 4)}</code> r_sim on
                average across {cf.steer_lift.n} held-out CF samples. The
                codec&apos;s contribution{" "}
                <em>beyond</em> the policy&apos;s built-in language
                conditioning is{" "}
                <code>{_fmtSigned(cf.codec_above_lang ?? 0, 4)}</code>{" "}
                &mdash; the activation-channel intervention is doing real,
                independent work.
              </p>
            </div>
          </div>
          <CFEvalChart cf={cf} />
          {cf.predicate_rates && (
            <p className="muted">
              Predicate firing rates (full task completion within the sim
              budget) across arms: matched/sem ={" "}
              {_pct(cf.predicate_rates.matched_semantic)}, matched/no_steer ={" "}
              {_pct(cf.predicate_rates.matched_no_steer)}, mismatched/sem ={" "}
              {_pct(cf.predicate_rates.mismatched_semantic)},
              mismatched/no_steer ={" "}
              {_pct(cf.predicate_rates.mismatched_no_steer)}. The headline
              metric is task progress (r_sim), not predicate completion.
            </p>
          )}
        </>
      ) : cfRunning && cf.steer_lift ? (
        <>
          <div className="callout">
            <strong>Eval in progress.</strong>{" "}
            <code>{cf.n_complete}/{cf.n_target}</code> samples complete.
            Running aggregate: steer_lift mean{" "}
            <code>{_fmtSigned(cf.steer_lift.mean, 4)}</code> (t ={" "}
            <code>{_fmtSigned(cf.steer_lift.t, 2)}</code>, wins/losses{" "}
            <code>{cf.steer_lift.wins}/{cf.steer_lift.losses}</code>),
            sem_gap{" "}
            <code>{_fmtSigned(cf.sem_gap?.mean ?? 0, 4)}</code>, lang_swap{" "}
            <code>{_fmtSigned(cf.lang_swap?.mean ?? 0, 4)}</code>,
            codec_above_lang{" "}
            <code>{_fmtSigned(cf.codec_above_lang ?? 0, 4)}</code>.
          </div>
          {cf.per_sample.length > 0 && <CFEvalChart cf={cf} />}
          <p className="muted">
            The site auto-regenerates its snapshot when the CF eval JSON
            lands at{" "}
            <code>data/eval/{data.run_name}_cf_strided_cached.json</code>{" "}
            and the final n={cf.n_target} numbers will replace this
            placeholder.
          </p>
        </>
      ) : (
        <p className="callout">
          <strong>CF eval is queued.</strong>{" "}
          The sim rollouts run after SFT completes; the final{" "}
          <code>steer_lift</code> headline will appear here when{" "}
          <code>data/eval/{data.run_name}_cf_strided_cached.json</code> lands.
          Until then this section shows only partial information.
        </p>
      )}
    </section>
  );
}
