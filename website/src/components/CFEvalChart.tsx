import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CFEvalResult } from "../types";

interface Props {
  cf: CFEvalResult;
}

function _signedColour(v: number) {
  if (v > 0) return "#1f7a3b"; // green for positive lift
  if (v < 0) return "#a93232"; // red for negative
  return "#888";
}

export default function CFEvalChart({ cf }: Props) {
  if (!cf.steer_lift || !cf.sem_gap || !cf.lang_swap) return null;

  const aggregateData = [
    {
      label: "steer_lift",
      sub: "matched/sem − matched/no_steer",
      mean: cf.steer_lift.mean,
      se: cf.steer_lift.se,
      t: cf.steer_lift.t,
      wins: cf.steer_lift.wins,
      losses: cf.steer_lift.losses,
    },
    {
      label: "sem_gap",
      sub: "matched/sem − mismatched/sem",
      mean: cf.sem_gap.mean,
      se: cf.sem_gap.se,
      t: cf.sem_gap.t,
      wins: cf.sem_gap.wins,
      losses: cf.sem_gap.losses,
    },
    {
      label: "lang_swap",
      sub: "matched/no_steer − mismatched/no_steer",
      mean: cf.lang_swap.mean,
      se: cf.lang_swap.se,
      t: cf.lang_swap.t,
      wins: cf.lang_swap.wins,
      losses: cf.lang_swap.losses,
    },
    {
      label: "codec_above_lang",
      sub: "sem_gap − lang_swap",
      mean: cf.codec_above_lang ?? 0,
      se: 0,
      t: 0,
      wins: 0,
      losses: 0,
    },
  ];

  const perSample = [...cf.per_sample]
    .sort((a, b) => a.steer_lift - b.steer_lift)
    .map((s, i) => ({
      idx: i,
      steer_lift: Number(s.steer_lift.toFixed(4)),
      sample_index: s.sample_index,
      target_task: s.target_task,
    }));

  return (
    <>
      <div className="card">
        <p className="chart-title">
          CF eval headline aggregates (n={cf.n_complete})
        </p>
        <p className="chart-sub">
          Error bars = ±1 standard error of the mean. <code>steer_lift</code>{" "}
          and <code>codec_above_lang</code> are the two publishable claims —
          both substantially positive here.
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={aggregateData}
            margin={{ top: 12, right: 16, bottom: 40, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11 }}
              interval={0}
              angle={-20}
              textAnchor="end"
            />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(2)}
              label={{
                value: "Δ r_sim (signed)",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
              }}
            />
            <ReferenceLine y={0} stroke="#444" strokeWidth={0.8} />
            <Tooltip
              formatter={(_: number, _name, item: any) => {
                const p = item?.payload;
                if (!p) return "—";
                return [
                  `${p.mean.toFixed(4)} (t=${p.t.toFixed(2)}, wins/losses=${
                    p.wins
                  }/${p.losses})`,
                  p.sub,
                ];
              }}
            />
            <Legend />
            <Bar dataKey="mean" name="signed mean Δ r_sim">
              {aggregateData.map((d, i) => (
                <Cell key={i} fill={_signedColour(d.mean)} />
              ))}
              <ErrorBar dataKey="se" stroke="#222" strokeWidth={1.2} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="source">
          Source: <code>data/eval/&lt;run&gt;_cf_strided_cached.json</code>{" "}
          aggregated by{" "}
          <code>scripts/website/export_site_data.py</code>.
        </p>
      </div>

      {perSample.length > 0 && (
        <div className="card">
          <p className="chart-title">
            Per-sample steer_lift, sorted ascending
          </p>
          <p className="chart-sub">
            Bars above zero are samples where injecting the codec made the
            matched-intent rollout score higher than its no-steer baseline.
            Green = win, red = loss. The win/loss tally here is{" "}
            <code>{cf.steer_lift?.wins ?? 0}</code> wins /{" "}
            <code>{cf.steer_lift?.losses ?? 0}</code> losses across n ={" "}
            {cf.steer_lift?.n ?? 0}.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={perSample}
              margin={{ top: 12, right: 16, bottom: 24, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="idx"
                tick={{ fontSize: 11 }}
                label={{
                  value: "sample (sorted)",
                  position: "insideBottom",
                  offset: -8,
                  fontSize: 11,
                }}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{
                  value: "Δ r_sim",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 11,
                }}
              />
              <ReferenceLine y={0} stroke="#444" strokeWidth={0.8} />
              <Tooltip
                formatter={(v: number) => v?.toFixed(4) ?? "—"}
                labelFormatter={(_, payload) => {
                  const p = payload?.[0]?.payload;
                  if (!p) return "";
                  return `sample ${p.sample_index}: ${p.target_task}`;
                }}
              />
              <Bar dataKey="steer_lift" name="steer_lift">
                {perSample.map((d, i) => (
                  <Cell key={i} fill={_signedColour(d.steer_lift)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  );
}
