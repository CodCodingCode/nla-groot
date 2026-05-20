# nla-groot technical writeup site

Static Vite + React site that explains the nla-groot pipeline (NLA on the
GR00T VLA backbone) and shows the V3 evaluation results, driven by a JSON
snapshot of the actual run artifacts under `data/`.

It mirrors the workshop paper at [`paper/main.tex`](../paper/main.tex) but is
designed to be browseable: charts are interactive, the pipeline diagram is
inline SVG, and every figure cites its source file.

## Layout

```
website/
  src/
    main.tsx, App.tsx          # entry + section composition
    index.css                  # minimal flat technical-doc styling
    components/                # PipelineDiagram + chart components
    sections/                  # Hero, Problem, Pipeline, Protocol, Results, Takeaway, Repro
    data/snapshot.json         # generated; committed
    types.ts
  public/figures/              # PNGs copied from data/eval/.../figures
  index.html, vite.config.ts, tsconfig.json
```

## Build a snapshot

The site never reads from `data/` at runtime. Every chart pulls from
[`src/data/snapshot.json`](src/data/snapshot.json). Regenerate it whenever
the underlying evals change:

```bash
# from repo root
python scripts/website/export_site_data.py
```

Inputs (paths overridable via CLI flags):

| Input                                                               | Used for                          |
| ------------------------------------------------------------------- | --------------------------------- |
| `data/sft/libero_4suite_v3/v3_scorecard.json`                        | Scorecard chart, hero badge       |
| `data/eval/steerability_v1_vs_v3/av_metrics.json`                    | Gold vs. AV judge chart           |
| `data/eval/steerability_v1_vs_v3/metrics.json`                       | Closed-loop steer chart           |
| `data/eval/steerability_v1_vs_v3/figures/*.png`                      | Per-object displacement / EE figs |
| `data/sft/libero_4suite_v3/metrics.jsonl` (val + final phases only)  | SFT training curve (optional)     |

Outputs:

* `website/src/data/snapshot.json` (committed; site builds offline)
* `website/public/figures/per_object_*.png` (copied from steerability run)

## Develop / build

```bash
cd website
npm install
npm run dev      # http://localhost:5173
npm run build    # static bundle in dist/
npm run preview  # serve dist/ locally
```

The Vite config sets `base: "./"`, so the `dist/` output works under any
GitHub Pages subpath (e.g. `https://<user>.github.io/<repo>/`).

## Deploy to GitHub Pages (one-shot)

The simplest path is the user-/project-pages branch flow:

```bash
cd website && npm install && npm run build
# Push dist/ to the gh-pages branch (use any tool, e.g. gh-pages npm pkg
# or a tiny CI workflow). dist/ is self-contained.
```

A GitHub Actions workflow that runs `python export_site_data.py` and
`npm run build` on push is straightforward to add later; keeping the
exported snapshot committed means CI does not need access to NFS data.

## Editing notes

* All numbers come from `snapshot.json`. Don't hand-edit them. Re-run the
  export script.
* Every chart component takes its data as props and adds a source caption.
  When you add a new metric, extend
  [`scripts/website/export_site_data.py`](../scripts/website/export_site_data.py)
  and the corresponding TypeScript type in
  [`src/types.ts`](src/types.ts).
* Section copy lives one-to-one with [`paper/main.tex`](../paper/main.tex);
  if you change the workshop paper's claim, update both.
