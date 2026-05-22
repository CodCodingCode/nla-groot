# nla-groot technical writeup site

Static Vite + React site for the **nla-groot** project: NLA on the GR00T VLA
backbone, the three-axis evaluation protocol, and the V3 negative result—with
interactive charts driven by a committed JSON snapshot of run artifacts.

**Live site:** https://codcodingcode.github.io/nla-groot/

It mirrors the papers at [`paper/main.tex`](../paper/main.tex) (workshop) and
[`paper/main_corl.tex`](../paper/main_corl.tex) (CoRL 2026 draft) but is
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
  public/
    figures/                   # PNGs copied from data/eval/.../figures
    papers/                    # PDFs copied from paper/ at build time
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
npm run build    # runs prebuild: copies paper PDFs into public/papers/
npm run dev      # http://localhost:5173
npm run preview  # serve dist/ locally
```

The Vite config sets `base: "./"`, so the `dist/` output works under the
GitHub Pages subpath `https://codcodingcode.github.io/nla-groot/`.

## Deploy to GitHub Pages

Deployment is automatic via GitHub Actions (`.github/workflows/deploy-pages.yml`):

1. Push to `main` (or run the workflow manually).
2. The workflow runs `npm ci && npm run build` in `website/` and publishes
   `website/dist/` to GitHub Pages.

**First-time setup** (repo admin, once):

1. **Settings → Pages → Build and deployment → Source:** GitHub Actions.
2. Or: `gh api repos/CodCodingCode/nla-groot/pages -X POST -f build_type=workflow`

The committed `snapshot.json` means CI does not need access to NFS `data/`.

## Editing notes

* All numbers come from `snapshot.json`. Don't hand-edit them. Re-run the
  export script.
* Every chart component takes its data as props and adds a source caption.
  When you add a new metric, extend
  [`scripts/website/export_site_data.py`](../scripts/website/export_site_data.py)
  and the corresponding TypeScript type in
  [`src/types.ts`](src/types.ts).
* Section copy should stay aligned with the papers; if you change a claim,
  update `paper/main_corl.tex`, `paper/main.tex`, and the relevant section here.
