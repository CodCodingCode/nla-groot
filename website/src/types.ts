export interface ScorecardMetric {
  name: string;
  value: number | null;
  threshold_pass: number;
  threshold_warn: number;
  higher_is_better: boolean;
  verdict: "PASS" | "WARN" | "FAIL" | "NA";
  required_for_overall: boolean;
}

export interface Scorecard {
  checkpoint: string;
  overall: "PASS" | "WARN" | "FAIL";
  metrics: ScorecardMetric[];
}

export interface JudgeRow {
  checkpoint: string;
  holdout: string;
  source: "gold" | "av_pred";
  grounding: number;
  appropriateness: number;
  template_distinguishable: number;
  n_rows: number;
}

export interface SimRow {
  condition: string;
  success_any: number;
  success_final: number;
  mean_steps: number;
  target_disp_m: number;
  target_min_ee_m: number;
  target_winner_rate: number;
  bowl_disp_m: number | null;
}

export interface TrainingPoint {
  step: number;
  fve: number | null;
  closed_greedy_fve: number | null;
}

export interface SiteSnapshot {
  generated_at: string;
  scorecard: Scorecard | null;
  judge: JudgeRow[];
  sim: SimRow[];
  training: TrainingPoint[];
}
