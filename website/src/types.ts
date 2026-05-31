// v9_combined-era types. The site is structured around the codec's
// absolute claims (cosine, MSE, intent specificity, causal steering)
// rather than a three-axis pass/fail scorecard.

export interface CodecMetricsAbsolute {
  // Headline numbers from the final SFT eval on held-out val.
  val_cosine: number;
  val_mse: number;
  val_fve: number;
  val_ce: number;
  closed_greedy_cosine: number;
  closed_greedy_mse: number;
  closed_greedy_fve: number;
  // Derived absolute interpretations (computed once at snapshot time).
  alpha_norm: number; // activation norm scale (= P75 of ||h|| from extraction stats)
  rms_error_raw_units: number; // sqrt(closed_greedy_mse)
  relative_norm_error_pct: number; // rms_error / alpha_norm * 100
  angle_degrees: number; // arccos(closed_greedy_cosine) * 180/pi
  // Provenance.
  step: number;
  total_steps: number;
}

export interface PairedCaptionSample {
  source_id: string;
  matched_intent: string;
  mismatched_intent: string;
  char_overlap: number;
  bullet_diff: number;
  caption_matched_preview: string;
  caption_mismatched_preview: string;
}

export interface CaptionDiagnostic {
  n_samples: number;
  mean_char_overlap: number;
  mean_bullet_diff_count: number;
  interpretation: string; // human-readable verdict from the script
  paired_samples: PairedCaptionSample[];
}

export interface StatSummary {
  mean: number;
  std: number;
  se: number;
  t: number;
  wins: number;
  losses: number;
  ties: number;
  n: number;
}

export interface CFArmRsim {
  sample_index: number;
  target_task: string;
  m_sem: number;
  m_nost: number;
  mm_sem: number;
  mm_nost: number;
  steer_lift: number;
  sem_gap: number;
  lang_swap: number;
}

export interface CFEvalResult {
  status: "complete" | "running" | "pending";
  n_complete: number;
  n_target: number;
  steer_lift: StatSummary | null;
  sem_gap: StatSummary | null;
  lang_swap: StatSummary | null;
  codec_above_lang: number | null;
  predicate_rates: {
    matched_semantic: number;
    matched_no_steer: number;
    mismatched_semantic: number;
    mismatched_no_steer: number;
  } | null;
  per_sample: CFArmRsim[];
}

export interface TrainingPoint {
  step: number;
  val_cosine: number | null;
  val_mse: number | null;
  val_fve: number | null;
  closed_greedy_cosine: number | null;
  closed_greedy_mse: number | null;
  closed_greedy_fve: number | null;
  train_loss: number | null;
  train_ar_mse: number | null;
  train_ce: number | null;
}

export interface SiteSnapshot {
  generated_at: string;
  run_name: string;
  codec_final: CodecMetricsAbsolute | null;
  training: TrainingPoint[];
  caption_diag: CaptionDiagnostic | null;
  cf_eval: CFEvalResult;
}
