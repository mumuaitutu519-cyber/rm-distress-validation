# Files omitted from the public edition

342 aggregate release. Each excluded source path is listed below. No restricted observation values appear in this inventory. Public results retain original values; no inference or statistical calculation was rerun. The source private checkout and manuscript are unchanged.

CSV rule: at most 500 data rows, with no stkcd/code/firm/case_id/prompt/response-like column names. The rule is conservative: aggregate `firms` counts and `reason_code` categories are excluded too. Company/call records, prompt text, replies, individual predictions, bootstrap draw rows and all Parquet files are excluded. Bootstrap means and confidence limits are aggregate statistics, not draw rows. Aggregate model performance, score means and calibration bins are not individual predictions.

## Filtered source artifacts

- `verify/VERIFICATION_sun21.csv`: original 793-row CSV excluded; [table partitions](verify/VERIFICATION_sun21/) retain 787 original records with identical fields. Six introductory individual-case score/probability records are excluded. All 766 table records are retained.
- `verify/MISMATCH_REPORT_sun21.md`: public excerpt excludes three individual-case probability rows and one related bullet. All other source lines retain their original text, figures and numbering; the public notice identifies the omission.
- `output/tables/Table_A7_object_13.csv`: prompt wording excluded. Table A7 has no numerical statistics; the verification partition contains structural labels only.

## Excluded whole source files

| Source path | Reason |
|---|---|
| `CHECKSUMS.sha256` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `DATA_REVIEW.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `EXCLUDED_FILES.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `MANIFEST.csv` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `data/calls/strict_timing_core_mistral_nemo_12b_4bit_batch1.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_core_qwen3_4b_4bit_batch1.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_core_qwen3_4b_4bit_batch1_reload.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_core_qwen3_4b_fp16_batch1.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_core_qwen3_8b_4bit_batch1.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_extended_qwen3_4b_4bit.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_information_factorial_missing_qwen3_4b_4bit.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_primary_qwen3_4b_4bit.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/calls/strict_timing_probability_qwen3_4b_4bit.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `data/derived/strict_timing_complete_candidates_2011_2024.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/derived/strict_timing_nested_cases_64.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/derived/strict_timing_panel_2011_2024.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/derived/strict_timing_primary_cases_256.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/derived/strict_timing_probability_cases_1500.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/derived/strict_timing_st_changes.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/frozen_draws/strict_timing_primary_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/frozen_draws/strict_timing_probability_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/frozen_draws/strict_timing_temporal_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/strict_timing_benchmark_holdout_predictions.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/strict_timing_prelandmark_regulatory_features.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `data/strict_timing_regulatory_event_linkage.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `logs/README.md` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/clean_environment_install.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/clean_run.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/clean_run_receipt.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/clean_run_receipt_sun21.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/clean_run_sun21.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/post_run_checks.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `logs/post_run_checks.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/analysis/DATA_DICTIONARY.csv` | Schema inventory, not aggregate statistics; the existing data/DATA_DICTIONARY.csv remains available. |
| `output/analysis/strict_timing_annual_frame.csv` | Conservative column-name rule: firms; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_artifact_case_ranges.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_benchmark_holdout_predictions.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_core_mistral_nemo_12b_4bit_batch1.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_core_qwen3_4b_4bit_batch1.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_core_qwen3_4b_4bit_batch1_reload.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_core_qwen3_4b_fp16_batch1.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_core_qwen3_8b_4bit_batch1.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_extended_qwen3_4b_4bit.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. CSV has 1920 data rows, above the 500-row release limit. |
| `output/analysis/strict_timing_information_factorial_missing_qwen3_4b_4bit.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. CSV has 1152 data rows, above the 500-row release limit. |
| `output/analysis/strict_timing_information_factorial_panel.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_order_case_metrics.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_prelandmark_regulatory_features.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_primary_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_primary_case_ranges.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_primary_qwen3_4b_4bit.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. CSV has 768 data rows, above the 500-row release limit. |
| `output/analysis/strict_timing_probability_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_probability_predictions.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_probability_qwen3_4b_4bit.csv` | Conservative column-name rule: case_id; omit even when a column denotes an aggregate count or category. CSV has 4500 data rows, above the 500-row release limit. |
| `output/analysis/strict_timing_regulatory_event_linkage.csv` | Conservative column-name rule: firm_id; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_sample_summary.csv` | Conservative column-name rule: firms; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_status_reason_decomposition.csv` | Conservative column-name rule: reason_code; omit even when a column denotes an aggregate count or category. |
| `output/analysis/strict_timing_temporal_bootstrap_draws.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/analysis/strict_timing_washout_risksets.csv` | Conservative column-name rule: firms; omit even when a column denotes an aggregate count or category. |
| `output/analysis/sun21_independent_capacity_bootstrap_draws.csv` | CSV has 6000 data rows, above the 500-row release limit. Individual bootstrap draw rows are restricted; interval summaries are allowed. |
| `output/analysis/supplement_values_sun21.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/bootstrap_mode.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/computed_cells.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_artifact_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_exhibits.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_extended_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_regulatory_validation.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/logs/strict_timing_tables.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `output/model_calls.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `output/tables/Table_A7_object_13.csv` | Contains actual unsupported-text prompt templates; Table A7 has no numerical statistics. |
| `public/README.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `raw_rebuild/inference_logs/anonymous_reproducibility_manifest.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_artifact_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_benchmarks.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_bootstrap.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_core_mistral_nemo_12b_4bit_batch1.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_core_qwen3_4b_4bit_batch1.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_core_qwen3_4b_4bit_batch1_reload.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_core_qwen3_4b_fp16_batch1.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_core_qwen3_8b_4bit_batch1.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_exhibits.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_extended_analysis.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_extended_qwen3_4b_4bit.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_information_factorial_build.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_information_factorial_missing_qwen3_4b_4bit.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_panel_build.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_primary_qwen3_4b_4bit.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_probability_qwen3_4b_4bit.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_prompt_build.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_regulatory_data_audit.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_regulatory_validation.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_robustness.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_submission_validation.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/inference_logs/strict_timing_tables.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `raw_rebuild/strict_timing_extended_prompts_64.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `raw_rebuild/strict_timing_information_factorial_missing_prompts.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `raw_rebuild/strict_timing_nested_core_prompts_64.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `raw_rebuild/strict_timing_primary_prompts_256.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `raw_rebuild/strict_timing_probability_prompts_1500.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `verify/BUILD_AUDIT.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/DATA_PROVENANCE.csv` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/FIGURE_VERIFICATION.csv` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/INTRO_EXAMPLE_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/LIMITATIONS.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/MISMATCH_CONTEXT.csv` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/MISMATCH_REPORT.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/SECURITY_SCAN_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/VERIFICATION_SUMMARY.md` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/VERIFICATION_sun21.csv` | CSV has 793 data rows, above the 500-row release limit. Also contains six individual-example score/probability records. Publish unchanged eligible records by table instead. |
| `verify/benchmark_refit_comparison.csv` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/figures/manuscript_vintage_strict_timing_annual_frame.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/manuscript_vintage_strict_timing_artifact_profiles.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/manuscript_vintage_strict_timing_primary_case_ranges.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `verify/figures/manuscript_vintage_strict_timing_reliability_bins.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/manuscript_vintage_strict_timing_threshold_crossing_curve.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/recomputed_strict_timing_annual_frame.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/recomputed_strict_timing_artifact_profiles.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/recomputed_strict_timing_primary_case_ranges.parquet` | Parquet is excluded without exception; restricted observation, prediction, prompt, case or draw material is not redistributed. |
| `verify/figures/recomputed_strict_timing_reliability_bins.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/figures/recomputed_strict_timing_threshold_crossing_curve.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/BUILD_AUDIT.md` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/LIMITATIONS.md` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/MISMATCH_CONTEXT.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/MISMATCH_REPORT.md` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/VERIFICATION.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/VERIFICATION_SUMMARY.md` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/extracted_tables.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/logs/clean_environment_install.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/logs/clean_run.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/logs/clean_run_receipt.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/logs/post_run_checks.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/logs/post_run_checks.log` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/source.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/legacy_sun15/verification_status.json` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/manuscript/blocks_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/extracted_tables_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/figures/image1.png` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/figures/image2.png` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/figures/image3.png` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/figures/image4.png` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/figures/image5.png` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/paragraphs_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/manuscript/source_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |
| `verify/source_results/strict_timing_annual_frame.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_artifact_profiles.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_artifact_variant_metrics.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_auc_difference_ci.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_batch_sensitivity.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_benchmark_metrics.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_calibration_coefficients.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_calibration_robustness.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_descriptives.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_economic_response.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_extended_paired_summary.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_forward_time_historical_bss.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_hgb_tuning.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_information_fe_coefficients.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_information_margins.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_logit_tuning.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_order_summary.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_outcome_robustness.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_output_precision.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_primary_bootstrap_ci.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_primary_metrics.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_primary_range_summary.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_primary_rosters.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_probability_bootstrap_ci.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_probability_metrics.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_probability_periods.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_quantization_sensitivity.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_regulatory_event_linkage.csv` | Licensed-source observation-level data, prompts, replies, predictions, draws or case records; not publicly redistributed. |
| `verify/source_results/strict_timing_regulatory_signal_validation.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_reliability_bins.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_reload_repeatability.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_sample_summary.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_status_reason_decomposition.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_target_definition_sensitivity.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_temporal_bootstrap_ci.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_threshold_crossing_curve.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_washout_risksets.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_xgboost_tuning.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/source_results/strict_timing_yearly_sample.csv` | Conservatively omit numerical result snapshots and execution evidence from the code-and-schema public edition. |
| `verify/verification_status_sun21.json` | Private audit/manuscript extraction or stale private inventory; omit or replace with public-edition documentation. |

`README.md` and `README_한국어.md` are public-edition explanations. Private review/inventory files are replaced by public inventories. Existing analysis/rebuild source code and the previously published manuscript PDF are retained unchanged. References to restricted input paths are not data exports. Authorized review access may be requested from the corresponding author. No private `.git` content was copied.
