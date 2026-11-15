# Capstone Research Report — Content Decline Ranking

- **Author:** rashmita-21
- **Repository:** https://github.com/rashmita-21/Flyrank-ML-Internship
- **Date:** 2026-11-15
- **Analysis scope:** 30,000-row anonymized starter sample (not the separate 79M-row warehouse)

## Abstract

Can a supervised model rank content items carrying a historical decline label more usefully than a simple age-and-visibility rule? I analyzed the repository's anonymized 30,000-row starter dataset, containing 44 source columns for 32 pseudonymized client groups and trailing-90-day activity measures. I trained a Logistic Regression pipeline with a client-grouped 80/20 holdout and compared it with the Week 4 rule on the exact same 6,163 test rows. On that holdout, Logistic Regression reached Precision@20 of 0.70 and Precision@50 of 0.72, compared with 0.40 and 0.52 for the rule, while average precision was 0.6044 versus 0.4980. The resulting scores and action queue are intended only to help a person prioritize review; temporal feature overlap remains unresolved, so this is not a certified prospective forecast or evidence that refreshing content causes recovery.

## 1. Introduction / Problem statement

The operational question is which existing content items should be inspected first when editorial review capacity is limited. The unit of analysis is one content item. The output is a ranked review list with reason codes and suggested actions; a human editor or SEO specialist decides whether to refresh, investigate, monitor, or take no action.

The prediction target is a historical decline label derived from impressions. It is not a Google ranking label, a forecast of future traffic, or a causal estimate of refresh impact.

## 2. Data

The model and playbook results use `data/raw/content_refresh_anonymized.csv`: 30,000 rows × 44 source columns, one row per pseudonymized content item, across 32 pseudonymized clients. The prepared feature vector has 52 columns. Activity metrics are trailing-90-day aggregates; trend direction compares the latest 30 days with the preceding 30 days. The local export timestamp is not recorded, so no exact calendar end date is asserted.

The repository separately documents a gated warehouse with 78,835,655 daily performance rows. Those warehouse tables were **not** queried to produce this paper's metrics. This report does not claim to be a 79-million-row analysis.

The target is `is_declining_label = 1` when `trend_direction == "down"`; all other categories are grouped as 0. `content_id` and `client_id` are used only for joins and grouping, never as features. `trend_direction` and `trend_pct` are excluded because they define the label. No client names, domains, URLs, page titles, keywords, or raw search queries are published.

## 3. Methodology

### Features and model

The Logistic Regression model uses 26 repository-defined features: 18 numeric and 8 categorical. Numeric variables include content age, days since update, search/engagement measures, and log-transformed activity totals. Categorical variables include content type, intent, and tier variables. Numeric preprocessing uses median imputation and standard scaling; categorical preprocessing uses most-frequent imputation and one-hot encoding. Class weighting is balanced; random state is 42.

### Baseline

The Week 4 baseline gives one point for content age ≥180 days and one point for impressions ≥500 over 90 days. Both methods rank the same held-out test items.

### Validation and leakage

Week 5 uses an 80/20 `GroupShuffleSplit` grouped by `client_id`, random state 42. Training: 23,837 rows / 25 clients. Test: 6,163 rows / 7 different clients. Client overlap: zero. Test decline-label rate: 51.1%.

Direct target-derived fields and IDs are excluded from the model feature matrix. Week 6 additionally reports five-fold GroupKFold out-of-fold metrics. However, 90-day activity features may overlap the 30-day windows used to construct the target. The snapshot does not provide a separate feature-availability timestamp proving those features predate the label window. Temporal leakage is therefore unresolved; this is retrospective classification/ranking, not a verified prospective forecast.

## 4. Results (model vs baseline)

Both methods are evaluated on the same 6,163 Week 5 test rows.

| Method | Precision@20 | Precision@50 | Average precision |
|---|---:|---:|---:|
| Week 4 age + visibility rule | 0.400 | 0.520 | 0.4980 |
| Logistic Regression | **0.700** | **0.720** | **0.6044** |

The test-set decline-label base rate is 51.1%. Week 6's five-fold GroupKFold out-of-fold sensitivity check reports average precision 0.6678, Precision@20 0.75, and Precision@50 0.74. This uses a different evaluation procedure and is not a direct baseline comparison.

Top Week 5 permutation importance values: log impressions 0.0734; log clicks 0.0585; average position 0.0372; content age 0.0310. These are predictive associations, not causal effects. The observed false negatives included pages with very few impressions; false positives included both older visible pages and newer low-traffic pages.

## 5. Limitations & honest framing

- The reported experiment uses the 30k starter sample, not the separate gated warehouse.
- The label is historical impression movement, not future ranking or refresh success.
- Temporal feature overlap is unresolved.
- Only 32 client groups are represented; the single holdout contains 7.
- Observational data cannot establish that refreshing causes recovery.
- Precision@20 is based on 20 items and should be interpreted alongside Precision@50, average precision, and the test base rate.
- Queue confidence tiers are operational thresholds, not calibrated probabilities of success.

Safe claim: on this anonymized sample, Logistic Regression ranked historical decline labels more effectively than the simple rule on the tested grouped holdout. This is directional decision-support evidence, not proof of future decline, ranking changes, or causal refresh impact.

## 6. Ranked recommendations

Week 7's playbook uses the pipeline-selected Random Forest probabilities and normalized baseline score:

`final_refresh_score = 100 × (0.70 × model_probability + 0.30 × normalized_baseline_score)`

This is a relative review-priority score, not a probability that a refresh will work.

| Suggested action | Items | Human next step |
|---|---:|---|
| monitor | 13,083 | Keep under observation; do not force an edit |
| refresh | 8,188 | Diagnose decline and verify freshness |
| refresh_and_review_ctr | 6,654 | Review search intent, title/snippet fit, position, and CTR |
| refresh_and_review_engagement | 1,993 | Review usefulness, structure, and page experience |
| expand_and_refresh | 82 | Check for missing user needs before expanding |

Confidence labels: 3,602 high, 11,398 medium, 15,000 low. Low-confidence items require additional investigation. Every item requires human approval.

**Human review:** confirm the page is live and relevant; verify reason codes; inspect impressions, clicks, sessions, CTR, and position; assess intent, accuracy, seasonality, tracking changes, cannibalization, effort, and likely value; record refresh, investigate, monitor, or no action.

**No-go:** no automatic publishing, rewriting, deletion, URL changes, redirects, canonical/schema/metadata edits, or sensitive editorial decisions. The model cannot override an editor, expert, or client approval.

**Monitoring:** track data quality, missingness, duplicates, feature availability, score distributions, client/content mix, reviewer acceptance/rejection, and later outcomes. Retraining requires consistent labels, verified feature timing, enough new labels, and fresh grouped/time-aware validation. A traffic change alone is not a retraining trigger.

## 7. Reproducibility

Repository: https://github.com/rashmita-21/Flyrank-ML-Internship

- [Week 5 model notebook](notebooks/w05_model.ipynb)
- [Week 6 validation audit](notebooks/w06_validation_audit.ipynb)
- [Week 7 action playbook](notebooks/w07_action_playbook.ipynb)
- [Capstone notebook](notebooks/capstone.ipynb)
- [Feature definitions](../scripts/ml_utils.py)
- [Baseline script](../scripts/02_baseline_score.py)
- [Training script](../scripts/03_train_model.py)

From a fresh clone: `pip install -r requirements.txt`, then `python scripts/run_all.py`. The Week 5 notebook reproduces the same-split Logistic Regression comparison with random state 42 and client-grouped 80/20 split. The Week 7 notebook rebuilds the ranked action playbook. The row-level queue is intentionally not published; aggregate counts and policy summaries are used here.

## 8. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). The anonymized starter sample is supplied for educational analysis; see [DATA_USE.md](../DATA_USE.md) for data-use and public-safety rules.
