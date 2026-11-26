# Methods

The main analysis compares three MLP representations using recorded validation scores, designates ECFP4 as the lead MLP, examines its learning-rate refinement and compares the refined setting with retrained PRnet. Both predictors use measured DMSO expression as input; unchanged DMSO expression is the no-change reference. Results are stored in `extension/balanced`. The internal and external test sets were examined during development, so the analyses are exploratory rather than independent confirmation. Additional fitted baselines, ablations and group-holdout experiments are described under [Supplementary analyses](docs/SUPPLEMENTARY_ANALYSES.md).

## Cohort and expression targets

The target is treated expression for the 978 directly measured LINCS L1000 landmark genes at 10 µM and 24 hours. Supplied sample IDs and compound assignments are preserved for retained rows. Targets remain on the supplied normalized Level 3 scale; preparation applies no additional logarithm or target standardization.

The screened cohort contains 210,438 training, 25,190 validation and 27,765 internal test profiles. Molecules above 0.7 ECFP4 Tanimoto similarity to the original external screening panel are removed. SKBR3 and the separately named SKB line are excluded from fitting. All seven expanded external study compounds are independently checked against the retained training set.

The internal test is compound-disjoint, not scaffold-disjoint. It contains 295 compounds with chirality-blind ECFP4 fingerprints identical to training compounds; 293 have matching molecular connectivity after stereochemistry is removed. The main low-similarity evaluation retains 474 compounds and 16,531 profiles with maximum training Tanimoto similarity at most 0.7. This is a covariate-defined subset of the existing test, not new independent data. Most internal test control groups also occur among training profiles; a separate supplementary holdout examines that dependence.

## Measured controls and the unchanged-DMSO reference

Quality-passed DMSO profiles are matched by detection plate, cell line, exposure time and bead batch. Missing matched controls lead to exclusion. Control replicates are averaged separately for each of the 978 genes. The resulting 978-value vector is the measured control expression supplied to each model.

Per-gene means and standard deviations for control-input scaling are fitted on training inputs only, with a standard-deviation floor of 1e-3. Residual MLP predictions add their learned response to the original, unstandardized control values so that prediction and treated target share the same scale. Cell context comes from expression, not a cell-ID embedding.

**Unchanged DMSO** directly predicts the measured control vector. It has no parameters, optimization or training seeds. Its predicted treatment response is exactly zero, so it abstains on up/down direction and has no external direction-accuracy score. This is distinct from the supplementary **control-only MLP**, which learns a response from control expression while omitting chemical inputs.

## Chemical representations and predictors

FCFP4 and ECFP4 are binary 1,024-bit Morgan fingerprints of radius 2, using feature and atom invariants respectively. ChemicalVAE uses the supplied frozen zinc encoder posterior mean, with 196 values. Every main sample is supported by all representations. The encoder's original pretraining exposure is not independently audited: unseen compounds are absent from response-model training, not necessarily from unsupervised encoder pretraining.

Each chemical MLP concatenates its chemical vector and standardized control expression. Hidden layers have 512 and 256 units, ReLU activation and 0.1 dropout after the first layer. A 978-unit mean head predicts a response added to the raw measured control. This head starts at zero. A softplus variance head adds a 1e-4 floor.

PRnet uses the published PGM classes with FCFP4 scaled by log10(1 + 10), 10-dimensional Gaussian decoder noise and project-specific standardized control inputs. Its final mean averages 32 fixed decoder draws; validation uses one fixed draw. This retrains the published architecture on the project cohort. It does not reproduce the paper's original dataset or numerical results, and the small shared tuning grid does not establish optimal PRnet performance. A separate source-recipe sensitivity run is documented in the supplement and is not pooled with the main comparison.

## Training and validation

The three MLP candidates and PRnet each use seeds 42, 43 and 44, AdamW, batch size 512 and early stopping after seven epochs without improvement in validation Gaussian negative log likelihood (NLL). The Gaussian constant is omitted. Equal tuning compares three (learning rate, weight decay) settings: (0.001, 0.01), (0.0003, 0.01), and (0.001, 0.0001). Seed 42 validation NLL selects a setting, which is then run with seeds 43 and 44. No test score selects a setting or checkpoint. Per-checkpoint metadata and complete validation curves are retained. NLL evaluates the predicted mean and variance, so a validation-preferred model is best under this criterion rather than necessarily under every test metric.

The initial epoch cap was 50. A validation-only convergence audit refitted any trial that reached that cap without patience stopping, using the same seed and a 150-epoch cap. Settings were then re-ranked on validation NLL; newly selected settings received the remaining seeds. Runs that had already stopped were unchanged. Fresh reruns allow 150 epochs from the start.

### Representation ranking and refinement

The three-encoding comparison is a project design choice. The brief asks for an MLP comparison against retrained PRnet and lists enhancement as optional. To present that sequence, this analysis first compares the three MLPs at the shared default setting using seed-42 validation NLL, then examines the preferred representation's tuning result.

| MLP | Default validation NLL | Best grid validation NLL | Best setting |
|---|---:|---:|---|
| FCFP4 | 0.108004 | 0.088339 | Learning rate 0.001; weight decay 0.0001 |
| ECFP4 | 0.084649 | 0.083725 | Learning rate 0.0003; weight decay 0.01 |
| ChemicalVAE | 0.110533 | 0.089919 | Learning rate 0.0003; weight decay 0.01 |

These are seed-42 checkpoint minima from [tuning_validation.csv](extension/balanced/tuning_validation.csv). ECFP4 has the lowest NLL among the three at the default and best-grid settings. The representations were tuned in parallel, with a setting selected separately for each. ECFP4 is identified here from the recorded validation scores. ChemicalVAE's test ranking does not select the representation.

The default learning rate and weight decay are 0.001 and 0.01. Reducing ECFP4's learning rate to 0.0003 improves validation NLL from 0.0846488068 to 0.0837246455. Its three-seed mean SKBR3 MSE nevertheless increases from 0.5802596472 to 0.6537275720, and its external correct count falls from 26.3 to 24.3 out of 42. This is a modest validation refinement, not demonstrated improvement across unseen data. The main comparison uses the validation-selected ECFP4 setting and validation-selected PRnet; other representation test results are supplementary. [Evaluation summaries](extension/balanced/summary/model_summary.csv) and [external counts](extension/balanced/summary/external_seed_counts.csv) retain both stages.

## Evaluation on unseen compounds and SKBR3

MSE averages squared prediction errors across profiles and genes. Macro-compound MSE first averages within each compound and then gives all compounds equal weight. Response Pearson correlation is calculated across genes within each profile after subtracting the measured control from target and prediction. Internal/SKBR3 direction accuracy uses absolute measured changes of at least 0.5 Level 3 units. Unchanged DMSO supplies no direction and is an abstaining reference.

The main compound-generalization table uses the low-similarity subset described above. Scores for the full original test are retained in the result tables. The SKBR3 set contains 574 profiles for 137 compounds, excluding six profiles with undisclosed structures. It tests transfer to an unseen cell line; it is not also a fully unseen-compound test, because some compounds were observed in other training cell lines.

## External gene-regulation evaluation

The dose/time-matched external panel has seven GEO studies, six cell lines and 42 directions: six genes per study, at 10 µM and 24 hours. The genes are AKT1, AURKB, EGFR, PIK3CA, SMAD3 and TP53. Studies were identified through PharmGEO. Norepinephrine/A549 is excluded because of training-analogue overlap.

Observed directions come from GEO treated-minus-control measurements. Model inputs use LINCS cell references formed by equally weighting available control-group means within each cell line. The predicted direction is the sign of predicted treated expression minus that LINCS reference. Matching signs count as correct. GEO measurements are not model inputs. Platform and experimental-context differences remain despite dose/time matching.

For RNA-seq, counts use median-ratio size factors from genes positive in every selected sample, followed by log2(normalized count + 0.5). Microarrays use the per-sample median of uniquely mapped gene probes on a log2 scale; GSE2450 is log2-transformed after a lower bound of 1. The measured response is treated mean minus control mean. These reconstructed effects are not formal differential-expression significance results.

Sensitivity analyses retain directions with separated treated/control replicates, then additionally require an absolute mean log-expression change of at least 0.5. These subsets contain 27 and 15 directions respectively. The descriptive rules do not define biologically unchanged genes or provide formal significance tests. No model is selected using these subsets.

## Uncertainty in model comparisons

Paired percentile intervals use 4,000 bootstrap draws with seed 26026. Losses or accuracies are averaged over training seeds before resampling. Internal and SKBR3 losses are profile-weighted and resampled by compound or by control group in separate analyses. External comparisons resample whole GEO studies, retaining all six genes together. Training seeds are repeated model fits, not independent biological experiments.

Intervals are descriptive and unadjusted for multiple comparisons. Separate compound and control-group analyses do not fully model crossed dependence. Seven external study clusters provide limited precision, and reused evaluation data remain exploratory regardless of the interval calculation.

## Robustness to noisy controls

Synthetic noise uses independent Gaussian gene perturbations at 0, 0.25, 0.5 and 1 times each gene's within-plate DMSO replicate standard deviation. One noisy control draw is shared by treated profiles on the same plate. Models and treated targets stay fixed. Each nonzero level uses 20 technical draws per selected model seed; unchanged DMSO is evaluated under the same input perturbations.

The main plot reports absolute SKBR3 MSE; its zero-noise points match the clean-input evaluation. The increase from clean to noisy inputs measures sensitivity and is distinct from the absolute error. This experiment does not represent all transcriptomic noise, gene covariance or uncertainty in a control mean of arbitrary sample size. Resampling actual DMSO replicates provides a separate supplementary diagnostic.

## Supplementary experiments

The [supplementary document](docs/SUPPLEMENTARY_ANALYSES.md) retains results and interpretation for the following additional checks. They inform the conclusions but are separate from the selected-MLP versus PRnet comparison:

- Test results for the other MLP representations, a fitted control-only MLP, global and cell-specific mean responses, ridge regression and external sign baselines.
- Chemical-input permutations and a 2 × 2 residual-prediction/initialization ablation.
- Joint compound/control-group holdouts, with their own reduced training cohort and train-fitted scaling.
- Empirical control-replicate resampling, predictive interval coverage and a SKBR3 repeatability reference.
- A separate PRnet source-recipe run with different preprocessing.

## Code and provenance

| Stage | Main files |
|---|---|
| Match and average controls | `MLP_PRnet_controls/match_controls.py`, `aggregate_controls.py` |
| Construct representations | `MLP_PRnet_controls/prepare_features.py`, `encoder.py` |
| Define and fit models | `extension/benchmark.py`, `balanced_train.py`, `tune_models.py` |
| Evaluate and summarize | `extension/evaluate_balanced.py`, `summarize_revision.py` |
| Reconstruct external measurements | `dose_matched/measure.py` |
| Check noise and other diagnostics | `extension/diagnose_final.py`, `calibration.py` |
| Verify saved results | `extension/verify_revision.py` |

[Data instructions](docs/DATA.md) describe the assets needed for prediction and full reproduction. Original data/model references and third-party credits are retained in [MLP_PRnet_controls/SOURCES.md](MLP_PRnet_controls/SOURCES.md), [extension/SOURCES.md](extension/SOURCES.md) and the corresponding source directories.
