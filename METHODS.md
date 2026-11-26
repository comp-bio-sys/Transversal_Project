# Methods

Main results are stored in `extension/balanced`. The internal and external test sets were examined during model development, so the analyses are exploratory rather than independent confirmation.

## Cohort and targets

The target is treated expression for the 978 measured LINCS L1000 landmark genes at 10 µM and 24 hours. Supplied sample IDs and compound split assignments are preserved for retained rows. Measured DMSO controls match detection plate, cell line, time and bead batch. Missing matched controls lead to exclusion. Targets remain on the supplied Level 3 scale.

The screened cohort has 210,438 training, 25,190 validation and 27,765 test profiles. Molecules above 0.7 ECFP4 Tanimoto similarity to the original external screening panel are removed. SKBR3 and the separately named SKB line are excluded. All seven expanded external study compounds are independently checked against the retained training set. The internal test is compound-disjoint, not scaffold-disjoint. It contains 295 test compounds with identical chirality-blind ECFP4 fingerprints to training compounds; 293 have matching molecular connectivity after removing stereochemistry. The stricter low-similarity evaluation retains 474 compounds / 16,531 profiles with maximum training Tanimoto similarity at most 0.7. This is a covariate-defined subset of the existing test, not new independent data. Most test control groups are also present among training profiles; a stricter secondary split addresses this separately.

## Models

FCFP4 and ECFP4 are binary 1,024-bit Morgan fingerprints of radius 2, using feature and atom invariants respectively. ChemicalVAE uses the supplied frozen zinc encoder posterior mean, with 196 values. All main samples are supported by every representation. The supplied encoder is frozen; its original pretraining exposure is not independently audited. Unseen compounds are absent from response-model training, not necessarily from unsupervised encoder pretraining.

Each residual MLP concatenates its chemical vector and standardized control expression. The hidden layers have 512 and 256 units with ReLU and 0.1 dropout after the first layer. A 978-unit mean head predicts a response added to the raw control. The head starts at zero. A softplus variance head adds a 1e-4 floor. The control-only model omits chemical features. Per-gene control scaling is fitted on training inputs only, with a standard-deviation floor of 1e-3. Cell context comes from control expression; these models have no cell-ID embedding.

PRnet uses the published PGM classes with FCFP4 scaled by log10(1 + 10), 10-dimensional Gaussian decoder noise and project-specific standardized control inputs. Its final mean averages 32 fixed decoder draws. Validation uses one fixed draw. This comparison uses the published architecture on this project cohort; it does not reproduce the paper's original dataset or numerical scores. A separate source-recipe run with different preprocessing is retained in `extension/reference` and is not pooled with the main results. Its dose-matched predictions can be reproduced with `python extension/evaluate_reference.py`; this separate run uses 32 fixed decoder draws and scores 24/42 eligible directions.

## Fitting and selection

Every main model has seeds 42, 43 and 44. All use AdamW, batch size 512, an initial 50-epoch cap and early stopping after seven epochs without improvement in validation Gaussian NLL. A validation-only convergence audit refits any trial that reaches the initial cap without patience stopping, using the same seed and a cap of 150 epochs. Settings are then re-ranked on validation NLL, and newly selected settings receive the remaining seeds. Runs that already stopped are unchanged. Fresh reruns allow 150 epochs from the start. The Gaussian constant is omitted. Equal tuning compares (learning rate, weight decay) settings (0.001, 0.01), (0.0003, 0.01), and (0.001, 0.0001). Seed 42 validation NLL selects a setting, which is then run for seeds 43 and 44. No test score selects a setting or checkpoint. Per-checkpoint metadata and complete validation curves are included.

The fixed-setting experiment crosses residual versus direct prediction with zero versus random mean-head initialization, using ECFP4 and three seeds. This factorial design separates the effects of the prediction rule and initialization.

## Baselines

Unchanged DMSO predicts the control itself. Global and cell-specific mean responses are fitted on training treated-minus-control targets; an unseen cell uses the global response. Ridge regression uses ECFP4 and standardized controls, an unpenalized intercept and validation-MSE-selected regularization from four candidates. A training gene-direction vote supplies an external sign baseline. These baselines never fit on held-out targets.

The leave-one-profile-out SKBR3 reference uses other test treatment profiles of the same compound. It is a repeatability diagnostic, not a trained predictor, deployable method or guaranteed noise ceiling.

## Evaluation and uncertainty

MSE averages profile-gene squared errors. Macro-compound MSE gives equal weight to each compound. Response Pearson correlation is computed across genes within each profile after subtracting the control from target and prediction. Internal/SKBR3 direction uses absolute measured changes of at least 0.5 Level 3 units. Unchanged DMSO predicts no sign and is an abstaining reference.

The SKBR3 set contains 574 profiles for 137 compounds, excluding six profiles with undisclosed structures. It tests an unseen cell line and includes compounds seen in other cells. The dose-matched external set has seven GEO studies, six genes per study, 42 directions and six cell lines. Norepinephrine/A549 is excluded for training-analogue overlap. Observed directions use GEO controls; model inputs use LINCS cell references formed by equally weighting available control-group means within each cell line. Platform and experimental-context differences remain.

External sensitivity analyses retain directions with separated treated/control replicates, then additionally require an absolute mean log-expression change of at least 0.5. This leaves 27 and 15 directions respectively. These descriptive rules are not formal differential-expression significance tests or unchanged-gene definitions.

Paired percentile intervals use 4,000 bootstrap draws with seed 26026. Per-run losses/accuracies are averaged over the three training seeds before resampling. Internal and SKBR3 losses are profile-weighted and resampled by compound or control group separately. External comparisons resample whole GEO studies, retaining their six genes. These are descriptive intervals without multiplicity correction; separate cluster analyses do not fully model crossed compound/group dependence.

## Diagnostics

Twenty chemical-input permutations within matched control groups keep targets, controls and decoder draws fixed. The control-only model must have exactly zero prediction change. For the low-similarity summary, only target rows are restricted; permutation donors come from the full internal test control group. The shuffling penalty measures chemical input dependence under this diagnostic, not causal mechanism or superiority over a separately trained baseline.

Synthetic control noise uses independent Gaussian gene noise at 0, 0.25, 0.5 and 1 times the within-plate DMSO replicate SD. A draw is shared by treated profiles on the same plate. Each nonzero level has 20 technical draws for each selected model seed. This does not represent all transcriptomic noise or uncertainty in a control mean of arbitrary sample size.

Marginal interval coverage uses mean ± 1.96 standard deviations. PRnet's interval is a Gaussian moment approximation using total mixture variance; its NLL uses the log of the average marginal density over 32 draws. Coverage is not simultaneous across the 978 genes and is not confidence in a causal drug effect.

The empirical control-resampling diagnostic uses 20 half-control subsets without replacement per plate, with at least one control retained. It is repeated for every selected model seed and unchanged DMSO.

A secondary joint holdout removes about 20% of control groups within training cell lines from fitting and validation, then tests original test compounds within those groups. Original compound assignments remain unchanged. Scaling is refitted on the reduced training set. The comparison with the original fixed models uses identical test rows, but training set size also changes. Control groups do not necessarily equal whole detection plates.

The joint-holdout results additionally report the subset with maximum similarity at most 0.7 to the original training cohort. This subset combines stronger chemical separation with unseen control-group identities without requiring a new model fit.
