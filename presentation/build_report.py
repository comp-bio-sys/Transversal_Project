"""Build the project report from the completed experiment tables."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BALANCED = PROJECT / "extension/balanced"
MLP_CANDIDATES = ["ResidualFCFP4", "Residual", "ResidualChemicalVAE"]
MAIN_MODELS = ["Residual", "PRnet"]
NAMES = {
    "ResidualFCFP4": "FCFP4 MLP", "Residual": "ECFP4 MLP",
    "ResidualChemicalVAE": "ChemicalVAE MLP", "PRnet": "PRnet",
    "DMSO": "Unchanged DMSO expression", "ControlOnly": "Control-only MLP",
    "GlobalMeanResponse": "Global mean response", "CellMeanResponse": "Cell mean response",
    "RidgeECFP4": "Linear ECFP4 + DMSO",
}


def main():
    summary = pd.read_csv(BALANCED / "summary/model_summary.csv")
    external = pd.read_csv(BALANCED / "summary/external_seed_counts.csv")
    sensitivity = pd.read_csv(BALANCED / "summary/external_label_sensitivity.csv")
    noise = pd.read_csv(BALANCED / "diagnostics/noise_seed_means.csv")
    selected = json.loads((BALANCED / "selected_settings.json").read_text())
    tuning = pd.read_csv(BALANCED / "tuning_validation.csv")
    default_mlp = tuning.loc[tuning.model.isin(MLP_CANDIDATES) & tuning.setting.eq("default")]
    lead_mlp = default_mlp.loc[default_mlp.val_nll.idxmin(), "model"]
    if lead_mlp != "Residual":
        raise ValueError("Revisit the report narrative: the validation-preferred MLP has changed.")
    if selected[lead_mlp] != "lower_lr":
        raise ValueError("Revisit the report narrative: the selected ECFP4 refinement has changed.")
    audit = json.loads((BALANCED / "generalization_audit.json").read_text())
    verification = json.loads((BALANCED / "verification.json").read_text())
    pages = []

    def page(title, subtitle=""):
        pages.append({"title": title, "subtitle": subtitle, "blocks": []})

    def p(text):
        pages[-1]["blocks"].append({"type": "paragraph", "text": text})

    def h(text):
        pages[-1]["blocks"].append({"type": "heading", "text": text})

    def table(headers, rows):
        pages[-1]["blocks"].append({"type": "table", "headers": headers, "rows": rows})

    def figure(name, caption, height=225):
        pages[-1]["blocks"].append({"type": "figure",
            "file": f"presentation/figures/{name}.png", "caption": caption, "height": height})

    def code(text):
        pages[-1]["blocks"].append({"type": "code", "text": text})

    def score(model, dataset, phase="tuned"):
        return summary.loc[summary.model.eq(model) & summary.dataset.eq(dataset)
                           & summary.phase.eq(phase)].iloc[0]

    def mse(model, dataset, phase="tuned", variability=True):
        row = score(model, dataset, phase)
        value = f"{row.mse_mean:.4f}"
        return value + (f" ± {row.mse_seed_sd:.4f}" if variability and row.seeds > 1 else "")

    def ext_rows(model, phase="tuned"):
        return external.loc[external.model.eq(model) & external.phase.eq(phase)].sort_values("seed")

    page("Predicting responses to chemical compounds", "M2 GENIOMHE-AI | Project report | 26 November 2026")
    p("Yernur Kenzhegazin · Alan Berdigaliyev · Hugo Telles<br/>Supervisor: Amaury Giard, IBISC")
    p("The project asks whether an MLP can predict chemical changes in gene expression and compare favorably with retrained PRnet. We compare FCFP4, ECFP4 and ChemicalVAE inputs, identify the validation-preferred MLP, examine its learning-rate refinement, and evaluate that predictor against PRnet. Both models receive measured DMSO expression and predict treated expression for 978 LINCS L1000 genes.")
    h("Choose the representation using validation")
    table(["MLP candidate", "Default validation NLL, seed 42"],
          [[NAMES[m], f"{default_mlp.loc[default_mlp.model.eq(m), 'val_nll'].iloc[0]:.5f}"] for m in MLP_CANDIDATES])
    p("ECFP4 has the lowest recorded default validation NLL. The representations were tuned in parallel; ECFP4 is identified here from the recorded validation scores. Test scores do not determine the designation.")
    h("Compare refined ECFP4 with retrained PRnet")
    table(["Model or reference", "Low-similarity MSE", "SKBR3 MSE"],
          [[NAMES[m], mse(m, "internal_low_similarity"), mse(m, "skbr3")] for m in MAIN_MODELS]
          + [[NAMES["DMSO"], mse("DMSO", "internal_low_similarity", "baseline"), mse("DMSO", "skbr3", "baseline")]])
    p("Network values are means ± sample standard deviations across three training seeds. Lower MSE is better. Unchanged DMSO directly predicts the matched 978-gene control mean; it has no fitted weights or training-seed variation. The low-similarity test has 474 compounds / 16,531 profiles, and SKBR3 has 137 compounds / 574 profiles.")
    p("Refined ECFP4 has lower MSE than PRnet in both tests and improves on unchanged DMSO for distant compounds. On SKBR3, unchanged DMSO is stronger. Lowering ECFP4's learning rate improves validation NLL but worsens SKBR3 and external scores relative to its default setting. External evidence does not establish a clear advantage over PRnet. The experiments therefore support a qualified comparison, not a validated drug-ranking method.")
    p("The tests were examined during development, so results are exploratory. Other representation scores and stronger simple baselines remain in the supplement and saved tables.")

    page("What the data represent")
    p("One expression profile is one bulk experimental sample with 978 gene values. Matrix rows are samples; columns are directly measured landmark genes. Values are supplied normalized LINCS Level 3 expression, not RNA-sequencing read counts, percentages or probabilities. Replicates and different cell lines may share a compound, so the number of profiles is not the number of independent compounds.")
    table(["Stage", "Retained data"], [
        ["Supplied usage manifest", "298,470 selected treatment profiles"],
        ["Matched DMSO cohort", "269,364 profiles; 47,995 DMSO measurements in 2,903 groups"],
        ["Screened training", "210,438 profiles; 8,168 compounds; 148 cell lines"],
        ["Validation", "25,190 profiles; 1,029 compounds"],
        ["Original internal test", "27,765 profiles; 1,019 compounds"],
        ["Low-similarity test subset", "16,531 profiles; 474 compounds; maximum training Tanimoto ≤0.7"],
        ["SKBR3 test", "574 profiles; 137 compounds; 226 measured DMSO controls"],
        ["External GEO panel", "Seven studies, six genes per study; 42 observed directions"],
    ])
    h("How DMSO expression is constructed")
    p("For each treatment, quality-passed vehicle controls are matched by detection plate, cell line, exposure time and bead batch. Replicates are averaged separately for each gene, producing a 978-value DMSO expression vector. This vector supplies the cellular context and also defines the unchanged-expression reference. The main cohort is restricted to 10 µM and 24 hours.")
    p("Treatment targets retain their supplied Level 3 scale, with no additional logarithm or target standardization. DMSO inputs are standardized gene by gene using training data only. The MLP response is added back to the original DMSO values. FCFP4 and ECFP4 descriptors are binary molecular features; ChemicalVAE descriptors are continuous molecular features. They are not expression measurements.")
    p(f"Original compound assignments are preserved for retained profiles. SKBR3, the separately named SKB line and close analogues of the external screening compounds are excluded from fitting. The original internal test still contains {audit['test_compounds_above_0_7_similarity_to_train']} compounds above 0.7 similarity to training and {audit['test_profiles_with_control_group_seen_in_train']:,} profiles sharing a control group with training. The low-similarity subset addresses chemical proximity; it does not establish scaffold or full experimental independence.")

    page("From the input files to a prediction")
    figure("pipeline", "Figure 1. The project procedure. Treatment samples are paired with matching DMSO replicates; their per-gene mean supplies the model input and unchanged-expression reference. Fixed compound assignments are screened; three encodings are compared by validation NLL. The preferred ECFP4 MLP is examined before and after refinement and compared with retrained PRnet. This sequence summarizes completed parallel tuning runs. Evaluation uses held-out measurements. Dish drawings, molecular examples and expression strips are schematic; sample counts come from the saved audit.", 220)
    table(["Input", "How it is used"], [
        ["l1000_level3_filtered_train_clean.feather", "Provides treatment targets and sample metadata"],
        ["compound_split.csv", "Assigns compounds to training, validation or test"],
        ["sample_usage.feather", "Identifies the selected sample IDs within each split"],
        ["Measured LINCS DMSO profiles", "Supply the matched per-gene replicate mean"],
        ["zinc.json and zinc_encoder.h5", "Define the frozen ChemicalVAE encoder"],
        ["External GEO contrasts", "Provide independent measured regulation directions"],
    ])
    p("Each molecular representation enters its own MLP; the three descriptors are not concatenated into one model. Training targets are measured treated profiles. At evaluation, a new structure and an appropriate DMSO profile produce a 978-gene prediction. GEO measurements are used for scoring only: external predictions use cell-line mean LINCS DMSO expression as the reference input.")
    code("chemical features + standardized DMSO expression\n  -> 512 hidden units -> 256 hidden units\n  -> predicted change and positive variance\npredicted treated expression = DMSO expression + predicted change")

    page("Select ECFP4 and assess its refinement")
    figure("model_selection", "Figure 2. Left: best seed-42 validation Gaussian NLL for the three MLPs at the shared default setting. Right: ECFP4 at the default and lower-learning-rate settings. Lower is better. These are single-seed checkpoint-selection values without uncertainty bars, not test scores. NLL assesses the predicted mean and variance; the preferred model is specific to this criterion.", 205)
    table(["ECFP4 setting", "Validation NLL", "SKBR3 MSE", "GEO correct / 42"], [
        ["Default", f"{tuning.loc[tuning.model.eq('Residual') & tuning.setting.eq('default'), 'val_nll'].iloc[0]:.5f}",
         mse("Residual", "skbr3", "fixed", variability=False), f"{ext_rows('Residual', 'fixed')['sum'].mean():.1f}"],
        ["Lower learning rate", f"{tuning.loc[tuning.model.eq('Residual') & tuning.setting.eq('lower_lr'), 'val_nll'].iloc[0]:.5f}",
         mse("Residual", "skbr3", variability=False), f"{ext_rows('Residual')['sum'].mean():.1f}"],
    ])
    p("Validation NLL is from seed 42; the two evaluation columns average seeds 42, 43 and 44. Reducing learning rate from 0.001 to 0.0003, at weight decay 0.01, improves validation NLL but worsens both listed evaluations. The main comparison retains the validation-selected setting instead of choosing retrospectively from test scores. ECFP4 is validation-preferred, not a universal test winner.")
    p("FCFP4 and ECFP4 have 1,024 radius-2 Morgan bits using functional-class and atom invariants respectively; ChemicalVAE supplies 196 frozen values. Each MLP concatenates its representation with standardized DMSO expression, uses 512 and 256 hidden units with ReLU and 0.1 dropout after the first layer, and predicts a response plus positive variance. The zero-initialized response is added to raw DMSO expression. There is no cell-ID embedding.")
    p("All models used AdamW, batch size 512 and patience seven epochs. Three learning-rate/weight-decay settings were evaluated in parallel for each representation; seed-42 validation NLL selected each setting. Every run uses its lowest-validation-NLL checkpoint. The Gaussian constant is omitted. Trials still improving at the initial 50-epoch cap were refitted with a 150-epoch cap before final validation selection. Full settings and curves are retained.")
    p("PRnet uses its published classes, dose-scaled FCFP4 and project-specific standardized DMSO inputs. Validation uses one fixed decoder draw; test means average 32 fixed draws. Its selected learning rate / decay is 0.0003 / 0.01. This retrains the architecture on our cohort; it does not reproduce the paper's original numerical results. The separate source-recipe run is supplementary.")

    page("Expression prediction depends on the test setting")
    figure("benchmark", "Figure 3. Treated-expression MSE for the refined ECFP4 MLP, retrained PRnet and unchanged DMSO expression on (a) the low-similarity compound subset and (b) excluded SKBR3. Bars show mean network scores over three training seeds; dots mark individual runs and whiskers span the smallest to largest seed score. These whiskers are seed ranges, not standard deviations or confidence intervals. DMSO expression is a measured reference without training variation. Lower values are better; both panels use the supplied Level 3 target scale.", 290)
    p("MSE is the squared difference between predicted and measured treated expression, averaged across the 978 genes and all eligible profiles. It measures reconstruction error in squared expression units. It does not directly measure a drug's usefulness, mechanism or biological effect strength.")
    p(f"Refined ECFP4 has MSE {mse('Residual', 'internal_low_similarity', variability=False)} on the low-similarity subset, compared with {mse('PRnet', 'internal_low_similarity', variability=False)} for PRnet and {mse('DMSO', 'internal_low_similarity', 'baseline')} for unchanged DMSO. On SKBR3, ECFP4 ({mse('Residual', 'skbr3', variability=False)}) again improves on PRnet ({mse('PRnet', 'skbr3', variability=False)}), but unchanged DMSO ({mse('DMSO', 'skbr3', 'baseline')}) is stronger than either model. Improvement over PRnet does not establish general transfer.")
    p("The compound subset is selected by maximum training ECFP4 Tanimoto similarity ≤0.7, using structures rather than observed model errors. It still includes shared control groups and is not scaffold-disjoint. SKBR3 tests an excluded cell line; 86 of its compounds were seen in other training cell lines and 51 were absent from training. It is not a uniformly joint unseen-compound/unseen-cell test.")
    p("Response Pearson correlation, retained in the result tables, subtracts DMSO expression from both predictions and targets before comparing the 978-gene patterns. This complementary metric distinguishes expression reconstruction from recovery of the treatment response.")

    page("External regulation: a small independent measurement panel")
    figure("external_comparison", "Figure 4. Correct regulation directions for the refined ECFP4 MLP and retrained PRnet among 42 gene-study observations in seven eligible GEO studies. Bars show the mean correct count across three training seeds, dots show individual runs and whiskers span their minimum and maximum. These are seed ranges, not confidence intervals over external studies. A correct direction means agreement between predicted and measured treatment-minus-reference signs. Unchanged DMSO expression predicts zero response and has no up/down classification score; it is not assigned an artificial zero-percent accuracy.", 220)
    table(["Model", "Correct directions, seeds 42 / 43 / 44", "Mean / 42"], [
        [NAMES[m], " / ".join(str(int(v)) for v in ext_rows(m)["sum"]), f"{ext_rows(m)['sum'].mean():.1f}"] for m in MAIN_MODELS
    ])
    p("The six genes are AKT1, AURKB, EGFR, PIK3CA, SMAD3 and TP53. The panel contains seven eligible GEO studies at 10 µM and 24 hours, spanning six cell lines. Norepinephrine/A549 is excluded because it has a training analogue with similarity 1.0. The panel was curated for available contrasts rather than sampled exhaustively from PharmGEO.")
    p("Observed signs come from GEO treatment-versus-control measurements. Predicted signs come from model-treated expression minus cell-mean LINCS DMSO expression. Dose and time are matched, but platform, cell culture and laboratory effects remain. GEO values are not substituted into the model as input, and expression magnitudes across these platforms are not treated as directly interchangeable.")
    p("ECFP4 averages 24.3 correct directions out of 42, compared with 23.0 for PRnet. The paired difference is 3.2 percentage points, with a descriptive study-bootstrap interval from -13.5 to +19.0 points. It therefore does not resolve a reliable external advantage. The seven studies, rather than the 42 correlated gene labels or repeated training seeds, determine the amount of independent external evidence.")

    page("What the external heatmap values mean")
    figure("external_studies", "Figure 5. Correct directions out of six genes for each compound/cell-line study, averaged over three training seeds. Rows are external studies; columns are refined ECFP4 and PRnet. Scores range from 0 to 6, with darker shading indicating more correct predictions. Values are counts of correct signs, not expression values or effect sizes.", 245)
    p("A value of 6.0 means all six directions were correct in every training run. A value such as 4.3 is a rounded average of integer counts from the three runs. Summing the unrounded study means yields the mean correct count out of 42. The six genes within a study are correlated, so uncertainty is estimated by resampling entire studies.")
    table(["Model", "All measured labels", "Clear-change subset"], [
        [NAMES[m], f"{sensitivity.loc[sensitivity.phase.eq('tuned') & sensitivity.model.eq(m) & sensitivity.subset.eq('all_measured'), 'sum'].mean():.1f} / 42",
         f"{sensitivity.loc[sensitivity.phase.eq('tuned') & sensitivity.model.eq(m) & sensitivity.subset.eq('clear_change'), 'sum'].mean():.1f} / 15"] for m in MAIN_MODELS
    ])
    p("Twenty-seven labels have completely separated treated and control replicate values. Fifteen also have an absolute mean log-expression change of at least 0.5. This is a descriptive sensitivity rule, not a differential-expression significance test. Counts above use the same predictions on two different label subsets; the smaller denominator must be considered.")
    p("RNA-seq counts are normalized by median-ratio size factors, then transformed as log2(count + 0.5). Microarrays use the per-sample median of uniquely mapped probes on a log2 scale; GSE2450 is log2-transformed after a lower bound of 1. Weak effects and few replicates make some directions uncertain. An unconvincing change is not proof of biological equivalence; a calibrated up/down/unchanged test remains missing.")

    page("How sensitive are predictions to noisy DMSO inputs?")
    figure("control_noise", "Figure 6. Total SKBR3 treated-expression MSE for refined ECFP4, PRnet and unchanged DMSO as Gaussian noise is added to DMSO inputs. Noise SD is 0, 0.25, 0.5 or 1 times each gene's measured within-plate DMSO replicate SD. Nonzero levels average 20 technical draws per training seed; bands span the seed means. DMSO has no training-seed band. At noise level zero, these are the clean SKBR3 scores in Figure 3. Lower is better.", 240)
    p("DMSO profiles are measured controls, not noiseless expression values. Their replicates vary. The project's robustness question is addressed by adding a known amount of noise to these inputs while keeping the fitted models and measured treated targets fixed. The vertical axis shows total MSE, so clean and perturbed predictions are directly comparable.")
    table(["Model or reference", "Clean SKBR3 MSE", "MSE at added noise SD = 1"], [
        [NAMES[m], mse(m, "skbr3", "baseline" if m == "DMSO" else "tuned", variability=False),
         f"{noise.loc[noise.model.eq(m) & noise.level.eq(1), 'mse'].mean():.4f}"]
        for m in MAIN_MODELS + ["DMSO"]
    ])
    p("ECFP4 error rises more with added noise, but remains lower than PRnet's total error. PRnet's flatter curve describes smaller sensitivity around an already higher clean-input error. This agrees with the clean SKBR3 comparison; a smaller change is not the same as a smaller prediction error.")
    p("A shared noisy reference is used for profiles on the same plate. Noise is applied independently across genes. This does not capture biological covariance, all assay errors or uncertainty in the mean of a particular replicate count. Twenty technical draws are not twenty independent experiments.")
    p("A separate diagnostic recomputes DMSO means from half of the observed control replicates. Predictive-interval coverage and chemical-input shuffling are also retained in the supplementary analyses. These checks answer different questions from synthetic input-noise sensitivity.")

    page("Coverage of the project objectives")
    table(["Objective in the brief", "What was completed", "Remaining limitation"], [
        ["Review perturbation models and metrics", "PRnet and MLP comparison; chemCPA, PertDiT and benchmarking literature discussed", "Focused review, not an exhaustive method benchmark"],
        ["Fit PRnet on the project dataset", "Published architecture retrained on the common screened LINCS cohort", "Project preprocessing and small tuning budget; not original-paper score reproduction"],
        ["Compare representations and refine an MLP", "Three default validation scores; ECFP4 learning-rate refinement; comparison with retrained PRnet", "Designation from completed parallel runs; validation gain does not guarantee transfer"],
        ["Test unseen compounds and SKBR3", "Compound-disjoint test, low-similarity subset and excluded cell line", "No scaffold holdout; one unseen cell line; some shared control groups"],
        ["Predict regulation of six genes", "Seven dose/time-matched GEO studies; 42 directions and stronger-label sensitivity subset", "Few studies; weak effects; no calibrated unchanged class"],
        ["Measure sensitivity to noise", "Gaussian DMSO noise and measured-replicate subsampling", "Limited noise model; no full biological uncertainty model"],
        ["Assess whether the approach is useful", "Compare with unchanged DMSO and supplementary learned/statistical baselines", "No prospective compound ranking or functional validation"],
    ])
    p("The completed experiments answer the available-data questions in the brief, but they do not make the predictors a validated drug-screening system. The clearest result is that ranking methods depends on the evaluation setting. The ability to use a chemical input is distinct from improving predictions with it.")
    h("Next scientific step")
    p("Before further model changes, freeze a larger external set of dose/time-matched contrasts. Separate compounds, cell lineages and control groups where feasible. Predefine meaningful effect and unchanged-label criteria, retain simple comparators, and evaluate the selected model once. A proposed compound ranking then requires an independent functional assay.")
    p("Internal and external tests were reused during development. Three seeds describe fitting variation on the same data, not independent biological replication. The reported uncertainty intervals remain descriptive and do not remove this limitation.")

    page("How to reproduce the analysis")
    table(["Code", "Purpose"], [
        ["MLP_PRnet_controls/match_controls.py", "Match DMSO measurements and retain supplied split assignments"],
        ["prepare_features.py; encoder.py", "Create fingerprints and frozen ChemicalVAE features"],
        ["extension/benchmark.py", "Define residual MLPs and construct PRnet"],
        ["balanced_train.py; tune_models.py", "Fit models and select settings/checkpoints with validation NLL"],
        ["scientific_data.py; evaluate_balanced.py", "Load the common cohort and calculate predictions and metrics"],
        ["dose_matched/measure.py", "Reconstruct external measurements from GEO files"],
        ["diagnose_final.py; calibration.py", "Evaluate chemical shuffling, input noise and predictive intervals"],
        ["summarize_revision.py; verify_revision.py", "Summarize paired comparisons and check result consistency"],
        ["presentation/", "Build scientific figures, report, notebook and poster"],
    ])
    code("python -m pip install -r requirements.txt\npython run.py figures\npython run.py report\npython run.py notebook\n\n# With the data and weights restored:\npython run.py check\npython run.py evaluate\npython run.py train --fresh")
    p("A GitHub clone contains code, saved result tables and presentation files. It can regenerate figures and the report without retraining. The full submission ZIP additionally contains prepared arrays, supplied split files, encoder weights and checkpoints. Restore these assets as described in docs/DATA.md before prediction, verification or full evaluation. Evaluation may take hours; full retraining may take days.")
    p(f"Analysis.ipynb contains executed explanations and outputs; its prediction example is skipped if assets are absent. Verification covers {verification['checked_checkpoints']} checkpoints, data separation, training-only scaling, selection rules and score arithmetic. The audit and checkpoint hashes are under extension/balanced. These are consistency checks, not independent replication.")
    p("Primary results are under extension/balanced; external observations are under dose_matched. The separate extension/reference experiment uses a different source-style preprocessing recipe and is retained as a sensitivity analysis, not pooled into the selected-MLP comparison.")

    page("Supplementary checks and negative evidence")
    p("The main test figures compare the validation-preferred ECFP4 MLP, PRnet and measured DMSO expression. FCFP4 and ChemicalVAE test results are retained here rather than used to choose the lead MLP. Additional experiments remain part of the project because they test whether chemistry contributes useful information. The control-only MLP is a separately fitted network without chemical inputs; it is different from unchanged DMSO expression.")
    table(["Supplementary comparator", "Low-similarity MSE", "SKBR3 MSE", "GEO correct / 42"], [
        [NAMES[m], mse(m, "internal_low_similarity", phase, variability=False), mse(m, "skbr3", phase, variability=False),
         f"{ext_rows(m, phase)['sum'].mean():.1f}"]
        for m, phase in [("ResidualFCFP4", "tuned"), ("ResidualChemicalVAE", "tuned"),
                         ("ControlOnly", "tuned"), ("GlobalMeanResponse", "baseline"),
                         ("CellMeanResponse", "baseline"), ("RidgeECFP4", "baseline")]
    ])
    p("ChemicalVAE has the lowest test MSE among the MLPs, but did not have the lowest validation NLL. The control-only network has lower low-similarity MSE than all three chemical MLPs and a higher mean external count. Linear ECFP4 + DMSO is strongest on the low-similarity subset; training mean-response baselines are strongest on SKBR3. These findings limit claims that chemical features add consistent predictive value, even though the chemical MLPs improve on PRnet.")
    table(["Additional analysis", "Interpretation"], [
        ["Chemical-input shuffling", "A shuffling penalty demonstrates input dependence, not a causal mechanism or superiority to a fitted context-only model"],
        ["Residual × initialization ablation", "Separates a residual connection from zero-initialized mean-head effects under shared fixed settings"],
        ["Joint compound/control-group holdout", "Reduces shared experimental context; the smaller fitting cohort is an additional change"],
        ["Paired compound and study intervals", "Descriptive uncertainty with clustered observations; not confirmatory evidence on reused tests"],
        ["Coverage and replicate subsampling", "Check predicted intervals and sensitivity to the available measured controls"],
        ["Source-recipe PRnet run", "Different preprocessing; one seed; 24/42 dose-matched directions; raw MSE is on a different scale"],
    ])
    p('Details, figures and source tables are indexed in <a href="docs/SUPPLEMENTARY_ANALYSES.md">docs/SUPPLEMENTARY_ANALYSES.md</a>. Full score tables retain the original internal test, stricter subsets and all supplementary comparators. Their different cohorts and fitting settings should not be pooled into one ranking.')
    p("For external chemical-MLP versus PRnet comparisons, all study-bootstrap intervals include zero. Only 15 of 42 labels meet the stronger-change rule. A reliable three-way up/down/unchanged classifier, broad cell-line generalization and prospective compound-selection utility remain unproven.")

    page("References")
    references = [
        "Giard, A. M2 Research Project (2026). Project brief, IBISC.",
        "Subramanian et al. A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles. Cell 171, 1437–1452.e17 (2017). doi:10.1016/j.cell.2017.10.049.",
        "Qi et al. Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery. Nature Communications 15, 9256 (2024). doi:10.1038/s41467-024-53457-1.",
        "Hetzel et al. Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution. Advances in Neural Information Processing Systems 35 (2022). doi:10.52202/068431-1937.",
        "Hu et al. Predicting drug-perturbed transcriptional responses using multi-conditional diffusion transformer. Quantitative Biology 14(1), e70016 (2026). doi:10.1002/qub2.70016.",
        "PRnet source code. https://github.com/Perturbation-Response-Prediction/PRnet. Apache-2.0 license.",
        "PharmGEO data release. Zenodo. doi:10.5281/zenodo.16556994.",
        "NCBI Gene Expression Omnibus: GSE153871, GSE2450, GSE200700, GSE59008, GSE225537, GSE42872 and GSE113505.",
        "Wei et al. Benchmarking algorithms for generalizable single-cell perturbation response prediction. Nature Methods 23, 451–464 (2026). doi:10.1038/s41592-025-02980-0.",
        "Wu et al. PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis. arXiv:2408.10609.",
    ]
    for number, reference in enumerate(references, 1):
        p(f"{number}. {reference}")
    (PROJECT / "report_content.json").write_text(json.dumps(
        {"title": "Chemical perturbation response prediction", "pages": pages},
        indent=2, ensure_ascii=False) + "\n")
    print("Prepared", len(pages), "report sections.")


if __name__ == "__main__":
    main()
