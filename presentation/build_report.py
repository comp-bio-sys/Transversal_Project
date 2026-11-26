"""Build the report's tables and narrative from the final experiment outputs."""

import json
from pathlib import Path

import pandas as pd

from figures import NAMES, MODELS

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BALANCED = PROJECT / "extension/balanced"


def main():
    summary = pd.read_csv(BALANCED / "summary/model_summary.csv")
    external = pd.read_csv(BALANCED / "summary/external_seed_counts.csv")
    paired = pd.read_csv(BALANCED / "summary/paired_mse_intervals.csv")
    ext_ci = pd.read_csv(BALANCED / "summary/paired_external_intervals.csv")
    shuffle = pd.read_csv(BALANCED / "diagnostics/permutation_seed_means.csv")
    sensitivity = pd.read_csv(BALANCED / "summary/external_label_sensitivity.csv")
    calibration = pd.read_csv(BALANCED / "results/calibration.csv")
    selected = json.loads((BALANCED / "selected_settings.json").read_text())
    audit = json.loads((BALANCED / "generalization_audit.json").read_text())
    joint = json.loads((BALANCED / "group_holdout/protocol.json").read_text())
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
        pages[-1]["blocks"].append(
            {
                "type": "figure",
                "file": f"presentation/figures/{name}.png",
                "caption": caption,
                "height": height,
            }
        )

    def code(text):
        pages[-1]["blocks"].append({"type": "code", "text": text})

    def score(model, dataset, phase="tuned"):
        return summary.loc[
            summary.model.eq(model)
            & summary.dataset.eq(dataset)
            & summary.phase.eq(phase)
        ].iloc[0]

    def mse(model, dataset, phase="tuned"):
        r = score(model, dataset, phase)
        return f"{r.mse_mean:.4f}" + (f" ± {r.mse_seed_sd:.4f}" if r.seeds > 1 else "")

    def count(model, phase="tuned"):
        r = external.loc[external.model.eq(model) & external.phase.eq(phase)]
        return " / ".join(str(int(v)) for v in r.sort_values("seed")["sum"]) + " of 42"

    page(
        "What chemical information adds to expression prediction",
        "M2 GENIOMHE-AI | Project report | 26 November 2026",
    )
    p(
        "Yernur Kenzhegazin · Alan Berdigaliyev · Hugo Telles<br/>Supervisor: Amaury Giard, IBISC"
    )
    p(
        "This project asks whether a model can predict the cellular response to a chemical compound accurately enough to help evaluate new compounds. We compare three molecular representations in multilayer perceptrons (MLPs) with PRnet and simple baselines. The main target is treated expression for 978 measured L1000 genes; the biological response is the difference from a measured vehicle control."
    )
    h("Experimental design")
    p(
        "All five main networks use three training seeds and the same small, validation-only tuning budget. Comparisons include a linear baseline and training-only mean-response baselines. Chemical-input shuffling, a factorial residual/initialization ablation, a joint compound/control-group holdout, input-noise tests and uncertainty diagnostics examine what drives model performance."
    )
    table(
        ["Model", "Low-similarity MSE", "SKBR3 MSE"],
        [
            [NAMES[m], mse(m, "internal_low_similarity"), mse(m, "skbr3")]
            for m in MODELS
        ],
    )
    p(
        "The low-similarity test contains 16,531 profiles for 474 compounds with maximum training ECFP4 similarity at most 0.7. Values are means ± sample standard deviations across three independently initialized training runs. This is training variability, not a confidence interval over new biological experiments. Detailed baselines, direction results and paired intervals follow."
    )
    best_internal = min(
        MODELS[:3], key=lambda m: score(m, "internal_low_similarity").mse_mean
    )
    best_sk = min(MODELS[:3], key=lambda m: score(m, "skbr3").mse_mean)
    p(
        f"The main finding is the strength of simple alternatives. On the low-similarity test, the best chemical MLP ({NAMES[best_internal]}) has MSE {score(best_internal,'internal_low_similarity').mse_mean:.4f}, compared with {score('ControlOnly','internal_low_similarity').mse_mean:.4f} for control-only and {score('RidgeECFP4','internal_low_similarity','baseline').mse_mean:.4f} for ridge. On SKBR3, the best chemical MLP has MSE {score(best_sk,'skbr3').mse_mean:.4f}, compared with {score('GlobalMeanResponse','skbr3','baseline').mse_mean:.4f} for the training mean response. Beating PRnet in this shared protocol therefore does not establish added predictive value from chemistry."
    )
    h("Scope of the conclusion")
    p(
        "This benchmark is exploratory: the external panel and internal tests were examined during model development. Repeated training measures variability on the available data, but does not provide independent confirmation. The results describe prediction, chemical-input dependence and transfer within the tested settings; they do not validate a stand-alone drug-discovery ranking system."
    )

    page("Why compare simple and generative models?")
    p(
        "The brief asks whether perturbation predictors are reliable enough to evaluate new compounds. Model complexity alone cannot answer that question. The comparison needs a common cohort, explicit generalization tests and baselines that reveal how much can be predicted from cellular context without chemical information."
    )
    table(
        ["Approach", "Published idea", "Implementation"],
        [
            [
                "PRnet",
                "A generative network conditions transcriptional predictions on chemical perturbations and cellular state.",
                "Published network classes are fitted on the project cohort; source-recipe sensitivity is kept separate.",
            ],
            [
                "chemCPA",
                "An encoder-decoder learns responses to unseen drugs and can transfer information from bulk screening data.",
                "Literature context for molecular encoders and transfer; chemCPA itself is not trained here.",
            ],
            [
                "PertDiT",
                "A diffusion transformer combines control transcriptomes with text-based chemical and dose representations.",
                "Literature context for richer conditional generation; PertDiT itself is not trained here.",
            ],
            [
                "Residual MLP and simple baselines",
                "Predict a response around the measured control, with or without chemical features.",
                "Test whether extra chemical representation and nonlinear capacity add useful information.",
            ],
        ],
    )
    p(
        'Primary sources: <a href="https://www.nature.com/articles/s41467-024-53457-1">Qi et al., PRnet (2024)</a>; <a href="https://papers.neurips.cc/paper_files/paper/2022/hash/aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html">Hetzel et al., chemCPA (NeurIPS 2022)</a>; <a href="https://onlinelibrary.wiley.com/doi/full/10.1002/qub2.70016">Hu et al., PertDiT</a>. The supplied Wei et al. benchmark motivates checking simple alternatives and complementary metrics. Its single-cell rankings are not transferred to our bulk-L1000 experiment.'
    )
    h("Why use several metrics?")
    p(
        "Treated-expression MSE measures reconstruction error, but a model can lower it by exploiting a stable cell state. Treatment-minus-control correlation asks whether the response pattern is recovered. Direction accuracy addresses the six external genes directly. Validation NLL evaluates a probabilistic prediction and selects checkpoints; empirical coverage checks whether its uncertainty matches observed outcomes. Chemical shuffling and a control-only model ask different questions: whether chemistry affects predictions, and whether adding it improves prediction."
    )
    p(
        "This is a focused methodological review, not a claim to benchmark every published method. The chosen experiments prioritize the models and data restrictions specified in the project brief."
    )

    page("The pipeline, from input files to scientific tests")
    figure(
        "pipeline",
        "Figure 1. Data assembly, control pairing, screening, encoding, validation-based selection and held-out evaluation. The blue dish represents the control input; the orange dish represents the measured treated target. Caffeine illustrates the molecular input. Control replicates are averaged per gene; output tiles denote predicted expression values. Dishes and colors are schematic; counts come from the audit.",
        290,
    )
    h("What enters the model")
    p(
        "The input files define the common experiment: l1000_level3_filtered_train_clean.feather supplies treated-expression targets, compound_split.csv supplies compound assignments and sample_usage.feather supplies selected sample IDs. After control pairing and screening, retained rows keep their original split labels. Training inputs alone determine control scaling; validation NLL selects settings and checkpoints. Test sets are evaluated after fitting."
    )
    p(
        "FCFP4 and ECFP4 supply 1,024 binary molecular features; the frozen ChemicalVAE encoder supplies 196 continuous values. Each residual MLP adds a learned 978-gene response to measured control expression. For external GEO scoring, predictions use a cell-specific mean of LINCS control groups. GEO treated-minus-control measurements provide the observed directions, not model inputs. This cross-platform comparison is separate from the plate-matched LINCS tests."
    )
    code(
        "chemical features + standardized control\n    -> 512 hidden units -> 256 hidden units\n    -> predicted change and positive variance\npredicted treated expression = control + predicted change"
    )

    page("Data preparation and what the splits actually test")
    p(
        "One profile is one measured sample, represented by 978 gene-expression values. Rows of the expression matrix are profiles; columns are landmark genes. A compound can occur in several profiles, cell lines and experimental replicates, so a profile count is not a count of independent compounds. The values are supplied LINCS Level 3 expression measurements, not RNA-sequencing read counts, percentages or probabilities."
    )
    table(
        ["Stage", "Retained profiles / details"],
        [
            ["Shared usage manifest", "298,470 selected profiles"],
            [
                "Matched measured controls",
                "269,364 profiles; 47,995 DMSO measurements in 2,903 groups",
            ],
            ["Screened training", "210,438 profiles; 8,168 compounds; 148 cell lines"],
            ["Screened validation", "25,190 profiles; 1,029 compounds"],
            ["Screened internal test", "27,765 profiles; 1,019 compounds"],
            ["SKBR3 test", "574 profiles; 137 compounds; 226 measured DMSO controls"],
            [
                "External GEO panel",
                "Seven eligible studies; six genes per study; 42 directions",
            ],
        ],
    )
    p(
        "Matched controls are quality-passed DMSO profiles from the same detection plate, cell line, 24-hour exposure and bead batch. Their mean is the 978-value control vector. Targets retain the supplied Level 3 scale: the preparation code applies no additional logarithm or target standardization. Control inputs are standardized gene by gene using training-only means and standard deviations. The residual prediction is then added to the original control values. Inferred genes are excluded."
    )
    p(
        "The original compound assignments are preserved for retained samples. Molecules above 0.7 ECFP4 Tanimoto similarity to the original external screening panel are excluded, as are SKBR3 and the separately named SKB line. The expanded seven-study panel was checked directly: all seven structures also remain at or below 0.7 similarity to the retained training molecules."
    )
    h("Compound separation is not full experimental separation")
    p(
        f"The internal audit found {audit['test_compounds_above_0_7_similarity_to_train']} of 1,019 test compounds above 0.7 similarity to a training compound. It also found {audit['test_profiles_with_control_group_seen_in_train']:,} of 27,765 test profiles with a matched control group also used by training profiles. Shared measured controls do not automatically leak a treated target, but they restrict the independence of this test. We therefore add a separate joint holdout and provide similarity-stratified results. Of the 1,019 test compounds, 295 have an identical ECFP4 vector to a training molecule; 293 share connectivity when stereochemistry is ignored. ECFP4 in this project does not encode chirality. The 474-compound low-similarity subset is therefore reported separately. No scaffold-disjoint claim is made."
    )
    p(
        "SKBR3 tests an unseen cell line; 86 compounds were seen in training in other cells and 51 were absent from training. Six of 580 retrieved treatment profiles have undisclosed structures and are excluded equally for all representations. Dose and time are 10 µM and 24 hours. These are different generalization questions from completely new compounds in completely new cell lines."
    )

    page("Models, fitting and the fair-comparison limits")
    table(
        ["Model", "Chemical input", "Core prediction rule"],
        [
            [
                "FCFP4 MLP",
                "1,024-bit radius-2 feature Morgan",
                "Control + learned response",
            ],
            [
                "ECFP4 MLP",
                "1,024-bit radius-2 atom Morgan",
                "Control + learned response",
            ],
            [
                "ChemicalVAE MLP",
                "196 frozen encoder values",
                "Control + learned response",
            ],
            ["Control-only MLP", "None", "Control + learned response"],
            [
                "PRnet",
                "FCFP4 scaled by log10(1 + 10 µM)",
                "Published PGM architecture with Gaussian decoder noise",
            ],
        ],
    )
    p(
        "The MLP trunk is 512 → 256 units with ReLU and 0.1 dropout after the first hidden layer. A mean head predicts a 978-gene response and a softplus variance head returns a positive variance. The main residual mean heads start at zero, so initial predictions equal the control. FCFP4 and ECFP4 have equal input width; ChemicalVAE has fewer input parameters. The supplied encoder is frozen and is not fitted to expression targets in this project. Its original pretraining exposure is not independently audited; unseen compounds here are absent from response-model training, not necessarily from unsupervised encoder pretraining."
    )
    p(
        "Every main model uses seeds 42, 43 and 44, batch size 512, patience of seven epochs and an initial 50-epoch cap. Any trial still improving at that cap is refitted from the same seed with a 150-epoch limit before final validation selection; already-stopped trials are unchanged. We compare three AdamW settings: learning rate 0.001 / decay 0.01; learning rate 0.0003 / decay 0.01; and learning rate 0.001 / decay 0.0001. Seed 42 validation Gaussian NLL selects the setting; the chosen setting is then evaluated with all three seeds. Test scores do not select the setting or checkpoint. The Gaussian constant is omitted from NLL."
    )
    table(
        ["Model", "Validation-selected setting", "Parameters"],
        [
            [
                NAMES[m],
                selected[m],
                f"{json.loads(((BALANCED/'weights' if selected[m]=='default' else BALANCED/'tuning'/selected[m])/f'{m}_42.json').read_text())['parameters']:,}",
            ]
            for m in MODELS
        ],
    )
    p(
        "PRnet uses its published network classes with project-specific standardization, fitting and data pairing. Validation uses one fixed decoder-noise draw; final prediction averages 32 fixed draws. This is a shared-budget comparison on this cohort, not a reproduction of the article’s published numerical scores or proof that PRnet is optimally tuned. A separate source-recipe run in extension/reference trained for 110 epochs and selected epoch 88 at seed 42. Its dose-matched evaluation averages 32 fixed decoder draws and scores 24/42 directions. It uses different expression preprocessing; this one-seed sensitivity result is not pooled with the main comparison, and its raw MSE is on a different scale. The separate 24/38 score in its original summary uses the original external labels and one decoder draw, so it is not directly comparable to the dose-matched score."
    )

    page("Performance across evaluation settings")
    p(
        "MSE is the mean squared difference between predicted and measured treated expression, averaged over profiles and genes. Lower is better; zero would mean exact prediction. Its units are squared Level 3 expression units. Values after ± show variation across the three training seeds. External counts instead measure how many of 42 gene directions were predicted correctly."
    )
    table(
        ["Model", "Internal MSE", "SKBR3 MSE", "GEO correct, seeds 42 / 43 / 44"],
        [[NAMES[m], mse(m, "internal"), mse(m, "skbr3"), count(m)] for m in MODELS]
        + [
            [
                NAMES[m],
                mse(m, "internal", "baseline"),
                mse(m, "skbr3", "baseline"),
                count(m, "baseline"),
            ]
            for m in ["GlobalMeanResponse", "CellMeanResponse", "RidgeECFP4"]
        ],
    )
    p(
        f"The unchanged-control baseline has internal MSE {mse('DMSO','internal','baseline')} and SKBR3 MSE {mse('DMSO','skbr3','baseline')}. Its predicted response is exactly zero, so it abstains on direction; an apparent 0% direction score would not be a meaningful up/down classifier comparison."
    )
    p(
        "The linear baseline uses ECFP4 and standardized controls to fit the treated-minus-control response. Its ridge penalty is selected by validation MSE from four candidates. Global and cell-specific mean responses use training targets only. The cell-specific baseline falls back to the global response for unseen SKBR3. Different selection objectives for ridge and the probabilistic networks are stated explicitly."
    )
    p(
        f"The linear model achieves internal MSE {mse('RidgeECFP4','internal','baseline')}, but its SKBR3 MSE is {mse('RidgeECFP4','skbr3','baseline')}. This illustrates why internal reconstruction accuracy alone cannot establish cell-line transfer. Representation rankings must be read separately for the internal test, SKBR3 and external direction."
    )
    h("Response metrics and repeatable signal")
    table(
        ["Model", "Internal response r", "SKBR3 response r", "SKBR3 direction"],
        [
            [
                NAMES[m],
                f"{score(m,'internal').response_pearson:.3f}",
                f"{score(m,'skbr3').response_pearson:.3f}",
                f"{100*score(m,'skbr3').direction_accuracy:.1f}%",
            ]
            for m in MODELS
        ],
    )
    p(
        "Response r is Pearson correlation across the 978 genes after subtracting the matched control from both prediction and measurement, then averaging across profiles. It ranges from -1 to 1: a higher value means better agreement in the response pattern. The SKBR3 direction percentage counts correct up/down signs only where the measured absolute change is at least 0.5 Level 3 units. It measures sign agreement, not effect size."
    )
    reference = summary.loc[summary.phase.eq("reference")]
    ref = reference.loc[reference.model.eq("LeaveOneProfileOutResponse")].iloc[0]
    dm = reference.loc[reference.model.eq("DMSO")].iloc[0]
    p(
        f"A leave-one-profile-out reference averages other test responses for the same compound. Across {int(ref.profiles)} eligible SKBR3 profiles, MSE is {ref.mse_mean:.4f}, versus {dm.mse_mean:.4f} for unchanged controls on those rows. This indicates repeatable signal, but uses test replicates: it is neither deployable nor a guaranteed noise ceiling."
    )

    page("A stricter test of unseen chemical structures")
    p(
        "The original compound labels are disjoint, but different stereochemical annotations can yield identical chirality-blind fingerprints. The stricter evaluation retains test compounds with maximum ECFP4 Tanimoto similarity to any training compound at most 0.7: 474 compounds and 16,531 profiles. This filter uses chemical inputs only; it does not inspect model errors or refit the networks. It is an exploratory subset of the existing test, not an independently collected test set."
    )
    table(
        ["Model", "Low-similarity MSE", "Macro-compound MSE", "Response r"],
        [
            [
                NAMES[m],
                mse(m, "internal_low_similarity"),
                f"{score(m,'internal_low_similarity').macro_compound_mse:.4f}",
                f"{score(m,'internal_low_similarity').response_pearson:.3f}",
            ]
            for m in MODELS
        ]
        + [
            [
                NAMES[m],
                mse(m, "internal_low_similarity", "baseline"),
                f"{score(m,'internal_low_similarity','baseline').macro_compound_mse:.4f}",
                f"{score(m,'internal_low_similarity','baseline').response_pearson:.3f}",
            ]
            for m in ["GlobalMeanResponse", "CellMeanResponse", "RidgeECFP4"]
        ],
    )
    p(
        f"Low-similarity MSE gives equal weight to each profile; macro-compound MSE first averages profiles within a compound, then weights every compound equally. Response r measures the agreement of gene-response patterns. Unchanged-control MSE on these rows is {mse('DMSO','internal_low_similarity','baseline')}. This subset still includes control groups seen during training and is distinct from the joint control-group holdout."
    )
    strict_intervals = paired.loc[
        paired.dataset.eq("internal_low_similarity")
        & paired.reference.eq("ControlOnly")
    ]
    table(
        ["Model vs control-only", "MSE difference", "Compound-bootstrap 95% interval"],
        [
            [
                NAMES[r.model],
                f"{r.difference:+.4f}",
                f"{r.lower:+.4f} to {r.upper:+.4f}",
            ]
            for r in strict_intervals.itertuples()
        ],
    )
    p(
        "Negative differences favor the chemical-input model. These paired intervals average seed-specific losses before compound resampling and remain descriptive. A low Tanimoto threshold does not guarantee different scaffolds, mechanisms or exposure to the frozen encoder’s original pretraining data."
    )

    page("Transfer depends on the compound")
    figure(
        "compound_response",
        "Figure 2. Each point is a SKBR3 compound. Model loss is averaged over three training seeds. Below the equality line, the model improves on unchanged controls for that compound. The horizontal axis also reflects effect strength and measurement variability, so this plot is descriptive rather than a causal correlation analysis. Axis ranges differ between panels.",
        330,
    )
    exposure_results = pd.read_csv(
        BALANCED / "summary/skbr3_compound_exposure_results.csv"
    )
    grouped = (
        exposure_results.loc[exposure_results.phase.eq("tuned")]
        .groupby(["model", "compound_seen_in_training"])
        .mse.mean()
        .unstack()
    )
    table(
        ["Model", "SKBR3: 86 seen compounds", "SKBR3: 51 unseen compounds"],
        [
            [NAMES[m], f"{grouped.loc[m,True]:.4f}", f"{grouped.loc[m,False]:.4f}"]
            for m in MODELS
        ],
    )
    p(
        "The supplementary internal-distance table uses the nearest training molecule and reports scores separately above and below 0.7 Tanimoto similarity. It creates descriptive subsets of the original test, not a new independently selected scaffold split. SKBR3 unseen-compound and seen-compound subsets contain different treatments and effect strengths; their MSE values should not be interpreted as a controlled causal estimate of chemical novelty."
    )

    page("Using chemistry is different from benefiting from it")
    figure(
        "chemistry_shuffle",
        "Figure 3. Twenty chemical-feature permutations per dataset and seed, within control groups. Controls, targets and decoder draws stay fixed. Low-similarity targets use donors from their full test group. Dots average technical draws, not independent experiments.",
        175,
    )
    table(
        ["Model", "Low-similarity MSE increase", "SKBR3 MSE increase"],
        [
            [
                NAMES[m],
                f"{shuffle.loc[shuffle.model.eq(m)&shuffle.dataset.eq('internal_low_similarity'),'mse_increase'].mean():+.5f}",
                f"{shuffle.loc[shuffle.model.eq(m)&shuffle.dataset.eq('skbr3'),'mse_increase'].mean():+.5f}",
            ]
            for m in MODELS
        ],
    )
    p(
        "A shuffling penalty shows dependence on the chemistry-response association. It does not establish a causal mechanism or improvement over the control-only model. Shuffling may create implausible inputs and is informative only within groups containing multiple compounds."
    )
    figure(
        "chemical_added_value",
        "Figure 4. Paired MSE differences for the three chemical MLPs versus control-only. Negative values favor chemical inputs. The three per-run losses are averaged before 4,000 compound-cluster bootstrap draws. These are descriptive intervals on previously examined data.",
        165,
    )
    p(
        "Additional control-group intervals are provided. Compound and group dependence are crossed; neither separate bootstrap fully captures both. Intervals are descriptive and unadjusted for multiple comparisons."
    )

    page("Reconstructing external measurements and label quality")
    p(
        "The curated convenience panel, rather than an exhaustive sample of PharmGEO, reconstructs six gene measurements from each eligible GEO study at 10 µM and 24 hours. It includes AKT1, AURKB, EGFR, PIK3CA, SMAD3 and TP53, including weak changes that were absent from a published differential-expression list. Norepinephrine/A549 is excluded because it has a training analogue with similarity 1.0. The seven eligible studies span six cell lines."
    )
    p(
        "RNA-seq counts use median-ratio size factors from genes positive in every selected sample, followed by log2(normalized count + 0.5). Microarrays use the per-sample median of uniquely mapped gene probes on a log2 scale; GSE2450 is log2-transformed after a lower bound of 1. The measured response is the treated mean minus the control mean. These are reconstructed effect estimates, not a formal differential-expression significance analysis."
    )
    measured = pd.read_csv(PROJECT / "dose_matched/measured_effects.csv")
    measured = measured.loc[measured.chemical_split.eq("eligible")]
    table(
        ["GEO study / cell", "Control / treated replicates", "Clear-change labels / 6"],
        [
            [
                f"{study} / {d.cell.iloc[0]}",
                f"{int(d.n_control.iloc[0])} / {int(d.n_treated.iloc[0])}",
                f"{int(d.clear_change.sum())} / 6",
            ]
            for study, d in measured.groupby("gse_id", sort=False)
        ],
    )
    p(
        "Twenty-seven directions have fully separated treated/control replicates. Fifteen also have an absolute mean log-expression change of at least 0.5; this is a descriptive sensitivity rule, not a differential-expression significance test. Clear-change coverage ranges from one to four observations per gene; PIK3CA has only one. Counts are small and platform-specific preprocessing matters. The full, separated-replicate and clear-change subsets are all retained without selecting a model on any subset."
    )

    page("External direction: seven studies, not 42 independent experiments")
    figure(
        "external_studies",
        "Figure 5. Rows are compound/cell-line studies; columns are prediction models. Each value is the mean number of correct up/down directions among six genes across three training seeds. Scores range from 0 to 6; higher is better. Shading repeats this same score.",
        235,
    )
    p(
        "The six genes are AKT1, AURKB, EGFR, PIK3CA, SMAD3 and TP53. For each gene, the observed sign comes from GEO treated-minus-control expression; the predicted sign comes from model-predicted treated expression minus its LINCS control reference. Matching signs count as correct. This tests regulation direction, not the size of the expression change."
    )
    p(
        "A value of 6.0 means that all six directions were correct in each of the three runs. A value such as 4.7 is a rounded average of integer counts across runs, not an expression level or a percentage. Summing the seven study scores before rounding gives the mean correct count out of 42 below. The six genes within a study are correlated, so uncertainty intervals resample whole studies, not individual heatmap cells."
    )
    table(
        ["Model", "All 42: mean correct", "Clear 15: mean correct"],
        [
            [
                NAMES[m],
                f"{sensitivity.loc[sensitivity.phase.eq('tuned')&sensitivity.model.eq(m)&sensitivity.subset.eq('all_measured'),'sum'].mean():.1f} / 42",
                f"{sensitivity.loc[sensitivity.phase.eq('tuned')&sensitivity.model.eq(m)&sensitivity.subset.eq('clear_change'),'sum'].mean():.1f} / 15",
            ]
            for m in MODELS
        ],
    )
    p(
        "The clear-change column scores the same predictions on the 15 stronger measured changes defined on the previous page. It uses a different denominator, so compare proportions or use the full 42-direction column when comparing total counts."
    )
    vote = external.loc[external.model.eq("TrainingGeneDirection")].iloc[0]
    p(
        f"The training gene-direction baseline gets {int(vote['sum'])}/42 correct; always predicting down gets 25/42. Platform, culture and treatment-context differences remain despite matched dose and time."
    )

    page("How much external evidence favors one model?")
    table(
        ["Model vs PRnet", "Difference in accuracy", "Descriptive 95% interval"],
        [
            [
                NAMES[r.model],
                f"{100*r.difference:+.1f} percentage points",
                f"{100*r.lower:+.1f} to {100*r.upper:+.1f} pp",
            ]
            for r in ext_ci.loc[ext_ci.reference.eq("PRnet")].itertuples()
        ],
    )
    p(
        "The intervals resample seven whole GEO studies, keeping the six genes in a study together. Seed-specific accuracies are averaged first; seeds are repeated model fits, not additional studies. With only seven study clusters, percentile intervals are approximate and can be unstable. These descriptive intervals do not turn this reused panel into confirmatory evidence."
    )
    p(
        "Every chemical-MLP versus PRnet interval in this panel spans zero. The external direction results therefore do not resolve an advantage over PRnet, despite higher point estimates. The control-only model has the highest mean count among the five networks, reinforcing the need to demonstrate a benefit beyond cellular context."
    )
    h("What remains missing")
    table(
        ["Limitation", "Implication", "Next experiment"],
        [
            [
                "Only seven eligible studies",
                "Limited precision and selected experimental contexts",
                "Freeze a larger independent panel before further tuning",
            ],
            [
                "Weak changes and few replicates",
                "Some signs are noisy; no formal equivalence labels",
                "Use replication and predeclared effect/uncertainty criteria",
            ],
            [
                "One unseen cell line",
                "Transfer to other lineages is unknown",
                "Hold out additional cell lines and their control groups",
            ],
            [
                "Fixed 10 µM / 24 h",
                "Dose or time extrapolation is untested",
                "Train and evaluate matched dose/time series",
            ],
            [
                "Expression endpoints only",
                "Correct signs do not establish drug benefit",
                "Test ranked compounds in a prospective functional assay",
            ],
        ],
    )
    p(
        "An unchanged gene cannot be identified just because a contrast fails a significance test. The current evaluation does not establish specificity for biologically unchanged genes, nor does it model a calibrated three-way up/down/unchanged decision. False-positive control and discovery utility remain open questions."
    )

    page("Noise sensitivity and predictive uncertainty")
    figure(
        "control_noise",
        "Figure 6. Gaussian noise is scaled by each gene’s within-plate DMSO replicate SD. A shared noisy control is used for profiles from the same plate. Lines average three training seeds and 20 technical draws; shading is the range of seed means. Smaller sensitivity is not better if clean predictions already have high error.",
        225,
    )
    p(
        "Noise levels are 0, 0.25, 0.5 and 1.0 times the measured replicate SD. These are independent Gaussian gene perturbations to the control input, not a complete model of RNA-seq noise, gene covariance or uncertainty in the mean of a particular number of replicates. Targets stay fixed. A second diagnostic uses 20 half-control subsamples from the actual measured DMSO replicates, applied to the same selected model seeds; those results are provided alongside the synthetic-noise results."
    )
    empirical = pd.read_csv(BALANCED / "diagnostics/empirical_control_resampling.csv")
    ec = empirical.loc[empirical.model.eq("Residual")]
    p(
        f"With half of the measured controls, the selected ECFP4 model has mean SKBR3 MSE {ec.half_control_mse.mean():.4f}, compared with {ec.clean_mse.mean():.4f} using all controls. The resampling removes control measurements without changing the treated targets; it measures sensitivity to this specific control-estimation procedure."
    )
    table(
        ["Model", "Internal 95% coverage", "SKBR3 95% coverage", "SKBR3 NLL"],
        [
            [
                NAMES[m],
                f"{100*calibration.loc[calibration.model.eq(m)&calibration.dataset.eq('internal'),'covered95'].mean():.1f}%",
                f"{100*calibration.loc[calibration.model.eq(m)&calibration.dataset.eq('skbr3'),'covered95'].mean():.1f}%",
                f"{calibration.loc[calibration.model.eq(m)&calibration.dataset.eq('skbr3'),'nll'].mean():.3f}",
            ]
            for m in MODELS
        ],
    )
    p(
        "Coverage is the percentage of measured profile-gene values inside the model's predicted 95% interval (mean ± 1.96 standard deviations). A calibrated interval should cover about 95%; high coverage alone can also reflect overly wide intervals. NLL scores the predictive distribution, with lower values preferred. PRnet uses total variance across 32 decoder draws for its Gaussian interval, but averages component densities before calculating NLL. Coverage applies to individual gene values, not all 978 genes simultaneously or confidence in a treatment mechanism."
    )

    page("Separating architecture and experimental context")
    figure(
        "residual_ablation",
        "Figure 7. A 2 × 2 ablation separates the residual connection from zero versus random mean-head initialization. All four use ECFP4 and the same fixed fitting settings. Dots are training seeds; bars mark their mean.",
        195,
    )
    p(
        "The factorial design separates the effects of residual prediction and mean-head initialization. All four combinations use shared default settings, independently of the validation-tuned main comparison."
    )
    figure(
        "control_group_holdout",
        "Figure 8. Both curves use identical low-similarity joint-holdout test rows (nearest-training Tanimoto at most 0.7). One retains the original training cohort; the other removes selected control groups from fitting and validation. Both curves use the shared default fitting settings, rather than the tuned main settings. The smaller training cohort is a second change, so the difference cannot be attributed solely to control-group reuse.",
        195,
    )
    joint_strict = score("Residual", "joint_low_similarity", "group_holdout")
    p(
        f"The plotted stricter subset contains {int(joint_strict.profiles):,} profiles for {int(joint_strict.compounds)} compounds. Its chemical similarity is measured against the original, larger training set, so it also satisfies the threshold against the reduced training set."
    )
    p(
        f"Within each training cell line, about 20% of control groups were selected before this secondary training. Original compound assignments are preserved. The joint split contains {joint['samples']['train']:,} training, {joint['samples']['val']:,} validation and {joint['samples']['test']:,} test profiles. Standardization is refitted on this smaller training set only. Control groups need not correspond to fully held-out detection plates or independent chemical scaffolds."
    )

    page("Training convergence and checkpoint selection")
    figure(
        "training_convergence",
        "Validation curves for the selected setting of each model. Stars mark the minimum validation NLL for each training seed. Positive curves use logarithmic y-axes; panels have separate scales. The selected checkpoint can precede an unstable later epoch.",
        300,
    )
    convergence_path = BALANCED / "convergence_audit.json"
    convergence = (
        json.loads(convergence_path.read_text())
        if convergence_path.exists()
        else {"extended_trials": []}
    )
    if convergence["extended_trials"]:
        p(
            "The original 50-epoch limit was checked before final evaluation. Every trial that reached it without completing the patience stopping rule was fitted again from the same seed with a 150-epoch limit. The same validation grid was then ranked again. This prevents a slowly improving candidate from being rejected solely because it reached the earlier limit."
        )
        table(
            [
                "Extended trial",
                "Original epochs",
                "Completed epochs",
                "Max NLL difference, first 50 epochs",
            ],
            [
                [
                    r["checkpoint"].replace("tuning/", "").replace(".json", ""),
                    r["old_epochs"],
                    r["new_epochs"],
                    f"{r['first_50_epoch_max_nll_difference']:.3g}",
                ]
                for r in convergence["extended_trials"]
            ],
        )
    else:
        p(
            "Fresh reruns allow 150 epochs from the start, with stopping after seven epochs without validation improvement. Full curves and selected checkpoint metadata are retained."
        )
    p(
        "Equal tuning budgets make this project comparison easier to interpret, but they do not guarantee that every architecture has reached its best possible performance. Validation NLL and treated-expression MSE also answer different questions. The report therefore retains simple baselines and reports transfer and response-direction metrics separately."
    )

    page("How the code produces the results")
    table(
        ["File / directory", "Responsibility"],
        [
            [
                "MLP_PRnet_controls/match_controls.py",
                "Join source metadata; retain eligible DMSO matches; preserve supplied split assignments",
            ],
            [
                "MLP_PRnet_controls/prepare_features.py and encoder.py",
                "Compute the two fingerprints and frozen ChemicalVAE features",
            ],
            [
                "extension/benchmark.py",
                "Residual/direct MLP definitions and PRnet construction",
            ],
            [
                "extension/balanced_train.py; tune_models.py",
                "Shared fitting, validation checkpointing and equal tuning budget",
            ],
            [
                "extension/fit_baselines.py; group_holdout.py",
                "Training-only baselines and stricter cohort",
            ],
            [
                "extension/scientific_data.py; evaluate_balanced.py",
                "Common data loaders, prediction and metrics",
            ],
            [
                "extension/diagnose_final.py; calibration.py",
                "Chemical shuffling, noise and uncertainty diagnostics",
            ],
            [
                "dose_matched/measure.py",
                "Reconstruct external measurements from supplied GEO files",
            ],
            [
                "extension/summarize_revision.py; verify_revision.py",
                "Paired comparisons, subgroup summaries and integrity checks",
            ],
            [
                "presentation/figures.py; poster.py",
                "Vector scientific figures, pipeline and poster",
            ],
        ],
    )
    p(
        "Checkpoints are selected by the lowest validation NLL. Complete learning curves are retained, including unstable runs. Verification checks sample and compound separation, chemical distances, training-only scaling, checkpoint selection, score arithmetic and negative controls. These are consistency checks, not independent scientific replication."
    )
    p(
        f"The audit covers {verification['checked_checkpoints']} checkpoints, seven eligible studies and 42 external labels. Original train, validation and test assignments have no target-compound overlap. Audit details and checkpoint hashes are in extension/balanced/verification.json and checkpoint_manifest.csv."
    )
    code(
        "python -m pip install -r requirements.txt\npython run.py check\npython run.py evaluate\npython run.py figures\npython run.py report\npython run.py notebook"
    )
    p(
        "Analysis.ipynb contains the executed analysis and plots. Run python run.py train --fresh to repeat training. The full submission ZIP contains prepared expression arrays, supplied splits, encoder weights, checkpoints and source-data checksums. Restore these assets before verification or evaluation from a GitHub clone; see docs/DATA.md. Figures and reports can be rebuilt from the saved tables alone. Main results are in extension/balanced; the separate source-recipe PRnet run is in extension/reference."
    )

    page("Relation to the brief and the next scientific step")
    table(
        ["Amaury project objective", "Evidence supplied", "Remaining qualification"],
        [
            [
                "Review models and useful metrics",
                "PRnet, three encodings, simple baselines, response and direction metrics",
                "Focused review; not an exhaustive survey",
            ],
            [
                "Train PRnet on LINCS and test external regulation",
                "Published network fitted on the screened cohort; separate source-recipe run retained",
                "Not an exact reproduction of the original article’s dataset or scores",
            ],
            [
                "Train an MLP and compare with PRnet",
                "Three representations, three seeds, equal small tuning budget",
                "Not an exhaustive architecture/optimizer search",
            ],
            [
                "Unseen compounds and SKBR3",
                "Compound-disjoint internal test; external analogue screening; excluded SKBR3",
                "No scaffold holdout; only one unseen cell line",
            ],
            [
                "Regulation of six specified genes",
                "Seven dose-matched studies, 42 labels and sensitivity subsets",
                "Limited independent evidence; no calibrated unchanged class",
            ],
            [
                "Robustness to noise",
                "Synthetic control-noise experiment; archived replicate subsampling",
                "Does not cover all biological and measurement noise",
            ],
            [
                "Optional enhancement",
                "Control-residual prediction and molecular representation comparison",
                "Mechanistic or prospective benefit is not established",
            ],
        ],
    )
    p(
        "The benchmark addresses the reliability questions in the brief by testing chemical-input dependence, added value over simple alternatives, transfer across evaluation settings and predictive uncertainty. The strongest conclusions concern the tested cohort; evidence for independent external generalization and practical compound selection remains limited."
    )
    h("The highest-value follow-up")
    p(
        "Freeze a new set of independent, dose/time-matched GEO contrasts before making another model change. Keep compounds, cell lineages and control groups distinct where feasible. Define biologically meaningful response and unchanged-label criteria before scoring. Compare the selected model against unchanged controls, mean response, a linear model and a control-only network. Finally, validate any proposed compound ranking using a functional experimental endpoint."
    )
    p(
        "The supplied analyses cover the available cohort. Independent external validation and prospective experimental evidence remain separate research steps."
    )

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
    for number, reference in enumerate(references, start=1):
        p(f"{number}. {reference}")
    (PROJECT / "report_content.json").write_text(
        json.dumps(
            {"title": "Chemical perturbation response prediction", "pages": pages},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print("Prepared", len(pages), "report sections.")


if __name__ == "__main__":
    main()
