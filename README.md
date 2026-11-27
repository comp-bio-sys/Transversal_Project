# Predicting gene-expression responses to chemical compounds

**Yernur Kenzhegazin · Alan Berdigaliyev · Hugo Telles**  
M2 GENIOMHE-AI · Supervisor: Amaury Giard, IBISC

Can a model predict how a chemical compound changes gene expression in cells it has not seen during training? This project compares three molecular representations in separate MLPs, examines the validation-selected model's learning-rate refinement, and evaluates it against PRnet trained on the same project dataset.

The models use chemical information and measured DMSO control expression to predict treated expression for **978 LINCS L1000 landmark genes**. Experiments cover unseen compounds, an excluded cell line, independent gene-regulation measurements and sensitivity to noise in the control input.

[Report](docs/REPORT.md) · [Report PDF](Research_Report.pdf) · [Notebook](Analysis.ipynb) · [Poster](presentation/Conference_Poster.pdf) · [Methods](METHODS.md) · [Project brief](docs/Project_brief.pdf)

## Workflow

![Data preparation, model training and evaluation](presentation/figures/pipeline.png)

1. Prepare normalized LINCS Level 3 treated profiles at **10 µM and 24 hours** and preserve the supplied compound assignments.
2. Match treatment profiles to DMSO controls and average control replicates gene by gene. Exclude SKBR3 and close analogues of the external screening compounds from training.
3. Train MLPs with FCFP4, ECFP4 or frozen ChemicalVAE features on the same cohort. Compare their validation scores and examine learning-rate tuning.
4. Compare the selected ECFP4 MLP with retrained PRnet across three training seeds. Evaluate expression error, external regulation direction and control-noise sensitivity.

The retained cohort contains **210,438 training profiles**, **25,190 validation profiles** and **27,765 internal test profiles**. The main compound test uses the 16,531 profiles whose compounds have nearest-training ECFP4 Tanimoto similarity ≤ 0.7. [Methods](METHODS.md) describes control matching, exclusions and scaling.

**DMSO has two roles.** Its measured expression is an input to every chemical model. The unchanged-DMSO reference also uses those values directly as a prediction, assuming no treatment effect. This reference has no fitted weights. The supplementary control-only MLP is a separate trained model.

## Model selection

| MLP | Chemical representation | Default validation NLL, seed 42 |
|---|---|---:|
| FCFP4 | 1,024 feature-Morgan bits, radius 2 | 0.10800 |
| ECFP4 | 1,024 atom-Morgan bits, radius 2 | **0.08465** |
| ChemicalVAE | 196 frozen encoder values | 0.11053 |

![Validation comparison and ECFP4 refinement](presentation/figures/model_selection.png)

*Validation Gaussian negative log likelihood (NLL); lower is better. Values are the best checkpoints from seed 42. These bars do not represent biological replicates.*

ECFP4 has the lowest validation NLL. Reducing its learning rate from 0.001 to 0.0003 lowers NLL from **0.08465 to 0.08372**, but worsens SKBR3 MSE from **0.5803 to 0.6537** and external direction accuracy from **26.3 to 24.3 correct out of 42**. The comparison retains the validation-selected setting.

All representations were tuned in parallel. Settings and checkpoints were selected using validation NLL; test scores did not select the lead representation. Training uses early stopping after seven epochs without validation improvement. [Full settings and selection records](METHODS.md#training-and-validation) are retained.

## Results

![ECFP4, PRnet and unchanged DMSO expression on two test sets](presentation/figures/benchmark.png)

*Treated-expression mean squared error (MSE); lower is better. Model bars average three training seeds, dots show individual runs and whiskers show their range. The compound test contains 474 compounds and 16,531 profiles. The SKBR3 test contains 137 compounds and 574 profiles.*

| Model or reference | Low-similarity compound MSE | SKBR3 MSE | External correct directions / 42 |
|---|---:|---:|---:|
| ECFP4 MLP | 0.6723 | 0.6537 | 24.3 |
| PRnet | 0.9055 | 2.1759 | 23.0 |
| Unchanged DMSO | 0.7357 | **0.5689** | Not applicable |

ECFP4 has lower MSE than PRnet in both expression tests and improves on unchanged DMSO for distant compounds. On SKBR3, unchanged DMSO has the lowest error among these three. Simpler fitted baselines also outperform the chemical MLPs in some tests; those results are included in the [supplementary analyses](docs/SUPPLEMENTARY_ANALYSES.md).

The [external evaluation](presentation/figures/external_studies.png) measures up/down regulation for six genes across seven dose- and time-matched GEO studies. Fractional scores are averages across training seeds. The ECFP4–PRnet accuracy difference is 3.2 percentage points, with a descriptive study-bootstrap interval of −13.5 to +19.0 points, so this panel does not establish a clear advantage. Unchanged DMSO predicts zero change and therefore has no up/down score.

The [noise experiment](presentation/figures/control_noise.png) adds Gaussian noise to SKBR3 DMSO inputs while keeping models and treated targets fixed. At one control-replicate standard deviation, ECFP4 MSE increases from **0.6537 to 0.9928**. PRnet changes less, from **2.1759 to 2.2292**, but retains higher total error.

### Limitations

These findings are exploratory. Evaluation sets were examined during development, the main compound test shares control groups with training, and only one unseen cell line was evaluated. Many SKBR3 compounds were seen in other training cell lines, so this test does not hold out both compounds and cell line together. The external panel is small and differs from LINCS in platform and experimental context. PRnet uses the published architecture with project-specific preprocessing and tuning; this is not a reproduction of the paper's numerical results. The results do not establish a reliable compound-ranking system or consistent added value from chemical features.

## Run the analysis

Use **Python 3.13**. The saved results, plots, report and executed notebook can be viewed without installing anything.

```sh
git clone https://github.com/comp-bio-sys/Transversal_Project.git
cd Transversal_Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py figures
python run.py notebook
```

These commands regenerate plots from saved result tables and execute the existing notebook. To open it interactively, install JupyterLab with `python -m pip install jupyterlab`, then run `jupyter lab Analysis.ipynb`.

Prediction, verification and retraining require the separate data-and-model bundle, which has no public download link at present. [Data instructions](docs/DATA.md) list the required files. After restoring them, use `python run.py check`, `python run.py evaluate` or `python run.py train --fresh`. The last command backs up existing experiment outputs before fitting. Evaluation can take hours; full training can take days.

## Repository guide

| Path | Contents |
|---|---|
| [Analysis.ipynb](Analysis.ipynb) | Results walkthrough and prediction example |
| [MLP_PRnet_controls/](MLP_PRnet_controls/) | Data preparation, molecular encodings and PRnet source |
| [extension/](extension/) | Training, evaluation and robustness analyses |
| [extension/balanced/](extension/balanced/) | Result tables, selected settings and checkpoint metadata |
| [dose_matched/](dose_matched/) | External-study curation and reconstructed measurements |
| [presentation/](presentation/) | Scientific plotting code, exported figures and poster |
| [docs/](docs/) | Report, project brief, supplementary analyses and data instructions |

Data and software references are listed in [benchmark sources](MLP_PRnet_controls/SOURCES.md) and [external-data sources](extension/SOURCES.md). Third-party licenses and attribution remain with the corresponding files.
