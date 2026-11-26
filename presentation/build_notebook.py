"""Create and execute the project walkthrough with an in-process IPython kernel."""

import os
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parent.parent


def main():
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text))

    md(
        """# Chemical perturbation response prediction

**Yernur Kenzhegazin · Alan Berdigaliyev · Hugo Telles**  
M2 GENIOMHE-AI · Supervisor: Amaury Giard, IBISC

This notebook explains the data, models and evaluation. It reads the saved results, displays the main figures and runs a small prediction example when the model and data files are available. Full retraining is described below.

The question is whether chemical structure adds useful information beyond measured control expression, and whether that benefit transfers to unseen compounds, SKBR3 and external experiments. These test sets were inspected during model development, so the analyses are exploratory rather than independent confirmation."""
    )
    code("""from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from IPython.display import display, Image

PROJECT = Path.cwd()
if not (PROJECT / 'extension/balanced').exists():
    raise FileNotFoundError('Open this notebook from the extracted project directory.')
BALANCED = PROJECT / 'extension/balanced'
sys.path.insert(0, str(PROJECT / 'extension'))
sys.path.insert(0, str(PROJECT / 'presentation'))
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, '| pandas:', pd.__version__)
print('Results:', BALANCED.name)""")
    md(
        """## 1. What the files mean

The treated-expression Feather file supplies targets. `compound_split.csv` fixes compound assignments and `sample_usage.feather` fixes eligible sample IDs. Measured DMSO profiles supply the untreated reference. A control group matches detection plate, cell, time and bead batch.

The original split labels are preserved for retained rows. Similarity screening and control availability remove rows; they do not create different training samples for the three encoders. The main comparison uses one common cohort."""
    )
    code(
        """audit = json.loads((BALANCED / 'generalization_audit.json').read_text())
verification = json.loads((BALANCED / 'verification.json').read_text())
display(pd.DataFrame({'profiles': audit['samples'], 'compounds': audit['compounds']}))
print('Integrity checks:', verification['status'])
print('Internal test profiles sharing a training control group:', audit['test_profiles_with_control_group_seen_in_train'])
print('Internal test compounds with a training analogue above 0.7:', audit['test_compounds_above_0_7_similarity_to_train'])
print('Test compounds with identical ECFP4 to training:', audit['test_compounds_with_identical_ECFP4_to_training'])"""
    )
    code(
        "display(Image(filename=str(PROJECT / 'presentation/figures/pipeline.png'), width=1100))"
    )
    md(
        """The diagram follows six steps:

1. Assemble the supplied treated profiles, compound assignments and sample IDs.
2. Match measured DMSO controls by plate, cell, time and bead batch, and average replicates.
3. Screen the cohort, retain the supplied split labels and estimate control scaling from training inputs only.
4. Encode each compound as FCFP4, ECFP4 or ChemicalVAE features for separate MLPs; fit PRnet with dose-scaled FCFP4.
5. Select settings and checkpoints using validation NLL, with the same tuning budget and three seeds.
6. Compare held-out predictions with measurements and baselines, then examine chemical dependence, noise and uncertainty.

Cell icons, feature grids and expression strips are schematic. Their colors do not represent measured values; sample counts and result plots come from the saved analyses. External GEO predictions use a cell-specific mean of LINCS control groups. GEO treated and control samples determine the observed gene directions, so platform differences remain."""
    )
    md(
        """## 2. Models and fitting

FCFP4 describes local functional classes, ECFP4 describes local atom environments, and ChemicalVAE supplies a frozen 196-value molecular embedding. FCFP4 and ECFP4 each have 1,024 bits and radius 2. Each representation enters a separate MLP with the measured control vector.

The residual MLP computes `treated_mean = control + response`. The response head starts at zero; a positive variance head supports Gaussian NLL training. Cell context is encoded by control expression, not a cell-ID embedding. PRnet uses the published PGM architecture, FCFP4 dose encoding and decoder noise. Its final mean averages 32 fixed draws.

Five main models receive three training seeds and the same three-setting AdamW tuning budget. Seed 42 validation NLL selects the setting; every checkpoint is selected by its own validation NLL. Test labels do not select settings. The comparison does not establish globally optimal tuning for any architecture."""
    )
    code(
        """selected = json.loads((BALANCED / 'selected_settings.json').read_text())
tuning = pd.read_csv(BALANCED / 'tuning_validation.csv')
display(tuning.assign(selected=tuning.apply(lambda r: selected[r['model']] == r['setting'], axis=1)))
display(Image(filename=str(PROJECT / 'presentation/figures/training_convergence.png'), width=1100))"""
    )
    md(
        """## 3. Run a saved model on held-out samples

The following cell loads eight internal test rows and the validation-selected ECFP4 checkpoint. It uses the same data loader and inference function as the full benchmark. These eight rows demonstrate execution; their MSE is not the headline test score. The example is skipped when large data or checkpoint files are absent. Restore these files from the full submission archive as described in `docs/DATA.md`. The saved tables and figures can be read without them."""
    )
    code(
        """setting = selected['Residual']
folder = BALANCED / 'weights' if setting == 'default' else BALANCED / 'tuning' / setting
required = [
    PROJECT / 'extension/split_indices.npz',
    PROJECT / 'extension/scaling.npz',
    PROJECT / 'MLP_PRnet_controls/data/sample_manifest.feather',
    PROJECT / 'MLP_PRnet_controls/prepared/indices.npz',
    PROJECT / 'MLP_PRnet_controls/data/chemical_features.npz',
    PROJECT / 'MLP_PRnet_controls/prepared/targets.npy',
    PROJECT / 'MLP_PRnet_controls/data/control_means.npy',
    folder / 'Residual_42.pt',
]
if not all(path.is_file() for path in required):
    print('Prediction example skipped: data or checkpoints are missing. '
          'Restore the assets from the full submission archive; see docs/DATA.md.')
else:
    import torch
    from benchmark import build
    from scientific_data import load_internal, infer

    torch.set_num_threads(4)
    indices = np.load(PROJECT / 'extension/split_indices.npz')['test'][:8]
    example = load_internal(indices)
    network = build('Residual')
    network.load_state_dict(torch.load(folder / 'Residual_42.pt', map_location='cpu', weights_only=True))
    prediction = infer(network, 'Residual', example['features']['ECFP4'], example['control'])
    assert prediction.shape == example['y'].shape == (8, 978)
    assert np.isfinite(prediction).all()
    display(pd.DataFrame({'sample_id': example['sample_id'],
                          'model_mse': ((prediction - example['y']) ** 2).mean(1),
                          'control_mse': ((example['control'] - example['y']) ** 2).mean(1)}))"""
    )
    md(
        """## 4. Full benchmark and simple baselines

The main scores below average three independently initialized model runs. They are not ensemble scores. The linear model, global mean response, cell mean response and unchanged-control baselines provide additional tests of complexity. Cell mean response falls back to the global training response for unseen SKBR3.

The low-similarity subset contains 474 compounds / 16,531 profiles with nearest-training ECFP4 similarity at most 0.7. It avoids fingerprint-identical training analogues present in the original compound-disjoint split, but still shares some control groups with training. It is a subset of the existing test, not newly collected evidence.

MSE measures treated-expression reconstruction. Response correlation and direction measure treatment-minus-control changes. Lower MSE need not imply better regulation prediction or useful drug ranking."""
    )
    code(
        """summary = pd.read_csv(BALANCED / 'summary/model_summary.csv')
main = summary.loc[summary.phase.isin(['tuned', 'baseline']) & summary.dataset.isin(['internal', 'internal_low_similarity', 'skbr3'])]
display(main[['dataset','model','seeds','mse_mean','mse_seed_sd','macro_compound_mse','response_pearson','direction_accuracy']])
fixed = summary.loc[summary.phase.eq('fixed') & summary.dataset.isin(['internal','skbr3'])]
display(fixed[['dataset','model','mse_mean','mse_seed_sd']])
display(Image(filename=str(PROJECT / 'presentation/figures/benchmark.png'), width=1200))
display(Image(filename=str(PROJECT / 'presentation/figures/compound_response.png'), width=1000))"""
    )
    md(
        """## 5. Does the model use chemical information?

For each trained model, chemical features are shuffled within matched control groups while targets, controls and PRnet decoder draws remain fixed. The control-only model must be unchanged. A shuffling penalty indicates input dependence under this diagnostic; it does not prove a causal mechanism or improvement over a separately fitted control-only model.

Paired intervals resample compounds after averaging per-run losses. Control-group intervals are provided separately. Neither approach completely models crossed dependence between chemical compounds and experimental groups."""
    )
    code(
        """permutation = pd.read_csv(BALANCED / 'diagnostics/permutation_seed_means.csv')
assert permutation.loc[permutation.model.eq('ControlOnly'),'mse_increase'].eq(0).all()
display(permutation.groupby(['dataset','model'])[['clean_mse','shuffled_mse','mse_increase']].mean())
display(Image(filename=str(PROJECT / 'presentation/figures/chemical_added_value.png'), width=1000))"""
    )
    md(
        """## 6. Dose-matched external regulation

Seven eligible GEO studies each contribute six genes at 10 µM and 24 hours. Forty-two directions are not 42 independent experiments: genes in the same study share samples and context. Twenty-seven directions have separated replicates; fifteen also pass the descriptive absolute-change threshold of 0.5 on their log-expression scale.

Observed directions use GEO treated and control samples. Predictions use LINCS cell-matched control references, so platform and culture differences remain. Norepinephrine is excluded because it has a training analogue. Small changes can have uncertain signs; the sensitivity subsets are not significance tests."""
    )
    md(
        """### Reading the external-study heatmap

Each row is one compound tested in one cell type; each column is a model. For AKT1, AURKB, EGFR, PIK3CA, SMAD3 and TP53, we ask whether the predicted increase or decrease relative to controls agrees with the direction measured in the GEO experiment.

Each cell counts correct directions out of six, averaged over three independently seeded training runs. Values range from 0 to 6: higher values and darker shading indicate more correct directions. A value of **6.0** means all six genes were correct in every run. **4.7** is a rounded mean across the three runs, not a gene-expression value or a percentage. This score measures direction, not the size of the expression change. The seven studies remain the independent clusters for uncertainty estimation."""
    )
    code(
        """labels = pd.read_csv(PROJECT / 'dose_matched/measured_effects.csv')
labels = labels.loc[labels.chemical_split.eq('eligible')]
display(labels.groupby('gse_id').agg(genes=('genes','size'), separated=('separated_replicates','sum'), clear=('clear_change','sum')))
display(pd.read_csv(BALANCED / 'summary/external_seed_counts.csv').query("phase == 'tuned'"))
display(pd.read_csv(BALANCED / 'summary/paired_external_intervals.csv'))
display(Image(filename=str(PROJECT / 'presentation/figures/external_studies.png'), width=850))"""
    )
    md(
        """## 7. Robustness, uncertainty and stricter separation

Synthetic Gaussian noise perturbs control inputs using the measured within-plate replicate SD. These technical repeats are not biological replicates and do not model every source of transcriptomic noise. Predictive intervals are marginal profile-gene intervals, not simultaneous coverage of the full expression vector.

A secondary split removes control groups from training and validation while preserving compound assignments. Its test groups and compounds are both unseen in fitting. The training set is also smaller, so any change cannot be attributed purely to group reuse. A 2 × 2 ablation separates the residual connection from zero mean-head initialization."""
    )
    code(
        """display(pd.read_csv(BALANCED / 'results/calibration.csv').groupby(['dataset','model'])[['nll','covered95','width95']].mean())
empirical = pd.read_csv(BALANCED / 'diagnostics/empirical_control_resampling.csv')
display(empirical.groupby('model')[['clean_mse','half_control_mse','prediction_rms_change']].mean())
display(Image(filename=str(PROJECT / 'presentation/figures/control_noise.png'), width=850))
display(Image(filename=str(PROJECT / 'presentation/figures/control_group_holdout.png'), width=850))
display(Image(filename=str(PROJECT / 'presentation/figures/residual_ablation.png'), width=1000))"""
    )
    md(
        """## 8. Recompute tables or retrain

Run these commands from the project directory. They are not started automatically by this notebook. Checking and evaluation require the data and checkpoints from the full submission archive; see `docs/DATA.md`.

```sh
python run.py check
python run.py evaluate
```

To fit all models again:

```sh
python run.py train --fresh
```

Retraining is substantially longer than reading saved results. Fixed-setting models, validation tuning and joint holdout are separate result phases.

## 9. What the project can and cannot conclude

The results address the brief through tests of unseen compounds, SKBR3, the six external genes, model components and control noise. They can support a qualified conclusion about chemical input dependence and transfer in this cohort. They do not validate prospective drug-discovery utility, scaffold generalization, unseen doses or a broad range of new cell lines.

The next strongest experiment is a frozen independent external panel with matched dose/time and credible effect labels, followed by functional validation of compound rankings. Repeating training characterizes variability but does not provide independent experimental evidence.

See **Research_Report.pdf** for the full interpretation, code map and references. The motivating single-cell paper is literature context; the result figures here come from the project's bulk L1000 and GEO analyses."""
    )
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
    )
    old = os.getcwd()
    os.chdir(ROOT)
    shell = InteractiveShell.instance()
    count = 0
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(
                    f"Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}"
                )
            outputs = []
            if captured.stdout:
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=captured.stdout
                    )
                )
            if captured.stderr:
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stderr", text=captured.stderr
                    )
                )
            for rich in captured.outputs:
                outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=rich.data, metadata=rich.metadata
                    )
                )
            cell.execution_count = count
            cell.outputs = outputs
    finally:
        os.chdir(old)
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / "Analysis.ipynb")
    print("Executed", count, "notebook cells without errors.")


if __name__ == "__main__":
    main()
