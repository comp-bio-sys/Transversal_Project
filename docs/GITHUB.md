# Repository organization

The repository contains the analysis code, experiment settings, result tables, figures, report and executed notebook.

| Directory | Contents |
|---|---|
| `MLP_PRnet_controls/` | Data preparation, molecular encoding and PRnet implementation |
| `extension/` | Model fitting, evaluation and robustness analyses |
| `extension/balanced/` | Saved settings, checkpoint metadata and result tables |
| `dose_matched/` | External-study curation and reconstructed measurements |
| `presentation/` | Plotting code, exported figures and finished poster |
| `docs/` | Project brief, full report and data instructions |

Start with the [project overview](../README.md), [methods](../METHODS.md) or [analysis notebook](../Analysis.ipynb). The saved result tables support regenerating plots and rerunning the notebook without retraining. Use `python run.py figures` to rebuild plots and `python run.py notebook` to execute the existing notebook.

Evaluation and training commands refresh the results, plots and notebook. The report text, PDF and poster are not regenerated; the HTML report displays the updated plot files.

Expression arrays, encoder weights and trained checkpoints are distributed separately in the full submission ZIP. [DATA.md](DATA.md) lists the files needed for prediction, evaluation and training. The repository retains their settings, metadata and source references; `.gitignore` excludes these binary assets from regular commits.
