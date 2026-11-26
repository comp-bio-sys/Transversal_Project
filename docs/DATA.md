# Data and trained models

The full submission ZIP contains the prepared expression data, molecular encoder and trained models used by the analysis. These binary assets are excluded from regular Git commits. The repository keeps their settings, metadata, source references and result tables.

## What works from a GitHub clone?

The executed `Analysis.ipynb`, report, poster and result tables can be read immediately. With the dependencies installed, figures and reports can be rebuilt from those tables. Running predictions, full verification, evaluation or training requires the bundled arrays and weights.

## Restore the experiment files

Extract the full submission ZIP outside the clone. Copy the files below into the same relative locations in the clone, preserving the directory structure.

| Location in the bundle | Files to restore | Purpose |
|---|---|---|
| `MLP_PRnet_controls/` | `l1000_level3_filtered_train_clean.feather` | Supplied treated profiles and original identifiers |
| `MLP_PRnet_controls/data/` | `.feather`, `.npy` and `.npz` files | Sample manifests, measured controls and molecular features |
| `MLP_PRnet_controls/prepared/` | `.npy` and `.npz` files | Targets, row indices and preparation arrays |
| `MLP_PRnet_controls/encoder/` | `zinc_encoder.h5` | Frozen ChemicalVAE encoder; `zinc.json` is versioned |
| `extension/` | `.npy` and `.npz` files | Screened split, training scaling and SKBR3 expression arrays |
| `extension/balanced/` and its subdirectories | `.pt` and `.npz` files | Trained networks, fitted baselines and joint-holdout arrays |
| `extension/reference/` | `PRnet.pt` | Separate PRnet source-recipe comparison |
| `dose_matched/source/` | `.gz` files | GEO expression files and platform annotations for reconstructing external measurements |

After restoring the files, run from the project root:

```sh
python run.py check
```

Keep the CSV and JSON metadata from the same bundle with its arrays and weights. Mixing experiment versions can change sample alignment or checkpoint selection. The file manifest in the submission bundle records file sizes and checksums.

## Data coverage

- LINCS Level 3: 978 directly measured landmark genes, treated profiles and matched measured DMSO controls.
- Screened cohort: 210,438 training, 25,190 validation and 27,765 internal test profiles.
- Low-similarity internal subset: 16,531 profiles for 474 compounds.
- SKBR3: 574 eligible treated profiles for 137 compounds, with 226 measured DMSO controls.
- External panel: seven eligible dose-matched GEO studies and six genes per study.

The full LINCS release is not included. The bundled prepared arrays are sufficient for the supplied fitting and evaluation workflow; reconstructing all upstream data joins may require the public source release. Source locations and download checks are documented in [SOURCES.md](../MLP_PRnet_controls/SOURCES.md) and the data audit files. [METHODS.md](../METHODS.md) explains the filters, control matching and external measurements.

No public bundle URL is configured in this package. Add the actual storage link here when the data-and-model bundle is published.
