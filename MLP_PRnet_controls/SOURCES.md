# Sources

- Original supplied files: `l1000_level3_filtered_train_clean.feather`, `sample_usage.feather`, and `compound_split.csv`.
- [CLUE release dashboard](https://clue.io/releases/data-dashboard) and its [public release index](https://s3.amazonaws.com/data.clue.io/releases/data-releases_v4.json).
- [LINCS 2020 sample metadata](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt).
- [Level 3 control matrix](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_ctl_n188708x12328.gctx).
- [Level 3 compound matrix](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_trt_cp_n1805898x12328.gctx), used to verify sample coverage and expression scale.
- [PRnet source repository](https://github.com/Perturbation-Response-Prediction/PRnet), Xiaoning Qi and contributors. The downloaded source and Apache-2.0 license are retained under `vendor/PRnet`. `prnet_core.py` contains the four network classes; AST equality with the downloaded source is checked in `extension/verify_revision.py`.
- [PRnet paper](https://www.nature.com/articles/s41467-024-53457-1).
- ChemicalVAE encoder calculation from the supplied `chemicalvae_mlp.py`; supplied `zinc.json` and `zinc_encoder.h5` are retained under `encoder`.

Source checksums, download verification, and target-scale checks are saved in the data audit files. Accessed 25 November 2026.

The retained `vendor/PRnet/README.md` is the upstream documentation. Its example notebooks are in the [original PRnet repository](https://github.com/Perturbation-Response-Prediction/PRnet), rather than this project's source subset. Follow the root README for this benchmark's commands.
