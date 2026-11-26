# Data and model sources

| Source | Use |
|---|---|
| [PharmGEO versioned archive](https://zenodo.org/records/16556994) | Published differential-expression lists, drug experiment annotations and PubChem-linked InChIs |
| [PharmGEO paper](https://doi.org/10.1021/acs.jcim.5c02241) | Cross-study pharmaco-transcriptomic resource and curation |
| [Broad release dashboard](https://clue.io/releases/data-dashboard) | Level 3 treatment and vehicle matrices, sample metadata and matched controls |
| [Broad chemical annotation](https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt) | Structures for the SKBR3 test treatments |
| [PRnet source](https://github.com/Perturbation-Response-Prediction/PRnet) | Published network and LINCS training recipe |
| Supplied project files | Original expression, compound assignment, sample usage and frozen ChemicalVAE encoder |

The PharmGEO archive is `PharmGEO.tar`, release 16556994, MD5 d3f689967df46af59839e2eafe19fefa as listed by Zenodo. The extraction script requests individual members by HTTP range. The training representation and gene order follow the supplied benchmark. File origins for that benchmark are listed in `../MLP_PRnet_controls/SOURCES.md`.
