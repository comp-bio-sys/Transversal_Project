"""Fetch the archived PharmGEO tables and Broad annotations used for curation.

Run from the project directory. R with readxl is needed for the study workbook.
"""

from pathlib import Path
import subprocess
import urllib.request
import tarfile

raw = Path("tmp/pharm")
raw.mkdir(parents=True, exist_ok=True)
url = "https://zenodo.org/api/records/16556994/files/PharmGEO.tar/content"
# Byte ranges identify specific files in the versioned, uncompressed tar.
for name, start, size in [
    ("genes.Rdata", 142848, 134640321),
    ("metadata.xlsx", 134784000, 907057),
    ("pubchem.Rdata", 135732224, 135340814),
]:
    request = urllib.request.Request(
        url + "?member=" + name, headers={"Range": f"bytes={start}-{start+size-1}"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        assert response.status == 206
        payload = response.read()
    assert len(payload) == size
    (raw / name).write_bytes(payload)
subprocess.run(
    [
        "Rscript",
        "-e",
        """
load('tmp/pharm/genes.Rdata')
targets=c('AKT1','AURKB','EGFR','PIK3CA','SMAD3','TP53')
rows=lapply(names(all_diffsig_list),function(id){
    x=all_diffsig_list[[id]]
    x=x[x$genes %in% targets,,drop=FALSE]
    if(nrow(x)){x$drug_id=id;return(x)}
})
write.csv(do.call(rbind,rows),'tmp/pharm/six.csv',row.names=FALSE)
write.csv(readxl::read_excel('tmp/pharm/metadata.xlsx'),'tmp/pharm/metadata.csv',row.names=FALSE)
load('tmp/pharm/pubchem.Rdata')
write.csv(drug_info[,c('Drug','CID','InChI','CanonicalSMILES','Title')],'tmp/pharm/structures.csv',row.names=FALSE)
""",
    ],
    check=True,
)
