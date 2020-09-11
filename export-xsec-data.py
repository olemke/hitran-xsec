#!/usr/bin/env python3

import json
from gzip import GzipFile
from hitran_xsec.xsec import XsecFileIndex, XsecFile

directory = "/scratch/uni/u237/users/olemke/xsect/all_xsecs/hitran.org/data/xsec/xsc"
species = "CF4"
outfile = f"/scratch/uni/u237/users/olemke/tmp/{species}-xsc.json.gz"

xfi = XsecFileIndex(directory, species)
bands = xfi.cluster_by_band()

bands2 = []
for b in bands:
    band2 = []
    x: XsecFile
    for x in b:
        band2.append(
            {
                "species": x.species,
                "wmin": x.wmin,
                "wmax": x.wmax,
                "fmin": x.fmin,
                "fmax": x.fmax,
                "pressure": x.pressure,
                "temperature": x.temperature,
                "xsec": list(x.data),
            }
        )
    bands2.append(band2)

with GzipFile(outfile, "w") as f:
    f.write(json.dumps(bands2).encode("utf-8"))


# Example for reading the data
# import json
# from gzip import GzipFile
#
# with GzipFile(f"/scratch/uni/u237/users/olemke/tmp/CF4-xsc.json.gz") as f:
#     cf4 = json.loads(f.read().decode("utf-8"))
