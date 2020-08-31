import os
import logging

import numpy as np
import typhon.arts.xml as axml
from typhon.physics import frequency2wavenumber
from .xsec import pascal_to_torr

logger = logging.getLogger(__name__)


def create_markdown_table(tabledata):
    sep = " | "
    ret = sep.join(tabledata[0].keys())
    ret += "\n"
    ret += sep.join(["---"] * len(tabledata[0].keys()))
    for d in tabledata:
        ret += "\n"
        ret += sep.join(d.values())

    return ret


# List of gases for CFC paper
gas_list = [
    # Main components
    "H2O",
    "O2",
    "O3",
    "N2",
    "CO",
    "CO2",
    "CH4",
    "N2O",
    # Chlorofluorocarbons
    "CFC11",
    "CFC12",
    "CFC113",
    "CFC114",
    "CFC115",
    # Hydrochlorofluorocarbons
    "HCFC22",
    "HCFC141b",
    "HCFC142b",
    # Hydrofluorocarbons
    "HFC23",
    "HFC32",
    "HFC125",
    "HFC134a",
    "HFC143a",
    "HFC152a",
    "HFC227ea",
    "HFC4310mee",
    # Chlorocarbons and Hydrochlorocarbons
    "CH3CCl3",
    "CCl4",
    "CH3Cl",
    "CH2Cl2",
    "CHCl3",
    # Bromocarbons, Hydrobromocarbons and Halons
    "CH3Br",
    "Halon1211",
    "Halon1301",
    "Halon2402",
    # Fully Fluorinated Species
    "NF3",
    "SF6",
    "SO2F2",
    "CF4",
    "C2F6",
    "C3F8",
    "cC4F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C8F18",
]


def create_data_overview(outdir, format="markdown", **_):
    infile = os.path.join(outdir, "cfc_combined.xml")
    logger.info(f"Reading {infile}")
    cfc_combined = axml.load(infile)

    infile = os.path.join(outdir, "cfc_averaged_coeffs.xml")
    logger.info(f"Reading {infile}")
    cfc_averaged_coeffs = axml.load(infile)

    tabledata = []
    for xsec in cfc_combined:
        tabledata.append(
            {
                "Species": xsec.species,
                "# of bands": str(len(xsec.xsec)),
                "f min [1/cm]": ", ".join(
                    f"{frequency2wavenumber(x)/100:.0f}" for x in xsec.fmin
                ),
                "f max [1/cm]": ", ".join(
                    f"{frequency2wavenumber(x)/100:.0f}" for x in xsec.fmax
                ),
                "Pressure fit": "no"
                if np.all(np.isclose(cfc_averaged_coeffs, xsec.coeffs))
                else "yes",
                "Temperature fit": ", ".join(
                    "yes" if len(t) > 1 else "no" for t in xsec.tfit_slope
                ),
                "Reference P [hPa]": ", ".join(
                    f"{x/100:.0f}" if x > 0 else "-" for x in xsec.refpressure
                ),
                "Reference P [Torr]": ", ".join(
                    f"{pascal_to_torr(x):.1f}" if x > 0 else "-" for x in xsec.refpressure
                ),
                "Reference T [K]": ", ".join(f"{x:.1f}" for x in xsec.reftemperature),
            }
        )

    tabledata_paper = [d for s in gas_list for d in tabledata if d["Species"] == s]
    if format == "markdown":
        print(create_markdown_table(tabledata_paper))
