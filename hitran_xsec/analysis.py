import os
import logging
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import typhon.arts.xml as axml
from typhon.arts.xsec import XsecRecord
from typhon.physics import frequency2wavenumber
from .xsec import xsec_config

logger = logging.getLogger(__name__)


def check_tfit(species, output_dir, cfc_combined):
    xsec_list = [
        x
        for x in cfc_combined
        if x.species == species.translate(str.maketrans(dict.fromkeys("-")))
    ]
    if len(xsec_list) == 0:
        logger.warning(f"Species {species} not found")
        return
    xsec: XsecRecord = xsec_list[0]
    logger.info(f"Running analysis for {xsec.species}")
    temps = [200, 240, 280, 320]

    for i in range(len(xsec.xsec)):
        if len(xsec.tfit_slope[i]) < 2:
            continue
        wmin = frequency2wavenumber(xsec.fmin[i]) / 100
        wmax = frequency2wavenumber(xsec.fmax[i]) / 100
        fig, ax = plt.subplots()
        ax.set_title(f"{species}")
        plotdir = os.path.join(output_dir, "..", "plots")
        os.makedirs(plotdir, exist_ok=True)
        plotfile = os.path.join(
            plotdir,
            f"{xsec.species}_applied_tfit_"
            f"{wmin:.0f}-{wmax:.0f}_"
            f"{xsec.refpressure[i]:.0f}P.pdf",
        )
        fgrid = np.linspace(wmin, wmax, len(xsec.xsec[i]))
        for t in temps:
            x = (
                xsec.tfit_slope[i] * (t - xsec.reftemperature[i])
                + xsec.tfit_intersect[i]
            )
            x = x / 10000
            x += xsec.xsec[i]
            if np.any(x < 0):
                logger.warning(
                    f"{species} contains {np.sum(x<0)} negative xsecs in band "
                    f"{wmin:.0f}-{wmax:.0f} @ {t:.0f}K"
                )
            ax.plot(fgrid, x, label=f"{t}", rasterized=True)
        if np.any(xsec.xsec[i] < 0):
            logger.warning(
                f"{species} contains {np.sum(xsec.xsec[i] < 0)} negative xsecs in ref band "
                f"{wmin:.0f}-{wmax:.0f} @ {t:.0f}K"
            )

        ax.plot(
            fgrid,
            xsec.xsec[i],
            label=f"{xsec.reftemperature[i]} Reference",
            rasterized=True,
        )
        ax.plot((wmin, wmax), (0, 0), color="black", lw=1)
        ax.legend(fontsize="xx-small")
        logger.info(f"Writing temperature fit plot {plotfile}")
        plt.savefig(plotfile, dpi=300)
        plt.close(fig)


def run_analysis(species, outdir, **_):
    infile = os.path.join(outdir, "cfc_combined.xml")
    logger.info(f"Reading {infile}")
    cfc_combined = axml.load(infile)

    with mp.Pool(processes=xsec_config.nprocesses) as pool:
        pool.starmap(
            check_tfit, ((s, os.path.join(outdir, s), cfc_combined) for s in species),
        )
