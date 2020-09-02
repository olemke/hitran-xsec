import os
import logging
import multiprocessing as mp
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import typhon.arts.xml as axml
from typhon.arts.xsec import XsecRecord
from typhon.physics import frequency2wavenumber
from typhon.plots import cmap2rgba
from .xsec import xsec_config, _cluster2, XsecFile, xsec_convolve_f, run_lorentz_f
from .calc import prepare_data
from .plotting import plot_xsec
from .fit import func_2straights

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
                f"{species} contains {np.sum(xsec.xsec[i] < 0)} "
                "negative xsecs in ref band "
                f"{wmin:.0f}-{wmax:.0f} @ {xsec.reftemperature:.0f}K"
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


def create_cf4_temperature_plot(xscdir, output_dir):
    species = "CF4"
    xfi = prepare_data(xscdir, output_dir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    bands = [band for band in xfi.cluster_by_band_and_pressure()]

    # for b in bands:
    b = [b for b in bands[1]]
    xs = b[0]
    tpressure = sorted(xs, key=lambda t: t.temperature)
    tpressure: List[XsecFile] = sorted(
        _cluster2(tpressure, 0, key=lambda n: n.nfreq),
        key=lambda n: len(n),
        reverse=True,
    )[0]

    fgrid = np.linspace(tpressure[0].wmin, tpressure[0].wmax, len(tpressure[0].data))

    xmaxi = np.argmax(tpressure[0].data)
    freqis = (xmaxi - 1400, xmaxi - 400, xmaxi - 150, xmaxi)

    fig, axes = plt.subplots(len(freqis) + 1, constrained_layout=True, figsize=(8, 12))
    fig.suptitle("CF4 temperature characteristics")
    ax = axes[0]
    for x in tpressure:
        x.data = x.data / 10000
        ax.plot(fgrid, x.data, label=f"{x.temperature:.1f}", rasterized=True, lw=1.5)

    ax.set_xlim(1275, 1284)

    ax.legend(fontsize="xx-small", ncol=3)

    for i in range(len(freqis)):
        ax2 = axes[i + 1]
        freqi = freqis[i]
        freq = fgrid[freqi]
        ax.plot((freq, freq), (0, 1.4e-20), color="black", lw=1)
        ax.annotate(f"({i+1})", (freq, 1.4e-20), fontsize="xx-small")
        temps = [x.temperature for x in tpressure]
        xsec = [x.data[freqi] for x in tpressure]
        ax2.plot(temps, xsec, marker="x", label=f"({i+1}) {freq:.1f}")
        ax2.legend(fontsize="xx-small")

    plotdir = os.path.join(output_dir, "plots")
    os.makedirs(plotdir, exist_ok=True)
    plotfile = os.path.join(
        plotdir,
        f"{tpressure[0].species}_temp_freq_"
        f"{tpressure[0].wmin:.0f}-{tpressure[0].wmax:.0f}_"
        f"{tpressure[0].pressure:.0f}P.pdf",
    )
    logger.info(f"Writing temperature/frequency plot {plotfile}")
    plt.savefig(plotfile, dpi=300)
    plt.close(fig)


def create_zoom_plot(
    outdir,
    xsecs,
    species,
    zoomx,
    zoomy,
    inset_coord,
    xlim=None,
    scale=None,
    scalestr=None,
    coeffs=None,
):
    fig, ax = plt.subplots(constrained_layout=True)

    axins = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=inset_coord,
        bbox_transform=ax.transData,
        loc=2,
        borderpad=0,
    )

    cmm = cmap2rgba("material", 20)
    cmq = cmap2rgba("qualitative1", 7)
    cols = [cmq[1], cmq[4], cmq[2]]
    ax.set_prop_cycle(color=cols)
    axins.set_prop_cycle(color=cols)

    zorder = 1
    for xs in xsecs:
        if scale:
            xs.data /= scale
        plot_xsec(
            xs,
            ax,
            rasterized=True,
            zorder=zorder,
            label=f"Ref, {xs.temperature:.0f} K, " f"{xs.pressure/100:.0f} hPa",
        )
        plot_xsec(
            xs,
            axins,
            rasterized=True,
            zorder=zorder,
            lw=1 if xs.pressure < 1000 else 1.5,
        )
        zorder += 1

    if coeffs is not None:
        fwhm = func_2straights(
            [xsecs[-1].pressure - xsecs[0].pressure], coeffs[0], coeffs[1], coeffs[2]
        )[0]
        xsec_conv, _, _ = xsec_convolve_f(xsecs[0], fwhm / 2, run_lorentz_f)

        plot_xsec(
            xsec_conv,
            ax,
            zorder=1.5,
            rasterized=True,
            label=f"Fit, {xsec_conv.temperature:.0f} K, "
            f"{xsecs[-1].pressure/100:.0f} hPa",
        )
        plot_xsec(xsec_conv, axins, zorder=1.5, lw=1.5, rasterized=True)

    axins.set_xlim(zoomx)
    axins.set_ylim(zoomy)
    axins.tick_params(axis="both", labelsize="xx-small")
    axins.set_xticks(axins.get_xticks()[1:-1])
    axins.set_yticks(axins.get_yticks()[::2])
    t = axins.yaxis.get_offset_text()
    t.set_size("xx-small")
    axins.set_facecolor("#eceff1")

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(xsecs[0].wmin, xsecs[0].wmax)
    ax.set_ylim(0, ax.get_ylim()[1])

    if scale and not scalestr:
        scalestr = scale
    ax.set_ylabel(f"Crosssection / {scalestr if scalestr else ''}m$^2$")
    ax.set_xlabel("Wavenumber / cm$^{-1}$")

    legend = ax.legend(title=species, fontsize="xx-small", loc="upper right")
    plt.setp(legend.get_title(), fontsize="x-small")

    zcolor = "ty:jetblack"
    line = mlines.Line2D(
        [zoomx[0], inset_coord[0]], [zoomy[1], inset_coord[1]], color=zcolor, lw=0.8,
    )
    ax.add_line(line)
    line = mlines.Line2D(
        [zoomx[1], inset_coord[0] + inset_coord[2]],
        [zoomy[1], inset_coord[1]],
        color=zcolor,
        lw=0.8,
    )
    ax.add_line(line)
    rect = Rectangle(
        (zoomx[0], zoomy[0]),
        zoomx[1] - zoomx[0],
        zoomy[1] - zoomy[0],
        linewidth=0.8,
        facecolor="#eceff1",
        edgecolor=zcolor,
    )
    ax.add_patch(rect)

    plotdir = os.path.join(outdir, "plots")
    os.makedirs(plotdir, exist_ok=True)
    plotfile = os.path.join(
        plotdir,
        f"{xsecs[0].species}_spectrum_zoom_"
        f"{xsecs[0].wmin:.0f}-{xsecs[0].wmax:.0f}_"
        f"{xsecs[0].temperature:.0f}K.pdf",
    )
    logger.info(f"Writing spectrum zoom-in plot {plotfile}")
    plt.savefig(plotfile, dpi=300)
    plt.close(fig)


def create_cfc12_zoom_plot(xscdir, outdir):
    scale = 1e-22
    zoomx = (888, 889)
    zoomy = (0.2e-22 / scale, 1.05e-22 / scale)
    inset_coord = (860, 2.5e-22 / scale, 50, 4.5e-22 / scale)
    species = "CFC-12"

    infile = os.path.join(outdir, "cfc_combined.xml")
    logger.info(f"Reading {infile}")
    cfc_combined = axml.load(infile)

    xfit = [x for x in cfc_combined if x.species == "CFC12"]
    if len(xfit) != 1:
        raise RuntimeError(f"{len(xfit)} matching species found in cfc data.")

    xfi = prepare_data(xscdir, outdir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    xsecs = [
        x
        for x in xfi.files
        if x.wmin == 850
        and x.wmax == 950
        and 273 < x.temperature < 274
        and (x.torr == 7.5 or x.torr == 760.5)
    ]
    print(f"Spacing: {(xsecs[0].wmax-xsecs[0].wmin)/xsecs[0].nfreq}")
    create_zoom_plot(
        outdir,
        xsecs,
        "CFC-12",
        zoomx,
        zoomy,
        inset_coord,
        scale=scale,
        scalestr="10$^{-22}$",
        coeffs=xfit[0].coeffs,
    )


def create_cfc11_zoom_plot(xscdir, outdir):
    zoomx = (836, 837)
    zoomy = (0.8e-22, 1.25e-22)
    inset_coord = (815, 2.5e-22, 24, 4e-22)
    species = "CFC-11"
    xfi = prepare_data(xscdir, outdir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    xsecs = [
        x
        for x in xfi.files
        if x.wmin == 810
        and x.wmax == 880
        and 272 < x.temperature < 274
        and (x.torr == 7.5 or x.torr == 760)
    ]
    for x in xsecs:
        print(f"Spacing: {(x.wmax-x.wmin)/x.nfreq}")
    xlim = (810, 870)
    create_zoom_plot(outdir, xsecs, "CFC-11", zoomx, zoomy, inset_coord, xlim)


def run_analysis(species, xscdir, outdir, fig1=False, fig2=False, fig3=False, **_):
    if fig1:
        infile = os.path.join(outdir, "cfc_combined.xml")
        logger.info(f"Reading {infile}")
        cfc_combined = axml.load(infile)

        with mp.Pool(processes=xsec_config.nprocesses) as pool:
            pool.starmap(
                check_tfit,
                ((s, os.path.join(outdir, s), cfc_combined) for s in species),
            )

    if fig2:
        create_cf4_temperature_plot(xscdir, outdir)

    if fig3:
        create_cfc12_zoom_plot(xscdir, outdir)
