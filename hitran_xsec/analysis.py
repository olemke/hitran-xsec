import os
import logging
import multiprocessing as mp
from typing import List
from copy import copy

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
from .fit import func_2straights, apply_tfit, apply_pressure_fit

# from pyarts.workspace import Workspace

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


def create_cf4_temperature_plot(
    xscdir, output_dir, ylim=None, narrow=True, linear=False, logspec=False
):
    species = "CF4"
    xfi = prepare_data(xscdir, output_dir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    bands = [band for band in xfi.cluster_by_band_and_pressure()]

    # for b in bands:
    b = [b for b in bands[1]]
    xs = b[0]
    xsec_tp = sorted(xs, key=lambda t: t.temperature)
    xsec_tp: List[XsecFile] = sorted(
        _cluster2(xsec_tp, 0, key=lambda n: n.nfreq),
        key=lambda n: len(n),
        reverse=True,
    )[0]

    refi = 2
    ref = xsec_tp[refi]
    xsec_ref = ref.data
    fgrid = np.linspace(ref.wmin, ref.wmax, len(ref.data))

    xmaxi = np.argmax(xsec_ref)
    if narrow:
        centerf = 5939
        freqis = [centerf + offset for offset in range(-2, 3)]
    else:
        # freqis = (np.argmax(xsec_ref[800:900])+825, xmaxi - 1400, xmaxi - 400, xmaxi - 150, xmaxi)
        freqis = (xmaxi - 1400, xmaxi - 400, xmaxi - 150, xmaxi)

    fig, axes = plt.subplots(3, constrained_layout=True, figsize=(8, 8))
    ax = axes[0]
    for x in xsec_tp:
        ax.plot(
            fgrid, x.data / 10000, label=f"{x.temperature:.1f}", rasterized=True, lw=1.5
        )
    legend = ax.legend(
        title="CF4 $T_{ref}$=" + f"{ref.temperature:.1f}", fontsize="xx-small", ncol=3
    )
    plt.setp(legend.get_title(), fontsize="x-small")
    ax.set_ylabel("$\\sigma$")
    ax.set_xlim(1250, 1290)
    if logspec:
        ax.set_yscale("log")

    ax = axes[1]
    for x in xsec_tp:
        if linear:
            x.data = x.data - xsec_ref
        else:
            x.data = np.log(x.data / xsec_ref)
        ax.plot(fgrid, x.data, label=f"{x.temperature:.1f}", rasterized=True, lw=1.5)

    if linear:
        lnsigma = "$\\sf \\sigma-\\sigma_{T_{ref}}$"
    else:
        lnsigma = "$\\sf ln(\\sigma/\\sigma_{T_{ref}})$"
    ax.set_ylabel(lnsigma)
    if narrow:
        ax.set_xlim(1280.89, 1281.21)
    else:
        ax.set_xlim(1250, 1290)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    ax.legend(fontsize="xx-small", ncol=3)

    xsec_values = [
        x.data[freqis[i]]
        for x in xsec_tp
        for i in range(len(freqis))
        if not (np.isnan(x.data[freqis[i]]) or np.isinf(x.data[freqis[i]]))
    ]

    # Frequency marker lines
    for j in range(len(freqis)):
        freqi = freqis[j]
        freq = fgrid[freqi]
        for ax in axes[0:2]:
            ax.plot((freq, freq), ax.get_ylim(), zorder=1, lw=1.5)
            ax.annotate(f"{j + 1}", (freq, ax.get_ylim()[1]), fontsize="xx-small")

    # Temperature comparison plots
    ax2 = axes[2]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax2.set_prop_cycle(color=colors[len(xsec_tp) :])
    for j in range(len(freqis)):
        freqi = freqis[j]
        freq = fgrid[freqi]
        temps = np.array([x.temperature for x in xsec_tp])
        xsec = np.array([x.data[freqi] for x in xsec_tp])
        digs = 3 if narrow else 1
        if linear:
            t = temps - temps[refi]
        else:
            t = np.log(temps / temps[refi])
        ax2.plot(t, xsec, marker="x", label=f"({j + 1}) {freq:.{digs}f}")
    ax2.legend(fontsize="xx-small", ncol=2)
    ax2.set_ylabel(lnsigma)
    if linear:
        ax2.set_xlabel("$\\sf T-T_{ref}$")
    else:
        ax2.set_xlabel("$\\sf ln(T/T_{ref})$")

    plotdir = os.path.join(output_dir, "plots")
    os.makedirs(plotdir, exist_ok=True)
    plotfile = os.path.join(
        plotdir,
        f"{xsec_tp[0].species}_{'linear' if linear else 'ln'}_"
        f"{xsec_tp[0].wmin:.0f}-{xsec_tp[0].wmax:.0f}_"
        f"{xsec_tp[0].pressure:.0f}P-{'narrow' if narrow else 'wide'}.pdf",
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


def compare_fit_vs_reference(xscdir, outdir):
    species = "CFC-12"
    bandmin = 850
    targettemp = 296

    # species = "CFC-11"
    # bandmin = 810
    # targettemp = 296

    artsspecies = species.translate(str.maketrans(dict.fromkeys("-")))
    infile = os.path.join(outdir, species, "cfc.xml")
    logger.info(f"Reading {infile}")
    cfc_combined = axml.load(infile)
    axsec = [x for x in cfc_combined if x.species == artsspecies][0]
    xi = 0

    xfi = prepare_data(xscdir, outdir, species).files
    ref = [
        x
        for x in xfi
        if abs(x.wmin - bandmin) < 1
        and abs(x.temperature - targettemp) < 1
        and x.pressure > 100000
    ]

    if len(ref) != 1:
        raise RuntimeError(f"Unexpected number ({len(ref)} of reference spectra")

    ref = ref[0]
    # ws = Workspace()
    # ws.ReadXML(ws.hitran_xsec_data, os.path.join(outdir, species, "cfc.xml"))
    # ws.abs_speciesSet(species=[artsspecies + "-HXSEC"])
    # ws.VectorNLinSpace(ws.f_grid, ref.nfreq, ref.fmin, ref.fmax)
    # ws.jacobianOff()
    # ws.ArrayOfIndexSet(ws.abs_species_active, [0])
    # ws.VectorSet(ws.abs_p, [ref.pressure])
    # ws.VectorSet(ws.abs_t, [ref.temperature])
    # ws.IndexSet(ws.abs_xsec_agenda_checked, 1)
    # ws.abs_xsec_per_speciesInit(nlte_do=0)
    # ws.abs_xsec_per_speciesAddHitranXsec(apply_tfit=1)
    # fit_t = ws.abs_xsec_per_species.value[0].flatten()
    # ws.abs_xsec_per_speciesInit(nlte_do=0)
    # ws.abs_xsec_per_speciesAddHitranXsec(apply_tfit=0)
    # fit_not = ws.abs_xsec_per_species.value[0].flatten()

    # xf = [x for x in xfi if "CFC-11_216.5K-7.5Torr_810.0-880.0_00.xsc" in x.filename][0]
    xf = [
        x
        for x in xfi
        if abs(x.fmin - axsec.fmin[xi]) < 1e9
        and abs(x.temperature - axsec.reftemperature[xi]) < 1
        and abs(x.pressure - axsec.refpressure[xi]) < 1
    ]
    if len(xf) != 1:
        raise RuntimeError(f"Unexpected number ({len(xf)} of reference spectra")
    xf = xf[0]

    tfit = apply_tfit(
        ref.temperature - axsec.reftemperature[xi],
        axsec.tfit_slope[xi],
        axsec.tfit_intersect[xi],
    )

    fit_t = apply_pressure_fit(
        xf.data + tfit,
        xf.fmin,
        xf.fmax,
        ref.pressure - axsec.refpressure[xi],
        axsec.coeffs,
    )
    fit_t /= 10000

    fit_not = apply_pressure_fit(
        xf.data, xf.fmin, xf.fmax, ref.pressure - axsec.refpressure[xi], axsec.coeffs
    )
    fit_not /= 10000

    fig, (axes) = plt.subplots(3, 1, constrained_layout=True, figsize=(8, 10))
    ref.data /= 10000
    ax = axes[0]

    fit = copy(ref)
    if xf.nfreq != ref.nfreq:
        fgrid_xf = np.linspace(xf.fmin, xf.fmax, xf.nfreq)
        fgrid_ref = np.linspace(ref.fmin, ref.fmax, ref.nfreq)
        fit_t = np.interp(fgrid_ref, fgrid_xf, fit_t)
        fit_not = np.interp(fgrid_ref, fgrid_xf, fit_not)

    fit.data = fit_t
    plot_xsec(
        fit,
        ax,
        rasterized=True,
        label=f"ARTS P+T fit, original spectrum: {axsec.reftemperature[xi]:.0f} K, "
        f"{axsec.refpressure[xi] / 100:.0f} hPa",
        lw=3,
    )

    # fit.data = fit_not
    fit.data = fit_not
    plot_xsec(
        fit,
        ax,
        rasterized=True,
        label=f"ARTS P fit, original spectrum: {axsec.reftemperature[xi]:.0f} K, "
        f"{axsec.refpressure[xi] / 100:.0f} hPa",
        lw=2,
    )

    plot_xsec(
        ref,
        ax,
        rasterized=True,
        label=f"Reference spectrum, {ref.temperature:.0f} K, "
        f"{ref.pressure / 100:.0f} hPa",
        lw=1,
    )

    legend = ax.legend(title=species, fontsize="xx-small")
    plt.setp(legend.get_title(), fontsize="x-small")

    ax = axes[1]

    fit = copy(ref)
    fit.data = fit_t - ref.data
    plot_xsec(fit, ax, rasterized=True, label="")

    fit.data = fit_not - ref.data
    plot_xsec(fit, ax, rasterized=True, label="")

    legend = ax.legend(
        title="Absolute difference $xsec_{fit}-xsec_{ref}$", fontsize="xx-small"
    )
    plt.setp(legend.get_title(), fontsize="x-small")

    ax = axes[2]

    fit = copy(ref)
    fit.data = (fit_t - ref.data) / ref.data * 100
    plot_xsec(fit, ax, rasterized=True, label="")

    fit.data = (fit_not - ref.data) / ref.data * 100
    plot_xsec(fit, ax, rasterized=True, label="")
    ax.set_ylim(-0.02, 0.02)

    legend = ax.legend(
        title="% Relative change $(xsec_{fit}-xsec_{ref})/xsec_{ref}$",
        fontsize="xx-small",
    )
    plt.setp(legend.get_title(), fontsize="x-small")

    plotdir = os.path.join(outdir, "plots")
    os.makedirs(plotdir, exist_ok=True)
    plotfile = os.path.join(
        plotdir,
        f"{ref.species}_fit_comparison_"
        f"{ref.wmin:.0f}-{ref.wmax:.0f}-{axsec.reftemperature[xi]}K.pdf",
    )
    logger.info(f"Saving plot {plotfile}")
    plt.savefig(plotfile, dpi=300)


def run_analysis(
    species, xscdir, outdir, fig1=False, fig2=False, fig3=False, fig4=False, **_
):
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
        create_cf4_temperature_plot(xscdir, outdir, narrow=True, ylim=(-0.5, 0.5))
        create_cf4_temperature_plot(xscdir, outdir, narrow=False)
        create_cf4_temperature_plot(
            xscdir, outdir, narrow=True, ylim=(-0.5e-17, 0.5e-17), linear=True
        )
        create_cf4_temperature_plot(xscdir, outdir, narrow=False, linear=True)

    if fig3:
        create_cfc12_zoom_plot(xscdir, outdir)

    if fig4:
        compare_fit_vs_reference(xscdir, outdir)
