import functools
import os
import sys

import matplotlib.pyplot as plt
import numpy
import typhon.arts.xml as axml
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from scipy.integrate import simps as integrate
from typhon.physics import frequency2wavenumber


def f2w(temp):
    return frequency2wavenumber(temp) / 100.0


def hz2thz(x, pos):
    return "{:.0f}".format(x / 1e12)


def convert_ax_to_wavenumber(ax1, ax2):
    x1, x2 = ax1.get_xlim()
    ax2.set_xlim(f2w(x1), f2w(x2))
    ax2.figure.canvas.draw()


def THzFormatter():
    @FuncFormatter
    def _THzFormatter(x, pos):
        return "{:g}".format(x / 1e12)

    return _THzFormatter


def func_2straights(x, x0, a, b):
    y = numpy.empty_like(x)
    for i, xi in enumerate(x):
        if xi <= x0:
            y[i] = a * xi
        else:
            y[i] = b * (xi - x0) + a * x0

    return y


def lorentz_pdf(x, x0, gamma):
    return gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def plot_spectrum(ax, x, y, **kwargs):
    ax2 = ax.twiny()
    ax.callbacks.connect(
        "xlim_changed", functools.partial(convert_ax_to_wavenumber, ax2=ax2)
    )

    ax.plot(x, y, **kwargs)

    ax.xaxis.set_major_formatter(THzFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%7.3g"))
    ax.set_xlabel("Frequency [THz]")
    ax2.set_xlabel("Wavenumber [cm$^{-1}$]")
    return ax2


def plot_xsec(inputdir):
    fgrid_s = axml.load(os.path.join(inputdir, "TestHitranXsec-simple.f_grid.xml"))
    xsec = axml.load(
        os.path.join(inputdir, "TestHitranXsec-simple.abs_xsec_per_species.xml")
    )
    species_names = axml.load(
        os.path.join(inputdir, "TestHitranXsec-simple.abs_species.xml")
    )

    fig, ax = plt.subplots()

    for species, name in zip(xsec, species_names):
        for band in range(species.shape[1]):
            ax.plot(
                fgrid_s,
                species[:, band],
                label=f"{name[0].split('-')[0]} {band}",
                linewidth=0.75,
            )

    ax.xaxis.set_major_formatter(THzFormatter())
    ax.set_xlabel("THz")

    fig.legend()
    fig.savefig("xsec.pdf", dpi=300)


def plot_y(directory):
    fgrid = axml.load(os.path.join(directory, "TestHitranXsec.f_grid.xml"))
    y = axml.load(os.path.join(directory, "TestHitranXsec.y.xml"))

    fgrid_nocfc = axml.load(os.path.join(directory, "TestHitranXsec-nocfc.f_grid.xml"))
    y_nocfc = axml.load(os.path.join(directory, "TestHitranXsec-nocfc.y.xml"))

    if numpy.max(y) < 100:
        unit = "radiance"
        ylabel = "Spectral radiance $[\\frac{W}{sr⋅m^2⋅Hz}]$"
        int_nocfc = integrate(y_nocfc, fgrid_nocfc)
        int_cfc = integrate(y, fgrid)
        nocfc_extra_label = f" Total: {int_nocfc:.2f} " + "$\\frac{W}{sr⋅m^2}$"
        cfc_extra_label = f" Total: {int_cfc:.2f} " + "$\\frac{W}{sr⋅m^2}$"
    else:
        unit = "bt"
        ylabel = "Brightness temperature $[B_T]$"
        nocfc_extra_label = ""
        cfc_extra_label = ""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0)

    fig.suptitle("Fascod profile without CFCs vs. with CFCs")
    ax1a = plot_spectrum(
        ax1, fgrid_nocfc, y_nocfc, label="$y$" + nocfc_extra_label, rasterized=True
    )
    ax1.plot(fgrid, y, label="$y_{cfc}$" + cfc_extra_label, rasterized=True)
    ax1.set_ylabel(ylabel)
    ax1.xaxis.set_ticks([])
    ax1.set_xlabel("")
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1a.spines["right"].set_visible(False)
    ax1a.spines["bottom"].set_visible(False)

    ax2.set_ylabel("Relative change $\\frac{y_{cfc} - y}{y} [\\%]$")
    ax1.legend()

    next(ax2._get_lines.prop_cycler)["color"]
    next(ax2._get_lines.prop_cycler)["color"]
    axtop = plot_spectrum(ax2, fgrid, (y - y_nocfc) / y_nocfc * 100, rasterized=True)
    axtop.xaxis.set_ticks([])
    axtop.set_xlabel("")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    axtop.spines["right"].set_visible(False)
    axtop.spines["top"].set_visible(False)

    # fig.tight_layout()
    fig.savefig(os.path.join(directory, "y." + unit + ".pdf"), dpi=300)


def plot_compare_coeffs(directory1, directory2):
    fgrid1 = axml.load(os.path.join(directory1, "TestHitranXsec.f_grid.xml"))
    y1 = axml.load(os.path.join(directory1, "TestHitranXsec.y.xml"))

    fgrid2 = axml.load(os.path.join(directory2, "TestHitranXsec.f_grid.xml"))
    y2 = axml.load(os.path.join(directory2, "TestHitranXsec.y.xml"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    fig.suptitle("Species specific coefficients $C_s$ vs. averaged coefficients $C_a$")
    plot_spectrum(ax1, fgrid1, y1, label="$C_s$", rasterized=True)
    ax1.plot(fgrid2, y2, label="$C_a$", rasterized=True)
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlabel("")

    if numpy.max(y1) < 100:
        unit = "radiance"
        ax1.set_ylabel("Spectral radiance $[\\frac{W}{sr⋅m^2⋅Hz}]$")
    else:
        unit = "bt"
        ax1.set_ylabel("Brightness temperature $[B_T]$")

    ax2.set_ylabel("Relative change $\\frac{y_{C_a} - y_{C_s}}{y_{C_s}} [\\%]$")
    ax1.legend()

    axtop = plot_spectrum(ax2, fgrid1, (y2 - y1) / y1 * 100, rasterized=True)
    axtop.xaxis.set_ticklabels([])
    axtop.set_xlabel("")

    fig.savefig(os.path.join("y_coeff_compare." + unit + ".pdf"), dpi=300)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        plot_compare_coeffs(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        plot_y(sys.argv[1])
    else:
        print(f"Usage: {sys.argv[0]} DIRECTORY1 [DIRECTORY2]")
        exit(1)
