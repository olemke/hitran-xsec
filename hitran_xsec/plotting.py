import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
from typhon.physics import frequency2wavenumber
from typhon.plots import HectoPascalFormatter

from .fit import calc_fwhm_and_pressure_difference, func_2straights, do_fit
from .xsec import (_LORENTZ_CUTOFF, xsec_convolve_f, run_lorentz_f,
                   calc_xsec_rms)

__all__ = [s for s in dir() if not s.startswith('_')]

logger = logging.getLogger('xsec')


def plot_available_xsecs(xsecfileindex, title=None, ax=None):
    """Plots the available temperatures and pressures of cross section data."""
    if ax is None:
        ax = plt.gca()

    ax.scatter([x.temperature for x in xsecfileindex.files],
               np.array([x.pressure for x in xsecfileindex.files]))
    ax.yaxis.set_major_formatter(HectoPascalFormatter())
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.set_xlabel('T [K]')
    ax.set_ylabel('P [hPa]')


def generate_rms_and_spectrum_plots(xsec_low, xsec_high, title, xsec_result,
                                    outdir=''):
    """Plots the RMS for different FWHMs of the Lorentz filter."""
    xsecs = {
        'low': xsec_low,
        'high': xsec_high,
    }

    xsec_name = (
        f"{xsecs['low']['longname']}_"
        f"{frequency2wavenumber(xsecs['low']['fmin'])/100.:.0f}"
        f"-{frequency2wavenumber(xsecs['low']['fmax'])/100.:.0f}"
        f"_{xsecs['low']['temperature']:.1f}K_"
        f"{xsecs['low']['pressure']:.0f}P_{xsecs['high']['temperature']:.1f}K_"
        f"{xsecs['high']['pressure']:.0f}P")

    fname = f"{xsec_name}_rms.pdf"
    fname = os.path.join(outdir, fname)

    fwhms = np.linspace(xsec_result['fwhm_min'], xsec_result['fwhm_max'],
                        xsec_result['fwhm_nsteps'], endpoint=True)

    rms = np.array(xsec_result['rms'])
    rms_min = np.min(rms)
    rms_min_i = np.argmin(rms)

    fig, ax = plt.subplots()
    if rms_min_i < rms.size:
        ax.plot((fwhms[rms_min_i] / 1e9, fwhms[rms_min_i] / 1e9),
                (np.min(rms), np.max(rms)),
                linewidth=1,
                label=f'Minimum RMS {rms_min:.2e}'
                      f'@{fwhms[rms_min_i]/1e9:1.2g} GHz')

    ax.plot(fwhms / 1e9, rms,
            label=f"T1: {xsecs['low']['temperature']:.1f}K "
                  f"P1: {xsecs['low']['pressure']:.0f}P "
                  f"f: {frequency2wavenumber(xsecs['low']['fmin']/100):1.0f}"
                  f"-{frequency2wavenumber(xsecs['low']['fmax']/100):1.0f}\n"
                  f"T2: {xsecs['high']['temperature']:.1f}K "
                  f"P2: {xsecs['high']['pressure']:.0f}P "
                  f"f: {frequency2wavenumber(xsecs['high']['fmin']/100):1.0f}"
                  f"-{frequency2wavenumber(xsecs['high']['fmax']/100):1.0f}",
            linewidth=0.5)

    # Simple approach
    df = (xsecs['low']['fmax'] - xsecs['low']['fmin']) / xsecs['low']['nfreq']
    xsec_simple_fwhm = (
            df * xsecs['high']['pressure'] / xsecs['low']['pressure']
            * np.sqrt(xsecs['low']['temperature']
                      / xsecs['high']['temperature']))
    xsec_simple, conv, width = xsec_convolve_f(
        xsecs['low'],
        xsec_simple_fwhm / 2.,
        run_lorentz_f,
        _LORENTZ_CUTOFF)
    xsec_simple['pressure'] = xsecs['high']['pressure']
    if len(xsec_simple['data']) != len(xsecs['high']['data']):
        fgrid_low = np.linspace(xsecs['low']['fmin'],
                                xsecs['low']['fmax'],
                                xsecs['low']['nfreq'])

        fgrid_high = np.linspace(xsecs['high']['fmin'],
                                 xsecs['high']['fmax'],
                                 xsecs['high']['nfreq'])

        xsec_high = deepcopy(xsecs['low'])
        xsec_high['data'] = np.interp(fgrid_low, fgrid_high,
                                      xsecs['high']['data'])
    else:
        xsec_high = xsecs['high']

    rms_simple = calc_xsec_rms(xsec_simple, xsec_high)

    ax.plot((fwhms[0] / 1e9, fwhms[-1] / 1e9),
            (rms_simple, rms_simple),
            linewidth=1,
            label=f'RMS simple approach {rms_simple:.2e}'
                  f'@{xsec_simple_fwhm/1e9:.1f} GHz')

    ax.yaxis.set_major_formatter(mplticker.FormatStrFormatter('%1.0e'))
    ax.legend(loc=1)
    ax.set_ylabel('RMS')
    ax.set_xlabel('FWHM of Lorentz filter [GHz]')
    ax.set_title(title + f' fspacing: {df/1e9:g} GHz')

    fig.savefig(fname)
    plt.close(fig)
    logger.info(f'Saved: {fname}')

    fig, ax = plt.subplots()

    linewidth = 0.5
    # Plot xsec at low pressure
    plot_xsec(ax, xsecs['low'], linewidth=linewidth)

    # Plot convoluted xsec
    xsecs['conv'], conv, width = xsec_convolve_f(xsecs['low'],
                                                 fwhms[rms_min_i] / 2,
                                                 run_lorentz_f, _LORENTZ_CUTOFF)

    plot_xsec(ax, xsecs['conv'], linewidth=linewidth,
              label=f'Lorentz FWHM {fwhms[rms_min_i]/1e9:1.2g} GHz')

    # Plot xsec at high pressure
    plot_xsec(ax, xsecs['high'], linewidth=linewidth)

    # Plot simple approach
    plot_xsec(ax, xsec_simple, linewidth=linewidth,
              label=f'Simple approach {xsec_simple_fwhm/1e9:1.2g} GHz')

    ax.legend(loc=1)
    ax.set_title(title + f' fspacing: {df/1e9:g} GHz')

    fname = f"{xsec_name}_xsec_conv.pdf"
    fname = os.path.join(outdir, fname)
    fig.savefig(fname)
    plt.close(fig)
    logger.info(f'File saved: {fname}')


def plot_xsec(xsec, ax=None, **kwargs):
    """Plot cross section data."""
    if ax is None:
        ax = plt.gca()

    fgrid = np.linspace(xsec.fmin, xsec.fmax, xsec.nfreq)

    if kwargs is None:
        kwargs = {}

    if 'label' not in kwargs:
        kwargs['label'] = (f"{xsec.temperature:.0f} K, "
                           f"{xsec.pressure:.0f} Pa")

    ax.plot(fgrid, xsec.data / 10000., **kwargs)  # Convert to m^2

    return ax


def plot_fit(ax, fwhm, pressure_diff, outliers=False):
    """Plot the fitting function."""
    fit_func = func_2straights
    popt, pcov, decision = do_fit(fwhm, pressure_diff, fit_func=fit_func,
                                  outliers=outliers)
    p = np.linspace(np.min(pressure_diff), np.max(pressure_diff), 200)
    ax.plot(p,
            fit_func(p, *popt) / 1e9,
            label=f'fit: 2straights\n'
                  f'x0 = {popt[0]:.2f}\n'
                  f'a = {popt[1]:.2e}\n'
                  f'b = {popt[2]:.2e}\n')
    ax.scatter(pressure_diff[~decision], fwhm[~decision] / 1e9,
               color='red')
    ax.xaxis.set_major_formatter(HectoPascalFormatter())


def scatter_plot(fwhm, pressure_diff, title=None, ax=None, **kwargs):
    """Scatter plot of Lorentz filter FWHM vs pressure difference."""
    if kwargs is None:
        kwargs = {}

    if ax is None:
        ax = plt.gca()

    ax.scatter(pressure_diff, fwhm / 1e9, **kwargs)
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')

    if title is not None:
        ax.set_title(title)


def scatter_and_fit(xsecfileindex, rmsoutput, species=None, outliers=False,
                    ax=None):
    """Scatter plot of the FWHM with the lowest RMS."""
    if not rmsoutput:
        raise RuntimeError('RMS output is empty')

    if species is None:
        species = xsecfileindex.files[0].species

    if ax is None:
        ax = plt.gca()

    bands = [(b[0].wmin, b[0].wmax) for b in xsecfileindex.cluster_by_band()]
    for band in bands:
        xsecs = [x for x in rmsoutput
                 if band[0] == x['wmin'] and band[1] == x['wmax']]
        scatter_plot(*calc_fwhm_and_pressure_difference(xsecs),
                     species, ax, label=f'{band[0]}-{band[1]}')
    plot_fit(ax, *calc_fwhm_and_pressure_difference(rmsoutput),
             outliers=outliers)

    ax.legend(fontsize='xx-small')

    ax.set_xlim((0, 110000))
    ax.xaxis.set_major_formatter(HectoPascalFormatter())

    return ax
