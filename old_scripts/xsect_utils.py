import glob
import json
import logging
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy
import typhon
import typhon.arts.xml
import typhon.arts.xsec
import typhon.physics
import typhon.plots
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from sklearn.ensemble import IsolationForest
from typhon.physics import frequency2wavenumber, wavenumber2frequency

_LORENTZ_CUTOFF = None

logger = logging.getLogger(__name__)

SPECIES_MAP = {
    'CFC11': 'CCl3F',
    'CFC12': 'CCl2F2',
    'HCFC22': 'CHClF2',
    'HFC134a': 'CFH2CF3',
}


class LorentzError(RuntimeError):
    pass


def func_linear(x, a, b):
    return a * x + b


def func_quad(x, a, b, c):
    return a * x * x + b * x + c


def func_log(x, a, b, c):
    return a * numpy.log(b * x + c)


def func_bent_line(x, x0, a, b):
    return a * x + b * (x - x0) ** 2 - b * x0 ** 2


def func_2straights(x, x0, a, b):
    y = numpy.empty_like(x)
    c1 = c2 = 0
    for i, xi in enumerate(x):
        if xi <= x0:
            y[i] = a * xi
            c1 += 1
        else:
            y[i] = b * (xi - x0) + a * x0
            c2 += 1

    return y


def scatter_plot(ax, fwhm, pressure_diff, title, **kwargs):
    if kwargs is None:
        kwargs = {}

    ax.scatter(pressure_diff, fwhm / 1e9, **kwargs)
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')

    ax.set_title(title)


def calc_simple_fwhm_and_pressure_difference(xsec_result):
    dfs = [(r['fmax'] - r['fmin']) / r['nfreq'] for r in xsec_result]
    fwhm = numpy.array([df * r['target_pressure'] / r['source_pressure']
                        * numpy.sqrt(r['source_temp']
                                     / r['target_temp']) for r, df in
                        zip(xsec_result, dfs)])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in xsec_result])

    return fwhm, pressure_diff


def calc_fwhm_and_pressure_difference(xsec_result):
    fwhm = numpy.array([r['optimum_fwhm'] for r in xsec_result])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in xsec_result])

    return fwhm, pressure_diff


def do_fit(fwhm, pressure_diff, fit_func=func_2straights, outliers=True):
    if outliers:
        data = numpy.hstack((pressure_diff.reshape(-1, 1), fwhm.reshape(-1, 1)))
        forrest = IsolationForest(contamination=0.001)
        forrest.fit(data)
        decision = forrest.predict(data) != -1
    else:
        decision = numpy.ones_like(fwhm, dtype='bool')
    # Apriori for fitting the two lines
    p0 = (30000., 1e6, 1e6)
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(fit_func, pressure_diff[decision], fwhm[decision],
                           p0=p0)
    return popt, pcov, decision


def plot_fit(ax, fwhm, pressure_diff, outliers=True):
    fit_func = func_2straights
    popt, pcov, decision = do_fit(fwhm, pressure_diff, fit_func=fit_func,
                                  outliers=outliers)
    p = numpy.linspace(numpy.min(pressure_diff), numpy.max(pressure_diff), 200)
    ax.plot(p,
            fit_func(p, *popt) / 1e9,
            label=f'fit: 2straights\n'
                  f'x0 = {popt[0]:.2f}\n'
                  f'a = {popt[1]:.2e}\n'
                  f'b = {popt[2]:.2e}\n')
    ax.scatter(pressure_diff[~decision], fwhm[~decision] / 1e9,
               color='red')
    ax.xaxis.set_major_formatter(typhon.plots.HectoPascalFormatter())


def xsec_select_band2(xsec_result, band, epsilon=10):
    return [x for x in xsec_result
            if band[0] - epsilon < frequency2wavenumber(x['fmin'] / 100)
            < band[0] + epsilon
            and band[1] - epsilon < frequency2wavenumber(x['fmax'] / 100)
            < band[1] + epsilon]


def xsec_select_pressure(xsec_result, pressure, epsilon=100):
    return [x for x in xsec_result
            if pressure - epsilon < x['pressure'] < pressure + epsilon]


def xsec_select_temperature(xsec_result, temperature, epsilon=3):
    return [x for x in xsec_result
            if temperature - epsilon < x['temperature'] < temperature + epsilon]


def xsec_select_band(xsec_result, band):
    return [x for x in xsec_result if
            band[0] - 1. < frequency2wavenumber(x['fmin'] / 100) < band[1] + 1.]


def scatter_and_fit(xsec_result, species, datadir):
    fig, ax = plt.subplots()

    if species == 'CFC11':
        bands = ((810, 880), (1050, 1120))
        outliers = False
    elif species == 'CFC12':
        bands = ((800, 1270), (850, 950), (1050, 1200))
        outliers = True
    elif species == 'HCFC22':
        bands = ((730, 1380), (760, 860), (1070, 1195))
        outliers = False
    elif species == 'HFC134a':
        bands = ((750, 1600), (1035, 1130), (1135, 1140))
        outliers = False
    else:
        raise RuntimeError('Unknown species')

    for band in bands:
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, band=band)),
                     species, label=f'{band[0]}-{band[1]}')
        scatter_plot(ax,
                     *calc_simple_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, band=band)),
                     species, label=f'Freddy\'s approach {band[0]}-{band[1]}')
    plot_fit(ax, *calc_fwhm_and_pressure_difference(xsec_result),
             outliers=outliers)

    ax.legend(fontsize='xx-small')

    ax.set_ylim((0, 6))
    ax.set_xlim((0, 110000))
    ax.xaxis.set_major_formatter(typhon.plots.HectoPascalFormatter())

    fig.savefig(os.path.join(datadir, 'xsec_scatter.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots()

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if x['source_temp'] <= 210]),
                 species, label='T ≤ 210K')

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if 210 < x['source_temp'] <= 240]),
                 species, label='210K < T ≤ 240K')
    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if 240 < x['source_temp'] <= 270]),
                 species, label='240K < T ≤ 270K')
    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if x['source_temp'] > 270]),
                 species, label='270K < T')

    ax.legend(fontsize='xx-small')

    ax.set_ylim((0, 6))
    ax.set_xlim((0, 110000))
    ax.xaxis.set_major_formatter(typhon.plots.HectoPascalFormatter())

    fig.savefig(os.path.join(datadir, 'xsec_scatter_temp.pdf'))
    plt.close(fig)


def convert_line_to_float(line):
    return list(map(float, line.split()))


def hitran_raw_xsec_to_dict(header, data):
    fields = header.split()
    if len(fields) == 10:
        fields_new = fields[:6]
        fields_new.append(fields[6][:9])
        fields_new.append(fields[6][9:])
        fields_new.append(fields[7:])
        fields = fields_new

    fieldnames = {
        'longname': str,
        'fmin': float,
        'fmax': float,
        'nfreq': int,
        'temperature': float,
        'pressure': float,
        'unknown1': float,
        'unknown2': float,
        'name': str,
        'broadener': str,
        'unknown3': int,
    }
    xsec_dict = {fname[0]: fname[1](
        field) for fname, field in zip(fieldnames.items(), fields)}
    xsec_dict['header'] = header
    xsec_dict['data'] = data
    # Recalculate number of frequency points based on actual data values since
    # header information is not correct for some files
    xsec_dict['nfreq'] = len(data)
    xsec_dict['pressure'] = torr_to_pascal(xsec_dict['pressure'])
    xsec_dict['fmin'] = wavenumber2frequency(xsec_dict['fmin'] * 100)
    xsec_dict['fmax'] = wavenumber2frequency(xsec_dict['fmax'] * 100)

    return xsec_dict


def read_hitran_xsec(filename):
    logger.info(f"Reading {filename}")
    with open(filename) as f:
        header = f.readline()
        data = numpy.hstack(
            list(map(convert_line_to_float, f.readlines())))
    xsec = hitran_raw_xsec_to_dict(header, data)
    xsec['filename'] = filename
    return hitran_raw_xsec_to_dict(header, data)


def read_hitran_xsec_multi(filepattern):
    """Read multiple cross section files."""
    if isinstance(filepattern, str):
        infiles = glob.glob(filepattern)
    else:
        infiles = []
        for i in filepattern:
            infiles.extend(glob.glob(i))
    return [read_hitran_xsec(f) for f in infiles]


def torr_to_pascal(torr):
    return torr * 101325. / 760.


def plot_compare_xsec_temp(inputs, title, outdir, diff=False):
    typhon.plots.styles.use('typhon')
    fig, ax = plt.subplots()
    last_temp = 0
    xsecs2plot = []
    last_pressure = None
    for xsec in inputs:
        if last_pressure is None:
            last_pressure = xsec['pressure']
        if numpy.abs(xsec['pressure'] - last_pressure) < 1000:
            if not numpy.isclose(xsec['temperature'], last_temp):
                last_temp = xsec['temperature']
                xsecs2plot.append(xsec)

    xsecs2plot = sorted(xsecs2plot, key=lambda x: x['temperature'])
    nt = 5
    if nt > len(xsecs2plot):
        nt = len(xsecs2plot)
    if diff:
        if len(xsecs2plot) > nt:
            xsecs2plot = numpy.array(xsecs2plot)[
                             numpy.linspace(0, len(xsecs2plot) - 1, num=nt,
                                            endpoint=True, dtype=int)][::-1]
        iref = -1
        # ref = xsecs2plot[len(xsecs2plot) // 2]['data'].copy()
        ref = xsecs2plot[iref]['data'].copy()
        reftemp = xsecs2plot[iref]['temperature']
        for x in xsecs2plot:
            if len(x['data']) != len(ref):
                logger.error('fail!!!')
            x['data'] = ref - x['data']
    else:
        if len(xsecs2plot) > nt:
            xsecs2plot = numpy.array(xsecs2plot)[
                numpy.linspace(0, len(xsecs2plot) - 1, num=nt, endpoint=True,
                               dtype=int)]
    ax.set_prop_cycle(color=typhon.plots.mpl_colors(
        'viridis' if diff else 'viridis_r', len(xsecs2plot)))
    for xsec in xsecs2plot:
        plot_xsec(ax, xsec, linewidth=1)

    typhon.plots.set_xaxis_formatter(typhon.plots.ScalingFormatter(1e9), ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Frequency [GHz]')
    if diff:
        ax.set_ylabel(f'Cross section difference [m^2]')
    else:
        ax.set_ylabel('Cross section [m^2]')
    ax.legend()

    sdiff = '_diff' if diff else ''
    figname = f'xsec_temp_compare_{title}{sdiff}.pdf'
    plt.savefig(os.path.join(outdir, figname))
    logger.info(f'Wrote {figname}')


def plot_compare_xsec_temp_at_freq(inputs, ifreq, title, outdir, reftype='freq',
                                   diff=False):
    typhon.plots.styles.use('typhon')
    fig, ax = plt.subplots()

    inputs = sorted(inputs, key=lambda x: x['temperature'])
    fgrid = numpy.linspace(inputs[0]['fmin'], inputs[0]['fmax'],
                           inputs[0]['nfreq'])
    if diff:
        iref = 0
        if reftype == 'freq':
            ref = numpy.array([i['data'][ifreq[iref]] / 10000 for i in inputs])
            reffreq = fgrid[ifreq[iref]]
        elif reftype == 'temp':
            ref = inputs[iref]['data'].copy()
            reftemp = inputs[iref]['temperature']
            for x in inputs:
                x['data'] = ref - x['data']

    # ax.set_prop_cycle(color=typhon.plots.mpl_colors('viridis', len(ifreq)))
    for i_frequency in ifreq:
        y = numpy.array([i['data'][i_frequency] / 10000 for i in inputs])
        if diff and reftype == 'freq':
            y = ref - y
        x = [i['temperature'] for i in inputs]
        p = [i['pressure'] for i in inputs]
        ax.plot(x, y, marker='x', label=f'{fgrid[i_frequency]/1e12:g} THz - '
                                        f'{frequency2wavenumber(fgrid[i_frequency])/100:.0f} 1/cm')

    # ax.set_prop_cycle(
    #     color=typhon.plots.mpl_colors('viridis_r', len(xsecs2plot)))
    # for xsec in xsecs2plot:
    #     plot_xsec(ax, xsec, linewidth=1)
    #
    # typhon.plots.set_xaxis_formatter(typhon.plots.scalingformatter(1e9), ax=ax)
    if diff and reftype == 'freq':
        ax.set_title(
            title + f', ref freq {frequency2wavenumber(reffreq)/100:.0f} cm^-1')
    elif diff and reftype == 'temp':
        ax.set_title(
            title + f', ref temp {reftemp:.0f} K')
    else:
        ax.set_title(title)

    ax.set_xlabel('Temperature [K]')
    if diff:
        ax.set_ylabel(f'Cross section difference [m^2]')
    else:
        ax.set_ylabel('Cross section [m^2]')
    ax.legend(fontsize='xx-small')
    #
    sdiff = f'_diff_{reftype}' if diff else ''
    figname = f'xsec_temp_freq_compare_{title}{sdiff}.pdf'
    plt.savefig(
        os.path.join(outdir, figname))
    logger.info(f'Wrote {figname}')


def plot_available_xsecs(inputs, title, outdir):
    fig, ax = plt.subplots()
    ax.scatter([x[0]['temperature'] for x in inputs],
               numpy.array([x[0]['pressure'] for x in inputs]),
               label='reference pressure')
    ax.scatter([x[1]['temperature'] for x in inputs],
               numpy.array([x[1]['pressure'] for x in inputs]),
               label='target pressures')
    ax.yaxis.set_major_formatter(typhon.plots.HectoPascalFormatter())
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('T [K]')
    ax.set_ylabel('P [hPa]')
    ax.legend()

    plt.savefig(os.path.join(outdir, 'xsec_datasets.pdf'))


def plot_xsec(ax, xsec, **kwargs):
    fgrid = numpy.linspace(xsec['fmin'], xsec['fmax'], xsec['nfreq'])

    if kwargs is None:
        kwargs = {}

    if 'label' not in kwargs:
        kwargs['label'] = (f"{xsec['temperature']:.0f} K, "
                           f"{xsec['pressure']:.0f} Pa")

    ax.plot(fgrid, xsec['data'] / 10000., **kwargs)  # Convert to m^2


def run_mean(npoints):
    return numpy.ones((npoints,)) / npoints


def lorentz_pdf(x, x0, gamma):
    return gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def run_lorentz_f(npoints, fstep, hwhm, cutoff=None):
    ret = lorentz_pdf(
        numpy.linspace(0, fstep * npoints, npoints, endpoint=True),
        fstep * npoints / 2,
        hwhm)
    if cutoff is not None:
        ret = ret[ret > numpy.max(ret) * cutoff]
    if len(ret) > 1:
        return ret / simps(ret)
    else:
        return numpy.array([1.])


def xsec_convolve_f(xsec1, hwhm, convfunc, cutoff=None):
    fstep = (xsec1['fmax'] - xsec1['fmin']) / xsec1['nfreq']

    conv_f = convfunc(int(xsec1['nfreq']), fstep, hwhm, cutoff=cutoff)
    width = len(conv_f)
    xsec_conv = deepcopy(xsec1)
    xsec_conv['data'] = fftconvolve(xsec1['data'], conv_f, 'same')

    return xsec_conv, conv_f, width


def xsec_convolve_simple(xsec1, hwhm, cutoff=None):
    fstep = (xsec1['fmax'] - xsec1['fmin']) / xsec1['nfreq']
    conv_f = run_lorentz_f(int(xsec1['nfreq']), fstep, hwhm,
                           cutoff=cutoff)
    width = len(conv_f)
    xsec_conv = deepcopy(xsec1)
    xsec_conv['data'] = fftconvolve(xsec1['data'], conv_f, 'same')

    return xsec_conv, conv_f, width


def calc_xsec_std(xsec1, xsec2):
    return numpy.std(
        xsec1['data'] / numpy.sum(xsec1['data']) - xsec2['data'] / numpy.sum(
            xsec2['data']))


def calc_xsec_log_rms(xsec1, xsec2):
    data1log = numpy.log(xsec1['data'])
    data1log[numpy.isinf(data1log)] = 0
    data2log = numpy.log(xsec2['data'])
    data2log[numpy.isinf(data2log)] = 0
    return numpy.sqrt(numpy.mean(numpy.square(
        data1log / numpy.sum(data1log) - data2log / numpy.sum(data2log))))


def calc_xsec_rms(xsec1, xsec2):
    return numpy.sqrt(numpy.mean(numpy.square(
        xsec1['data'] / numpy.sum(xsec1['data']) - xsec2['data'] / numpy.sum(
            xsec2['data']))))


def generate_rms_and_spectrum_plots(xsec_low, xsec_high, title, xsec_result,
                                    outdir=''):
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

    fwhms = numpy.linspace(xsec_result['fwhm_min'], xsec_result['fwhm_max'],
                           xsec_result['fwhm_nsteps'], endpoint=True)

    rms = numpy.array(xsec_result['rms'])
    rms_min = numpy.min(rms)
    rms_min_i = numpy.argmin(rms)

    fig, ax = plt.subplots()
    if rms_min_i < rms.size:
        ax.plot((fwhms[rms_min_i] / 1e9, fwhms[rms_min_i] / 1e9),
                (numpy.min(rms), numpy.max(rms)),
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
            * numpy.sqrt(xsecs['low']['temperature']
                         / xsecs['high']['temperature']))
    xsec_simple, conv, width = xsec_convolve_simple(
        xsecs['low'],
        xsec_simple_fwhm / 2.,
        _LORENTZ_CUTOFF)
    xsec_simple['pressure'] = xsecs['high']['pressure']
    if len(xsec_simple['data']) != len(xsecs['high']['data']):
        fgrid_low = numpy.linspace(xsecs['low']['fmin'],
                                   xsecs['low']['fmax'],
                                   xsecs['low']['nfreq'])

        fgrid_high = numpy.linspace(xsecs['high']['fmin'],
                                    xsecs['high']['fmax'],
                                    xsecs['high']['nfreq'])

        xsec_high = deepcopy(xsecs['low'])
        xsec_high['data'] = numpy.interp(fgrid_low, fgrid_high,
                                         xsecs['high']['data'])
    else:
        xsec_high = xsecs['high']

    rms_simple = calc_xsec_rms(xsec_simple, xsec_high)

    ax.plot((fwhms[0] / 1e9, fwhms[-1] / 1e9),
            (rms_simple, rms_simple),
            linewidth=1,
            label=f'RMS simple approach {rms_simple:.2e}@{xsec_simple_fwhm/1e9:.1f} GHz')

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


def optimize_xsec(xsec_low, xsec_high):
    xsecs = {
        'low': xsec_low,
        'high': xsec_high,
    }

    fwhm_min = 0.01e9
    fwhm_max = 20.01e9
    fwhm_nsteps = 1000

    xsec_name = (
        f"{xsecs['low']['longname']}_"
        f"{frequency2wavenumber(xsecs['low']['fmin'])/100.:.0f}"
        f"-{frequency2wavenumber(xsecs['low']['fmax'])/100.:.0f}_"
        f"{xsecs['low']['temperature']:.1f}K_"
        f"{xsecs['low']['pressure']:.0f}P_{xsecs['high']['temperature']:.1f}K_"
        f"{xsecs['high']['pressure']:.0f}P")
    logger.info(f"Calc {xsec_name}")

    rms = numpy.zeros((fwhm_nsteps,))
    fwhms = numpy.linspace(fwhm_min, fwhm_max, fwhm_nsteps)

    fgrid_conv = numpy.linspace(xsecs['low']['fmin'],
                                xsecs['low']['fmax'],
                                xsecs['low']['nfreq'])

    fgrid_high = numpy.linspace(xsecs['high']['fmin'],
                                xsecs['high']['fmax'],
                                xsecs['high']['nfreq'])

    if len(xsecs['high']['data']) != len(fgrid_high):
        logger.error(f"Size mismatch in data (skipping): nfreq: "
                     f"{xsecs['high']['nfreq']} "
                     f"datasize: {len(xsecs['high']['data'])} "
                     f"header: {xsecs['high']['header']}")
        return None

    for i, fwhm in enumerate(fwhms):
        # logger.info(f"Calculating {fwhm/1e9:.3f} for {xsec_name}")
        xsecs['conv'], conv, width = xsec_convolve_f(xsecs['low'], fwhm / 2,
                                                     run_lorentz_f,
                                                     _LORENTZ_CUTOFF)
        # logger.info(f"Calculating done {fwhm/1e9:.3f} for {xsec_name}")
        if width < 10:
            logger.warning(
                f"Very few ({width}) points used in Lorentz function for "
                f"{xsec_name} at FWHM {fwhm/1e9:.2} GHz.")

        xsecs['high_interp'] = deepcopy(xsecs['conv'])
        xsecs['high_interp']['data'] = numpy.interp(fgrid_conv, fgrid_high,
                                                    xsecs['high']['data'])

        rms[i] = calc_xsec_rms(xsecs['conv'], xsecs['high_interp'])

    rms_optimum_fwhm = fwhms[numpy.argmin(rms)]

    logger.info(f"Done {xsec_name}")

    return {
        'source_pressure': float(xsecs['low']['pressure']),
        'target_pressure': float(xsecs['high']['pressure']),
        'source_temp': float(xsecs['low']['temperature']),
        'target_temp': float(xsecs['high']['temperature']),
        'fmin': float(xsecs['low']['fmin']),
        'fmax': float(xsecs['low']['fmax']),
        'nfreq': int(xsecs['low']['nfreq']),
        'optimum_fwhm': rms_optimum_fwhm,
        'fwhm_min': fwhm_min,
        'fwhm_max': fwhm_max,
        'fwhm_nsteps': int(fwhm_nsteps),
        'rms': rms.tolist(),
    }


def xsec_select(xsecs, freq, freq_epsilon, temp, temp_epsilon):
    xsec_sel = [x for x in xsecs if
                numpy.abs(x['fmin'] - freq) < freq_epsilon and numpy.abs(
                    x['temperature'] - temp) < temp_epsilon]

    return sorted(xsec_sel, key=lambda xsec: xsec['pressure'])


def combine_inputs(xsecs, temps, freqs, name):
    """Create list of inputs.

    Puts low pressure and high pressure data together in pairs.
    """
    inputs = []
    for temperature in temps:
        for freq in freqs:
            xsecs_sel = xsec_select(
                xsecs,
                wavenumber2frequency(freq * 100),
                wavenumber2frequency(10 * 100),
                temperature, 2)
            for t in ((xsecs_sel[0], x2, name) for x2 in xsecs_sel[1:]):
                inputs.append(t)
    return inputs


def save_output(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f)


def load_output(filename):
    with open(filename) as f:
        results = json.load(f)
    return results


def print_usage():
    print(f'usage: {sys.argv[0]} COMMAND SPECIES OUTDIR\n'
          '\n'
          '  COMMAND: rms, scatter, plot or avail\n'
          '  SPECIES: CFC11, CFC12, HFC134a or HCFC22')
