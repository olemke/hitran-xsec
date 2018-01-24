import glob
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys

import matplotlib as mpl
from sklearn.ensemble import IsolationForest

mpl.use('Agg')  # noqa

import matplotlib.ticker as mplticker
import numpy
import typhon
import typhon.physics
import typhon.plots
import typhon.arts.xml
import typhon.arts.xsec
from typhon.physics import frequency2wavenumber, wavenumber2frequency
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.integrate import simps

import matplotlib.pyplot as plt

_lorentz_cutoff = None


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

    ax.set_ylim((0, 6))
    ax.set_xlim((0, 1100))
    ax.scatter(pressure_diff / 100, fwhm / 1e9, **kwargs)
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')

    ax.set_title(title)


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
    ax.plot(p / 100,
            fit_func(p, *popt) / 1e9,
            label=f'fit: 2straights\n'
                  f'x0 = {popt[0]:.2f}\n'
                  f'a = {popt[1]:.2e}\n'
                  f'b = {popt[2]:.2e}\n')
    ax.scatter(pressure_diff[~decision] / 100, fwhm[~decision] / 1e9,
               color='red')


def xsec_select_band(xsec_result, flower=0, fupper=numpy.Inf):
    return [x for x in xsec_result if
            flower < frequency2wavenumber(x['fmin'] / 100) < fupper]


def scatter_and_fit(xsec_result, species, datadir):
    fig, ax = plt.subplots()

    if species == 'CFC11':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=800,
                                          fupper=1000)),
                     species, label='810-880')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=1000)),
                     species, label='1050-1120')
        plot_fit(ax, *calc_fwhm_and_pressure_difference(xsec_result),
                 outliers=False)
    elif species == 'CFC12':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, fupper=840)),
                     species, label='800-1270')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=840,
                                          fupper=1000)),
                     species, label='850-950')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=1000)),
                     species, label='1050-1200')
        fwhm, pressure_diff = calc_fwhm_and_pressure_difference(xsec_result)
        plot_fit(ax, fwhm, pressure_diff)
    elif species == 'HCFC22':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result,
                                          flower=725, fupper=1385)),
                     species, label='730-1380')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result,
                                          flower=755, fupper=865)),
                     species, label='760-860')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result,
                                          flower=1065, fupper=1200)),
                     species, label='1070-1195')
        plot_fit(ax, *calc_fwhm_and_pressure_difference(xsec_result))
    elif species == 'HFC134a':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result,
                                          flower=745, fupper=1605)),
                     species, label='750-1600')
    else:
        raise RuntimeError('Unknown species')

    ax.legend()

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

    ax.legend()

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
    logging.info(f"Reading {filename}")
    with open(filename) as f:
        header = f.readline()
        data = numpy.hstack(
            list(map(convert_line_to_float, f.readlines())))

    return hitran_raw_xsec_to_dict(header, data)


def torr_to_pascal(torr):
    return torr * 101325. / 760.


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
    ax.legend()

    plt.savefig(os.path.join(outdir, 'available.pdf'))


def plot_xsec(ax, xsec, **kwargs):
    fgrid = numpy.linspace(xsec['fmin'], xsec['fmax'], xsec['nfreq'])

    if kwargs is None:
        kwargs = {}

    if 'label' not in kwargs:
        kwargs['label'] = (f"{xsec['temperature']:.0f} K,"
                           f"{xsec['pressure']:.0f} Pa")

    ax.plot(fgrid, xsec['data'], **kwargs)


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
    xsec_conv = xsec1.copy()
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

    ax.yaxis.set_major_formatter(mplticker.FormatStrFormatter('%1.0e'))
    ax.legend(loc=1)
    ax.set_ylabel('RMS')
    ax.set_xlabel('FWHM of Lorentz filter [GHz]')
    ax.set_title(title)

    fig.savefig(fname)
    plt.close(fig)
    logging.info(f'Saved: {fname}')

    fig, ax = plt.subplots()

    linewidth = 0.5
    # Plot xsec at low pressure
    plot_xsec(ax, xsecs['low'], linewidth=linewidth)

    # Plot xsec at high pressure
    plot_xsec(ax, xsecs['high'], linewidth=linewidth)

    # Plot convoluted xsec
    xsecs['conv'], conv, width = xsec_convolve_f(xsecs['low'],
                                                 fwhms[rms_min_i] / 2,
                                                 run_lorentz_f, _lorentz_cutoff)

    plot_xsec(ax, xsecs['conv'], linewidth=linewidth,
              label=f'Lorentz FWHM {fwhms[rms_min_i]/1e9:1.2g} GHz')

    ax.legend(loc=1)
    ax.set_title(title)

    fname = f"{xsec_name}_xsec_conv.pdf"
    fname = os.path.join(outdir, fname)
    fig.savefig(fname)
    plt.close(fig)
    logging.info(f'File saved: {fname}')


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
    logging.info(f"Calc {xsec_name}")

    rms = numpy.zeros((fwhm_nsteps,))
    fwhms = numpy.linspace(fwhm_min, fwhm_max, fwhm_nsteps, endpoint=True)

    fgrid_conv = numpy.linspace(xsecs['low']['fmin'],
                                xsecs['low']['fmax'],
                                xsecs['low']['nfreq'])

    fgrid_high = numpy.linspace(xsecs['high']['fmin'],
                                xsecs['high']['fmax'],
                                xsecs['high']['nfreq'])

    if len(xsecs['high']['data']) != len(fgrid_high):
        logging.error(f"Size mismatch in data (skipping): nfreq: "
                      f"{xsecs['high']['nfreq']} "
                      f"datasize: {len(xsecs['high']['data'])} "
                      f"header: {xsecs['high']['header']}")
        return None

    for i, fwhm in enumerate(fwhms):
        # logging.info(f"Calculating {fwhm/1e9:.3f} for {xsec_name}")
        xsecs['conv'], conv, width = xsec_convolve_f(xsecs['low'], fwhm / 2,
                                                     run_lorentz_f,
                                                     _lorentz_cutoff)
        # logging.info(f"Calculating done {fwhm/1e9:.3f} for {xsec_name}")
        if width < 10:
            logging.warning(
                f"Very few ({width}) points used in Lorentz function for "
                f"{xsec_name} at FWHM {fwhm/1e9:.2} GHz.")

        xsecs['high_interp'] = xsecs['conv'].copy()
        xsecs['high_interp']['data'] = numpy.interp(fgrid_conv, fgrid_high,
                                                    xsecs['high']['data'])

        rms[i] = calc_xsec_rms(xsecs['conv'], xsecs['high_interp'])

    rms_optimum_fwhm = fwhms[numpy.argmin(rms)]

    logging.info(f"Done {xsec_name}")

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


def combine_inputs(filepattern, temps, freqs, name):
    """Create list of inputs.

    Puts low pressure and high pressure data together in pairs.
    """
    if isinstance(filepattern, str):
        infiles = glob.glob(filepattern)
    else:
        infiles = []
        for i in filepattern:
            infiles.extend(glob.glob(i))
    xsecs = [read_hitran_xsec(f) for f in infiles]
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
          '  COMMAND: rms, scatter or plot\n'
          '  SPECIES: CFC11, CFC12, HFC134a or HCFC22')


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s',
                        datefmt='%b %d %H:%M:%S')

    p = mp.Pool(processes=16)
    # p = mp.Pool()

    logging.info('Reading cross section files')
    if len(sys.argv) > 3 and sys.argv[2] == 'CFC11':
        species = sys.argv[2]
        inputs = combine_inputs(
            'cfc11/*00.xsc',
            (190, 201, 208, 216, 225, 232, 246, 260, 272),
            (810, 1050),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'CFC12':
        species = sys.argv[2]
        inputs = combine_inputs(
            'cfc12/*00.xsc',
            (190, 201, 208, 216, 225, 232, 246, 260, 268, 272),
            (800, 850, 1050),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'HCFC22':
        species = sys.argv[2]
        inputs = combine_inputs(
            ['hcfc22/*_730*.xsc', 'hcfc22/*_760*.xsc', 'hcfc22/*_1070*.xsc'],
            (181, 190, 200, 208, 216, 225, 233, 251, 270, 296),
            (730, 760, 1070,),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'HFC134a':
        species = sys.argv[2]
        inputs = combine_inputs(
            ['hfc134a/*_750*.xsc'],
            (190, 200, 216, 231, 250, 270, 295),
            (750,),
            species)
    else:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]
    outdir = sys.argv[3]
    outfile = os.path.join(outdir, 'output.txt')

    if command == 'debug':
        fig, (ax1, ax2) = plt.subplots(2, 1)
        a = inputs[0][0]
        b = inputs[76][0]
        ax1.set_title('CFC11')
        ax1.plot(a['data'],
                 label=f"T: {a['temperature']:.0f}, p: {a['pressure']/100:.0f}",
                 rasterized=True)
        ax1.plot(b['data'],
                 label=f"T: {b['temperature']:.0f}, p: {b['pressure']/100:.0f}",
                 rasterized=True)
        ax1.set_xticks([])
        ax1.legend()
        ax2.plot((b['data'] - a['data']) / a['data'], rasterized=True)
        ax2.set_ylim([-1, 3])
        ax2.set_xticks([])
        fig.savefig('diff.pdf', dpi=300)
    if command == 'debug2':
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        a = inputs[0][0]
        ax1.set_title('CFC11')
        fgrid = numpy.linspace(a['fmin'], a['fmax'], len(a['data']))
        linewidth = 0.75
        fwhm = 2.35e9
        ax1.plot(a['data'],
                 label=f"T: {a['temperature']:.0f}, p: {a['pressure']/100:.0f}",
                 rasterized=True, linewidth=linewidth)
        data, conv1, width = xsec_convolve_f(a, fwhm / 2, run_lorentz_f,
                                             _lorentz_cutoff)
        ax1.plot(data['data'],
                 label=f"T: {a['temperature']:.0f}, p: {a['pressure']/100:.0f}",
                 rasterized=True, linewidth=linewidth)
        ax1.set_xticks([])
        b = a.copy()
        nth = 42
        b['nfreq'] = a['nfreq'] / nth
        print("fstep:",
              (wavenumber2frequency(b['fmax'] - b['fmin']) / b[
                  'nfreq'] * 100) / 1e9)
        b['data'] = numpy.interp(fgrid[::nth], fgrid, a['data'])
        ax1.legend()
        ax2.plot(b['data'],
                 label=f"T: {a['temperature']:.0f}, p: {a['pressure']/100:.0f}",
                 rasterized=True, linewidth=linewidth)
        data, conv2, width = xsec_convolve_f(b, fwhm / 2, run_lorentz_f,
                                             _lorentz_cutoff)
        ax2.plot(data['data'],
                 label=f"T: {a['temperature']:.0f}, p: {a['pressure']/100:.0f}",
                 rasterized=True, linewidth=linewidth)
        ax2.legend()
        ax2.set_ylim(ax1.get_ylim())
        fwidth = a['nfreq'] * a['fmin'] * a['fmax']
        ax3.plot(
            numpy.linspace(0., fwidth, a['nfreq'], endpoint=True),
            conv1,
            label=f"lorentz 1",
            rasterized=True, linewidth=linewidth)
        ax3.plot(
            numpy.linspace(0., fwidth, b['nfreq'], endpoint=True),
            conv2,
            label=f"lorentz 2",
            rasterized=True, linewidth=linewidth)
        ax3.legend()
        fig.savefig('compare.pdf', dpi=300)
    elif command == 'avail':
        os.makedirs(outdir, exist_ok=True)
        plot_available_xsecs(inputs, species, outdir)
    elif command == 'rms':
        os.makedirs(outdir, exist_ok=True)

        logging.info(f'Calculating RMS')
        res = [p.apply_async(optimize_xsec, args[0:2]) for args in inputs]
        results = [r.get() for r in res if r]
        logging.info(f'Done {len(results)} calculations')

        save_output(outfile, results)
        logging.info(f'Saved output to {outfile}')
    elif command == 'genarts':
        print("Generating ARTS XML file")
        results = load_output(outfile)
        xsecs = []
        refpressure = []
        reftemperature = []
        fmin = []
        fmax = []
        for xsec in inputs:
            if (230 <= xsec[0]['temperature'] <= 235
                    and xsec[0]['fmin'] not in fmin):
                xsecs.append(xsec[0]['data'])
                refpressure.append(xsec[0]['pressure'])
                reftemperature.append(xsec[0]['temperature'])
                fmin.append(xsec[0]['fmin'])
                fmax.append(xsec[0]['fmax'])

        fwhm, pressure_diff = calc_fwhm_and_pressure_difference(results)
        popt, pcov, decision = do_fit(fwhm, pressure_diff)
        xsec_data = typhon.arts.xsec.XsecRecord(
            sys.argv[2],
            popt,
            wavenumber2frequency(numpy.array(fmin) * 100.),
            wavenumber2frequency(numpy.array(fmax) * 100.),
            numpy.array(refpressure),
            numpy.array(reftemperature),
            xsecs)
        typhon.arts.xml.save((xsec_data,),
                             os.path.join(outdir, sys.argv[2] + '.xml'))

    elif command == 'testarts':
        xsecdata = typhon.arts.xml.load(
            os.path.join(outdir, sys.argv[2] + '.xml'))
        print(xsecdata)
    elif command == 'scatter':
        logging.info(f'Loading results from {outfile}')
        results = load_output(outfile)
        logging.info(f'Creating scatter plot and fit')
        scatter_and_fit(results, species, outdir)
    elif command == 'plot':
        logging.info(f'Loading results from {outfile}')
        results = load_output(outfile)
        logging.info(f'Plotting RMS and Xsecs')
        res = [p.apply_async(generate_rms_and_spectrum_plots,
                             (*args, result, ioutdir))
               for args, result, ioutdir in
               zip(inputs, results, itertools.repeat(outdir))]
        [r.get() for r in res if r]
    else:
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
