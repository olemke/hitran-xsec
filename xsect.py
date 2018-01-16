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
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


_hwhm_i_width = 20

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

    # ax.set_ylim((0, 6))
    ax.scatter(pressure_diff / 100, fwhm / 1e9, **kwargs)
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')

    ax.set_title(title)


def calc_fwhm_and_pressure_difference_orig(xsec_result, hwhm=_hwhm_i_width):
    fwhm = numpy.array(
        [typhon.physics.wavenumber2frequency(
            2 * r['optimum_width']
            * (r['fmax'] - r['fmin'])
            / r['nfreq'] * 100) / float(hwhm) for r in
         xsec_result])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in
         xsec_result])

    return fwhm, pressure_diff


def calc_fwhm_and_pressure_difference(xsec_result, hwhm=20):
    fwhm = 2 * numpy.array(
        [typhon.physics.wavenumber2frequency(
            r['optimum_width'] / float(hwhm)
            * (r['fmax'] - r['fmin'])
            / r['nfreq'] * 100) for r in
            xsec_result])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in
         xsec_result])

    return fwhm, pressure_diff


def do_fit(fwhm, pressure_diff, fit_func=func_2straights, outliers=True):
    if outliers:
        data = numpy.hstack((pressure_diff.reshape(-1, 1), fwhm.reshape(-1, 1)))
        forrest = IsolationForest(contamination=0.001)
        forrest.fit(data)
        decision = forrest.predict(data) != -1
    else:
        decision = numpy.ones_like(fwhm, dtype='bool')
    popt, pcov = curve_fit(fit_func, pressure_diff[decision], fwhm[decision],
                           p0=(30000, 1e6, 1e6))
    return popt, pcov, decision


def plot_fit(ax, fwhm, pressure_diff, outliers=True):
    # fit_func = func_bent_line
    # popt, pcov = curve_fit(fit_func, pressure_diff, fwhm)

    # p0=(300, 1e+06, 1e+06))
    # p0=(1, 1, 1, 0.5))
    # p0=(3e8, 2.16e-03, 1.04e+00))
    # p0=(-7e+10, 3.5e+09, 2e+06, 1e+09))
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
    return [x for x in xsec_result if x['fmin'] > flower and x['fmin'] < fupper]


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
        # s = fwhm < 5e9
        # fwhm = fwhm[s]
        # pressure_diff = pressure_diff[s]
        plot_fit(ax, fwhm, pressure_diff)
    elif species == 'CFC13':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, fupper=770)),
                     species, label='765-805')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=770,
                                          fupper=1000)),
                     species, label='1065-1140')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         xsec_select_band(xsec_result, flower=1100)),
                     species, label='1170-1235')
        plot_fit(ax, *calc_fwhm_and_pressure_difference(xsec_result))
    else:
        raise RuntimeError('Unknown species')

    ax.legend()

    fig.savefig(os.path.join(datadir, 'xsec_scatter.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots()

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if x['source_temp'] <= 240]),
                 species, label='T ≤ 240K')

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if
                      x['source_temp'] > 240 and x['source_temp'] <= 250]),
                 species, label='240K < T ≤ 250K')
    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if
                      x['source_temp'] > 250 and x['source_temp'] <= 270]),
                 species, label='250K < T ≤ 270K')
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
    # header information if not correct for some files
    xsec_dict['nfreq'] = len(data)
    xsec_dict['pressure'] = torr_to_pascal(xsec_dict['pressure'])
    return xsec_dict


def read_hitran_xsec(filename):
    with open(filename) as f:
        header = f.readline()
        data = numpy.hstack(
            list(map(convert_line_to_float, f.readlines())))

    return hitran_raw_xsec_to_dict(header, data)


def torr_to_pascal(torr):
    return torr * 101325 / 760


def plot_available_xsecs():
    files = 'cfc11/*.xsc'
    data = {}

    for f in map(open, glob.iglob(files)):
        name = f.readline()
        data[name] = numpy.hstack(
            list(map(convert_line_to_float, f.readlines())))

    xsecs = list(map(lambda x: hitran_raw_xsec_to_dict(x, data[x]), data))
    fig, ax = plt.subplots()
    xsecs_sel = [x for x in xsecs if x['fmin'] < 1000]
    ax.scatter([x['temperature'] for x in xsecs_sel],
               numpy.array([x['pressure'] for x in xsecs_sel]),
               alpha=0.5)
    xsecs_sel = [x for x in xsecs if x['fmin'] >= 1000]
    ax.scatter([x['temperature'] for x in xsecs_sel],
               numpy.array([x['pressure'] for x in xsecs_sel]),
               alpha=0.5)
    ax.yaxis.set_major_formatter(typhon.plots.HectoPascalFormatter())
    ax.invert_yaxis()

    plt.savefig('available.pdf')


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


def lorentz(x, x0, gamma, i=1):
    return i * gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def lorentz_pdf(x, x0, gamma):
    return gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def run_lorentz(npoints):
    ret = lorentz_pdf(numpy.linspace(0, npoints, npoints),
                      npoints / 2,
                      npoints / _hwhm_i_width)
    ret /= numpy.sum(ret)
    return ret


def xsec_convolve_f(xsec1, fwhm, owidth, convfunc):
    fstep = typhon.physics.wavenumber2frequency(
        (xsec1['fmax'] - xsec1['fmin']) / xsec1['nfreq'] * 100)

    width = _hwhm_i_width * int(fwhm / fstep)
    if width == 0: width = 1
    print("width:", width, "owidth:", owidth)

    conv_f = convfunc(width)
    xsec_extended = numpy.hstack(
        (numpy.ones((width // 2,)) * xsec1['data'][0],
         xsec1['data'],
         numpy.ones(((width + 1) // 2 - 1,)) * xsec1['data'][-1]))
    xsec_conv = xsec1.copy()
    xsec_conv['data'] = numpy.convolve(xsec_extended, conv_f, 'valid')

    return xsec_conv


def xsec_convolve(xsec1, width, convfunc):
    conv_f = convfunc(width)
    xsec_extended = numpy.hstack(
        (numpy.ones((width // 2,)) * xsec1['data'][0],
         xsec1['data'],
         numpy.ones(((width + 1) // 2 - 1,)) * xsec1['data'][-1]))
    xsec_conv = xsec1.copy()
    xsec_conv['data'] = numpy.convolve(xsec_extended, conv_f, 'valid')
    return xsec_conv


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
        f"{xsecs['low']['longname']}_{xsecs['low']['fmin']:.0f}"
        f"-{xsecs['low']['fmax']:.0f}_{xsecs['low']['temperature']:.1f}K_"
        f"{xsecs['low']['pressure']:.0f}P_{xsecs['high']['temperature']:.1f}K_"
        f"{xsecs['high']['pressure']:.0f}P_{xsec_result['npoints_max']}")

    fname = f"{xsec_name}_rms.pdf"
    fname = os.path.join(outdir, fname)

    npoints_arr = range(xsec_result['npoints_min'], xsec_result['npoints_max'],
                        xsec_result['npoints_step'])

    rms = numpy.array(xsec_result['rms'])
    rms_min = numpy.min(rms)
    rms_min_i = numpy.argmin(rms)
    rms_min_n = npoints_arr[numpy.argmin(rms)]

    fig, ax = plt.subplots()
    fwhm = 2 * typhon.physics.wavenumber2frequency(
        numpy.arange(xsec_result['npoints_min'],
                     xsec_result['npoints_max'],
                     xsec_result['npoints_step'])
        * (xsecs['low']['fmax'] - xsecs['low']['fmin'])
        / xsecs['low']['nfreq'] * 100) / float(_hwhm_i_width)

    if rms_min_n < rms.size:
        ax.plot((fwhm[rms_min_i] / 1e9, fwhm[rms_min_i] / 1e9),
        # ax.plot((rms_min_i, rms_min_i),
                (numpy.min(rms), numpy.max(rms)),
                linewidth=1,
                label=f'Minimum {rms_min:.2e}@{fwhm[rms_min_i]:1.2g} GHz')

    ax.plot(fwhm / 1e9, rms,
    # ax.plot(numpy.arange(xsec_result['npoints_min'],
    #                      xsec_result['npoints_max'],
    #                      xsec_result['npoints_step']),
    #         rms,
            label=f"T1: {xsecs['low']['temperature']:.1f}K "
                  f"P1: {xsecs['low']['pressure']:.0f}P "
                  f"f: {xsecs['low']['fmin']:1.0f}"
                  f"-{xsecs['low']['fmax']:1.0f}\n"
                  f"T2: {xsecs['high']['temperature']:.1f}K "
                  f"P2: {xsecs['high']['pressure']:.0f}P "
                  f"f: {xsecs['high']['fmin']:1.0f}"
                  f"-{xsecs['high']['fmax']:1.0f}",
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
    npoints = rms_min_n
    fmin = xsecs['low']['fmin']
    fmax = xsecs['low']['fmax']
    nf = xsecs['low']['nfreq']
    print(
        f"hpress: {xsecs['high']['pressure']} width: {2*typhon.physics.wavenumber2frequency(float(npoints)/_hwhm_i_width*(fmax-fmin)/nf*100)/1e9} - fwhm: {fwhm[rms_min_i]/1e9}")
    # xsecs['conv'] = xsec_convolve(xsecs['low'], npoints, run_lorentz)
    xsecs['conv'] = xsec_convolve_f(xsecs['low'], fwhm[rms_min_i] / 2, npoints,
                                    run_lorentz)

    plot_xsec(ax, xsecs['conv'], linewidth=linewidth,
              label=f'Lorentz FWHM {fwhm[rms_min_i]/1e9:1.2g} GHz')

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

    npoints_min = 1
    npoints_max = 1000
    step = 2

    xsec_name = (
        f"{xsecs['low']['longname']}_{xsecs['low']['fmin']:.0f}"
        f"-{xsecs['low']['fmax']:.0f}_{xsecs['low']['temperature']:.1f}K_"
        f"{xsecs['low']['pressure']:.0f}P_{xsecs['high']['temperature']:.1f}K_"
        f"{xsecs['high']['pressure']:.0f}P_{npoints_max}")
    logging.info(f"Calc {xsec_name}")

    rms = numpy.zeros((round((npoints_max - npoints_min) / step),))
    npoints_arr = range(npoints_min, npoints_max, step)

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
        return {}

    for i, npoints in enumerate(npoints_arr):
        xsecs['conv'] = xsec_convolve(xsecs['low'], npoints, run_lorentz)

        xsecs['high_interp'] = xsecs['conv'].copy()
        xsecs['high_interp']['data'] = numpy.interp(fgrid_conv, fgrid_high,
                                                    xsecs['high']['data'])

        rms[i] = calc_xsec_rms(xsecs['conv'],
                               xsecs['high_interp'])

    rms_min_n = npoints_arr[numpy.argmin(rms)]

    logging.info(f"Done {xsec_name}")

    return {
        'source_pressure': float(xsecs['low']['pressure']),
        'target_pressure': float(xsecs['high']['pressure']),
        'source_temp': float(xsecs['low']['temperature']),
        'target_temp': float(xsecs['high']['temperature']),
        'fmin': float(xsecs['low']['fmin']),
        'fmax': float(xsecs['low']['fmax']),
        'nfreq': int(xsecs['low']['nfreq']),
        'optimum_width': int(rms_min_n),
        'npoints_min': int(npoints_min),
        'npoints_max': int(npoints_max),
        'npoints_step': int(step),
        'rms': rms.tolist(),
    }


def xsec_select(xsecs, freq, freq_epsilon, temp, temp_epsilon):
    xsec_sel = [x for x in xsecs if
                numpy.abs(x['fmin'] - freq) < freq_epsilon and numpy.abs(
                    x['temperature'] - temp) < temp_epsilon]

    return sorted(xsec_sel, key=lambda xsec: xsec['pressure'])


def combine_inputs(infiles, temps, freqs, name):
    """Create list of inputs.

    Puts low pressure and high pressure data together in pairs.
    """
    infiles = glob.glob(infiles)
    xsecs = [read_hitran_xsec(f) for f in infiles]
    inputs = []
    for temperature in temps:
        for freq in freqs:
            xsecs_sel = xsec_select(xsecs, freq, 10, temperature, 2)
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
          '  SPECIES: CFC11,12 or 13')


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s',
                        datefmt='%b %d %H:%M:%S')

    p = mp.Pool()

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
    elif len(sys.argv) > 3 and sys.argv[2] == 'CFC13':
        species = sys.argv[2]
        inputs = combine_inputs(
            'cfc13/*01.xsc',
            (203, 213, 233, 253, 273, 293),
            (765, 1065, 1170),
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
        ax1.plot(a['data'], label=f"T: {a['temperature']:.0f}", rasterized=True)
        ax1.plot(b['data'], label=f"T: {b['temperature']:.0f}", rasterized=True)
        ax1.set_xticks([])
        ax1.legend()
        ax2.plot((b['data'] - a['data']) / a['data'], rasterized=True)
        ax2.set_ylim([-1, 3])
        ax2.set_xticks([])
        fig.savefig('diff.pdf', dpi=300)
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
            if xsec[0]['temperature'] >= 230 and xsec[0][
                'temperature'] <= 235 and xsec[0]['fmin'] not in fmin:
                xsecs.append(xsec[0]['data'])
                refpressure.append(xsec[0]['pressure'])
                reftemperature.append(xsec[0]['temperature'])
                fmin.append(xsec[0]['fmin'])
                fmax.append(xsec[0]['fmax'])

        fwhm, pressure_diff = calc_fwhm_and_pressure_difference(results)
        popt, pcov, decision = do_fit(fwhm, pressure_diff)
        xsec_data = typhon.arts.xsec.XsecRecord(sys.argv[2],
                                                popt,
                                                typhon.physics.wavenumber2frequency(
                                                    numpy.array(fmin) * 100.),
                                                typhon.physics.wavenumber2frequency(
                                                    numpy.array(fmax) * 100.),
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
        if len(sys.argv) >= 4:
            plotname = sys.argv[3]
        scatter_and_fit(results, species, outdir)
    elif command == 'plot':
        logging.info(f'Loading results from {outfile}')
        results = load_output(outfile)
        logging.info(f'Plotting RMS and Xsecs')
        res = [p.apply_async(generate_rms_and_spectrum_plots,
                             (*args, result, ioutdir))
               for args, result, ioutdir in
               zip(inputs, results, itertools.repeat(outdir))]
        results = [r.get() for r in res if r]
    else:
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
