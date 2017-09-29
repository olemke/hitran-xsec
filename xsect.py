import glob
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys

import matplotlib as mpl
import numpy

mpl.use('Agg')  # noqa

import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import typhon
import typhon.physics
import typhon.plots


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
                      npoints / 20)
    ret /= numpy.sum(ret)
    return ret


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
    fwhm = typhon.physics.wavenumber2frequency(
        2 * numpy.arange(xsec_result['npoints_min'],
                         xsec_result['npoints_max'],
                         xsec_result['npoints_step'])
        * (xsecs['low']['fmax'] - xsecs['low']['fmin'])
        / xsecs['low']['nfreq'] * 100) / float(20) / 1e9

    if rms_min_n < rms.size:
        ax.plot((fwhm[rms_min_i], fwhm[rms_min_i]),
                (numpy.min(rms), numpy.max(rms)),
                linewidth=1,
                label=f'Minimum {rms_min:.2e}@{fwhm[rms_min_i]:1.2g} GHz')

    ax.plot(fwhm, rms,
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
    xsecs['conv'] = xsec_convolve(xsecs['low'], npoints, run_lorentz)
    plot_xsec(ax, xsecs['conv'], linewidth=linewidth,
              label=f'Lorentz FWHM {fwhm[rms_min_i]:1.2g} GHz')

    ax.legend(loc=1)
    ax.set_title(title)

    fname = f"{xsec_name}_xsec.pdf"
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
    step = 1

    xsec_name = (
        f"{xsecs['low']['longname']}_{xsecs['low']['fmin']:.0f}"
        f"-{xsecs['low']['fmax']:.0f}_{xsecs['low']['temperature']:.1f}K_"
        f"{xsecs['low']['pressure']:.0f}P_{xsecs['high']['temperature']:.1f}K_"
        f"{xsecs['high']['pressure']:.0f}P_{npoints_max}")
    logging.info(f"Calc {xsec_name}")

    rms = numpy.zeros(((npoints_max - npoints_min) // step,))
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


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(asctime)s:%(message)s',
                        datefmt='%b %d %H:%M:%S')

    p = mp.Pool()

    logging.info('Reading cross section files')
    if len(sys.argv) > 3 and sys.argv[2] == 'cfc11':
        inputs = combine_inputs(
            'cfc11/*00.xsc',
            (190, 201, 208, 216, 225, 232, 246, 260, 272),
            (810, 1050),
            'CFC-11')
    elif len(sys.argv) > 3 and sys.argv[2] == 'cfc12':
        inputs = combine_inputs(
            'cfc12/*00.xsc',
            (190, 201, 208, 216, 225, 232, 246, 260, 268, 272),
            (800, 850, 1050),
            'CFC-12')
    else:
        print(f'usage: {sys.argv[0]} COMMAND SPECIES OUTDIR\n'
              '\n'
              '  COMMAND: rms, plot or fit\n'
              '  SPECIES: cfc11 or cfc12')
        sys.exit(1)

    command = sys.argv[1]
    outdir = sys.argv[3]
    outfile = os.path.join(outdir, 'output.txt')

    if command == 'rms':
        os.makedirs(outdir, exist_ok=True)

        res = [p.apply_async(optimize_xsec, args[0:2]) for args in inputs]
        results = [r.get() for r in res if r]
        logging.info(f'{len(results)} calculations')

        save_output(outfile, results)
        logging.info(f'Saved output to {outfile}')
    elif command == 'plot':
        logging.info(f'Loading results from {outfile}')
        load_output(outfile)
        res = [p.apply_async(generate_rms_and_spectrum_plots,
                             (*args, result, ioutdir))
               for args, result, ioutdir in
               zip(inputs, results, itertools.repeat(outdir))]
        results = [r.get() for r in res if r]
    elif command == 'fit':
        logging.info(f'Loading results from {outfile}')
        load_output(outfile)


if __name__ == '__main__':
    main()
