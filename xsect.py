import glob
import json
import multiprocessing as mp
import os
import sys

import matplotlib as mpl
import numpy

mpl.use('Agg')

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
    # header infomation if not correct for some files
    xsec_dict['nfreq'] = len(data)
    xsec_dict['pressure'] = torr_to_pascal(xsec_dict['pressure'])
    return xsec_dict


def read_hitran_xsect(filename):
    with open(filename) as f:
        header = f.readline()
        data = numpy.hstack(
            list(map(convert_line_to_float, f.readlines())))

    return hitran_raw_xsec_to_dict(header, data)


def torr_to_pascal(torr):
    return torr * 101325 / 760


def plot_available_xsects():
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


def plot_xsect(ax, xsect, **kwargs):
    fgrid = numpy.linspace(xsect['fmin'], xsect['fmax'], xsect['nfreq'])

    if kwargs is None:
        kwargs = {}

    if 'label' not in kwargs:
        kwargs[
            'label'] = f"{xsect['temperature']:.0f} K, {xsect['pressure']:.0f} Pa"

    ax.plot(fgrid, xsect['data'], **kwargs)


def run_mean(npoints):
    return numpy.ones((npoints,)) / npoints


def lorentz(x, x0, gamma, i=1):
    return i * gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def lorentz_pdf(x, x0, gamma):
    return gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def run_lorentz(npoints):
    return lorentz_pdf(numpy.linspace(0, npoints, npoints),
                       npoints / 2,
                       npoints / 20)


def xsect_convolve(xsect1, width, convfunc):
    conv_f = convfunc(width)
    xsect_extended = numpy.hstack(
        (numpy.ones((width // 2,)) * xsect1['data'][0],
         xsect1['data'],
         numpy.ones(((width + 1) // 2 - 1,)) * xsect1['data'][-1]))
    xsect_conv = xsect1.copy()
    xsect_conv['data'] = numpy.convolve(xsect_extended, conv_f, 'valid')
    return xsect_conv


def calc_xsect_std(xsect1, xsect2):
    return numpy.std(
        xsect1['data'] / numpy.sum(xsect1['data']) - xsect2['data'] / numpy.sum(
            xsect2['data']))


def calc_xsect_log_rms(xsect1, xsect2):
    data1log = numpy.log(xsect1['data'])
    data1log[numpy.isinf(data1log)] = 0
    data2log = numpy.log(xsect2['data'])
    data2log[numpy.isinf(data2log)] = 0
    return numpy.sqrt(numpy.mean(numpy.square(
        data1log / numpy.sum(data1log) - data2log / numpy.sum(data2log))))


def calc_xsect_rms(xsect1, xsect2):
    return numpy.sqrt(numpy.mean(numpy.square(
        xsect1['data'] / numpy.sum(xsect1['data']) - xsect2['data'] / numpy.sum(
            xsect2['data']))))


def optimize_xsect(xsect_low, xsect_high, title):
    xsects = {
        'low': xsect_low,
        'high': xsect_high,
    }

    npoints_min = 1
    npoints_max = 1000
    step = 1

    fname = f"{xsects['low']['longname']}_{xsects['low']['fmin']:.0f}-{xsects['low']['fmax']:.0f}_{xsects['low']['temperature']:.1f}K_{xsects['low']['pressure']:.0f}P_{xsects['high']['temperature']:.1f}K_{xsects['high']['pressure']:.0f}P_{npoints_max}_rms.pdf"

    rms = numpy.zeros(((npoints_max - npoints_min) // step,))
    npoints_arr = range(npoints_min, npoints_max, step)

    fgrid_conv = numpy.linspace(xsects['low']['fmin'],
                                xsects['low']['fmax'],
                                xsects['low']['nfreq'])

    fgrid_high = numpy.linspace(xsects['high']['fmin'],
                                xsects['high']['fmax'],
                                xsects['high']['nfreq'])
    if len(xsects['high']['data']) != len(fgrid_high):
        print(
            f"Size mismatch in data (skipping): nfreq: {xsects['high']['nfreq']} datasize: {len(xsects['high']['data'])} header: {xsects['high']['header']}")
        return {}

    for i, npoints in enumerate(npoints_arr):
        xsects['conv'] = xsect_convolve(xsects['low'], npoints, run_lorentz)

        xsects['high_interp'] = xsects['conv'].copy()
        xsects['high_interp']['data'] = numpy.interp(fgrid_conv, fgrid_high,
                                                     xsects['high']['data'])

        rms[i] = calc_xsect_rms(xsects['conv'],
                                xsects['high_interp'])

    rms_min = numpy.min(rms)
    rms_min_i = numpy.argmin(rms)
    rms_min_n = npoints_arr[numpy.argmin(rms)]
    # first_local_min = scipy.signal.argrelmin(rms)[0][0]
    # rms_min = rms[first_local_min]
    # rms_min_n = first_local_min + npoints_min

    if os.path.exists(fname):
        print(f"Skipping plotting: {xsect_high}")
    else:
        fig, ax = plt.subplots()
        width_vec = typhon.physics.wavenumber2frequency(
               2 * numpy.arange(npoints_min, npoints_max, step)
               * (xsects['low']['fmax'] - xsects['low']['fmin'])
               / xsects['low']['nfreq'] * 100) / float(20) / 1e9

        if rms_min_n < rms.size:
            ax.plot((width_vec[rms_min_i], width_vec[rms_min_i]), (numpy.min(rms), numpy.max(rms)),
                    linewidth=1,
                    label=f'Minimum {rms_min:.2e}@{width_vec[rms_min_i]:1.2g} GHz')
        #ax.plot(range(npoints_min, npoints_max, step), rms,
        ax.plot(width_vec, rms,
                label=f"T1: {xsects['low']['temperature']:.1f}K P1: {xsects['low']['pressure']:.0f}P f: {xsects['low']['fmin']:1.0f}-{xsects['low']['fmax']:1.0f}\nT2: {xsects['high']['temperature']:.1f}K P2: {xsects['high']['pressure']:.0f}P f: {xsects['high']['fmin']:1.0f}-{xsects['high']['fmax']:1.0f}",
                linewidth=0.5)

        ax.yaxis.set_major_formatter(mplticker.FormatStrFormatter('%1.0e'))
        # ax.set_xlim(npoints_min, npoints_max)
        ax.legend(loc=1)
        ax.set_ylabel('RMS')
        ax.set_xlabel('FWHM of Lorentz filter [GHz]')
        ax.set_title(title)

        # ax.set_ylim(1e-6, 9e-6)
        fig.savefig(fname)
        print(f'File saved: {fname}')

        fig, ax = plt.subplots()

        linewidth = 0.1
        # Plot xsect at low pressure
        plot_xsect(ax, xsects['low'], linewidth=linewidth)

        # Plot convoluted xsect
        npoints = rms_min_n
        xsects['conv'] = xsect_convolve(xsects['low'], npoints, run_lorentz)
        plot_xsect(ax, xsects['conv'], linewidth=linewidth,
                   label=f'Lorentz FWHM {width_vec[rms_min_i]:1.2g} GHz')

        # Plot xsect at high pressure
        plot_xsect(ax, xsects['high'], linewidth=linewidth)

        ax.legend(loc=1)
        ax.set_title(title)

        fname = f"{xsects['low']['longname']}_{xsects['low']['fmin']:.0f}-{xsects['low']['fmax']:.0f}_{xsects['low']['temperature']:.1f}K_{xsects['low']['pressure']:.0f}P_{xsects['high']['temperature']:.1f}K_{xsects['high']['pressure']:.0f}P_{npoints}_xsec.pdf"
        fig.savefig(fname)
        print(f'File saved: {fname}')

    # print(f"{xsect_high} done")

    return {
        'source_pressure': float(xsects['low']['pressure']),
        'target_pressure': float(xsects['high']['pressure']),
        'source_temp': float(xsects['low']['temperature']),
        'target_temp': float(xsects['high']['temperature']),
        'fmin': float(xsects['low']['fmin']),
        'fmax': float(xsects['low']['fmax']),
        'nfreq': int(xsects['low']['nfreq']),
        'optimum_width': int(rms_min_n),
        'rms_min': float(rms_min),
    }


def xsect_select(xsects, freq, freq_epsilon, temp, temp_epsilon):
    xsect_sel = [x for x in xsects if
                 numpy.abs(x['fmin'] - freq) < freq_epsilon and numpy.abs(
                     x['temperature'] - temp) < temp_epsilon]

    return sorted(xsect_sel, key=lambda xsect: xsect['pressure'])


def get_cfc11_inputs():
    infiles = glob.glob('cfc11/*00.xsc')
    xsects = [read_hitran_xsect(f) for f in infiles]
    inputs = []
    for temperature in (190, 201, 208, 216, 225, 232, 246, 260, 272):
        for freq in (810, 1050):
            xsects_sel = xsect_select(xsects, freq, 10, temperature, 2)
            for t in ((xsects_sel[0], x2, 'CFC-11') for x2 in xsects_sel[1:]):
                inputs.append(t)
    return inputs


def get_cfc12_inputs():
    infiles = glob.glob('cfc12/*00.xsc')
    xsects = [read_hitran_xsect(f) for f in infiles]
    inputs = []
    for temperature in (190, 201, 208, 216, 225, 232, 246, 260, 268, 272):
        for freq in (800, 850, 1050):
            xsects_sel = xsect_select(xsects, freq, 10, temperature, 2)
            for t in ((xsects_sel[0], x2, 'CFC-12') for x2 in xsects_sel[1:]):
                inputs.append(t)
    return inputs


if __name__ == '__main__':
    p = mp.Pool()

    if len(sys.argv) > 1 and sys.argv[1] == 'cfc11':
        inputs = get_cfc11_inputs()
    elif len(sys.argv) > 1 and sys.argv[1] == 'cfc12':
        inputs = get_cfc12_inputs()
    else:
        raise RuntimeError('Unknown species')

    # for i in inputs:
    # print(f"{i[0]['header']}{i[1]['header']}")
    res = [p.apply_async(optimize_xsect, args) for args in inputs]
    results = [r.get() for r in res if r]
    print(f'{len(results)} calculations')

    with open('output.txt', 'w') as f:
        json.dump(results, f)
