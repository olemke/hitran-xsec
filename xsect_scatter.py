import json
import sys

import matplotlib as mpl

mpl.use('Agg')
import numpy
import typhon.physics

import matplotlib.pyplot as plt


def scatter_plot(ax, xsect_result, hwhm, title, **kwargs):
    if kwargs is None:
        kwargs = {}
    width_vec = numpy.array(
        [typhon.physics.wavenumber2frequency(
            2 * r['optimum_width']
            * (r['fmax'] - r['fmin'])
            / r['nfreq'] * 100) / float(hwhm) / 1e9 for r in
         # [r['optimum_width'] for r in
         xsect_result])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in
         xsect_result]) / 100
    ax.set_ylim((0, 6))
    ax.scatter(pressure_diff, width_vec, **kwargs)
    # ax.semilogx()
    # ax.set_ylabel('∆λ FWHM of Lorentz filter [Hz]')
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')
    ax.set_title(title)


def main():
    with open('output.txt') as f:
        xsect_result = json.load(f)

    fig, ax = plt.subplots()

    species = sys.argv[1]
    if species == 'cfc11':
        title = 'CFC-11'
        scatter_plot(ax, [x for x in xsect_result if x['fmin'] > 800 and x[
            'fmin'] < 1000], sys.argv[2], title, label='810-880')
        scatter_plot(ax, [x for x in xsect_result if x['fmin'] > 1000],
                     sys.argv[2], title, label='1050-1120')
    elif species == 'cfc12':
        title = 'CFC-12'
        scatter_plot(ax, [x for x in xsect_result if x['fmin'] < 840],
                     sys.argv[2], title, label='800-1270')
        scatter_plot(ax, [x for x in xsect_result if x['fmin'] > 840 and x[
            'fmin'] < 1000], sys.argv[2], title, label='850-950')
        scatter_plot(ax, [x for x in xsect_result if x['fmin'] > 1000],
                     sys.argv[2], title, label='1050-1200')
    else:
        raise RuntimeError('Unknown species')

    ax.legend()

    fig.savefig('xsect_scatter.pdf')

    fig, ax = plt.subplots()

    scatter_plot(ax, [x for x in xsect_result if x['source_temp'] <= 240],
                 sys.argv[2], title, label='T ≤ 240K')

    scatter_plot(ax, [x for x in xsect_result if
                      x['source_temp'] > 240 and x['source_temp'] <= 250],
                 sys.argv[2], title, label='240K < T ≤ 250K')
    scatter_plot(ax, [x for x in xsect_result if
                      x['source_temp'] > 250 and x['source_temp'] <= 270],
                 sys.argv[2], title, label='250K < T ≤ 270K')
    scatter_plot(ax, [x for x in xsect_result if x['source_temp'] > 270],
                 sys.argv[2], title, label='270K < T')

    ax.legend()

    fig.savefig('xsect_scatter_temp.pdf')


if __name__ == '__main__':
    main()
