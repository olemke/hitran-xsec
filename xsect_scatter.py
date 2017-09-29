import json
import os
import sys

import matplotlib as mpl

mpl.use('Agg')
import numpy
import typhon.physics

import matplotlib.pyplot as plt


def scatter_plot(ax, fwhm, pressure_diff, title, **kwargs):
    if kwargs is None:
        kwargs = {}

    ax.set_ylim((0, 6))
    ax.scatter(pressure_diff, fwhm / 1e9, **kwargs)
    ax.set_ylabel('FWHM of Lorentz filter [GHz]')
    ax.set_xlabel('∆P [hPa]')
    ax.set_title(title)


def calc_fwhm_and_pressure_difference(xsec_result, hwhm):
    fwhm = numpy.array(
        [typhon.physics.wavenumber2frequency(
            2 * r['optimum_width']
            * (r['fmax'] - r['fmin'])
            / r['nfreq'] * 100) / float(hwhm) for r in
         xsec_result])
    pressure_diff = numpy.array(
        [r['target_pressure'] - r['source_pressure'] for r in
         xsec_result]) / 100

    return fwhm, pressure_diff


def main():
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} SPECIES TITLE DATADIR')
        print(f'  SPECIES: cfc11 or cfc12')
        sys.exit(1)

    datadir = sys.argv[3]

    with open(os.path.join(datadir, 'output.txt')) as f:
        xsec_result = json.load(f)

    fig, ax = plt.subplots()

    hwhm = 20
    species = sys.argv[1]
    title = sys.argv[2]
    if species == 'cfc11':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         [x for x in xsec_result if x['fmin'] > 800 and x[
                             'fmin'] < 1000], hwhm),
                     title, label='810-880')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         [x for x in xsec_result if x['fmin'] > 1000], hwhm),
                     title, label='1050-1120')
    elif species == 'cfc12':
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         [x for x in xsec_result if x['fmin'] < 840], hwhm),
                     title, label='800-1270')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         [x for x in xsec_result if x['fmin'] > 840 and x[
                             'fmin'] < 1000], hwhm),
                     title, label='850-950')
        scatter_plot(ax,
                     *calc_fwhm_and_pressure_difference(
                         [x for x in xsec_result if x['fmin'] > 1000], hwhm),
                     title, label='1050-1200')
    else:
        raise RuntimeError('Unknown species')

    ax.legend()

    fig.savefig(os.path.join(datadir, 'xsec_scatter.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots()

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if x['source_temp'] <= 240],
                     hwhm),
                 title, label='T ≤ 240K')

    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if
                      x['source_temp'] > 240 and x['source_temp'] <= 250],
                     hwhm),
                 title, label='240K < T ≤ 250K')
    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if
                      x['source_temp'] > 250 and x['source_temp'] <= 270],
                     hwhm),
                 title, label='250K < T ≤ 270K')
    scatter_plot(ax,
                 *calc_fwhm_and_pressure_difference(
                     [x for x in xsec_result if x['source_temp'] > 270], hwhm),
                 title, label='270K < T')

    ax.legend()

    fig.savefig(os.path.join(datadir, 'xsec_scatter_temp.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
