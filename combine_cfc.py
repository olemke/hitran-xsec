import functools
import os
import sys

import matplotlib.pyplot as plt
import numpy
import typhon.arts.xml as axml
from matplotlib import ticker
from typhon.physics import frequency2wavenumber


def f2w(temp):
    return frequency2wavenumber(temp) / 100.


def hz2thz(x, pos):
    return '{:.0f}'.format(x / 1e12)


def convert_ax_to_wavenumber(ax1, ax2):
    x1, x2 = ax1.get_xlim()
    ax2.set_xlim(f2w(x1), f2w(x2))
    ax2.figure.canvas.draw()


def combine_xsec(outfile):
    cfcs = []
    for species in (('CFC11', 'output_cfc11_full'),
                    ('CFC12', 'output_cfc12_full'),
                    ('HFC134a', 'output_hfc134a_full'),
                    ('HCFC22', 'output_hcfc22_full'),
                    ):
        cfcs.extend(axml.load(os.path.join(species[1], species[0] + '.xml')))

    axml.save(cfcs, outfile + '.xml', format='binary')

    avg_coeffs = numpy.sum(numpy.array([x.coeffs for x in cfcs]), axis=0) / len(
        cfcs)
    for x in cfcs:
        x.coeffs = avg_coeffs

    print(f'Average coeffs: {avg_coeffs}')
    axml.save(cfcs, outfile + '.avg.xml', format='binary')


def plot_xsec(inputfile):
    xsecs = axml.load(inputfile + '.xml')

    fig, ax = plt.subplots()

    ax2 = ax.twiny()
    ax.callbacks.connect("xlim_changed",
                         functools.partial(convert_ax_to_wavenumber, ax2=ax2))

    for xsec in xsecs:
        for fmin, fmax, xsecdata in zip(xsec.fmin, xsec.fmax, xsec.xsec):
            ax.plot(numpy.linspace(fmin, fmax, len(xsecdata), endpoint=True),
                    xsecdata, label=f'{xsec.species} '
                                    f'{f2w(fmin):.0f}-{f2w(fmax):.0f}',
                    rasterized=True)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(hz2thz))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1g'))
    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel('Cross section [m$^2$]')
    ax2.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax.legend()
    fig.savefig(inputfile + '.pdf', dpi=300)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} OUTFILE')
        exit(1)

    combine_xsec(sys.argv[1])
    plot_xsec(sys.argv[1])
