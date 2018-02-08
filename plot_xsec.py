import sys

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy
import functools
import typhon.arts.xml as axml
from typhon.physics import frequency2wavenumber


def f2w(temp):
    return frequency2wavenumber(temp) / 100.

def hz2thz(x, pos):
    return '{:.0f}'.format(x / 1e12)

def convert_ax_to_wavenumber(ax1, ax2):
    x1, x2 = ax1.get_xlim()
    ax2.set_xlim(f2w(x1), f2w(x2))
    ax2.figure.canvas.draw()

def main(inputfile):
    xsecs = axml.load(inputfile)

    fig, ax = plt.subplots()

    ax2 = ax.twiny()
    ax.callbacks.connect("xlim_changed", functools.partial(convert_ax_to_wavenumber, ax2=ax2))

    for xsec in xsecs:
        for fmin, fmax, xsecdata in zip(xsec.fmin, xsec.fmax, xsec.xsec):
            ax.plot(numpy.linspace(fmin, fmax, len(xsecdata), endpoint=True),
                    xsecdata, label=f'{xsec.species} '
                                    f'{f2w(fmin):.0f}-{f2w(fmax):.0f}')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(hz2thz))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1g'))
    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel('Cross section [m$^2$]')
    ax2.set_xlabel('Wavenumber [cm$^{-1}$]')
    ax.legend()
    fig.savefig(inputfile + '.pdf', boundary_box='tight')


if __name__ == '__main__':
    main(sys.argv[1])
