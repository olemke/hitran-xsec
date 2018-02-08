import functools

import matplotlib.pyplot as plt
import numpy
import typhon.arts.xml as axml
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
from typhon.physics import frequency2wavenumber


def f2w(temp):
    return frequency2wavenumber(temp) / 100.


def hz2thz(x, pos):
    return '{:.0f}'.format(x / 1e12)


def convert_ax_to_wavenumber(ax1, ax2):
    x1, x2 = ax1.get_xlim()
    ax2.set_xlim(f2w(x1), f2w(x2))
    ax2.figure.canvas.draw()


def THzFormatter():
    @FuncFormatter
    def _THzFormatter(x, pos):
        return '{:g}'.format(x / 1e12)

    return _THzFormatter


def func_2straights(x, x0, a, b):
    y = numpy.empty_like(x)
    for i, xi in enumerate(x):
        if xi <= x0:
            y[i] = a * xi
        else:
            y[i] = b * (xi - x0) + a * x0

    return y


def lorentz_pdf(x, x0, gamma):
    return gamma / numpy.pi / ((x - x0) ** 2 + gamma ** 2)


def plot_spectrum(ax, x, y, **kwargs):
    ax2 = ax.twiny()
    ax.callbacks.connect("xlim_changed",
                         functools.partial(convert_ax_to_wavenumber, ax2=ax2))

    ax.plot(x, y, **kwargs)

    ax.xaxis.set_major_formatter(THzFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%7.3g'))
    ax.set_xlabel('Frequency [THz]')
    ax2.set_xlabel('Wavenumber [cm$^{-1}$]')
    return ax2


inputdir = "arts-example/"
fgrid_s = axml.load(inputdir + 'TestHitranXsec-simple.f_grid.xml')
xsec = axml.load(inputdir + 'TestHitranXsec-simple.abs_xsec_per_species.xml')
species_names = axml.load(inputdir + 'TestHitranXsec-simple.abs_species.xml')

fig, ax = plt.subplots()

for species, name in zip(xsec, species_names):
    for band in range(species.shape[1]):
        ax.plot(fgrid_s, species[:, band],
                label=f"{name[0].split('-')[0]} {band}", linewidth=0.75)

ax.xaxis.set_major_formatter(THzFormatter())
ax.set_xlabel('THz')

fig.legend()
fig.savefig('xsec.pdf', dpi=300)

#postfix = '.radiance'
postfix = '.planck'

fgrid = axml.load(inputdir + 'TestHitranXsec.f_grid' + postfix + '.xml')
y = axml.load(inputdir + 'TestHitranXsec.y' + postfix + '.xml')

fgrid_nocfc = axml.load(
    inputdir + 'TestHitranXsec-nocfc.f_grid' + postfix + '.xml')
y_nocfc = axml.load(inputdir + 'TestHitranXsec-nocfc.y' + postfix + '.xml')

fig, (ax1, ax2) = plt.subplots(2, 1)  # , figsize=(5, 10))

plot_spectrum(ax1, fgrid, y, label='w/ CFCs', rasterized=True)
ax1.plot(fgrid_nocfc, y_nocfc, label='w/o CFCs', rasterized=True)
ax1.xaxis.set_ticklabels([])
ax1.set_xlabel('')
if postfix == '.radiance':
    ax1.set_ylabel('$[\\frac{W}{sr⋅m^2⋅Hz}]$')
    ax2.set_ylabel('Spectral radiance')
else:
    ax1.set_ylabel('$[K]$')
    ax2.set_ylabel('Brightness temperature')

ax1.legend()

axtop = plot_spectrum(ax2, fgrid, y - y_nocfc, rasterized=True)
axtop.xaxis.set_ticklabels([])
axtop.set_xlabel('')

fig.tight_layout()
fig.savefig('y.pdf', dpi=300)
