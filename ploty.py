import matplotlib.pyplot as plt
import numpy
import typhon.arts.xml as axml


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


# x = numpy.linspace(20e12, 30e12, num=2000)
# y = lorentz_pdf(x, 25e12, 4e9)
# plt.plot(x, y)
# print(quad(lorentz_pdf, 20e12, 30e12, args=(25e12, 0.1e12)))
# print(numpy.sum(y))
# plt.show()
# exit(0)
xsec_orig = axml.load('output_cfc12/CFC12.xml')
# fwhm = func_2straights([86993], *xsec[0].coeffs);
# print(fwhm)
# print(lorentz_pdf(24e12, 24e12, fwhm / 2))

inputdir = "arts-example/"
fgrid = axml.load(inputdir + 'TestHitranXsec.f_grid.xml')
fgrid_s = axml.load(inputdir + 'TestHitranXsec-simple.f_grid.xml')
xsec = axml.load(inputdir + 'TestHitranXsec-simple.abs_xsec_per_species.xml')
y = axml.load(inputdir + 'TestHitranXsec.y.xml')
print("nfgrid:", len(fgrid_s))
print("orig nfgrid: ", len(xsec_orig[0].xsec[0]))
# plt.plot(typhon.physics.frequency2wavenumber(fgrid)/100, y, linewidth=0.5)  # , rasterized=True)
plt.plot(fgrid_s / 1e12, xsec[0][:, 0], label="1", linewidth=0.75,
         rasterized=True)
plt.plot(fgrid_s / 1e12, xsec[0][:, 1], label="2", linewidth=0.75,
         rasterized=True)
xi = 0
# plt.plot(numpy.linspace(xsec_orig[0].fmin[xi], xsec_orig[0].fmax[xi],
#                         num=len(xsec_orig[0].xsec[xi])) / 1e12, xsec_orig[0].xsec[xi],
#          linewidth=0.5)  # , rasterized=True)
plt.savefig('y.pdf', dpi=300)
plt.legend()
plt.show()
