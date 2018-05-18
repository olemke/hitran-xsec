import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from typhon.arts.xsec import XsecRecord

__all__ = ['gen_arts']


def func_2straights(x, x0, a, b):
    y = np.empty_like(x)
    c1 = c2 = 0
    for i, xi in enumerate(x):
        if xi <= x0:
            y[i] = a * xi
            c1 += 1
        else:
            y[i] = b * (xi - x0) + a * x0
            c2 += 1

    return y


def calc_fwhm_and_pressure_difference(xsec_result):
    fwhm = np.array([r['optimum_fwhm'] for r in xsec_result])
    pressure_diff = np.array(
        [r['target_pressure'] - r['source_pressure'] for r in xsec_result])

    return fwhm, pressure_diff


def do_fit(fwhm, pressure_diff, fit_func=func_2straights, outliers=False):
    if outliers:
        data = np.hstack((pressure_diff.reshape(-1, 1), fwhm.reshape(-1, 1)))
        forrest = IsolationForest(contamination=0.001)
        forrest.fit(data)
        decision = forrest.predict(data) != -1
    else:
        decision = np.ones_like(fwhm, dtype='bool')
    # Apriori for fitting the two lines
    p0 = (30000., 1e6, 1e6)
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(fit_func, pressure_diff[decision], fwhm[decision],
                           p0=p0)
    return popt, pcov, decision


def gen_arts(xsecfileindex, rmsoutput, reftemp=230):
    # Find reference profiles for each band
    bands = xsecfileindex.cluster_by_band_and_temperature()
    # Convert generators to lists and sort by pressure
    lbands = [[sorted(l, key=lambda x: x.pressure) for l in t]
              for t in [list(b) for b in bands]]

    # Get list of temperatures in each band
    temps = [[t[0].temperature for t in b] for b in lbands]

    # Select profiles closest to reference temperature
    mins = [tlist.index(min(tlist, key=lambda x: abs(x - reftemp))) for tlist in temps]
    xsec_ref = [band[index][0] for band, index in zip(lbands, mins)]

    if not len(xsec_ref):
        raise RuntimeError('No matching xsecs found.')

    print(f'{len(xsec_ref)} profiles selected.')
    fwhm, pressure_diff = calc_fwhm_and_pressure_difference(rmsoutput)
    popt, pcov, decision = do_fit(fwhm, pressure_diff)
    return XsecRecord(xsec_ref[0].species,
                      popt,
                      np.array([x.fmin for x in xsec_ref]),
                      np.array([x.fmax for x in xsec_ref]),
                      np.array([x.pressure for x in xsec_ref]),
                      np.array([x.temperature for x in xsec_ref]),
                      [x.data / 10000. for x in xsec_ref])
