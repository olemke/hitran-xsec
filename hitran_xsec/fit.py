import logging
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from typhon.arts.xsec import XsecRecord

from .xsec import XsecError
from .xsec_species_info import XSEC_SPECIES_INFO

logger = logging.getLogger(__name__)


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
        [r['target_pressure'] - r['ref_pressure'] for r in xsec_result])

    return fwhm, pressure_diff


def do_rms_fit(fwhm, pressure_diff, fit_func=func_2straights, outliers=False):
    if outliers:
        data = np.hstack((pressure_diff.reshape(-1, 1), fwhm.reshape(-1, 1)))
        forrest = IsolationForest(contamination=0.001)
        forrest.fit(data)
        non_outliers = forrest.predict(data) != -1
    else:
        non_outliers = np.ones_like(fwhm, dtype='bool')
    # Apriori for fitting the two lines
    p0 = (30000., 1e6, 1e6)
    # noinspection PyTypeChecker
    popt, pcov = curve_fit(fit_func, pressure_diff[non_outliers],
                           fwhm[non_outliers], p0=p0)
    return popt, pcov, non_outliers


def fit_temp_func(x, a, b):
    return x * a + b


def do_temperture_fit(xsecs, xref=None):
    if xref is None:
        xref = xsecs[0]

    xsec_diff = np.array([x.data - xref.data for x in xsecs])
    t_diff = [x.temperature - xref.temperature for x in xsecs]

    fit = np.array([linregress(t_diff, y) for y in xsec_diff.T])

    # Return slope and intersection only
    return fit[:, 0:2]


def gen_arts(xsecfileindex, rmsoutput, tfitoutput=None, reftemp=230):
    if not rmsoutput:
        raise XsecError('RMS output is empty')

    # Find reference profiles for each band
    bands = xsecfileindex.cluster_by_band_and_temperature()

    # Convert generators to lists and sort by pressure
    lbands = [[sorted(l, key=lambda x: x.pressure) for l in t]
              for t in [list(b) for b in bands]]

    # Get list of temperatures in each band
    temps = [[t[0].temperature for t in b] for b in lbands]

    # Select profiles closest to reference temperature
    mins = [tlist.index(min(tlist, key=lambda x: abs(x - reftemp))) for tlist in
            temps]

    species = XSEC_SPECIES_INFO[lbands[0][0][0].species]
    if 'arts_bands' in species:
        xsec_ref = [band[index][0] for band, index in zip(lbands, mins) if
                    (band[index][0].wmin, band[index][0].wmax) in
                    species['arts_bands']]
        logger.info(f"{lbands[0][0][0].species}:{len(xsec_ref)} bands out of "
                    f"{len(lbands)} selected for ARTS.")
    else:
        xsec_ref = [band[index][0] for band, index in zip(lbands, mins)]

    if not len(xsec_ref):
        raise XsecError('No matching xsecs found.')

    logger.info(f'{len(xsec_ref)} profiles selected: '
                f'{[os.path.basename(x.filename) for x in xsec_ref]}.')
    fwhm, pressure_diff = calc_fwhm_and_pressure_difference(rmsoutput)
    popt, pcov, decision = do_rms_fit(fwhm, pressure_diff)
    return XsecRecord(
        xsec_ref[0].species.translate(str.maketrans(dict.fromkeys('-'))),
        popt,
        np.array([x.fmin for x in xsec_ref]),
        np.array([x.fmax for x in xsec_ref]),
        np.array([x.pressure for x in xsec_ref]),
        np.array([x.temperature for x in xsec_ref]),
        [x.data / 10000. for x in xsec_ref])
