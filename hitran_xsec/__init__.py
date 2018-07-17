"""Package for processing HITRAN cross section data files.
"""

from . import fit
from . import plotting
from .xsec import *

if __name__ == "__main__":
    import multiprocessing as mp
    import logging

    logging.basicConfig(level=logging.WARN)
    xfi = XsecFileIndex(directory='cfc12', ignore='.*[^0-9._].*')

    bands = xfi.cluster_by_band_and_temperature()

    with mp.Pool() as pool:
        result = pool.starmap(optimize_xsec,
                              build_pairs_with_lowest_pressure(bands))

    save_rms_data("output.json", result)
