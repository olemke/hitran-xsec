"""Package for processing HITRAN cross section data files.
"""

import multiprocessing as mp

from .plotting import *
from .xsec import *
from .fit import *

__all__ = [s for s in dir() if not s.startswith('_')]

logger = logging.getLogger('xsec')

_LORENTZ_CUTOFF = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    xfi = XsecFileIndex(directory='cfc12', ignore='.*[^0-9._].*')

    bands = xfi.cluster_by_band_and_temperature()

    with mp.Pool() as pool:
        result = pool.starmap(optimize_xsec,
                              build_pairs_with_lowest_pressure(bands))

    save_output("output.json", result)
