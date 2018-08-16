import itertools
import json
import logging
import multiprocessing as mp
import os
import re
from copy import deepcopy
from glob import glob

import numpy as np
from scipy.integrate import simps
from scipy.signal import fftconvolve
from typhon.physics import wavenumber2frequency

from .xsec_species_info import XSEC_SPECIES_INFO

__all__ = [
    'XsecError',
    'XsecFile',
    'XsecFileIndex',
    'load_rms_data',
    'optimize_xsec',
    'optimize_xsec_multi',
    'save_rms_data',
]

logger = logging.getLogger(__name__)

LORENTZ_CUTOFF = None


class XsecError(RuntimeError):
    """Cross section related RuntimeError."""
    pass


class XsecFile:
    """HITRAN cross section file."""

    def __init__(self, filename):
        """Lazy-load cross section file."""
        self.filename = filename
        # noinspection PyUnusedLocal
        rnum = r'[0-9]+\.?[0-9]*'
        m = re.search(
            f'(?P<species>[^_]*)_(?P<T>{rnum})K?[-_](?P<P>{rnum})(Torr)?[-_]'
            f'(?P<wmin>{rnum})[-_](?P<wmax>{rnum})(?P<extra>_.*)?\.xsc',
            os.path.basename(self.filename))
        try:
            self.species = m.group('species')
            self.temperature = float(m.group('T'))
            self.torr = float(m.group('P'))
            self.pressure = torr_to_pascal(self.torr)
            self.wmin = float(m.group('wmin'))
            self.wmax = float(m.group('wmax'))
            self.fmin = wavenumber2frequency(self.wmin * 100)
            self.fmax = wavenumber2frequency(self.wmax * 100)
            self.extra = m.group('extra')
            self._header = None
            self._data = None
            self._nfreq = None
        except AttributeError:
            raise XsecError(f'Error parsing filename {filename}')

    def __repr__(self):
        return 'XsecFile:' + self.filename

    def __hash__(self):
        return hash(f'{self.species}{self.pressure}{self.temperature}'
                    f'{self.wmin}{self.wmax}')

    def __eq__(self, x):
        return (self.species == x.species
                and self.pressure == x.pressure
                and self.temperature == x.temperature
                and self.wmin == x.wmin
                and self.wmax == x.wmax)

    def read_hitran_xsec(self):
        """Read HITRAN cross section data file."""
        if self._data is not None:
            return

        logger.info(f"Reading {self.filename}")
        with open(self.filename) as f:
            header = f.readline()
            data = np.hstack(
                list(map(lambda l: list(map(float, l.split())), f.readlines())))

        fields = header.split()
        if len(fields) == 10:
            fields_new = fields[:6]
            fields_new.append(fields[6][:9])
            fields_new.append(fields[6][9:])
            fields_new.append(fields[7:])
            fields = fields_new

        fieldnames = {
            'longname': str,
            'fmin': float,
            'fmax': float,
            'nfreq': int,
            'temperature': float,
            'pressure': float,
            'unknown1': str,
            'unknown2': str,
            'name': str,
            'broadener': str,
            'unknown3': int,
        }
        xsec_dict = {fname[0]: fname[1](
            field) for fname, field in zip(fieldnames.items(), fields)}
        self._header = header
        self._data = data
        # Recalculate number of frequency points based on actual data values
        # since header information is not correct for some files
        self._nfreq = len(data)
        xsec_dict['pressure'] = torr_to_pascal(xsec_dict['pressure'])
        xsec_dict['fmin'] = wavenumber2frequency(xsec_dict['fmin'] * 100)
        xsec_dict['fmax'] = wavenumber2frequency(xsec_dict['fmax'] * 100)

        return xsec_dict

    @property
    def nfreq(self):
        if self._nfreq is None:
            self.read_hitran_xsec()
        return self._nfreq

    @property
    def header(self):
        if self._header is None:
            self.read_hitran_xsec()
        return self._header

    @property
    def data(self):
        if self._data is None:
            self.read_hitran_xsec()
        return self._data

    @data.setter
    def data(self, val):
        self._data = val


class XsecFileIndex:
    """Database of HITRAN cross section files."""

    def __init__(self, directory=None, species=None, ignore=None):
        self.files = []
        self.ignored_files = []
        self.failed_files = []
        if directory is not None and species is not None:
            if 'altname' in XSEC_SPECIES_INFO[species]:
                speciesname = XSEC_SPECIES_INFO[species]['altname']
            else:
                speciesname = species

            for f in glob(os.path.join(directory, '*.xsc')):
                try:
                    xsec_file = XsecFile(f)
                    if xsec_file.species != speciesname:
                        pass
                    elif ignore is not None and re.match(ignore,
                                                         xsec_file.extra):
                        self.ignored_files.append(f)
                    else:
                        self.files.append(xsec_file)
                        if species != speciesname:
                            xsec_file.species = species
                except XsecError:
                    self.failed_files.append(f)
        self.uniquify()

    @classmethod
    def from_list(cls, xsec_file_list):
        obj = cls()
        obj.files = xsec_file_list
        return obj

    def __repr__(self):
        return '\n'.join([f.filename for f in self.files])

    def uniquify(self):
        nfiles = len(self.files)
        checked = {}
        uniqfiles = []
        for item in self.files:
            marker = item
            if marker in checked:
                continue
            checked[marker] = 1
            uniqfiles.append(item)
        nuniqfiles = len(uniqfiles)
        if nuniqfiles < nfiles:
            logger.info(f'Removed {nfiles - nuniqfiles} duplicate data files.')
            self.files = uniqfiles

    def find_file(self, filename):
        ret = [x for x in self.files if x.filename == filename]
        return ret if len(ret) > 1 else ret[0]

    def find(self, wmin=None, wmax=None, temperature=None, pressure=None):
        """Find cross sections that match the criteria."""
        return [x for x in self.files if
                (not wmin or x.wmin == wmin)
                and (not wmax or x.wmax == wmax)
                and (not temperature or x.temperature == temperature)
                and (not pressure or x.torr == pressure)]

    def cluster_by_band(self, wgap=1):
        """Combine files for each band in a list."""
        return _cluster2(self.files, wgap, key=lambda x: x.wmin,
                         key2=lambda x: x.wmax)

    def cluster_by_temperature(self, tgap=3):
        """Combine files for each temperature in a list."""
        return _cluster2(self.files, tgap, key=lambda x: x.temperature)

    def cluster_by_band_and_pressure(self, wgap=1, pgap=100):
        """Combine files for each band and pressure in a nested list."""
        return (_cluster2(l, pgap, key=lambda x: x.pressure)
                for l in _cluster2(self.files, wgap, key=lambda x: x.wmin,
                                   key2=lambda x: x.wmax))

    def cluster_by_band_and_temperature(self, wgap=1, tgap=3):
        """Combine files for each band and temperature in a nested list."""
        return (_cluster2(l, tgap, key=lambda x: x.temperature)
                for l in _cluster2(self.files, wgap, key=lambda x: x.wmin,
                                   key2=lambda x: x.wmax))


def torr_to_pascal(torr):
    """Convert Torr to Pascal."""
    return torr * 101325. / 760.


def _pairify(it):
    """Build pairs."""
    it0, it1 = itertools.tee(it, 2)
    first = next(it0)
    return zip(itertools.chain([first, first], it0), it1)


def _cluster2(iterable, maxgap, key=lambda x: x, key2=None):
    """Cluster sequence elements by distance."""
    prev = None
    group = []
    for item in sorted(
            iterable,
            key=lambda x: (key(x), key2(x)) if key2 is not None else key(x)):
        if not prev or (key(item) - key(prev) <= maxgap and
                        (not key2 or key2(item) - key2(prev) <= maxgap)):
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def lorentz_pdf(x, x0, gamma):
    return gamma / np.pi / ((x - x0) ** 2 + gamma ** 2)


def run_lorentz_f(npoints, fstep, hwhm, cutoff=None):
    ret = lorentz_pdf(
        np.linspace(0, fstep * npoints, npoints, endpoint=True),
        fstep * npoints / 2,
        hwhm)
    if cutoff is not None:
        ret = ret[ret > np.max(ret) * cutoff]
    if len(ret) > 1:
        return ret / simps(ret)
    else:
        return np.array([1.])


def xsec_convolve_f(xsec1, hwhm, convfunc, cutoff=None):
    """Convolve cross section data with the given function."""
    fstep = (xsec1.fmax - xsec1.fmin) / xsec1.nfreq

    conv_f = convfunc(int(xsec1.nfreq), fstep, hwhm, cutoff=cutoff)
    width = len(conv_f)
    xsec_conv = deepcopy(xsec1)
    xsec_conv.data = fftconvolve(xsec1.data, conv_f, 'same')

    return xsec_conv, conv_f, width


def xsec_convolve_simple(xsec1, hwhm, cutoff=None):
    """Convolve cross section data with the Lorentz function."""
    return xsec_convolve_f(xsec1, hwhm, run_lorentz_f, cutoff)


def calc_xsec_rms(xsec1, xsec2):
    """Calculate RMS between two cross sections."""
    return np.sqrt(np.mean(np.square(
        xsec1.data / np.sum(xsec1.data) - xsec2.data / np.sum(xsec2.data))))


def optimize_xsec(xsec_low, xsec_high,
                  fwhm_min=0.01e9, fwhm_max=20.01e9, fwhm_nsteps=1000):
    """Find the broadening width with lowest RMS."""

    xsec_name = (
        f"{xsec_low.species}_"
        f"{xsec_low.wmin:.0f}"
        f"-{xsec_low.wmax:.0f}_"
        f"{xsec_low.temperature:.1f}K_"
        f"{xsec_low.pressure:.0f}P_{xsec_high.temperature:.1f}K_"
        f"{xsec_high.pressure:.0f}P")
    logger.info(f"Calc {xsec_name}")

    rms = np.zeros((fwhm_nsteps,))
    fwhms = np.linspace(fwhm_min, fwhm_max, fwhm_nsteps)

    fgrid_conv = np.linspace(xsec_low.fmin, xsec_low.fmax, xsec_low.nfreq)
    fgrid_high = np.linspace(xsec_high.fmin, xsec_high.fmax, xsec_high.nfreq)

    if xsec_low == xsec_high:
        logger.info(f'{xsec_high} and {xsec_low} are identical. Ignoring.')
        return None

    if len(xsec_high.data) != len(fgrid_high):
        logger.error(f"Size mismatch in data (skipping): nfreq: "
                     f"{xsec_high.nfreq} "
                     f"datasize: {len(xsec_high.data)} "
                     f"header: {xsec_high.header}")
        return None

    for i, fwhm in enumerate(fwhms):
        # logger.info(f"Calculating {fwhm/1e9:.3f} for {xsec_name}")
        xsec_conv, conv, width = xsec_convolve_f(xsec_low, fwhm / 2,
                                                 run_lorentz_f,
                                                 LORENTZ_CUTOFF)
        # logger.info(f"Calculating done {fwhm/1e9:.3f} for {xsec_name}")
        if width < 10:
            logger.warning(
                f"Very few ({width}) points used in Lorentz function for "
                f"{xsec_name} at FWHM {fwhm/1e9:.2} GHz.")

        xsec_high_interp = deepcopy(xsec_conv)
        xsec_high_interp.data = np.interp(fgrid_conv, fgrid_high,
                                          xsec_high.data)

        rms[i] = calc_xsec_rms(xsec_conv, xsec_high_interp)

    rms_optimum_fwhm_index = np.argmin(rms)
    rms_optimum_fwhm = fwhms[rms_optimum_fwhm_index]

    logger.info(f"Done {xsec_name}")

    return {
        'ref_pressure': float(xsec_low.pressure),
        'target_pressure': float(xsec_high.pressure),
        'ref_filename': xsec_low.filename,
        'target_filename': xsec_high.filename,
        'ref_temp': float(xsec_low.temperature),
        'target_temp': float(xsec_high.temperature),
        'wmin': float(xsec_low.wmin),
        'wmax': float(xsec_low.wmax),
        'fmin': float(xsec_low.fmin),
        'fmax': float(xsec_low.fmax),
        'nfreq': int(xsec_low.nfreq),
        'optimum_fwhm': rms_optimum_fwhm,
        'optimum_fwhm_index': int(rms_optimum_fwhm_index),
        'fwhm_min': fwhm_min,
        'fwhm_max': fwhm_max,
        'fwhm_nsteps': int(fwhm_nsteps),
        'rms': rms.tolist(),
    }


def build_pairs_with_lowest_pressure(iterable):
    """Pairs lowest pressure xsec with each higher pressure.

    Cross sections with pressure 0 are ignored.
    """
    for b in iterable:
        for t in b:
            xsec_list = sorted(t, key=lambda x: x.pressure)
            xsec_no_p0 = list(filter(lambda x: x.pressure != 0, xsec_list))
            if len(xsec_no_p0) > 1:
                yield from ((xsec_no_p0[0], xsec2) for xsec2 in
                            itertools.islice(xsec_no_p0, 1, len(xsec_no_p0)))
            else:
                logger.warning(f'Not enough xsecs for '
                               f'temperature {xsec_list[0].temperature} in '
                               f'band {xsec_list[0].wmin} - {xsec_list[0].wmax}'
                               )


def optimize_xsec_multi(xsecfileindex, processes=None):
    """Calculate best broadening width."""
    bands = xsecfileindex.cluster_by_band_and_temperature()
    pressure_pairs = build_pairs_with_lowest_pressure(bands)
    with mp.Pool(processes=processes) as pool:
        return pool.starmap(optimize_xsec, pressure_pairs)


def save_rms_data(filename, results):
    """Save calculated RMS data for cross sections to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f)


def load_rms_data(filename):
    """Load calculated RMS data for cross sections from JSON file."""
    with open(filename) as f:
        results = json.load(f)
    return results
