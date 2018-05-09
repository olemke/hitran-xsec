"""Package for processing HITRAN cross section data files.
"""

import os
import re
from glob import glob

from typhon.physics import wavenumber2frequency


class XsecError(RuntimeError):
    pass


class XsecFile:
    """HITRAN cross section file."""

    def __init__(self, filename):
        self.filename = filename
        rnum = r'[0-9]+\.?[0-9]*'
        m = re.search(
            f'(?P<species>[^_]*)_(?P<T>{rnum})K?[-_](?P<P>{rnum})(Torr)?[-_](?P<wmin>{rnum})[-_](?P<wmax>{rnum})(?P<extra>_.*)?\.xsc',
            os.path.basename(self.filename))
        try:
            self.species = m.group('species')
            self.temperature = float(m.group('T'))
            self.pressure = float(m.group('P'))
            self.wmin = float(m.group('wmin'))
            self.wmax = float(m.group('wmax'))
            self.fmin = wavenumber2frequency(self.wmin * 100)
            self.fmax = wavenumber2frequency(self.wmax * 100)
            self.extra = m.group('extra')
        except AttributeError:
            raise XsecError(f'Error parsing filename {filename}')

    def __repr__(self):
        return (f'{self.filename}: S{self.species}_T{self.temperature}'
                f'_P{self.pressure}_F{self.wmin}-{self.wmax}@{self.extra}')


class XsecFileIndex:
    def __init__(self, directory, ignore='extra'):
        self.files = []
        self.ignored_files = []
        for f in glob(os.path.join(directory, '*.xsc')):
            try:
                xsec_file = XsecFile(f)
                # if ignore == 'extra' and xsec_file.extra:
                #     self.ignored_files.append(f)
                # else:
                #     self.files.append(xsec_file)
                self.files.append(xsec_file)
            except XsecError:
                self.ignored_files.append(f)


if __name__ == "__main__":
    xfi = XsecFileIndex('cfc11')
    print(xfi.files)
    print(xfi.ignored_files)
    for i in xfi.files:
        print(i)
