import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import typhon.arts.xml as axml

import hitran_xsec as hx

logging.basicConfig(level=logging.INFO)


def parse_args():
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('species', help='Name of species to process.')
    args.add_argument('-d', '--directory', default='.',
                      help='Directory with cross section data files.')
    args.add_argument('-o', '--output', metavar='OUTPUT_DIRECTORY',
                      default='output',
                      help='Output directory.'
                           'A subdirectory SPECIES will be created inside.')
    return args.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()

    output_dir = os.path.join(args.output, args.species)
    os.makedirs(output_dir, exist_ok=True)

    xfi = hx.XsecFileIndex(directory=args.directory, species=args.species,
                           ignore='.*[^0-9._].*')

    hx.plotting.plot_available_xsecs(xfi, title=args.species)
    plt.gcf().savefig(os.path.join(output_dir, 'xsec_datasets.pdf'))
    plt.gcf().clear()

    rms_file = os.path.join(output_dir, 'optimized.json')
    if os.path.exists(rms_file):
        logger.info(f'Reading precalculated RMS values form {rms_file}.')
        rms_result = hx.xsec.load_rms_data(rms_file)
    else:
        rms_result = hx.optimize_xsec_multi(xfi)
        hx.save_rms_data(rms_file, rms_result)

    hx.plotting.scatter_and_fit(xfi, rms_result, outliers=False)
    plt.gcf().savefig(os.path.join(output_dir, 'xsec_scatter.pdf'))
    plt.gcf().clear()

    axml.save(hx.fit.gen_arts(xfi, rms_result),
              os.path.join(output_dir, 'cfc.xml'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
