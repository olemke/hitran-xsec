import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import typhon.arts.xml as axml

import hitran_xsec as hx

logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('species', metavar='SPECIES', nargs='+',
                      help='Name of species to process.')
    args.add_argument('-d', '--directory', default='.',
                      help='Directory with cross section data files.')
    args.add_argument('-i', '--ignore-rms', action='store_true',
                      help='Ignore existing RMS file.')
    args.add_argument('-o', '--output', metavar='OUTPUT_DIRECTORY',
                      default='output',
                      help='Output directory.'
                           'A subdirectory SPECIES will be created inside.')
    return args.parse_args()


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()

    for species in args.species:
        output_dir = os.path.join(args.output, species)
        os.makedirs(output_dir, exist_ok=True)

        xfi = hx.XsecFileIndex(directory=args.directory, species=species,
                               ignore='.*[^0-9._].*')

        # Scatter plot of available cross section data files
        hx.plotting.plot_available_xsecs(xfi, title=species)
        plt.gcf().savefig(os.path.join(output_dir, 'xsec_datasets.pdf'))
        plt.gcf().clear()

        rms_file = os.path.join(output_dir, 'xsec_rms.json')
        if os.path.exists(rms_file) and not args.ignore_rms:
            logger.info(f'Reading precalculated RMS values form {rms_file}.')
            rms_result = hx.xsec.load_rms_data(rms_file)
        else:
            rms_result = hx.optimize_xsec_multi(xfi)
            hx.save_rms_data(rms_file, rms_result)

        for r in rms_result:
            hx.plotting.generate_rms_and_spectrum_plots(
                xfi, title=species, xsec_result=r, outdir=output_dir)

        # Plot of best FWHM vs. pressure difference and the fit
        hx.plotting.scatter_and_fit(xfi, rms_result, outliers=False)
        plt.gcf().savefig(os.path.join(output_dir, 'xsec_scatter.pdf'))
        plt.gcf().clear()

        axml.save(hx.fit.gen_arts(xfi, rms_result),
                  os.path.join(output_dir, 'cfc.xml'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
