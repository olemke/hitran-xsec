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
    args.add_argument('-r', '--rms-plots', action='store_true',
                      help='Generate cross section and rms plots.')
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
        plotfile = os.path.join(output_dir, 'xsec_datasets.pdf')
        hx.plotting.plot_available_xsecs(xfi, title=species)
        plt.gcf().savefig(plotfile)
        plt.gcf().clear()
        logger.info(f'Wrote {plotfile}')

        rms_file = os.path.join(output_dir, 'xsec_rms.json')
        if os.path.exists(rms_file) and not args.ignore_rms:
            logger.info(f'Reading precalculated RMS values form {rms_file}.')
            rms_result = hx.xsec.load_rms_data(rms_file)
        else:
            rms_result = hx.optimize_xsec_multi(xfi)
            if rms_result:
                hx.save_rms_data(rms_file, rms_result)
                logger.info(f'Wrote {rms_file}')
            else:
                logger.warning(f'No results for {species}')

        # Plot of best FWHM vs. pressure difference and the fit
        if rms_result:
            xml_file = os.path.join(output_dir, 'cfc.xml')
            axml.save(hx.fit.gen_arts(xfi, rms_result), xml_file)
            logger.info(f'Wrote {xml_file}')

            plotfile = os.path.join(output_dir, 'xsec_scatter.pdf')
            hx.plotting.scatter_and_fit(xfi, rms_result, outliers=False)
            plt.gcf().savefig(plotfile)
            plt.gcf().clear()
            logger.info(f'Wrote {plotfile}')

            if args.rms_plots:
                for r in rms_result:
                    hx.plotting.generate_rms_and_spectrum_plots(
                        xfi, title=species, xsec_result=r, outdir=output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
