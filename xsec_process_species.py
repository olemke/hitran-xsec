import argparse
import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import typhon
import typhon.arts.xml as axml

import hitran_xsec as hx

logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optional commandline arguments
    parser.add_argument('-d', '--directory', default='.',
                        help='Directory with HITRAN cross section data files.')
    parser.add_argument('-o', '--output', metavar='OUTPUT_DIRECTORY',
                        default='output',
                        help='Output directory. A subdirectory named SPECIES '
                             'will be created inside.')
    parser.add_argument('-s', '--style', type=str, default=None,
                        help='Custom matplotlib style name.')

    # RMS command parser
    subparsers = parser.add_subparsers()
    subparser = subparsers.add_parser('rms',
                                      help='Calculate rms and perform fitting.')
    subparser.add_argument('-i', '--ignore-rms', action='store_true',
                           help='Ignore existing RMS file (recalculate).')
    subparser.add_argument('-r', '--rms-plots', action='store_true',
                           help='Generate cross section and rms plots.')
    subparser.set_defaults(command='rms')
    subparser.set_defaults(execute=rms_and_fitting)

    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    # tfit command parser
    subparser = subparsers.add_parser('tfit',
                                      help='Analyze temperature behaviour.')
    subparser.add_argument('-t', '--tref', type=int, default=0,
                           help='Reference temperature.')
    subparser.set_defaults(command='tfit')
    subparser.set_defaults(execute=compare_different_temperatures)

    # Required commandline argument
    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    # arts export command parser
    subparser = subparsers.add_parser('arts', help='Combine data for ARTS.')
    subparser.set_defaults(command='arts')
    subparser.set_defaults(execute=combine_data_for_arts)

    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    return parser.parse_args()


def prepare_data(directory, output_dir, species):
    xfi = hx.XsecFileIndex(directory=directory, species=species,
                           ignore='.*[^0-9._].*')
    if xfi.files:
        os.makedirs(output_dir, exist_ok=True)
    return xfi


def combine_data_for_arts(species, args):
    active_species = {k: v for k, v in hx.RFMIP_SPECIES.items()
                      if k in species
                      and (('active' in v and v[
        'active']) or 'active' not in v)}

    combined_xml_file = os.path.join(args.output, 'cfc_combined.xml')
    all_species = []
    for s in species:
        cfc_file = os.path.join(args.output, s, 'cfc.xml')
        try:
            data = axml.load(cfc_file)
        except:
            logger.warning(f"No xml file found for species {s}, ignoring")
        else:
            all_species.append(data)
            logger.info(f'Added {s}')

    axml.save(list(itertools.chain(*all_species)), combined_xml_file)
    logger.info(f'Wrote {combined_xml_file}')


def compare_different_temperatures(species, args):
    output_dir = os.path.join(args.output, species)

    xfi = prepare_data(args.directory, output_dir, species)
    if not xfi.files:
        logger.warning(f'No input files found for {species}.')
        return

    tfit_result = hx.plotting.temperature_fit_multi(xfi, args.tref, output_dir,
                                                    species, 1)

    tfit_file = os.path.join(output_dir, 'xsec_tfit.json')
    if tfit_result:
        hx.save_rms_data(tfit_file, tfit_result)
        logger.info(f'Wrote {tfit_file}')


def rms_and_fitting(species, args):
    output_dir = os.path.join(args.output, species)

    xfi = prepare_data(args.directory, output_dir, species)
    if not xfi.files:
        logger.warning(f'No input files found for {species}.')
        return

    # Scatter plot of available cross section data files
    plotfile = os.path.join(output_dir, 'xsec_datasets.pdf')
    plt.figure()
    hx.plotting.plot_available_xsecs(xfi, title=species)
    plt.savefig(plotfile)
    logger.info(f'Wrote {plotfile}')

    rms_file = os.path.join(output_dir, 'xsec_rms.json')
    if os.path.exists(rms_file) and not args.ignore_rms:
        logger.info(f'Reading precalculated RMS values form {rms_file}.')
        rms_result = hx.xsec.load_rms_data(rms_file)
    else:
        rms_result = [x for x in hx.optimize_xsec_multi(xfi) if x]
        if rms_result:
            hx.save_rms_data(rms_file, rms_result)
            logger.info(f'Wrote {rms_file}')
        else:
            logger.warning(f'No results for {species}')

    # Plot of best FWHM vs. pressure difference and the fit
    if rms_result:
        xml_file = os.path.join(output_dir, 'cfc.xml')
        xsec_records = (hx.fit.gen_arts(xfi, rms_result),)
        axml.save(xsec_records, xml_file)
        logger.info(f'Wrote {xml_file}')

        plotfile = os.path.join(output_dir, 'xsec_bands.pdf')
        plt.figure()
        hx.plotting.plot_xsec_records(xsec_records)
        plt.savefig(plotfile)
        logger.info(f'Wrote {plotfile}')

        plotfile = os.path.join(output_dir, 'xsec_scatter.pdf')
        plt.figure()
        hx.plotting.scatter_and_fit(xfi, rms_result, outliers=False)
        plt.savefig(plotfile)
        logger.info(f'Wrote {plotfile}')

        if args.rms_plots:
            for r in rms_result:
                hx.plotting.generate_rms_and_spectrum_plots(
                    xfi, title=species, xsec_result=r, outdir=output_dir)


def main():
    typhon.plots.styles.use('typhon')

    global logger
    logger = logging.getLogger(__name__)
    args = parse_args()

    if args.style:
        plt.style.use(args.style)

    if args.species[0] == 'rfmip':
        args.species = hx.RFMIP_SPECIES.keys()

    if args.command == 'arts':
        args.execute(args.species, args)
    else:
        for species in args.species:
            args.execute(species, args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
