import argparse
import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import typhon
import typhon.arts.xml as axml

import hitran_xsec as hx


def set_default_logging_format(level=None, include_timestamp=False,
                               include_function=True):
    """Generate decently looking logging format string."""

    if level is None:
        level = logging.INFO

    color = "\033[1;%dm"
    reset = "\033[0m"
    black, red, green, yellow, blue, magenta, cyan, white = [
        color % (30 + i) for i in range(8)]
    logformat = '['
    if include_timestamp:
        logformat += f'{red}%(asctime)s.%(msecs)03d{reset}:'
    logformat += (f'{yellow}%(filename)s{reset}'
                  f':{blue}%(lineno)s{reset}')
    if include_function:
        logformat += f':{green}%(funcName)s{reset}'
    logformat += f'] %(message)s'

    logging.basicConfig(
        format=logformat,
        level=level,
        datefmt='%H:%M:%S')


set_default_logging_format(level=logging.INFO,
                           include_timestamp=True,
                           include_function=True),
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optional commandline arguments
    parser.add_argument('-d', '--xscdir', default='.',
                        help='Directory with HITRAN cross section data files.')
    parser.add_argument('-o', '--outdir', metavar='OUTPUT_DIRECTORY',
                        default='output',
                        help='Output directory. A subdirectory named SPECIES '
                             'will be created inside.')
    parser.add_argument('-s', '--style', type=str, default=None,
                        help='Custom matplotlib style name.')

    # RMS command parser
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.metavar = 'COMMAND'
    subparser = subparsers.add_parser('rms',
                                      help='Calculate rms and perform fitting.')
    subparser.add_argument('-i', '--ignore-rms', action='store_true',
                           help='Ignore existing RMS file (recalculate).')
    subparser.add_argument('-r', '--rms-plots', action='store_true',
                           help='Generate cross section and rms plots.')
    subparser.set_defaults(command='rms')
    subparser.set_defaults(execute=calc_broadening)

    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    # tfit command parser
    subparser = subparsers.add_parser('tfit',
                                      help='Analyze temperature behaviour.')
    subparser.add_argument('-t', '--tref', type=int, default=-1,
                           help='Reference temperature.')
    subparser.set_defaults(command='tfit')
    subparser.set_defaults(execute=calc_temperature_correction)

    # Required commandline argument
    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    # average command parser
    # TODO: Implement averaging command
    subparser = subparsers.add_parser(
        'avg', help='Calculate average coefficients from reference species.')

    # arts export command parser
    subparser = subparsers.add_parser('arts', help='Combine data for ARTS.')
    subparser.set_defaults(command='arts')
    subparser.set_defaults(execute=combine_data_for_arts)

    subparser.add_argument('species', metavar='SPECIES', nargs='+',
                           help='Name of species to process. '
                                'Pass "rfmip" for all RFMIP species.')

    return parser.parse_args()


def prepare_data(directory, output_dir, species):
    # Uses all available spectra. To only use those for air broadening, add
    # keyword ignore='.*[^0-9._].*' below
    xfi = hx.XsecFileIndex(directory=directory, species=species)
    if xfi.files:
        os.makedirs(output_dir, exist_ok=True)
    if xfi.ignored_files:
        logger.info(f'Ignored xsec files: {xfi.ignored_files}')
    return xfi


def combine_data_for_arts(species, outdir, **_):
    # FIXME: How to handle the active flag?
    # active_species = {k: v for k, v in hx.XSEC_SPECIES_INFO.items()
    #                   if k in species
    #                   and (('active' in v and v[
    #     'active']) or 'active' not in v)}

    combined_xml_file = os.path.join(outdir, 'cfc_combined.xml')
    all_species = []
    for s in species:
        cfc_file = os.path.join(outdir, s, 'cfc.xml')
        try:
            data = axml.load(cfc_file)
        except FileNotFoundError:
            logger.warning(f"No xml file found for species {s}, ignoring")
        else:
            all_species.append(data)
            logger.info(f'Added {s}')

    axml.save(list(itertools.chain(*all_species)), combined_xml_file)
    logger.info(f'Wrote {combined_xml_file}')


def calc_temperature_correction(species, xscdir, outdir, tref=-1, **_):
    output_dir = os.path.join(outdir, species)

    xfi = prepare_data(xscdir, output_dir, species)
    if not xfi.files:
        logger.warning(f'No input files found for {species}.')
        return

    tfit_result = hx.plotting.temperature_fit_multi(xfi, tref,
                                                    output_dir, species, 1)
    tfit_result = [x for x in tfit_result if x]

    tfit_file = os.path.join(output_dir, 'xsec_tfit.json')
    if tfit_result:
        hx.save_rms_data(tfit_file, tfit_result)
        logger.info(f'Wrote {tfit_file}')


def calc_broadening(species, xscdir, outdir, ignore_rms=False, rms_plots=False,
                    **_):
    output_dir = os.path.join(outdir, species)

    xfi = prepare_data(xscdir, output_dir, species)
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
    if os.path.exists(rms_file) and not ignore_rms:
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

        # Load temperature fit if available
        try:
            tfit_file = os.path.join(output_dir, 'xsec_tfit.json')
            tfit_result = hx.xsec.load_rms_data(tfit_file)
            logger.info(f'Loaded temperature fit data for {species}')
        except FileNotFoundError:
            logger.info(f'No temperature fit data for {species}')
            tfit_result = None

        xsec_records = (hx.fit.gen_arts(xfi, rms_result, tfit_result),)
        axml.save(xsec_records, xml_file)
        logger.info(f'Wrote {xml_file}')

        plotfile = os.path.join(output_dir, 'xsec_bands.pdf')
        plt.figure()
        hx.plotting.plot_xsec_records(xsec_records)
        plt.savefig(plotfile)
        logger.info(f'Wrote {plotfile}')

        plotfile = os.path.join(output_dir, 'xsec_scatter.pdf')
        plt.figure()
        hx.plotting.scatter_plot_by_pressure_difference(xfi, rms_result,
                                                        outliers=False)
        plt.savefig(plotfile)
        logger.info(f'Wrote {plotfile}')

        plotfile = os.path.join(output_dir, 'xsec_scatter_temp.pdf')
        plt.figure()
        hx.plotting.scatter_plot_by_temperature(xfi, rms_result)
        plt.savefig(plotfile)
        logger.info(f'Wrote {plotfile}')

        if rms_plots:
            for r in rms_result:
                hx.plotting.generate_rms_and_spectrum_plots(
                    xfi, title=species, xsec_result=r, outdir=output_dir)


def main():
    typhon.plots.styles.use('typhon')

    args = parse_args()

    if args.style:
        plt.style.use(args.style)

    species = []
    for s in args.species:
        if s in hx.XSEC_SPECIES_INFO.keys():
            species.append(s)
        elif s in hx.SPECIES_GROUPS.keys():
            species += hx.SPECIES_GROUPS[s]
        else:
            raise RuntimeError(f'Unknown xsec species {s}. '
                               'Not found in XSEC_SPECIES_INFO.')

    if args.command == 'arts':
        args.execute(**vars(args))
    else:
        for s in species:
            args.species = s
            args.execute(**vars(args))

    return 0


if __name__ == '__main__':
    sys.exit(main())
