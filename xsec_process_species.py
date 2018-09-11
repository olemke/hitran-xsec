import argparse
import logging
import sys

import matplotlib.pyplot as plt
import typhon

from hitran_xsec import (calc_broadening, calc_temperature_correction,
                         combine_data_for_arts, SPECIES_GROUPS,
                         XSEC_SPECIES_INFO, set_default_logging_format)

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


def main():
    typhon.plots.styles.use('typhon')

    args = parse_args()

    if args.style:
        plt.style.use(args.style)

    species = []
    for s in args.species:
        if s in XSEC_SPECIES_INFO.keys():
            species.append(s)
        elif s in SPECIES_GROUPS.keys():
            species += SPECIES_GROUPS[s]
        else:
            raise RuntimeError(f'Unknown xsec species {s}. '
                               'Not found in XSEC_SPECIES_INFO.')

    if args.command == 'arts':
        args.species = species
        args.execute(**vars(args))
    else:
        for s in species:
            args.species = s
            args.execute(**vars(args))

    return 0


if __name__ == '__main__':
    sys.exit(main())
