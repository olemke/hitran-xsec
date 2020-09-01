import argparse
import logging
import multiprocessing as mp
import sys

import matplotlib.pyplot as plt
import typhon

from hitran_xsec import (
    calc_broadening,
    calc_temperature_correction,
    combine_data_for_arts,
    calc_average_coeffs,
    create_data_overview,
    run_analysis,
    SPECIES_GROUPS,
    XSEC_SPECIES_INFO,
    xsec_config,
    set_default_logging_format,
)

set_default_logging_format(
    level=logging.INFO, include_timestamp=True, include_function=True
),
logger = logging.getLogger(__name__)


def add_rms_parser_args(subparsers):
    """RMS command parser"""
    subparser = subparsers.add_parser("rms", help="Calculate rms and perform fitting.")
    subparser.add_argument(
        "-i",
        "--ignore-rms",
        action="store_true",
        help="Ignore existing RMS file (recalculate).",
    )
    subparser.add_argument(
        "-r",
        "--rms-plots",
        action="store_true",
        help="Generate cross section and rms plots.",
    )
    subparser.add_argument(
        "-a",
        "--averaged",
        action="store_true",
        help="Force use of averaged coefficients.",
    )
    subparser.set_defaults(command="rms")
    subparser.set_defaults(execute=calc_broadening)

    subparser.add_argument(
        "species",
        metavar="SPECIES",
        nargs="+",
        help="Name of species to process. " 'Pass "rfmip" for all RFMIP species.',
    )


def add_tfit_parser_args(subparsers):
    """tfit command parser"""
    subparser = subparsers.add_parser("tfit", help="Analyze temperature behaviour.")
    subparser.add_argument(
        "-t", "--tref", type=int, default=-1, help="Reference temperature."
    )
    subparser.set_defaults(command="tfit")
    subparser.set_defaults(execute=calc_temperature_correction)

    # Required commandline argument
    subparser.add_argument(
        "species",
        metavar="SPECIES",
        nargs="+",
        help="Name of species to process. " 'Pass "rfmip" for all RFMIP species.',
    )


def add_average_parser_args(subparsers):
    """Average command parser"""
    # TODO: Implement averaging command
    subparser = subparsers.add_parser(
        "avg", help="Calculate average coefficients from reference species."
    )
    subparser.set_defaults(command="average")
    subparser.set_defaults(execute=calc_average_coeffs)

    # Required commandline argument
    subparser.add_argument(
        "species",
        metavar="SPECIES",
        nargs="+",
        help="Name of species to process. " 'Pass "reference" for all default species.',
    )


def add_arts_parser_args(subparsers):
    """ARTS export command parser"""
    subparser = subparsers.add_parser("arts", help="Combine data for ARTS.")
    subparser.set_defaults(command="arts")
    subparser.set_defaults(execute=combine_data_for_arts)

    subparser.add_argument(
        "species",
        metavar="SPECIES",
        nargs="+",
        help="Name of species to process. " 'Pass "rfmip" for all RFMIP species.',
    )


def add_overview_parser_args(subparsers):
    """ARTS export command parser"""
    subparser = subparsers.add_parser("overview", help="Create overview of xsec data.")
    subparser.set_defaults(command="overview")
    subparser.set_defaults(execute=create_data_overview)
    subparser.set_defaults(species=[])


def add_analysis_parser_args(subparsers):
    """ARTS export command parser"""
    subparser = subparsers.add_parser("analysis", help="Analysis of xsec data.")
    subparser.set_defaults(command="analysis")
    subparser.set_defaults(execute=run_analysis)
    subparser.set_defaults(species=[])

    subparser.add_argument(
        "--fig1",
        action="store_true",
        help="Apply and plot tfit at different tempertures.",
    )

    subparser.add_argument(
        "--fig2",
        action="store_true",
        help="Plot intensity at different temperatures.",
    )

    subparser.add_argument(
        "species",
        metavar="SPECIES",
        nargs="+",
        help="Name of species to process. " 'Pass "rfmip" for all RFMIP species.',
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Optional commandline arguments
    parser.add_argument(
        "-d",
        "--xscdir",
        default=".",
        help="Directory with HITRAN cross section data files.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        metavar="OUTPUT_DIRECTORY",
        default="output",
        help="Output directory. A subdirectory named SPECIES "
        "will be created inside.",
    )
    parser.add_argument(
        "-p", "--processes", type=int, help="Maximum number of processes."
    )
    parser.add_argument(
        "-s", "--style", type=str, default=None, help="Custom matplotlib style name."
    )

    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.metavar = "COMMAND"

    add_rms_parser_args(subparsers)
    add_tfit_parser_args(subparsers)
    add_average_parser_args(subparsers)
    add_arts_parser_args(subparsers)
    add_overview_parser_args(subparsers)
    add_analysis_parser_args(subparsers)

    return parser.parse_args()


def main():
    typhon.plots.styles.use("typhon")

    args = parse_args()

    if args.style:
        plt.style.use(args.style)

    xsec_config.nprocesses = args.processes

    species = []
    for s in args.species:
        if s in XSEC_SPECIES_INFO.keys():
            species.append(s)
        elif s in SPECIES_GROUPS.keys():
            species += SPECIES_GROUPS[s]
        else:
            raise RuntimeError(
                f"Unknown xsec species {s}. " "Not found in XSEC_SPECIES_INFO."
            )

    if args.command in ("arts", "average", "analysis"):
        args.species = species
        args.execute(**vars(args))
    elif args.command in ("overview"):
        args.execute(**vars(args))
    elif args.command in ("tfit"):
        with mp.Pool(processes=xsec_config.nprocesses) as pool:
            pool.starmap(
                calc_temperature_correction,
                ((s, args.xscdir, args.outdir, args.tref) for s in species),
            )
    else:
        for s in species:
            args.species = s
            args.execute(**vars(args))

    return 0


if __name__ == "__main__":
    sys.exit(main())
