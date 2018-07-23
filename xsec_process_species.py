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
    # Required commandline argument
    args.add_argument('species', metavar='SPECIES', nargs='+',
                      help='Name of species to process.'
                           'Pass "rfmip" for all RFMIP species.')
    # Optional commandline arguments
    args.add_argument('-d', '--directory', default='.',
                      help='Directory with HITRAN cross section data files.')
    args.add_argument('-i', '--ignore-rms', action='store_true',
                      help='Ignore existing RMS file (recalculate).')
    args.add_argument('-o', '--output', metavar='OUTPUT_DIRECTORY',
                      default='output',
                      help='Output directory. A subdirectory named SPECIES '
                           'will be created inside.')
    args.add_argument('-r', '--rms-plots', action='store_true',
                      help='Generate cross section and rms plots.')
    return args.parse_args()


RFMIP_SPECIES = [
    # Alcohols, ethers and other oxygenated hydrocarbons

    # Bromocarbons, Hydrobromocarbons, and Halons
    'CBrClF2',  # RFMIP name: HALON1211, no air broadening, only N2
    'CBrF3',  # RFMIP name: HALON1301, no air broadening, only N2
    'C2Br2F4',  # RFMIP name: HALON2402, not in Hitran

    # Chlorocarbons and Hydrochlorocarbons
    'CCl4',  # RFMIP Name: CARBON_TETRACHLORIDE
    #          +++++ fit ok +++++, use only band 700-860
    'CH2Cl2',  # no air broadening, only N2
    'CH3CCl3',  # not available in Hitran
    'CHCl3',  # not available in Hitran

    # Chlorofluorocarbons (CFCs)
    'CFC-11',  # +++++ fit ok +++++
    'CFC11EQ',  # not in Hitran
    'CFC-12',  # +++++ fit ok +++++
    'CFC12EQ',  # not in Hitran
    'CFC-113',  # only data for 0 torr
    'CFC-114',  # only data for 0 torr
    'CFC-115',  # only data for 0 torr

    # Fully Fluorinated Species
    'C2F6',  # !!!!! bad fit !!!!!
    'C3F8',  # no air broadening, only N2
    'C4F10',  # no air broadening, only N2
    'n-C5F12',  # no air broadening, only N2
    'n-C6F14',  # no air broadening, only N2
    'C7F16',  # not in Hitran
    'C8F18',  # no air broadening, only N2
    'c-C4F8',  # only data for 0 Torr
    'CF4',  # +++++ fit ok +++++
    'NF3',  # no air broadening, only N2
    'SO2F2',  # no air broadening, only N2

    # Halogenated Alcohols and Ethers

    # Hydrocarbons

    # Hydrochlorofluorocarbons (HCFCs)
    'HCFC-141b',  # only data for 0 torr
    'HCFC-142b',  # only data for 0 torr
    'HCFC-22',  # !!!!! bad fit !!!!!, no high pressure differences available

    # Hydrofluorocarbons (HFCs)
    'CFH2CF3',  # RFMIP name: HFC-134a, also available in Hitran under that
    #             name, but without the 750-1600 band. This gives a better fit.
    #             +++++ fit ok +++++. Use band 750-1600.
    'HFC125'  # not available in Hitran
    'HFC134AEQ',  # not available in Hitran
    'HFC-143a',  # not enough xsecs available
    'HFC-152a',  # only data for 0 torr
    'HFC227EA',  # not available in Hitran
    'HFC236FA',  # not available in Hitran
    'CHF3',  # RFMIP name: HFC-23
    'HFC245FA',  # not available in Hitran
    'HFC-32',  # not enough xsecs available
    'CH3CF2CH2CF3',  # RFMIP name: HFC365MFC, not enough xsecs available
    'HFC4310MEE',  # not available in Hitran

    # Iodocarbons and hydroiodocarbons

    # Nitriles, amines and other nitrogenated hydrocarbons

    # Other molecules

    # Sulfur-containing species

]


def main():
    logger = logging.getLogger(__name__)
    args = parse_args()

    if args.species[0] == 'rfmip':
        args.species = RFMIP_SPECIES
    for species in args.species:
        output_dir = os.path.join(args.output, species)

        xfi = hx.XsecFileIndex(directory=args.directory, species=species,
                               ignore='.*[^0-9._].*')
        if xfi.files:
            os.makedirs(output_dir, exist_ok=True)
        else:
            logger.warning(f'No input files found for {species}.')
            continue

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
            rms_result = [x for x in hx.optimize_xsec_multi(xfi) if x]
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
