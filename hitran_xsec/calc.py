"""High-level functions for cross section data processing"""

import itertools
import logging
import os

import matplotlib.pyplot as plt
import typhon.arts.xml as axml

from .fit import gen_arts, calc_fwhm_and_pressure_difference, do_rms_fit
from .plotting import (
    plot_available_xsecs,
    plot_xsec_records,
    scatter_plot_by_pressure_difference_per_band,
    scatter_plot_by_temperature,
    generate_rms_and_spectrum_plots,
    temperature_fit,
    scatter_plot_by_pressure_difference,
)
from .xsec import (
    XsecFileIndex,
    XsecError,
    save_rms_data,
    load_rms_data,
    optimize_xsec_multi,
)

logger = logging.getLogger(__name__)


def prepare_data(directory, output_dir, species):
    # Uses all available spectra. To only use those for air broadening, add
    # keyword ignore='.*[^0-9._].*' below
    xfi = XsecFileIndex(directory=directory, species=species)
    if xfi.files:
        os.makedirs(output_dir, exist_ok=True)
    if xfi.ignored_files:
        logger.info(f"Ignored xsec files: {xfi.ignored_files}")
    return xfi


def combine_data_for_arts(species, outdir, **_):
    # FIXME: How to handle the active flag?
    # active_species = {k: v for k, v in XSEC_SPECIES_INFO.items()
    #                   if k in species
    #                   and (('active' in v and v[
    #     'active']) or 'active' not in v)}

    combined_xml_file = os.path.join(outdir, "cfc_combined.xml")
    all_species = []
    for s in species:
        cfc_file = os.path.join(outdir, s, "cfc.xml")
        try:
            data = axml.load(cfc_file)
        except FileNotFoundError:
            logger.warning(f"No xml file found for species {s}, ignoring")
        else:
            all_species.append(data)
            logger.info(f"Added {s}")

    axml.save(list(itertools.chain(*all_species)), combined_xml_file, format="binary")
    logger.info(f"Wrote {combined_xml_file}")


def calc_temperature_correction(species, xscdir, outdir, tref=-1, **_):
    output_dir = os.path.join(outdir, species)

    xfi = prepare_data(xscdir, output_dir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    bands = xfi.cluster_by_band_and_pressure()
    tfit_result = [
        temperature_fit(x, output_dir, species, tref) for band in bands for x in band
    ]
    tfit_result = [x for x in tfit_result if x]

    tfit_file = os.path.join(output_dir, "xsec_tfit.json")
    if tfit_result:
        save_rms_data(tfit_file, tfit_result)
        logger.info(f"Wrote {tfit_file}")


def calc_broadening(
    species, xscdir, outdir, ignore_rms=False, rms_plots=False, averaged=False, **_
):
    output_dir = os.path.join(outdir, species)

    xfi = prepare_data(xscdir, output_dir, species)
    if not xfi.files:
        logger.warning(f"No input files found for {species}.")
        return

    averaged_coeffs_xml_file = os.path.join(outdir, "cfc_averaged_coeffs.xml")
    if os.path.exists(averaged_coeffs_xml_file):
        avg_coeffs = axml.load(averaged_coeffs_xml_file)
    else:
        avg_coeffs = None

    # Scatter plot of available cross section data files
    plotfile = os.path.join(output_dir, "xsec_datasets.pdf")
    fig = plt.figure()
    plot_available_xsecs(xfi, title=species)
    plt.savefig(plotfile)
    plt.close(fig)
    logger.info(f"Wrote {plotfile}")

    rms_file = os.path.join(output_dir, "xsec_rms.json")
    rms_result = None
    if os.path.exists(rms_file) and not ignore_rms:
        logger.info(f"Reading precalculated RMS values form {rms_file}.")
        rms_result = load_rms_data(rms_file)
    elif not averaged:
        rms_result = [x for x in optimize_xsec_multi(xfi) if x]
        if rms_result:
            save_rms_data(rms_file, rms_result)
            logger.info(f"Wrote {rms_file}")
        else:
            logger.warning(f"No results for {species}")

    # Plot of best FWHM vs. pressure difference and the fit
    if rms_result and rms_plots:
        for r in rms_result:
            generate_rms_and_spectrum_plots(
                xfi, title=species, xsec_result=r, outdir=output_dir
            )

    # Load temperature fit if available
    try:
        tfit_file = os.path.join(output_dir, "xsec_tfit.json")
        tfit_result = load_rms_data(tfit_file)
        logger.info(f"Loaded temperature fit data for {species}")
    except FileNotFoundError:
        logger.info(f"No temperature fit data for {species}")
        tfit_result = None

    try:
        xsec_records = (
            gen_arts(xfi, rms_result, tfit_result, averaged_coeffs=avg_coeffs),
        )
    except XsecError as e:
        logger.warning(str(e))
        logger.warning(f"No RMS calculation possible for {species}")
        return

    xml_file = os.path.join(output_dir, "cfc.xml")
    axml.save(xsec_records, xml_file)
    logger.info(f"Wrote {xml_file}")

    plotfile = os.path.join(output_dir, "xsec_bands.pdf")
    fig = plt.figure()
    plot_xsec_records(xsec_records)
    plt.savefig(plotfile)
    plt.close(fig)
    logger.info(f"Wrote {plotfile}")

    if rms_result:
        plotfile = os.path.join(output_dir, "xsec_scatter.pdf")
        fig = plt.figure()
        scatter_plot_by_pressure_difference_per_band(xfi, rms_result, outliers=False)
        plt.savefig(plotfile)
        plt.close(fig)
        logger.info(f"Wrote {plotfile}")

        plotfile = os.path.join(output_dir, "xsec_scatter_temp.pdf")
        fig = plt.figure()
        scatter_plot_by_temperature(xfi, rms_result)
        plt.savefig(plotfile)
        plt.close(fig)
        logger.info(f"Wrote {plotfile}")


def calc_average_coeffs(species, outdir, **_):
    """Calculate averaged coefficients"""
    averaged_coeffs_xml_file = os.path.join(outdir, "cfc_averaged_coeffs.xml")
    averaged_species_xml_file = os.path.join(outdir, "cfc_averaged_species.xml")
    all_species = []
    for s in species:
        rms_file = os.path.join(outdir, s, "xsec_rms.json")
        try:
            data = load_rms_data(rms_file)
        except FileNotFoundError:
            logger.warning(f"No RMS file found for species {s}, ignoring")
        else:
            all_species.extend(data)
            logger.info(f"Added {s}")

    plotfile = os.path.join(outdir, "xsec_avg_scatter.pdf")
    fig = plt.figure()
    scatter_plot_by_pressure_difference(
        all_species, species=",".join(species), outliers=False
    )
    plt.savefig(plotfile)
    plt.close(fig)
    logger.info(f"Wrote {plotfile}")

    fwhm, pressure_diff = calc_fwhm_and_pressure_difference(all_species)
    avg_coeffs, _, _ = do_rms_fit(fwhm, pressure_diff)

    axml.save(avg_coeffs, averaged_coeffs_xml_file)
    logger.info(f"Wrote {averaged_coeffs_xml_file}")
    axml.save(species, averaged_species_xml_file)
    logger.info(f"Wrote {averaged_species_xml_file}")
