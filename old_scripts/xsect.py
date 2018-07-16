#!/usr/bin/env python3

import copy
import itertools
import logging
import multiprocessing as mp
import os
import sys

import numpy
import typhon

import xsect_utils as xu

logging.basicConfig(level=logging.WARN)
# format='%(asctime)s:%(levelname)s: %(message)s',
# datefmt='%b %d %H:%M:%S')

xulogger = logging.getLogger(xu.__name__)
xulogger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

XSECINFO = {
    'CFC11': {
        'molname': 'CCl3F',
        'bands': ((810, 880), (1050, 1120)),
    },
    'CFC12': {
        'molname': 'CCl2F2',
        'bands': ((800, 1270), (850, 950), (1050, 1200)),
    },
    'HCFC22': {
        'molname': 'CHClF2',
        'bands': ((730, 1380), (760, 860), (1070, 1195)),
    },
    'HFC134a': {
        'molname': 'CFH2CF3',
        'bands': ((750, 1600), (1035, 1130), (1135, 1140)),
    },
}


def main():
    #p = mp.Pool(processes=16)
    p = mp.Pool()

    logger.info('Reading cross section files')
    if len(sys.argv) > 3 and sys.argv[2] == 'CFC11':
        species = sys.argv[2]
        xsecs = xu.read_hitran_xsec_multi('cfc11/*00.xsc')
        inputs = xu.combine_inputs(
            xsecs,
            (190, 201, 208, 216, 225, 232, 246, 260, 272),
            (810, 1050),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'CFC12':
        species = sys.argv[2]
        xsecs = xu.read_hitran_xsec_multi('cfc12/*00.xsc')
        inputs = xu.combine_inputs(
            xsecs,
            (190, 201, 208, 216, 225, 232, 246, 260, 268, 272),
            (800, 850, 1050),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'HCFC22':
        species = sys.argv[2]
        xsecs = xu.read_hitran_xsec_multi(
            ['hcfc22/*_730*.xsc', 'hcfc22/*_760*.xsc', 'hcfc22/*_1070*.xsc'])
        inputs = xu.combine_inputs(
            xsecs,
            (181, 190, 200, 208, 216, 225, 233, 251, 270, 296),
            (730, 760, 1070),
            species)
    elif len(sys.argv) > 3 and sys.argv[2] == 'HFC134a':
        species = sys.argv[2]
        xsecs = xu.read_hitran_xsec_multi(
            ['hfc134a/*_750*.xsc', 'hfc134a/*_1035*.xsc',
             'hfc134a/*_1135*.xsc'])
        inputs = xu.combine_inputs(
            xsecs,
            (190, 200, 208, 216, 225, 231, 245, 250, 261, 271, 284, 295),
            (750, 1035, 1135),
            species)
    else:
        xu.print_usage()
        sys.exit(1)

    command = sys.argv[1]
    outdir = sys.argv[3]
    outfile = os.path.join(outdir, 'output.txt')

    if command == 'rms':
        os.makedirs(outdir, exist_ok=True)

        logger.info(f'Calculating RMS')
        res = [p.apply_async(xu.optimize_xsec, args[0:2]) for args in inputs]
        results = [r.get() for r in res if r]
        logger.info(f'Done {len(results)} calculations')

        xu.save_output(outfile, results)
        logger.info(f'Saved output to {outfile}')
    elif command == 'genarts':
        print("Generating ARTS XML file")
        results = xu.load_output(outfile)
        xsecs = []
        refpressure = []
        reftemperature = []
        fmin = []
        fmax = []

        if sys.argv[2] == 'HFC134a':
            xsec_refs = ((750, 1600, 250, 3266),)
        elif sys.argv[2] == 'HCFC22':
            xsec_refs = ((730, 1380, 232, 1000),)
        elif sys.argv[2] == 'CFC11':
            xsec_refs = ((810, 880, 231, 1000),
                         (1050, 1120, 231, 1000),)
        elif sys.argv[2] == 'CFC12':
            xsec_refs = ((800, 1270, 233, 1000),)

        for xsec_ref in xsec_refs:
            for xsec in inputs:
                if (xsec_ref[2] - 1 <= xsec[0]['temperature'] <= xsec_ref[2] + 1
                        and xsec_ref[0] - 1 <= xu.frequency2wavenumber(
                            xsec[0]['fmin']) / 100 <= xsec_ref[0] + 1
                        and xsec_ref[1] - 1 <= xu.frequency2wavenumber(
                            xsec[0]['fmax']) / 100 <= xsec_ref[1] + 1
                        and xsec_ref[3] - 1 <= xsec[0]['pressure'] <= xsec_ref[
                            3] + 1
                        and xsec[0]['fmin'] not in fmin
                        and xsec[0]['fmax'] not in fmax):
                    xsecs.append(
                        xsec[0]['data'] / 10000.)  # Convert to m2 for ARTS
                    refpressure.append(xsec[0]['pressure'])
                    reftemperature.append(xsec[0]['temperature'])
                    fmin.append(xsec[0]['fmin'])
                    fmax.append(xsec[0]['fmax'])

        if not len(xsecs):
            raise RuntimeError('No matching xsecs found.')

        print(f'{len(xsec)} profiles selected.')
        fwhm, pressure_diff = xu.calc_fwhm_and_pressure_difference(results)
        popt, pcov, decision = xu.do_fit(fwhm, pressure_diff)
        xsec_data = typhon.arts.xsec.XsecRecord(
            sys.argv[2],
            popt,
            numpy.array(fmin),
            numpy.array(fmax),
            numpy.array(refpressure),
            numpy.array(reftemperature),
            xsecs)
        typhon.arts.xml.save((xsec_data,),
                             os.path.join(outdir, sys.argv[2] + '.xml'))

    elif command == 'avail':
        os.makedirs(outdir, exist_ok=True)
        xu.plot_available_xsecs(inputs, species, outdir)
    elif command == 'comparetemp':
        os.makedirs(outdir, exist_ok=True)
        bands = XSECINFO[sys.argv[2]]['bands']

        for band in bands:
            xsec_sel = sorted(xu.xsec_select_band2(xsecs, band),
                              key=lambda x: x['pressure'])
            xu.plot_compare_xsec_temp(
                copy.deepcopy(xsec_sel),
                species + f' {band[0]}-{band[1]}', outdir, diff=True)
            xu.plot_compare_xsec_temp(
                copy.deepcopy(xsec_sel),
                species + f' {band[0]}-{band[1]}', outdir, diff=False)
    elif command == 'comparetempfreq':
        os.makedirs(outdir, exist_ok=True)
        bands = XSECINFO[sys.argv[2]]['bands']
        for band in bands:
            xsec_sel = sorted(xu.xsec_select_band2(xsecs, band),
                              key=lambda x: x['pressure'])
            xsec_sel = xu.xsec_select_pressure(xsec_sel,
                                               xsec_sel[0]['pressure'])
            nf = xsec_sel[0]['nfreq']
            for diff, reftype in ((True, 'temp'), (True, 'freq'), (False, '')):
                xu.plot_compare_xsec_temp_at_freq(
                    copy.deepcopy(xsec_sel),
                    numpy.linspace(nf // 10, nf - nf // 10, num=10,
                                   endpoint=True,
                                   dtype=int),
                    species + f' {band[0]}-{band[1]}',
                    outdir, diff=diff, reftype=reftype)
    elif command == 'testarts':
        xsecdata = typhon.arts.xml.load(
            os.path.join(outdir, sys.argv[2] + '.xml'))
        print(xsecdata)
    elif command == 'scatter':
        logger.info(f'Loading results from {outfile}')
        results = xu.load_output(outfile)
        logger.info(f'Creating scatter plot and fit')
        xu.scatter_and_fit(results, species, outdir)
    elif command == 'plot':
        logger.info(f'Loading results from {outfile}')
        results = xu.load_output(outfile)
        logger.info(f'Plotting RMS and Xsecs')
        res = [p.apply_async(xu.generate_rms_and_spectrum_plots,
                             (*args, result, ioutdir))
               for args, result, ioutdir in
               zip(inputs, results, itertools.repeat(outdir))]
        [r.get() for r in res if r]
    else:
        xu.print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
