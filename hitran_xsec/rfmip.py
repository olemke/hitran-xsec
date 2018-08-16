"""Metadata for RFMIP species."""

RFMIP_SPECIES = {
    # Alcohols, ethers and other oxygenated hydrocarbons

    # Bromocarbons, Hydrobromocarbons, and Halons
    'CBrClF2': {},  # RFMIP name: HALON1211, no air broadening, only N2
    'CBrF3': {},  # RFMIP name: HALON1301, no air broadening, only N2
    'C2Br2F4': {},  # RFMIP name: HALON2402, not in Hitran

    # Chlorocarbons and Hydrochlorocarbons
    'CCl4': {  # RFMIP Name: CARBON_TETRACHLORIDE
        'active': True,
        'reference': True,
        'arts_bands': ((700, 860),),
    },
    #          +++++ fit ok +++++, use only band 700-860
    'CH2Cl2': {},  # no air broadening, only N2
    'CH3CCl3': {},  # not available in Hitran
    'CHCl3': {},  # not available in Hitran

    # Chlorofluorocarbons (CFCs)
    'CFC-11': {  # +++++ fit ok +++++
        'active': True,
        'reference': True,
    },
    'CFC11EQ': {},  # not in Hitran
    'CFC-12': {  # +++++ fit ok +++++
        'active': True,
        'reference': True,
    },
    'CFC12EQ': {},  # not in Hitran
    'CFC-113': {  # only data for 0 torr
        'active': True,
    },
    'CFC-114': {},  # only data for 0 torr
    'CFC-115': {},  # only data for 0 torr

    # Fully Fluorinated Species
    'C2F6': {
        'arts_bands': ((680, 750), (1061, 1165), (1170, 1380),),
    },  # !!!!! bad fit !!!!! no high pressure differences available
    'C3F8': {},  # no air broadening, only N2
    'C4F10': {},  # no air broadening, only N2
    'n-C5F12': {},  # no air broadening, only N2
    'n-C6F14': {},  # no air broadening, only N2
    'C7F16': {},  # not in Hitran
    'C8F18': {},  # no air broadening, only N2
    'c-C4F8': {},  # only data for 0 Torr
    'CF4': {  # +++++ fit ok +++++
        'active': True,
        'reference': True,
    },
    'NF3': {},  # no air broadening, only N2
    'SO2F2': {},  # no air broadening, only N2

    # Halogenated Alcohols and Ethers

    # Hydrocarbons

    # Hydrochlorofluorocarbons (HCFCs)
    'HCFC-141b': {},  # only data for 0 torr
    'HCFC-142b': {},  # only data for 0 torr
    'HCFC-22': {
        'arts_bands': ((760, 860), (1060, 1210), (1275, 1380),)
    },  # !!!!! bad fit !!!!! no high pressure differences available

    # Hydrofluorocarbons (HFCs)
    'HFC-134a': {  # RFMIP name: HFC-134a, also available in Hitran under that
        #            +++++ fit ok +++++. Use band 750-1600.
        'altname': 'CFH2CF3',
        'active': True,
        'reference': True,
        'arts_bands': ((750, 1600),),
    },
    #             name, but without the 750-1600 band. This gives a better fit.
    'HFC125': {},  # not available in Hitran
    'HFC134AEQ': {},  # not available in Hitran
    'HFC-143a': {
        'arts_bands': ((580, 630), (694, 1504),)
    },  # not enough xsecs available
    'HFC-152a': {},  # only data for 0 torr
    'HFC227EA': {},  # not available in Hitran
    'HFC236FA': {},  # not available in Hitran
    'CHF3': {},  # RFMIP name: HFC-23
    'HFC245FA': {},  # not available in Hitran
    'HFC-32': {},  # !!!!! bad fit !!!!! not enough xsecs available
    'CH3CF2CH2CF3': {},  # RFMIP name: HFC365MFC, not enough xsecs available
    'HFC4310MEE': {},  # not available in Hitran

    # Iodocarbons and hydroiodocarbons

    # Nitriles, amines and other nitrogenated hydrocarbons

    # Other molecules

    # Sulfur-containing species

}
