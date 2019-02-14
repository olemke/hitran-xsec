"""Metadata for RFMIP species."""

XSEC_SPECIES_INFO = {
    # Alcohols, ethers and other oxygenated hydrocarbons

    # Bromocarbons, Hydrobromocarbons, and Halons
    'CBrClF2': {  # no air broadening, only N2
        'rfmip': 'halon1211_GM',
    },
    'CBrF3': {  # no air broadening, only N2
        'rfmip': 'halon1301_GM',
    },
    'CBrF2CBrF2': {
        'rfmip': 'halon2402_GM',
    },

    # Chlorocarbons and Hydrochlorocarbons
    'CCl4': {  # +++++ fit ok +++++, use only band 700-860
        'active': True,
        'arts_bands': ((700, 860),),
        'rfmip': 'carbon_tetrachloride_GM',
    },
    'CH2Cl2': {  # no air broadening, only N2
        'rfmip': 'ch2cl2_GM',
    },
    'CH3CCl3': {  # not available in Hitran
        'rfmip': 'ch3ccl3_GM',
    },
    'CHCl3': {  # not available in Hitran
        'rfmip': 'chcl3_GM',
    },

    # Chlorofluorocarbons (CFCs)
    'CFC-11': {  # +++++ fit ok +++++
        'active': True,
        'rfmip': 'cfc11_GM',
    },
    'CFC-12': {  # +++++ fit ok +++++
        'active': True,
        'rfmip': 'cfc12_GM',
    },
    'CFC-113': {  # only data for 0 torr
        'active': True,
        'rfmip': 'cfc113_GM',
    },
    'CFC-114': {  # only data for 0 torr
        'rfmip': 'cfc114_GM',
    },
    'CFC-115': {  # only data for 0 torr
        'rfmip': 'cfc115_GM',
    },

    # Fully Fluorinated Species
    'C2F6': {  # !!!!! bad fit !!!!! no high pressure differences available
        # 'arts_bands': ((1061, 1165), (1170, 1380),),
        'rfmip': 'c2f6_GM',
    },
    'C3F8': {  # no air broadening, only N2
        'rfmip': 'c3f8_GM',
    },
    'C4F10': {  # no air broadening, only N2
        'rfmip': 'c4f10_GM',
    },
    'n-C5F12': {  # no air broadening, only N2
        'rfmip': 'c5f12_GM',
    },
    'n-C6F14': {  # no air broadening, only N2
        'rfmip': 'c6f14_GM',
    },
    'C8F18': {  # no air broadening, only N2 at 0 Torr
        'rfmip': 'c8f18_GM',
    },
    'c-C4F8': {  # only data for 0 Torr
        'rfmip': 'c_c4f8_GM',
    },
    'CF4': {  # +++++ fit ok +++++
        'active': True,
        'arts_bands': ((1250, 1290),),
        'rfmip': 'cf4_GM',
    },
    'NF3': {  # no air broadening, only N2
        'rfmip': 'nf3_GM',
    },
    'SF6': {
        'rfmip': 'sf6_GM',
    },
    'SO2F2': {  # no air broadening, only N2
        'rfmip': 'so2f2_GM',
    },

    # Halogenated Alcohols and Ethers

    # Hydrocarbons

    # Hydrochlorofluorocarbons (HCFCs)
    'HCFC-141b': {  # only data for 0 torr
        'rfmip': 'hcfc141b_GM',
    },
    'HCFC-142b': {  # only data for 0 torr
        'rfmip': 'hcfc142b_GM',
    },
    'HCFC-22': {  # !!!!! bad fit !!!!! no high pressure differences available
        'arts_bands': ((760, 860), (1060, 1210), (1275, 1380),),
        'rfmip': 'hcfc22_GM',
    },

    # Hydrofluorocarbons (HFCs)
    'HFC-125': {
        'rfmip': 'hfc125_GM',
    },
    'HFC-134a': {  # +++++ fit ok +++++. Use band 750-1600.
        'altname': 'CFH2CF3',
        'active': True,
        'arts_bands': ((750, 1600),),
        'rfmip': 'hfc134a_GM',
    },
    'HFC-143a': {  # not enough xsecs available
        'arts_bands': ((580, 630), (694, 1504),),
        'rfmip': 'hfc143a_GM',
    },
    'HFC-152a': {
        'rfmip': 'hfc152a_GM',
    },
    'CF3CHFCF3': {
        'rfmip': 'hfc227ea_GM',
    },
    'CF3CH2CF3': {
        'rfmip': 'hfc236fa_GM',
    },
    'HFC-23': {
        'rfmip': 'hfc23_GM',
    },
    'CHF2CH2CF3': {  # Only one profile
        'rfmip': 'hfc245fa_GM',
    },
    'HFC-32': {  # !!!!! bad fit !!!!! not enough xsecs available
        'rfmip': 'hfc32_GM',
    },
    'CH3CF2CH2CF3': {  # Only one profile
        'rfmip': 'hfc365mfc_GM',
    },
    'CF3CHFCHFCF2CF3': {  # not available in Hitran
        'rfmip': 'hfc4310mee_GM',
    },

    # Iodocarbons and hydroiodocarbons

    # Nitriles, amines and other nitrogenated hydrocarbons

    # Other molecules
    'N2O': {
        'rfmip': 'nitrous_oxide_GM',
    },

    # Sulfur-containing species

}

SPECIES_GROUPS = {
    'reference': [
        'CCl4',
        'CF4',
        'CFC-11',
        'CFC-12',
        #  'HFC-134a',
    ],
    'rfmip-names': [
        'c2f6_GM',
        'c3f8_GM',
        'c4f10_GM',
        'c5f12_GM',
        'c6f14_GM',
        'c7f16_GM',
        'c8f18_GM',
        'c_c4f8_GM',
        'carbon_dioxide_GM',
        'carbon_tetrachloride_GM',
        'cf4_GM',
        'cfc113_GM',
        'cfc114_GM',
        'cfc115_GM',
        'cfc11_GM',
        'cfc11eq_GM',
        'cfc12_GM',
        'cfc12eq_GM',
        'ch2cl2_GM',
        'ch3ccl3_GM',
        'chcl3_GM',
        'halon1211_GM',
        'halon1301_GM',
        'halon2402_GM',
        'hcfc141b_GM',
        'hcfc142b_GM',
        'hcfc22_GM',
        'hfc125_GM',
        'hfc134a_GM',
        'hfc134aeq_GM',
        'hfc143a_GM',
        'hfc152a_GM',
        'hfc227ea_GM',
        'hfc236fa_GM',
        'hfc23_GM',
        'hfc245fa_GM',
        'hfc32_GM',
        'hfc365mfc_GM',
        'hfc4310mee_GM',
        'methane_GM',
        'methyl_bromide_GM',
        'methyl_chloride_GM',
        'nf3_GM',
        'nitrous_oxide_GM',
        'sf6_GM',
        'so2f2_GM',
    ],
}

RFMIPMAP = {v['rfmip']: k for k, v in XSEC_SPECIES_INFO.items() if
            'rfmip' in v.keys()}
SPECIES_GROUPS['rfmip'] = [RFMIPMAP[k] for k in SPECIES_GROUPS['rfmip-names'] if
                           k in RFMIPMAP.keys()]
