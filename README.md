HITRAN cross-section data for ARTS
=

Calculating input file for ARTS RFMIP calculation:

1. Calculate RMS for reference species
2. Calculate average coefficients from reference species
4. Calculate temperature fit for rfmip species [optional]
3. Calculate RMS for all rfmip species
5. Combine data for ARTS


```bash
python xsec_process_species.py -p 12 -d all_xsecs/hitran.org/data/xsec/xsc -o output-rfmip rms reference
python xsec_process_species.py -p 12 -d all_xsecs/hitran.org/data/xsec/xsc -o output-rfmip avg reference
python xsec_process_species.py -p 12 -d all_xsecs/hitran.org/data/xsec/xsc -o output-rfmip tfit rfmip
python xsec_process_species.py -p 12 -d all_xsecs/hitran.org/data/xsec/xsc -o output-rfmip rms rfmip
python xsec_process_species.py -p 12 -d all_xsecs/hitran.org/data/xsec/xsc -o output-rfmip arts rfmip
```

