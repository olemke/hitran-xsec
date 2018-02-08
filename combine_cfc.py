import os
import typhon.arts.xml as axml

cfcs = []
for species in (('CFC11', 'output_cfc11_full'),
                ('CFC12', 'output_cfc12_full'),
                ('HFC134a', 'output_hfc134a_full'),
                ('HCFC22', 'output_hcfc22_full'),
                ):
    cfcs.extend(axml.load(os.path.join(species[1], species[0]+'.xml')))

axml.save(cfcs, 'arts-example/CFC.xml', format='binary')

