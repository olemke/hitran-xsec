import os

import matplotlib.pyplot as plt
import typhon

arts_exe = os.path.join(os.environ["HOME"], "Hacking/arts-build/clang-debug/src/arts")

stdout = typhon.arts.run_arts(controlfile="TestHitranXsec.arts", arts=arts_exe)[1]
print(stdout)

f_grid = typhon.arts.xml.load("TestHitranXsec.f_grid.xml")
y = typhon.arts.xml.load("TestHitranXsec.y.xml")

fig, ax = plt.subplots(1, 1)
ax.plot(typhon.physics.frequency2wavenumber(f_grid) / 100, y)
ax.set_xlabel("Wavenumber")
fig.savefig("xsec.pdf")
