import matplotlib.pyplot as plt
import numpy
import pickle

mytestvariable = 10
nc = 10
f = numpy.ones((100,))
c = numpy.ones((nc,)) / nc

f2 = numpy.hstack(
    (numpy.ones((nc // 2,)) * f[0], f, numpy.ones((nc // 2 - 1,)) * f[-1]))
fc = numpy.convolve(f2, c, 'valid')
fig, ax = plt.subplots()

ax.plot(fc)

plt.savefig('convtest.pdf')

print("test")
