
#  Created on August 08, 2020
#  Author: Jailson Santana
#  E-mail: Jailson.Oliveira.Fisica@outlook.com
#  Coded in Python 3.7

'''=========================================================================================
||                                                                                        ||
||     This code integrates part of computational production developed as a test in       ||
||     the course "computational methods in physics" in my undergraduation in Physics     ||
||                                                                                        ||
||     -- Professor: Felipe Mondaini, D. Sc.                                              ||
||     -- Student: Jailson Oliveira Santana.                                              ||
||                                                                                        ||
||     Computational implementation of the Velocity Verlet algorithm for the case         ||
||     of the One-dimensional Harmonic Oscillator.                                        ||
||                                                                                        ||
========================================================================================'''



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('classic')

N, m, k, tmax = 8500, 1, 3, 10
epsilon, sigma = 1, 1
x_0, vx_0 = 2, 1
ax_0 = -k*x_0
dt = tmax/N
ax, vx, x = np.empty(N+1), np.empty(N+1), np.empty(N+1)
ax[0], vx[0], x[0] = ax_0, vx_0, x_0

# Iterating over the time steps:
for i in range(0, N):
    ax[i+1] = -k*x[i]/m
    vx[i+1] = vx[i] + 1/2*(ax[i] + ax[i+1])*dt
    x[i+1] = x[i] + vx[i]*dt + 1/2*ax[i]*dt**2
    # print("x:","%10.6f" % x[i+1],"   ", "vx:",  # This can make the execution of this code very slow! Try not to use it!
    #       "%10.6f" % vx[i+1],"   ", "ax:",      # Use this only for data inspection to be aware of the correct calculated result.
    #       "%10.6f" % ax[i+1])

plt.plot(x, '-', color='blue', label=r'$x(t)$')
plt.plot(vx, '-', color='red', label=r'$v(t)$')
plt.plot(ax, '-', color='black', label=r'$a(t)$')
plt.xlabel('$Time \:\:-\:\: t[s]$', fontsize=14)
plt.ylabel('$Approximations \:\:for \:\: \\vec{a},\: \\vec{v}\:\: and \:\: \\vec{x}$', fontsize=14)
plt.legend(fancybox=True, loc='upper right', scatterpoints=1, numpoints=1)
plt.title(r"$One-dimensional \:\: Harmonic \:\: Oscillator$", fontdict={'fontsize': 17, 'fontweight': 'normal'})
plt.grid(True)
plt.show()
