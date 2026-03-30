
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

# Defining Physical Parameters:
N, m, k, max_t = 8500, 1, 3, 10
epsilon, sigma = 1, 1
x_0, vx_0 = 2, 1
ax_0 = -k*x_0
dt = max_t/N
ax, vx, x = np.empty(N+1), np.empty(N+1), np.empty(N+1)
ax[0], vx[0], x[0] = ax_0, vx_0, x_0              # Entering the initial conditions at index 0:
omega = np.sqrt(k / m)                            # Angular frequency (w = sqrt(k/m))


for i in range(0, N):                             # Iterating over the time steps.
    ax[i+1] = -k*x[i]/m
    vx[i+1] = vx[i] + 1/2*(ax[i] + ax[i+1])*dt
    x[i+1] = x[i] + vx[i]*dt + 1/2*ax[i]*dt**2
    # print("x:","%10.6f" % x[i+1],"   ", "vx:",  # This can make the execution of this code very slow! Try not to use it!
    #       "%10.6f" % vx[i+1],"   ", "ax:",      # Use this only for data inspection to be aware of the correct calculated result.
    #       "%10.6f" % ax[i+1])


t = np.linspace(0, max_t, N + 1)
# Analytical equations for the Simple Harmonic Oscillator:
x_ana  = x_0 * np.cos(omega * t) + (vx_0 / omega) * np.sin(omega * t)
vx_ana = -x_0 * omega * np.sin(omega * t) + vx_0 * np.cos(omega * t)

# Plotting Analytical Solutions with solid lines
plt.plot(t, x_ana, label=r' $x(t)\:\:(Analytical)$', color='black', linewidth=1.5)
plt.plot(t, vx_ana, label=r'$v_x(t)\:\:(Analytical)$', color='orange', linewidth=1.5)

# Plotting Numerical Approximations with with semi-transparent dots:
plt.plot(t,  x, label=r' $x(t)\:\:(Numerical)$', color='blue', linestyle='none', marker='o', markersize=5, alpha=0.6)
plt.plot(t, vx, label=r'$v_x(t)\:\:(Numerical)$', color='red', linestyle='none', marker='o', markersize=5, alpha=0.6)
plt.plot(t, ax, label=r'$a_x(t)\:\:(Numerical)$', color='green', linestyle='none', marker='o', markersize=5, alpha=0.6)
#if you don't want to use LaTeX typography, use the text inside the single or double quote.

plt.xlabel("Time [s]")
plt.ylabel("a(t), v(t) and x(t) amplitudes")
plt.ylabel(r'a(t), v$_{\mathregular{x}}$(t) and x(t) amplitudes')
plt.title("One-Dimensional Harmonic Oscillator")
plt.legend(fancybox=True, loc='upper right', scatterpoints=1, numpoints=1)
plt.grid(True)
plt.show()
