
# Created on May 7, 2020
# Author: Jailson Oliveira Santana
# Coded in Python 3.7.3
# Revised on Mar 29 2026


#'''======================================================================================
#|                                                                                       |
#|       This code is part of the computational work I carried out during my first       |
#|        undergraduate research project.                                                |
#|       -- Advisor: Felipe Mondaini, D. Sc.                                             |
#|       -- Student: Jailson Oliveira Santana                                            |
#|                                                                                       |
#|                                                                                       |
#|       The problem of the damped harmonic oscillator is solved numerically using       |
#|       Euler’s method, determining the                                                 |
#|       classical phase space for four damping coefficients.                            |
#|                                                                                       |
#======================================================================================'''


import numpy as np
import matplotlib.pyplot as plt
from pylab import legend, grid

# DEFINING PHYSICAL PARAMETERS:
m = 1                                               # Mass of the body (Moment of inertia).
k = 8                                               # Spring constant (Elastic medium).
xt0 = 10                                            # Initial position of the block ----> x(t0) = xt0 = 10cm.
vt0 = 0                                             # Initial velocity of the block ----> V(t0) = X'(t0) = X0' = V0.
steps = 850                                         # Number of iterations.

# Function that generates the phase space for different gamma values:
def PhaseSpaceGen(gamma, lcolor, filename, A=1/2, B=-1, tmax=999, iterations=100000):
    
    dt = tmax / iterations          # The time step used as the “iteration step”.
    v = np.empty(steps + 1)         # Creates a vector that will hold the velocity values.
    x = np.empty(steps + 1)         # Creates a vector that will hold the position values.
    momentum = np.empty(steps + 1)  # Creates a vector that will hold the momentum values.
    
    # Initial Conditions:
    v[0] = vt0                      # Defines the initial velocity.
    x[0] = xt0                      # Defines the initial position.
    momentum[0] = m * vt0           # Defines the initial momentum.
    
    # Applying Euler's Algorithm for Dynamic Evolution:
    for i in range(0, steps):
        v[i + 1] = (-k / m) * dt * x[i] + v[i] - v[i] * dt * gamma / m
        x[i + 1] = dt * v[i] + x[i]
        momentum[i + 1] = m * v[i]

    # Configuring the vectors for the phase field (QuiverPlot):
    x2, y = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-30, 30, 30))
    '''It represents the phase space field based on the decomposition of the
        second-order ODE into a system of two first-order ODEs.'''
    U = A * y
    V = B * x2

    # Plot for each case in phase space:
    plt.figure(figsize=(12, 6))
    plt.ylim(-30, 30)
    plt.xlim(-15, 15)
    plt.title(f'$Phase\:Space\:for\:Different\:Drag\:Coefficients\:-\:(\gamma = {gamma})$', fontsize=14)

    plt.quiver(x2, y, U, V, color='black')
    plt.plot(x, momentum, label=f'$\gamma = {gamma}$', linewidth=1.8, color=lcolor)
    plt.xlabel('$X(t)$', fontsize=14)
    plt.ylabel('$V(t)$', fontsize=14)
    legend(fancybox=True, loc='upper left')
    grid(True)
    plt.savefig(filename, transparent=False)
    plt.show()
    
    return x, v

# Running the function for each case using a simple loop:

# Lists of the parameters that vary in each graph:
gammas = [0.1, 1.05, 2.7, 6.0]
lcolors = ['blue', 'red', 'green', 'purple']
files = ['Phase_Space_1.pdf', 'Phase_Space_2.pdf', 'Phase_Space_3.pdf', 'Phase_Space_4.pdf']

# Lists that will hold the X and V results for each gamma.
x_values = []
v_values = []

# Loop that will run 4 times, once for each gamma
for i in range(4):
    # Call the PhaseSpaceGen function, calculating for each gamma.
    # Then store each x, v, and momentum in the final list.
    x_calculated, v_calculated = PhaseSpaceGen(gammas[i], lcolors[i], files[i])
    
    x_values.append(x_calculated)
    v_values.append(v_calculated)


### Multiplot Combining all results:
fig = plt.figure(figsize=(9, 6))
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x_values[0], v_values[0], label=f'$\gamma = {gammas[0]}$', linewidth=1.8, color=lcolors[0])
ax1.plot(x_values[1], v_values[1], label=f'$\gamma = {gammas[1]}$', linewidth=1.8, color=lcolors[1])
ax1.plot(x_values[2], v_values[2], label=f'$\gamma = {gammas[2]}$', linewidth=1.8, color=lcolors[2])
ax1.plot(x_values[3], v_values[3], label=f'$\gamma = {gammas[3]}$', linewidth=1.8, color=lcolors[3])

plt.title('$Phase\:Space\:for\:Different\:Gamma\:Values$', fontsize=14)
plt.xlim(-15.6, 15.6)
plt.ylim(-33, 33)
plt.xlabel('$X(t)$', fontsize=14)
plt.ylabel('$V(t)$', fontsize=14)
legend(fancybox=True, loc='upper left')
grid(True)
plt.savefig('PhaseSpace_All.pdf', transparent=False)
plt.show()