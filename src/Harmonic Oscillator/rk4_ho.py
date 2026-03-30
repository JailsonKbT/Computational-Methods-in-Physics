
# Created on May 7, 2020
# Author: Jailson Oliveira Santana
# Coded in Python 3.7.3
# Revised on Mar 29 2026


'''=======================================================================================
#|                                                                                       |
#|       This code is part of the computational work I carried out during my first       |
#|        undergraduate research project.                                                |
#|       -- Advisor: Felipe Mondaini, D. Sc.                                             |
#|       -- Student: Jailson Oliveira Santana                                            |
#|                                                                                       |
#|      Initially, this algorithm provides a simple implementation of an integrator      |
#|      for the harmonic oscillator problem using the Runge-Kutta 4th Order method.      |
#|      Furthermore, it can be adapted to any other problem involving the calculation    |
#|      of a numerical approximation to the analytical solution of a first-order         |
#|      ordinary differential equation.                                                  |
#|                                                                                       |
#======================================================================================'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('classic')

# Defining Physical Parameters:
m = 1                       # Mass of the body (Moment of inertia).
k = 8                       # Spring constant (Elastic medium).
max_t = 10                  # Maximum time evolution parameter.
steps = 950                 # Number of iterations.
dt = max_t / steps          # The time step used as the “iteration step”.
omega = np.sqrt(k / m)      # Angular frequency (w = sqrt(k/m))

# Initial Conditions:
xt0, yt0 = 2.0, 2.0
vxt0, vyt0 = 1.0, 1.0

vx = np.zeros(steps + 1)    # Creates a vector that will hold the velocity values (in x direction).
vy = np.zeros(steps + 1)    # Creates a vector that will hold the velocity values (in y direction).
x = np.zeros(steps + 1)     # Creates a vector that will hold the position values (in x direction).
y = np.zeros(steps + 1)     # Creates a vector that will hold the position values (in y direction).

# Entering the initial conditions at index 0:
vx[0], vy[0] = vxt0, vyt0   # Defines the initial velocity (for both x and y).
x[0], y[0] = xt0, yt0       # Defines the initial position (for both x and y).

# Applying 4th Order Runge-Kutta Algorithm for Dynamic Evolution:
for i in range(steps):
    
    # K1: Derivatives evaluated at the beginning of the partition
    k1_x  = vx[i] * dt
    k1_y  = vy[i] * dt
    k1_vx = -(k/m) * x[i] * dt
    k1_vy = -(k/m) * y[i] * dt

    # K2: Derivatives evaluated at the midpoint of the partition (using K1)
    k2_x  = (vx[i] + k1_vx/2) * dt
    k2_y  = (vy[i] + k1_vy/2) * dt
    k2_vx = -(k/m) * (x[i] + k1_x/2) * dt
    k2_vy = -(k/m) * (y[i] + k1_y/2) * dt

    # K3: Derivatives evaluated at the midpoint of the partition (using K2)
    k3_x  = (vx[i] + k2_vx/2) * dt
    k3_y  = (vy[i] + k2_vy/2) * dt
    k3_vx = -(k/m) * (x[i] + k2_x/2) * dt
    k3_vy = -(k/m) * (y[i] + k2_y/2) * dt

    # K4: Derivatives evaluated at the final of the partition (using K3)
    k4_x  = (vx[i] + k3_vx) * dt
    k4_y  = (vy[i] + k3_vy) * dt
    k4_vx = -(k/m) * (x[i] + k3_x) * dt
    k4_vy = -(k/m) * (y[i] + k3_y) * dt

    # Calculating the next steps (complete solution) using k1, k2, k3, and k4:
    x[i+1]  = x[i]  + (1/6) * (k1_x  + 2*k2_x  + 2*k3_x  + k4_x)
    y[i+1]  = y[i]  + (1/6) * (k1_y  + 2*k2_y  + 2*k3_y  + k4_y)
    vx[i+1] = vx[i] + (1/6) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
    vy[i+1] = vy[i] + (1/6) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)


# Checking the results produced using the method:
t = np.linspace(0, max_t, steps + 1)

# Analytical equations for the Simple Harmonic Oscillator:
x_ana = xt0 * np.cos(omega * t) + (vxt0 / omega) * np.sin(omega * t)
vx_ana = -xt0 * omega * np.sin(omega * t) + vxt0 * np.cos(omega * t)

# Plotting Numerical Approximations with with semi-transparent dots:
plt.plot(t, x, label=r' $\:x(t)\:\:(Numerical)$', color='blue', linestyle='none', marker='o', markersize=5, alpha=0.6)
plt.plot(t, vx, label=r'$v_x(t)\:\:(Numerical)$', color='red', linestyle='none', marker='o', markersize=5, alpha=0.6)

# Plotting Analytical Solutions with solid lines
plt.plot(t, x_ana, label=r' $\:x(t)\:\:(Analytical)$', color='black', linewidth=1.5)
plt.plot(t, vx_ana, label=r'$v_x(t)\:\:(Analytical)$', color='orange', linewidth=1.5)

plt.title('Harmonic Oscillator via RK4')
plt.xlabel('Time [s]')
plt.ylabel(r'x(t) and v$_{\mathregular{x}}$(t) amplitudes')
plt.legend(fancybox=True, loc='upper right', scatterpoints=1, numpoints=1)

plt.grid(True)
plt.show()