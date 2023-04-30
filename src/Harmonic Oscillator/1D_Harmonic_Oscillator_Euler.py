
import warnings
import numpy as np
import cmath as cplx
import matplotlib.pyplot as plt
from matplotlib import style
style.use('classic')
warnings.filterwarnings('ignore')


'''============================================================================================
||                                                                                            ||
||       This code integrates part of computational production developed as a test in         ||
||       the course "computational methods in physics" in my undergraduation in Physics       ||
||                                                                                            ||
||       -- Professor: Felipe Mondaini, Dsc.                                                  ||
||       -- Student:   Jailson Oliveira Santana                                               ||
||                                                                                            ||
||       Numerical simulation of a classic damped harmonic oscillator using Euler's           ||
||       numerical method. This simulation includes numerical approximations for the          ||
||       Position, Velocity and Acceleration.                                                 ||
||                                                                                            ||
============================================================================================='''


# Setting the global parameters:
m, k, gamma = 1, 8, 1.05                                            # Body mass (inertia element). | # Spring Elastic constant. | # Drag coefficient for the block in the air. 
x_0, v_0 = 10, 0                                                    # Block starting position and velocity ----> x(t0) = x_0 = 10cm | V(t0) = X'(t0) = V_0 which in given in [cm/s].
a_0 = (k * x_0)/m                                                   # Block acceleration in the position where it is immediately released.
omega_0 = np.sqrt(k/m)                                              # Definition of Omega_0 to save space in analytical definitions.
steps, t_max = 2000, 200                                           # Number of iterations over time. | # Maximum time of the t-axis.                                        
dt = t_max/steps                                                    # The time step used for iterations over time.
E_0 = (1/2) * k * x_0**2 + (1/2)*m*v_0**2                           # Initial Mechanical energy allocated to the system.
U_0 = (1/2) * k * x_0**2                                            # Potential elastic energy.
K_0 = (1/2) * m * v_0**2                                            # Initial Kinnect Energy
A_x0, A_v0, A_a0 = x_0, v_0, a_0                                    # Setting the initial value for the position, velocity and acceleration amplitudes.
impr = np.empty(t_max)                                              # A vector that I needed to create to print the computed values on the termnial.
t = np.linspace(0, t_max, steps)                                    # Vector for time values.
A_x, A_v, A_a = np.empty(steps), np.empty(steps), np.empty(steps)   # A vector that will receive the position, velocity and acceleration amplitude values in time.
E, U, K = np.empty(steps), np.empty(steps), np.empty(steps)         # A vector that will receive the mechanical, potential elastic and Kinnect energy values.
a, v, x = np.empty(steps), np.empty(steps), np.empty(steps)         # The vectors that will receive the acceleration, velocity and position values.
Critical_dampimg = None
Overdamed = None
Underdamped = None


# Initial Conditions:
v[0], x[0], a[0] = v_0, x_0, a_0
E[0], U[0], K[0] = E_0, U_0, K_0
A_x[0], A_v[0], A_a[0] = x_0, v_0, a_0                              # Defining the amplitude of the position, velocity and acceleration at time t=0s.


# Damping type classification:
if (gamma**2 - 4*m*k == 0):
    Critical_dampimg = True
    print("Critical Damping !")
elif (gamma**2 - 4*m*k != 0):
    critical_dampimg = False

if (gamma**2 - 4*m*k > 0):
    Overdamped = True
    print("Overdamping !")
elif (gamma**2 <= 4*m*k):
    Overdamped = False

if (gamma**2 - 4*m*k < 0):
    Underdamped = True
    print("Underdamping !")
elif (gamma**2 >= 4*m*k):
    Underdamped = False


# Euler's numeric method:
def Euler(a,v,x,t):
    for i in range(0, steps-1):
        a[i] = -k*x[i] - gamma*v[i]/m
        v[i+1] = v[i] + a[i]*dt
        x[i+1] = x[i] + v[i]*dt
        U[i+1] = (1/2) * k*x[i]**2
        K[i+1] = (1/2) * m*v[i]**2
        E[i+1] = (1/2) * k*x[i]**2 + (1/2)*m*v[i]**2
    return a,v,x,U,K,E
a_aux_euler, v_aux_euler, x_aux_euler, U_aux_euler, K_aux_euler, E_aux_euler = Euler(a,v,x,t)


# Setting the Analyticals expressions for the one-dimensional harmonic oscillator:
if (Critical_dampimg == True):
    A_a = -2*x_0*np.exp((-omega_0*t))*omega_0+(t*x_0+x_0) * np.exp((-omega_0*t))*omega_0**2
    A_v = x_0*np.exp((-omega_0*t))-(x_0*t+x_0)*np.exp((-omega_0*t))*omega_0
    A_x = (x_0*t + x_0)*np.exp((-omega_0*t))
elif (Overdamped == True):
    A_a = -k*x-gamma*v
    A_v = (-k*x_0*(np.exp((1/2)*(gamma-np.sqrt(gamma**2-4*omega_0**2))*t)/((1/2)*gamma-(1/2)*np.sqrt(gamma**2-4*omega_0**2))+np.exp((1/2)*(gamma+np.sqrt(gamma**2-4*omega_0**2))*t)/((1/2)*gamma+(1/2)*np.sqrt(gamma**2-4*omega_0**2))) + k*x_0*(1/((1/2)*gamma-(1/2)*np.sqrt(gamma**2-4*omega_0**2))+1/((1/2)*gamma+(1/2)*np.sqrt(gamma**2-4*omega_0**2))))*np.exp(-gamma*t)
    A_x = (x_0*(m*gamma+np.sqrt(gamma**2*m**2-4*k*m)))*np.exp(-(m*gamma-np.sqrt(gamma**2*m**2-4*k*m))*t/(2*m))/(2*np.sqrt(gamma**2*m**2-4*k*m)) - (x_0*(m*gamma-np.sqrt(gamma**2*m**2-4*k*m)))*np.exp(-(m*gamma+np.sqrt(gamma**2*m**2-4*k*m))*t/(2*m))/(2*np.sqrt(gamma**2*m**2-4*k*m))
elif (Underdamped == True):    
    A_a = -k*x-gamma*v
    A_v = -x_0*k*m*(np.exp((m*gamma+cplx.sqrt(m*(gamma**2*m-4*k)))*t/(2*m))-np.exp((m*gamma-cplx.sqrt(m*(gamma**2*m-4*k)))*t/(2*m)))*np.exp(-gamma*t)/cplx.sqrt(m*(gamma**2*m-4*k))
    A_x = x_0*(m*gamma+cplx.sqrt(gamma**2*m**2-4*k*m))*np.exp(-(m*gamma-cplx.sqrt(gamma**2*m**2-4*k*m))*t/(2*m))/(2*cplx.sqrt(gamma**2*m**2-4*k*m))-(m*gamma-cplx.sqrt(gamma**2*m**2-4*k*m))*x_0*np.exp(-(m*gamma+cplx.sqrt(gamma**2*m**2-4*k*m))*t/(2*m))/(2*cplx.sqrt(gamma**2*m**2-4*k*m))
else:
    print("ERROR!")


# Setting the envelope of the underdamped oscillation:
Env_Sup_X = max(x) * np.exp(-gamma/2*t)
Env_Inf_X = -max(x) * np.exp(-gamma/2*t)
Env_Sup_V = (abs(min(v))+2) * np.exp(-gamma/2*t)
Env_Inf_V = (min(v)-2) * np.exp(-gamma/2*t)


# Generating the data outputs:
Euler_Pos = open("/Users/jailsonoliveira/Desktop/Euler_Pos.dat", "w+", encoding="utf8")
Euler_Vel = open("/Users/jailsonoliveira/Desktop/Euler_Vel.dat", "w+", encoding="utf8")
Euler_Acc = open("/Users/jailsonoliveira/Desktop/Euler_Acc.dat", "w+", encoding="utf8")
Euler_E = open("/Users/jailsonoliveira/Desktop/Euler_Energy.dat", "w+", encoding="utf8")
def output(a,v,x,E,t):
    for i in np.arange(t_max):
        Euler_Pos.write("%6.4f" "," "%6.4f\n" % (x[i+1], t[i]))
        Euler_Vel.write("%6.4f" "," "%6.4f\n" % (v[i+1], t[i]))
        Euler_Acc.write("%6.4f" "," "%6.4f\n" % (a[i], t[i]))
        Euler_E.write("%6.4f" "," "%6.4f\n" % (E[i+1], t[i]))
    return Euler_Pos, Euler_Vel, Euler_Acc, Euler_E
out_pos, out_vel, out_acc, out_E = output(a,v,x,E,t)
Euler_Pos.close()
Euler_Vel.close()
Euler_Acc.close()
Euler_E.close()


# Generating the plots for the Positions, Speeds and Accelerations, each separately:
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = False

# Acceleration plot:
plt.suptitle(r"$Acceleration\:using\:Euler\:(dt={0:.2g})$".format(dt), fontsize = 18, fontweight = "normal")
plt.plot(A_a, '-', label = r"$Analytical\:\:-\:\vec{a}_{Analyt.}(t)$", color = 'black', linewidth = 1.1)
plt.plot(a, '-', label = r"$Numerical\:-\:\vec{a}_{Numeric.}(t)$", color = 'green', linewidth = 1.1)
plt.xlabel(r"$Time\:-\:[\:s\:]$", fontsize = 18)
plt.ylabel(r"$\vec{a}_{Numeric.}(t)\:-\:\vec{a}_{Analyt.}(t)$", fontsize=18)
plt.xlim(-1, t_max)
plt.ylim(min(a)-5, max(a)+5)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/Comparison_Analyt_Numeric_Acc.png', transparent=True)
plt.show()

# Velocity plot:
plt.suptitle(r"$Velocity\:using\:Euler\:(dt={0:.2g})$".format(dt), fontsize = 18, fontweight = "normal")
plt.plot(A_v, '-', label = r"$Analytical\:\:-\:\vec{v}_{Analyt.}(t)$", color = 'black', linewidth = 1.1)
plt.plot(v, '-', label = r"$Numerical\:-\:\vec{v}_{Numeric.}(t)$", color = 'red', linewidth = 1.1)
plt.xlabel(r"$Time\:-\:[\:s\:]$", fontsize = 18)
plt.ylabel(r"$\vec{v}_{Numeric.}(t)\:-\:\vec{v}_{Analyt.}(t)$", fontsize = 18)
plt.xlim(-1, t_max)
plt.ylim(min(v)-5, max(v)+5)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/Comparison_Analyt_Numeric_Vel.png', transparent=True)
plt.show()

# Position plot:
plt.suptitle(r"$Position\:using\:Euler\:(dt={0:.2g})$".format(dt), fontsize = 18, fontweight = "normal")
plt.plot(A_x, '-', label = r"$Analytical\:\:-\:\vec{x}_{Analyt.}(t)$", color = 'black', linewidth = 1.1)
plt.plot(x_aux_euler, '-', label = r"$Numerical\:-\:\vec{x}_{Numeric.}(t)$", color = 'blue', linewidth = 1.1)
plt.xlabel(r"$Time\:-\:[\:s\:]$", fontsize = 18)
plt.ylabel(r"$\vec{x}_{Numeric.}(t)\:-\:\vec{x}_{Analyt.}(t)$", fontsize = 18)
plt.xlim(-1, t_max)
plt.ylim(min(x)-5, max(x)+5)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/Comparison_Analyt_Numeric_Pos.png', transparent=True)
plt.show()

# A,V and X overlay plot:
plt.figure(figsize=(9, 6))
plt.suptitle(r"$Position,Velocity\:and\:Acceleration\:using\:Euler\:(dt={0:.2g})$".format(dt), fontsize=18, fontweight="normal")
plt.plot(v, '-', label=r"$Velocity$", color='red', linewidth=1.1)
plt.plot(x, '-', label=r"$Position$", color='blue', linewidth=1.1)
plt.plot(a, '-', color='green', label=r"$Acceleration$", linewidth=1.1)
plt.xlabel(r"$Time\:-\:[\:s\:]$", fontsize=18)
plt.ylabel(r"$\vec{a}(t)\:-\:\vec{v}(t)\:-\:\vec{x}(t)$", fontsize=18)
plt.xlim(-1, t_max)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/Pos_Vel_Acc_Numeric.png', transparent=True)
plt.show()


# Energy decay and enclosure of the energetic relation:
plt.figure(figsize=(9, 6))
plt.suptitle(r'$The\:Energy\:Balance\:(\:U(t)\:,\:K(t)\:\:and\:\:E_{{{}}}(t)\:-\:\:dt=%0.2g\:)$'.format('mec.',dt) % (dt), fontsize = 18, fontweight = "normal")
plt.plot(E, '-', label=r"$E.mec.(t)$", linewidth=1.4)
plt.plot(U, '-', label=r"$U(t)$", linewidth=1.4, color = 'red')
plt.plot(K, '-', label=r"$K(t)$", linewidth=1.4, color = 'black')
plt.xlim(0, 100)
plt.ylim(0, max(E)+10)
plt.xlabel(r"$Time\:-\:[\:s\:]$", fontsize = 18)
plt.ylabel(r"$Energy(t)\:-\:[\:$J$\:]$", fontsize = 18)
plt.legend()
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/E_Time_Decay.png', transparent=True)
plt.show()


# Testing multiplot with two figures (Position and Velocity):
# Position.
plt.figure(figsize=(7, 7))
plt.subplot(2, 1, 1)
if (Underdamped == True):
    plt.plot(Env_Sup_X, color = 'black', label = r"$\vec{x}(t)_{Envelope}$")
    plt.plot(Env_Inf_X, color = 'black')
    plt.plot(A_x, color = 'black', label = r"$\vec{x}(t)_{Analyt.}$")
else:
    plt.plot(A_x, color = 'black', label = r"$\vec{x}(t)_{Analyt.}$")
plt.plot(x, '-', label=r"$x(t)_{Numeric}$", color='red', linewidth=1.2)
plt.suptitle(r"$Amplitude\:Envelope\:for\:\vec{v}(t)\:and\:\:\vec{x}(t)$", fontsize=17, fontweight="normal")
plt.ylabel(r"$Position\:-\:\vec{x}(t)$", fontsize=18)
plt.xlim(-10, steps/4)
plt.ylim(min(x)-5, max(x)+5)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)

# Velocity.
plt.subplot(2, 1, 2)
if (Underdamped == True):
    plt.plot(Env_Sup_V, color = 'black', label = r"$\vec{v}(t)_{Envelope}$")
    plt.plot(Env_Inf_V, color = 'black')
    plt.plot(A_v, color = 'black', label = r"$\vec{v}(t)_{Analyt.}$")
else:
    plt.plot(A_v, color = 'black', label = r"$\vec{v}(t)_{Analyt.}$")
plt.plot(v, '-', label=r"$v(t)_{Numeric}$", color='blue', linewidth=1.2)
plt.xlabel(r"$Time\:-\:t$", fontsize = 18)
plt.ylabel(r"$Velocity\:-\:\vec{v}(t)$", fontsize = 18)
plt.xlim(-10, steps/4)
plt.ylim(min(v)-5, abs(min(v))+5)
plt.legend(fancybox=True, loc='upper right')
plt.grid(True)
plt.savefig('/Users/jailsonoliveira/Desktop/Multiplot_x_v.png', transparent=True)
plt.show()
