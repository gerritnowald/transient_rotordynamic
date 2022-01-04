# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a rigid rotor in linear elastic bearings

Created on Jan 04 2022

@author: Gerrit Nowald
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81

# -----------------------------------------------------------------------------
# parameters

m = 0.1        # mass of rotor / kg
epsilonR = 5e-6	# unbalance radius / m
c = 1e5

q0 = [0, 0, -m*g/c, 0]   # initial conditions [displ. x, speed x, displ. y, speed y]
fmax = 300  # max rotational frequency / Hz
tmax = 1     # max. time of calculation / s

# -----------------------------------------------------------------------------
# functions

def ODE(t, q, m, epsilonR, c, arot, g):
    # ode in state space formulation
    DX, VX, DY, VY = q
    FX = c*DX
    FY = c*DY
    return [
        VX,
        - FX/m + epsilonR*( arot*np.sin(0.5*arot*t**2) + (arot*t)**2*np.cos(0.5*arot*t**2)) ,
        VY,
        - FY/m + epsilonR*(-arot*np.cos(0.5*arot*t**2) + (arot*t)**2*np.sin(0.5*arot*t**2)) - g
        ]

# -----------------------------------------------------------------------------
# calculation

arot = 2*np.pi*fmax/tmax   # acceleration of rotor speed / rad/s**2


# N = 300    # number of result points
# res = solve_ivp(ODE, [0, tmax], q0, args=(m, epsilonR, c, arot, g),
#                 t_eval=np.linspace(0, tmax, N), rtol=1e-6, atol=1e-6 )
res = solve_ivp(ODE, [0, tmax], q0, args=(m, epsilonR, c, arot, g), 
                rtol=1e-6, atol=1e-6 )

# -----------------------------------------------------------------------------
# plot

plt.figure()

# displacement over time
plt.subplot(221)
plt.plot(res.t, res.y[2]*1e3 )
plt.title("displacement over time")
plt.xlabel("time / s")
plt.ylabel("y / mm")
plt.grid()

# phase diagram
plt.subplot(222)
plt.plot(res.y[0]*1e3, res.y[2]*1e3 )
plt.title("orbit")
plt.xlabel("x / mm")
plt.ylabel("y / mm")
plt.grid()