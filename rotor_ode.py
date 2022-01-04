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

g = 9.81    # gravitational acceleration

# -----------------------------------------------------------------------------
# parameters

m = 0.1         # mass of rotor / kg
epsilonR = 5e-6	# unbalance radius / m
c = 1e4         # bearing stiffness / N/m

tmax = 2        # max. time of calculation / s

fmax = 1.5*np.sqrt(c/m)/2/np.pi     # max rotational frequency / Hz (based on natural frequency)

# -----------------------------------------------------------------------------
# functions

def bearing(x,y,c):
    # linear elastic bearing
    return - c*x, - c*y # horizontal, vertical bearing force

def ODE(t, q, m, epsilonR, c, arot, g):
    # ode in state space formulation
    x, y, xd, yd = q        # states
    Fx, Fy = bearing(x,y,c) # bearing forces
    return [ xd, yd,
        Fx/m + epsilonR*( arot*np.sin(0.5*arot*t**2) + (arot*t)**2*np.cos(0.5*arot*t**2)) ,
        Fy/m + epsilonR*(-arot*np.cos(0.5*arot*t**2) + (arot*t)**2*np.sin(0.5*arot*t**2)) - g
        ]

# -----------------------------------------------------------------------------
# calculation

q0   = [0, -m*g/c, 0, 0]    # initial conditions [displ. x, displ. y, speed x, speed y] (static equilibrium)
arot = 2*np.pi*fmax/tmax    # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# numerical integration
res = solve_ivp(ODE, [0, tmax], q0, args=(m, epsilonR, c, arot, g),
                t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6 )

# -----------------------------------------------------------------------------
# plot

plt.figure()

# displacement over time
plt.subplot(221)
plt.plot(res.t, res.y[1]*1e3 )
plt.title("displacement over time")
plt.xlabel("time / s")
plt.ylabel("y / mm")
plt.grid()

# phase diagram
plt.subplot(222)
plt.plot(res.y[0]*1e3, res.y[1]*1e3 )
plt.title("orbit")
plt.xlabel("x / mm")
plt.ylabel("y / mm")
plt.grid()