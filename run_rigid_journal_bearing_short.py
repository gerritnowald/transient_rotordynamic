# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a rigid rotor in journal bearings (analytical short bearing solution)

Created on Jan 04 2022

@author: Gerrit Nowald
"""

import rotordynamic as rd

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import time

g = 9.81    # gravitational acceleration

# -----------------------------------------------------------------------------
# parameters

m   = 0.1       # mass of rotor / kg
eps = m*1e-6    # center of mass eccentricity / m (unbalance)

B = 3.5e-3      # journal width / m
D = 7e-3        # journal diameter / m
C = 15e-6       # bearing gap / m
eta = 1e-2      # dyn. oil viscosity / Ns/m^2

tmax = 1                    # max. time of calculation / s
fmax = 700                 # max rotational frequency / Hz
arot = 2*np.pi*fmax/tmax    # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# functions

def rotor_rigid(t, q):
    FB = 2*rd.bearing_journal_short(q,B,D,C,eta,arot*t)   # bearing forces
    FU = rd.unbalance_const_acc(t,eps,arot) # unbalance forces
    return np.hstack(( q[-2:],              # ode in state space formulation
        ( FB + FU)/m - np.array([0,g]) ))

# -----------------------------------------------------------------------------
# initial conditions [displ. x, displ. y, speed x, speed y]

q0  = np.zeros(4)

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_rigid, [0, tmax], q0,
                t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6, method='BDF' )
print(f"elapsed time: {time.time() - start_time} s")

# -----------------------------------------------------------------------------
# plot

plt.close('all')

plt.figure()

# displacement over time
plt.subplot(121)
plt.plot(res.t, np.sqrt(res.y[0]**2+res.y[1]**2)/C )
plt.title("journal eccentricity")
plt.xlabel("time / s")
plt.ylabel("epsilon")
plt.grid()

# phase diagram
angle = np.linspace(0,2*np.pi,100)
plt.subplot(122)
plt.plot(res.y[0]/C, res.y[1]/C )
plt.plot(np.cos(angle), np.sin(angle), color='k' )
plt.title("journal orbit")
plt.xlabel("x/C")
plt.ylabel("y/C")
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()