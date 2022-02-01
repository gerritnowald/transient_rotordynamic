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

BB  = 3.5e-3    # journal width / m
DB  = 7e-3      # journal diameter / m
CB  = 15e-6     # bearing gap / m
eta = 1e-2      # dyn. oil viscosity / Ns/m^2

tmax = 1                   # max. time of calculation / s
fmax = 700                 # max rotational frequency / Hz
arot = 2*np.pi*fmax/tmax   # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# rotor ODE

def rotor_rigid(t, q):
    # bearing state vector
    qB = np.zeros(6, dtype=np.float64)
    qB[0:4] = q
    qB[4]   = arot*t
    # external forces
    FB = rd.bearing_journal_short(qB,BB,DB,CB,eta)      # bearing forces & torque
    FU = rd.unbalance_const_acc(t,eps,arot)             # unbalance forces
    # ode in state space formulation
    qd = np.empty(4, dtype=np.float64)
    qd[0:2] = q[-2:]
    qd[2:]  = (2*FB[:2] + FU)/m
    qd[-1] -= g
    return qd

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_rigid, [0, tmax], np.zeros(4),
                t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6, method='BDF' )
print(f"elapsed time: {time.time() - start_time} s")

# -----------------------------------------------------------------------------
#%% plot

plt.close('all')

plt.figure()

# eccentricity over time
plt.subplot(121)
plt.plot(res.t, np.sqrt(res.y[0]**2+res.y[1]**2)/CB )
plt.title("journal eccentricity")
plt.xlabel("time / s")
plt.ylabel("epsilon")
plt.ylim((0, 1))
plt.grid()

# orbit
plt.subplot(122)
plt.plot(res.y[0]/CB, res.y[1]/CB )
rd.plot_circ()
plt.title("journal orbit")
plt.xlabel("x/C")
plt.ylabel("y/C")
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()