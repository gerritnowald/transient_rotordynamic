# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a rigid rotor in linear elastic bearings

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
eps = m*5e-6    # center of mass eccentricity / m (unbalance)

cb = 1e4                    # bearing stiffness / N/m
d  = 1e-2*np.sqrt(cb*m)     # damping coefficient / Ns/m (modal damping)

tmax = 5                            # max. time of calculation / s
fmax = 1.5*np.sqrt(cb/m)/2/np.pi    # max rotational frequency / Hz (based on natural frequency)
arot = 2*np.pi*fmax/tmax            # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# functions

def rotor_rigid(t, q):
    FB = rd.bearing_lin_elast(q[0:2],cb)    # bearing forces
    FU = rd.unbalance_const_acc(t,eps,arot) # unbalance forces
    return np.hstack(( q[-2:],              # ode in state space formulation
        (-d*q[-2:] + FB + FU)/m - np.array([0,g]) ))

# -----------------------------------------------------------------------------
# initial conditions (static equilibrium)

q0  = [0, -m*g/cb, 0, 0]    # [displ. x, displ. y, speed x, speed y]

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_rigid, [0, tmax], q0,
                t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6 )
print(f"elapsed time: {time.time() - start_time} s")

# -----------------------------------------------------------------------------
# plot

plt.close('all')

plt.figure()

# displacement over time
plt.subplot(121)
plt.plot(res.t, res.y[1]*1e3 )
plt.title("displacement")
plt.xlabel("time / s")
plt.ylabel("y / mm")
plt.grid()

# orbit
plt.subplot(122)
plt.plot(res.y[0]*1e3, res.y[1]*1e3 )
plt.title("orbit")
plt.xlabel("x / mm")
plt.ylabel("y / mm")
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()