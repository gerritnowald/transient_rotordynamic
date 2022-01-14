# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a Jeffcott rotor in fixed bearings

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
c   = 1e4       # shaft stiffness / N/m

d   = 2e-2*np.sqrt(c*m)     # damping coefficient / Ns/m (modal damping)

tmax = 5                         # max. time of calculation / s
fmax = 2*np.sqrt(c/m)/2/np.pi    # max rotational frequency / Hz (based on natural frequency)
arot = 2*np.pi*fmax/tmax         # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# rotor ODE

def rotor_Jeffcott(t, q):
    FU   = rd.unbalance_const_acc(t,eps,arot)       # unbalance forces
    fvec = np.hstack(( np.zeros(2), Minv @ FU ))    # external forces state space
    return A @ q + fvec - gvec

# -----------------------------------------------------------------------------
# system matrices [x, y]

M = m*np.eye(2)
D = d*np.eye(2)
C = c*np.eye(2)

A, Minv = rd.state_space(M,D,C)              # state space matrix
gvec    = g*np.hstack(( np.zeros(3), 1 ))    # gravity state space

# -----------------------------------------------------------------------------
# initial conditions (static equilibrium)
# q0 = [x, y, xd, yd]

q0 = np.linalg.solve(A, gvec)

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_Jeffcott, [0, tmax], q0,
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