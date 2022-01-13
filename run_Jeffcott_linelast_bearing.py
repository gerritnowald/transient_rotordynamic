# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a Jeffcott rotor in linear elastic bearings

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
mj  = 1e-2      # journal mass / kg
eps = m*5e-6    # center of mass eccentricity / m (unbalance)
cs  = 1e5       # shaft stiffness / N/m
d   = 1e-2*np.sqrt(cs*m)     # damping coefficient / Ns/m (modal damping)

cb = 1e4        # bearing stiffness / N/m

tmax = 5                             # max. time of calculation / s
fmax = 1.5*np.sqrt(cb/m)/2/np.pi     # max rotational frequency / Hz (based on natural frequency)
arot = 2*np.pi*fmax/tmax             # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# functions

def rotor_Jeffcott(t, q):
    # FB   = rd.bearing_lin_elast(q[[0,2]],cb)        # bearing forces
    FB   = np.array([ 0, 0 ])                       # bearing stiffness in C
    FU   = rd.unbalance_const_acc(t,eps,arot)       # unbalance forces
    Fvec = np.array([ FB[0], FU[0], FB[1], FU[1] ]) # external forces physical space
    fvec = np.hstack(( np.zeros(4), Minv @ Fvec ))  # external forces state space
    return A @ q + fvec - gvec

# -----------------------------------------------------------------------------
# system matrices [xj, xm, yj, ym]

M = np.diag([mj,m,mj,m])

O = np.array([[2,-1], [-1,2]])
D = np.vstack(( np.hstack((  d*O, np.zeros((2,2)) )), np.hstack(( np.zeros((2,2)),  d*O )) ))

O = np.array([[1,-1], [-1,1]])
C = np.vstack(( np.hstack(( cs*O, np.zeros((2,2)) )), np.hstack(( np.zeros((2,2)), cs*O )) ))
C[0,0] += cb    # bearing stiffness x
C[2,2] += cb    # bearing stiffness y

A, Minv = rd.state_space(M,D,C)                             # state space matrix
gvec    = g*np.hstack(( np.zeros(6), np.array([1,1]) ))     # gravity state space

# -----------------------------------------------------------------------------
# initial conditions (static equilibrium)
# q0 = [xj, xm, yj, ym, xdj, xdm, ydj, ydm]

# y0j = - (m+mj)*g/cb
# y0m = y0j - m*g/cs
# q0  = [0, 0, y0j, y0m, 0, 0, 0, 0]    # analytical solution

q0 = np.linalg.solve(A, gvec)   # only if bearing stiffness in C

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
plt.plot(res.t, res.y[3]*1e3 )
plt.title("rotor displacement")
plt.xlabel("time / s")
plt.ylabel("y / mm")
plt.grid()

# orbit
plt.subplot(122)
plt.plot(res.y[1]*1e3, res.y[3]*1e3 )
plt.title("rotor orbit")
plt.xlabel("x / mm")
plt.ylabel("y / mm")
plt.axis('equal')
plt.grid()

plt.tight_layout()
plt.show()