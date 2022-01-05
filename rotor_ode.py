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

import time

g = 9.81    # gravitational acceleration

# -----------------------------------------------------------------------------
# parameters

m   = 0.1       # mass of rotor / kg
mj  = 1e-2      # journal mass / kg
eps = m*5e-6    # center of mass eccentricity / m (unbalance)
cs  = 1e5       # shaft stiffness / N/m
D   = 1e-2      # modal damping

cb = 1e4        # bearing stiffness / N/m

tmax = 5        # max. time of calculation / s
fmax = 1.5*np.sqrt(cb/m)/2/np.pi     # max rotational frequency / Hz (based on natural frequency)

# -----------------------------------------------------------------------------
# functions

def state_space(M,D,C):
    Minv = np.linalg.inv(M)
    A = np.vstack((      # state space matrix
        np.hstack(( np.zeros(np.shape(M)), np.eye(np.shape(M)[1]) )),
        np.hstack(( - Minv @ C, - Minv @ D ))
        ))
    return A, Minv

def bearing_lin_elast(q,cb):
    return - cb*q    # horizontal, vertical bearing force

def unbalance_const_acc(t,eps,arot):
    return np.array([    # horizontal, vertical unbalance force
        eps*( arot*np.sin(0.5*arot*t**2) + (arot*t)**2*np.cos(0.5*arot*t**2)) ,
        eps*(-arot*np.cos(0.5*arot*t**2) + (arot*t)**2*np.sin(0.5*arot*t**2))
        ])

def rotor_rigid(t, q, m, eps, d, arot, g, cb):
    FB = bearing_lin_elast(q[0:2],cb)       # bearing forces
    FU = unbalance_const_acc(t,eps,arot)    # unbalance forces
    return np.hstack(( q[-2:],              # ode in state space formulation
                      (-d*q[-2:] + FB + FU)/m - np.array([0,g]) 
                      ))

def rotor_Jeffcott(t, q, A, Minv, eps, arot, cb, gvec):
    FB = bearing_lin_elast(q[[0,2]],cb)             # bearing forces
    FU = unbalance_const_acc(t,eps,arot)            # unbalance forces
    Fvec = np.array([ FB[0],FU[0],FB[1],FU[1] ])    # external forces
    fvec = np.hstack(( np.zeros(4), Minv @ Fvec ))
    return A @ q + fvec - gvec

# -----------------------------------------------------------------------------
# calculation

arot = 2*np.pi*fmax/tmax    # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)
d    = D*np.sqrt(cs*m)      # damping coefficient / Ns/m

# -----------------------------------------------------------------------------
# rigid rotor

# q0  = [0, -m*g/cb, 0, 0]    # initial conditions [displ. x, displ. y, speed x, speed y] (static equilibrium)

# # numerical integration
# start_time = time.time()
# res = solve_ivp(rotor_rigid, [0, tmax], q0, args=(m, eps, d, arot, g, cb),
#                 t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
#                 rtol=1e-6, atol=1e-6 )
# print("--- %s seconds ---" % (time.time() - start_time))

# # plot
# plt.figure()

# # displacement over time
# plt.subplot(221)
# plt.plot(res.t, res.y[1]*1e3 )
# plt.title("displacement over time")
# plt.xlabel("time / s")
# plt.ylabel("y / mm")
# plt.grid()

# # phase diagram
# plt.subplot(222)
# plt.plot(res.y[0]*1e3, res.y[1]*1e3 )
# plt.title("orbit")
# plt.xlabel("x / mm")
# plt.ylabel("y / mm")
# plt.grid()

# -----------------------------------------------------------------------------
# Jeffcott rotor

# system matrices [xj, xm, yj, ym]
M = np.diag([mj,m,mj,m])

O = np.array([[2,-1], [-1,2]])
D = np.vstack(( np.hstack((  d*O, np.zeros((2,2)) )), np.hstack(( np.zeros((2,2)),  d*O )) ))

O = np.array([[1,-1], [-1,1]])
C = np.vstack(( np.hstack(( cs*O, np.zeros((2,2)) )), np.hstack(( np.zeros((2,2)), cs*O )) ))

A, Minv = state_space(M,D,C)

gvec = g*np.hstack(( np.zeros(6), np.array([1,1]) ))    # gravity

# initial conditions (static equilibrium)
y0j = - (m+mj)*g/cb
y0m = y0j - m*g/cs
q0  = [0, 0, y0j, y0m, 0, 0, 0, 0]    # [xj, xm, yj, ym, xdj, xdm, ydj, ydm]

# numerical integration
start_time = time.time()
res = solve_ivp(rotor_Jeffcott, [0, tmax], q0, args=(A, Minv, eps, arot, cb, gvec),
                t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6 )
print("--- %s seconds ---" % (time.time() - start_time))

# plot
plt.figure()

# displacement over time
plt.subplot(221)
plt.plot(res.t, res.y[3]*1e3 )
plt.title("displacement over time")
plt.xlabel("time / s")
plt.ylabel("y / mm")
plt.grid()

# phase diagram
plt.subplot(222)
plt.plot(res.y[1]*1e3, res.y[3]*1e3 )
plt.title("orbit")
plt.xlabel("x / mm")
plt.ylabel("y / mm")
plt.grid()