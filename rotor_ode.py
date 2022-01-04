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
epsR = 5e-6     # unbalance radius / m

mj = 1e-2       # journal mass / kg
cs = 1e5        # shaft stiffness / N/m

cb = 1e4        # bearing stiffness / N/m

tmax = 5        # max. time of calculation / s
fmax = 1.5*np.sqrt(cb/m)/2/np.pi     # max rotational frequency / Hz (based on natural frequency)

# -----------------------------------------------------------------------------
# functions

def bearing_lin_elast(x,y,cb):
    return [    # horizontal, vertical bearing force
        - cb*x,
        - cb*y
        ]

def unbalance_const_acc(epsR,arot,t):
    return [    # horizontal, vertical unbalance force
        epsR*( arot*np.sin(0.5*arot*t**2) + (arot*t)**2*np.cos(0.5*arot*t**2)) ,
        epsR*(-arot*np.cos(0.5*arot*t**2) + (arot*t)**2*np.sin(0.5*arot*t**2))
        ]

def rotor_rigid(t, q, m, epsR, cb, d, arot, g):
    x, y, xd, yd = q                        # states
    FB = bearing_lin_elast(x,y,cb)          # bearing forces
    FU = unbalance_const_acc(epsR,arot,t)   # unbalance forces
    return [ xd, yd,                        # ode in state space formulation
            FB[0]/m + FU[0] - d*xd ,
            FB[1]/m + FU[1] - d*yd - g
            ]

def rotor_Jeffcott(t, q, m, epsR, mj, cs, cb, d, arot, g):
    xm, ym, xj, yj, xdm, ydm, xdj, ydj = q  # states
    FB = bearing_lin_elast(xj,yj,cb)        # bearing forces
    FU = unbalance_const_acc(epsR,arot,t)   # unbalance forces
    return [ xdm, ydm, xdj, ydj,            # ode in state space formulation
            - d*(xdm-xdj)/m  - cs*(xm-xj)/m  - d*xdm + FU[0]     ,
            - d*(ydm-ydj)/m  - cs*(ym-yj)/m  - d*ydm + FU[1] - g , 
            - d*(xdj-xdm)/mj - cs*(xj-xm)/mj - d*xdj + FB[0]/mj  ,
            - d*(ydj-ydm)/mj - cs*(yj-ym)/mj - d*ydj + FB[1]/mj -g
            ]

# -----------------------------------------------------------------------------
# calculation

arot = 2*np.pi*fmax/tmax    # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)
d    = 1e-2*np.sqrt(cs/m)   # damping coefficient / Ns/m

# rigid rotor
# q0   = [0, -m*g/cb, 0, 0]    # initial conditions [displ. x, displ. y, speed x, speed y] (static equilibrium)
# res = solve_ivp(rotor_rigid, [0, tmax], q0, args=(m, epsR, cb, d, arot, g),
#                 t_eval = np.linspace(0, tmax, int(tmax*fmax*30) ),    # points of orbit at highest frequency
#                 rtol=1e-6, atol=1e-6 )

# Jeffcott rotor
y0 = - (m+mj)*g/cb
q0  = [0, y0, 0, y0, 0, 0, 0, 0]    # initial conditions [displ. x, displ. y, speed x, speed y] (static equilibrium)
res = solve_ivp(rotor_Jeffcott, [0, tmax], q0, args=(m, epsR, mj, cs, cb, d, arot, g),
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