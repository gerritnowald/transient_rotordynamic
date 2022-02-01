# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a Jeffcott rotor in journal bearings (analytical short bearing solution)

Created on Jan 04 2022

@author: Gerrit Nowald
"""

import rotordynamic as rd

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg    import block_diag

import time

g = 9.81    # gravitational acceleration

# -----------------------------------------------------------------------------
# parameters

m   = 0.1       # mass of rotor / kg
mj  = 1e-5      # journal mass / kg
eps = m*1e-6    # center of mass eccentricity / m (unbalance)
cs  = 1e5       # shaft stiffness / N/m
d   = 1e-2*np.sqrt(cs*m)     # damping coefficient / Ns/m (modal damping)

BB = 3.5e-3     # journal width / m
DB = 7e-3       # journal diameter / m
CB = 15e-6      # bearing gap / m
eta = 1e-2      # dyn. oil viscosity / Ns/m^2

tmax = 1                    # max. time of calculation / s
fmax = 500                  # max rotational frequency / Hz (optional)
arot = 2*np.pi*fmax/tmax    # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# rotor ODE

def rotor_Jeffcott(t, q):
    # bearing state vector
    qB = np.zeros(6, dtype=np.float64)
    qB[0:4] = q[[0,2,4,6]]
    qB[4]   = arot*t
    # external forces
    FB   = rd.bearing_journal_short(qB,BB,DB,CB,eta)        # bearing forces & torque
    FU   = rd.unbalance_const_acc(t,eps,arot)               # unbalance forces
    Fvec = np.array([ 2*FB[0], FU[0], 2*FB[1], FU[1] ])     # external forces
    # ode in state space formulation
    qd      = A @ q - gvec
    qd[4:] += Minv @ Fvec   # adding external forces
    return qd

# -----------------------------------------------------------------------------
# system matrices [xj, xm, yj, ym]

M = np.diag([mj,m,mj,m])

O = np.array([[1,-1], [-1,1]])
D = block_diag(  d*O, d*O )
C = block_diag( cs*O,cs*O )

A, Minv = rd.state_space(M,D,C)                                 # state space matrix
gvec    = g*np.hstack(( np.zeros(np.shape(M)[0]), 0,0,1,1 ))    # gravity state space

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_Jeffcott, [0, tmax], np.zeros(np.shape(A)[0]) + 1e-10,
                t_eval = np.linspace(0, tmax, int(tmax**2*arot/2/np.pi*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6, method='LSODA' )
print(f"elapsed time: {time.time() - start_time} s")

# -----------------------------------------------------------------------------
#%% plot

plt.close('all')

plt.figure()

# eccentricity over time
plt.subplot(221)
plt.plot(res.t, np.sqrt(res.y[0]**2+res.y[2]**2)/CB )
plt.title("journal eccentricity")
plt.xlabel("time / s")
plt.ylabel("epsilon")
plt.ylim((0, 1))
plt.grid()

# orbit
plt.subplot(222)
plt.plot(res.y[0]/CB, res.y[2]/CB )
rd.plot_circ()
plt.title("journal orbit")
plt.xlabel("x/C")
plt.ylabel("y/C")
plt.grid()

# horiz. displacement disc
plt.subplot(223)
plt.plot(res.t, res.y[1]*1e3, label='horiz.' )
plt.plot(res.t, res.y[3]*1e3, label='vert.' )
plt.legend()
plt.title("displacement disc")
plt.xlabel("time / s")
plt.ylabel("x, y / mm")
plt.grid()

# vert. displacement disc
plt.subplot(224)
plt.specgram(res.y[3], Fs=len(res.y[3])/max(res.t), detrend='mean',
             NFFT=512, pad_to=4096, noverlap=256 )
plt.ylim((0, arot*max(res.t)/(2*np.pi) ))
plt.title("spectogram disc")
plt.xlabel("time / s")
plt.ylabel("frequency / Hz")

plt.tight_layout()
plt.show()