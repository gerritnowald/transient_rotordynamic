# -*- coding: utf-8 -*-
"""
run-up with constant rotational acceleration 
of a Jeffcott rotor in floating ring journal bearings
(analytical short bearing solution)

Created on Jan 25 2022

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

m   = 3         # mass of rotor / kg
mj  = 1e-2      # journal mass / kg
eps = m*1e-5    # center of mass eccentricity / m (unbalance)
cs  = 1e7       # shaft stiffness / N/m
d   = 1e-2*np.sqrt(cs*m)     # shaft damping coefficient / Ns/m (modal damping)

BBi = 20e-3     # inner journal width / m
DBi = 25e-3     # inner journal diameter / m
CBi = 50e-6     # inner bearing gap / m
etai = 1e-2     # inner dyn. oil viscosity / Ns/m^2

BBo = 20e-3     # outer journal width / m
DBo = 40e-3     # outer journal diameter / m
CBo = 70e-6     # outer bearing gap / m
etao = 2e-2     # outer dyn. oil viscosity / Ns/m^2

mf  = 80e-3     # floating ring mass / kg
jf  = 20e-6     # floating ring moment of intertia / kgm^2

tmax = 10           # max. time of calculation / s
arot = 2*np.pi*200  # acceleration of rotor speed / rad/s**2 (reach fmax in tmax)

# -----------------------------------------------------------------------------
# rotor ODE

def rotor_Jeffcott(t, q):
	# bearing state vectors
    qBi  = np.hstack(( q[0]-q[4], q[2]-q[5], q[7]-q[11], q[9]-q[12], arot*t, q[6] ))    # bearing state vector
    qBo  = np.hstack(( q[4], q[5], q[11], q[12], q[6], 0 )) # bearing state vector
	# external forces
	FBi  = rd.bearing_journal_short(qBi,BBi,DBi,CBi,etai)   # inner bearing forces & torque
    FBo  = rd.bearing_journal_short(qBo,BBo,DBo,CBo,etao)   # outer bearing forces & torque
    FU   = rd.unbalance_const_acc(t,eps,arot)               # unbalance forces
	# ode in state space formulation
    Fvec = np.array([ 2*FBi[0], FU[0], 2*FBi[1], FU[1], 2*(FBo[0]-FBi[0]), 2*(FBo[1]-FBi[1]), 2*(FBo[2]-FBi[2]) ])   # external forces physical space
    fvec = np.hstack(( np.zeros(7), Minv @ Fvec ))          # external forces state space
    return A @ q + fvec - gvec

# -----------------------------------------------------------------------------
# system matrices [xj, xm, yj, ym, xf, yf, omf]

M = np.diag([mj,m,mj,m,mf,mf,jf])

O = np.array([[1,-1], [-1,1]])
D = block_diag(  d*O,  d*O, np.zeros((3,3)) )
C = block_diag( cs*O, cs*O, np.zeros((3,3)) )

A, Minv = rd.state_space(M,D,C)                                       # state space matrix
gvec    = g*np.hstack(( np.zeros(np.shape(M)[0]), 0,0,1,1,0,1,0 ))    # gravity state space

# -----------------------------------------------------------------------------
# numerical integration

start_time = time.time()
res = solve_ivp(rotor_Jeffcott, [0, tmax], np.zeros(np.shape(A)[0]) + 1e-10,
				t_eval = np.linspace(0, tmax, int(tmax**2*arot/2/np.pi*30) ),    # points of orbit at highest frequency
                rtol=1e-6, atol=1e-6, method='BDF' )
print(f"elapsed time: {time.time() - start_time} s")

# -----------------------------------------------------------------------------
#%% plot

plt.close('all')

plt.figure()

# deflection disc
plt.subplot(221)
plt.plot(res.t, (res.y[1]**2+res.y[3]**2)*1e3 )
plt.legend()
plt.title("deflection disc")
plt.xlabel("time / s")
plt.ylabel("r / mm")
plt.grid()

# spectogram
plt.subplot(222)
plt.specgram(res.y[3], Fs=len(res.y[3])/max(res.t), detrend='mean',
              NFFT=512, pad_to=4096, noverlap=256 )
plt.ylim((0, arot*max(res.t)/(2*np.pi) ))
plt.title("spectogram disc")
plt.xlabel("time / s")
plt.ylabel("frequency / Hz")

# eccentricity over time
plt.subplot(223)
plt.plot(res.t, np.sqrt((res.y[0]-res.y[4])**2+(res.y[2]-res.y[5])**2)/CBi, label='inner' )
plt.plot(res.t, np.sqrt(           res.y[4]**2+           res.y[5]**2)/CBo, label='outer' )
plt.legend()
plt.title("journal eccentricity")
plt.xlabel("time / s")
plt.ylabel("epsilon")
plt.ylim((0, 1))
plt.grid()

# relative floating ring speed
plt.subplot(224)
plt.plot(res.t, res.y[6]/arot/res.t )
plt.title("relative floating ring speed")
plt.xlabel("time / s")
plt.ylabel("omf/omj")
plt.grid()

plt.tight_layout()
plt.show()