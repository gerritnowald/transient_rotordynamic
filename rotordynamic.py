# -*- coding: utf-8 -*-
"""
Rotordynamic Module

Created on Wed Jan  5 16:30:25 2022

@author: Gerrit Nowald
"""

import numpy as np
import matplotlib.pyplot as plt

from numba import njit

# -----------------------------------------------------------------------------
# math

@njit
def cos_sin(q):
    # q = [x,y]
    denominator = np.sqrt(q[0]*q[0]+q[1]*q[1])
    angle = np.empty(2, dtype=np.float64)
    angle[0] = q[0]/denominator     # cos
    angle[1] = q[1]/denominator     # sin
    return angle

def plot_circ( R=1, C=(0,0), color='k', points=50 ):
    angle = np.linspace(0, 2*np.pi, points)
    x = C[0] + R*np.cos(angle)
    y = C[1] + R*np.sin(angle)
    plt.plot(x,y, color=color )
    plt.axis('equal')
    
def state_space(M,D,C):
    Minv = np.linalg.inv(M)
    A = np.vstack((      # state space matrix
        np.hstack(( np.zeros(np.shape(M)), np.eye(np.shape(M)[0]) )),
        np.hstack(( - Minv @ C, - Minv @ D )) ))
    return A, Minv

# -----------------------------------------------------------------------------
# rotor

def unbalance_const_acc(t,eps,arot):
    alphad = arot*t
    alpha  = 0.5*arot*t**2
    return np.array([    # horizontal, vertical unbalance force
        eps*( arot*np.cos(alpha) - alphad**2*np.sin(alpha) ) ,
        eps*( arot*np.sin(alpha) + alphad**2*np.cos(alpha) ) ])

# -----------------------------------------------------------------------------
# short bearing

@njit
def short_bearing_forces(eps,epsS,phiS):
# journal bearing forces short bearing theory
# Vrande, van de, B. L. (2001). Nonlinear dynamics of elementary rotor systems 
# with compliant plain journal bearings. Technische Universiteit Eindhoven.
# https://doi.org/10.6100/IR550147
    vs = - np.sqrt(epsS**2+(eps*(phiS-0.5))**2)             # effective squeeze speed
    cos_alpha, sin_alpha = cos_sin([epsS, eps*(0.5-phiS)])  # effective squeeze angle
    if cos_alpha < 0:
        delta = -1
    else:
        delta = 1
    A = (eps+sin_alpha)/(1+eps*sin_alpha)
    B = (eps-sin_alpha)/(1-eps*sin_alpha)
    # integrals
    I001 = (np.arccos(-delta*A) + np.arccos(-delta*B))/np.sqrt(1-eps**2)
    I023 = 1/(2*(1-eps**2)**2)*((1+2*eps**2)*I001 + 2*eps*cos_alpha*(3+(2-5*eps**2)*sin_alpha**2)/(1-eps**2*sin_alpha**2)**2)
    I113 = -2*eps*sin_alpha**3/(1-eps**2*sin_alpha**2)**2
    I203 = 1/(2*(1-eps**2))*(I001 + 2*eps*cos_alpha*(1-(2-eps**2)*sin_alpha**2)/(1-eps**2*sin_alpha**2)**2)
    # dimensionless forces
    f = np.empty(2, dtype=np.float64)
    f[0] = 2*vs*(I023*cos_alpha - I113*sin_alpha)   # fr
    f[1] = 2*vs*(I113*cos_alpha - I203*sin_alpha)   # fphi
    return f

def bearing_journal_short(qB,B,D,C,eta):
    # bearing state vector qB = [x, y, xd, yd, omj, oms]
    # kinematics
    omega0 = np.abs(qB[4]+qB[5]) + 1e-10    # reference angular velocity
    d = qB[0:2]/C                           # relative journal displacements [x, y]
    v = qB[2:4]/C/omega0                    # relative journal speeds [xd, yd]
    eps  = np.sqrt(np.sum(d**2)) + 1e-10    # journal eccentricity
    epsS = d @ v/eps                        # dimensionless squeeze speed
    phiS = np.cross(d,v)/eps**2             # dimensionless whirl speed
    cos_delta, sin_delta = d/eps            # absolute angle of widest gap
    # dimensionless bearing forces
    fb = (B/D)**2*short_bearing_forces(eps,epsS,phiS)
    # dimensional forces transformed into absolute coordinates
    F = np.empty(3, dtype=np.float64)
    F[0:2] = 0.25*D**3*B*eta/C**2*omega0*np.array([
        [ cos_delta, -sin_delta ],
        [ sin_delta,  cos_delta ]
        ]) @ fb
    # bearing torque
    F[2] = - eta*np.pi*B*D**3/C/4*(qB[4]-qB[5])/np.sqrt(1-eps**2)
    return F