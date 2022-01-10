# -*- coding: utf-8 -*-
"""
Rotordynamic Module

Created on Wed Jan  5 16:30:25 2022

@author: Gerrit Nowald
"""

import numpy as np

# -----------------------------------------------------------------------------
# math

def cos_sin(q):                       # q = [x,y]
    return q/np.sqrt(np.sum(q**2))    # [cos, sin]

# -----------------------------------------------------------------------------
# rotor

def unbalance_const_acc(t,eps,arot):
    alphad = arot*t
    alpha  = 0.5*arot*t**2
    return np.array([    # horizontal, vertical unbalance force
        eps*( arot*np.cos(alpha) - alphad**2*np.sin(alpha) ) ,
        eps*( arot*np.sin(alpha) + alphad**2*np.cos(alpha) ) ])

def state_space(M,D,C):
    Minv = np.linalg.inv(M)
    A = np.vstack((      # state space matrix
        np.hstack(( np.zeros(np.shape(M)), np.eye(np.shape(M)[1]) )),
        np.hstack(( - Minv @ C, - Minv @ D )) ))
    return A, Minv

# -----------------------------------------------------------------------------
# linear elastic bearing

def bearing_lin_elast(q,cb):
    return - cb*q    # horizontal, vertical bearing force

# -----------------------------------------------------------------------------
# short bearing

def short_bearing_forces(eps,epsS,phiS):
# journal bearing forces short bearing theory
    if epsS == 0 and phiS == 0:    # rotation only
        fr   = 2*eps**2/(1-eps**2)**2
        fphi = - 0.5*np.pi*eps/(1-eps**2)**1.5
    else:                      # rotation + squeeze
        cos_alpha, sin_alpha = cos_sin(np.array([2*epsS, eps*(1-2*phiS)]))
        cos_alpha, sin_alpha = - sin_alpha, - cos_alpha     # needs to be corrected
        # integrals
        angle = np.arctan2(np.sqrt(1-eps**2),eps*sin_alpha)
        I1 = 2*eps*cos_alpha**3/(1-eps**2*cos_alpha**2)**2
        I2 = - eps*sin_alpha*(1-(2-eps**2)*cos_alpha**2)/(1-eps**2)/(1-eps**2*cos_alpha**2)**2 + angle/(1-eps**2)**1.5
        I3 = - eps*sin_alpha*(4-eps**2*(1+(2+eps**2)*cos_alpha**2))/(1-eps**2)**2/(1-eps**2*cos_alpha**2)**2 + angle/(1-eps**2)**2.5*(2+eps**2)
        # dimensionless forces
        fr   = I2*eps*(1-2*phiS) - 2*epsS*I1
        fphi = I1*eps*(1-2*phiS) + 2*epsS*(I2-I3)
        # forces fr and fphi are switched! correct:
        # fr	--> fphi
        # fphi	-->	-fr
    return np.array([fr, fphi])

# print(short_bearing_forces(0.7,1,2) )

def bearing_journal_short(q,B,D,C,eta,omega0):
    # state vector q = [x, y, xd, yd]
    offset = 1e-10      # against singularities
    omega0 = np.abs(omega0) + offset
    d = q[0:2]/C        # journal displacements [x, y]
    v = q[2:]/C/omega0  # journal speeds [xd, yd]
    eps  = np.sqrt(np.sum(d**2)) + offset
    epsS = d@v/eps
    phiS = np.cross(d,v)/eps**2
    cos_theta, sin_theta = d/eps
    # dimensional forces transformed into absolute coordinates
    return 0.25*D**3*B*eta/C**2*omega0*np.array([
        [-sin_theta, cos_theta],
        [ cos_theta, sin_theta]
        ]) @ short_bearing_forces(eps,epsS,phiS)*(B/D)**2
