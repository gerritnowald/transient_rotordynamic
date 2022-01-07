# -*- coding: utf-8 -*-
"""
Rotordynamic Module

Created on Wed Jan  5 16:30:25 2022

@author: Gerrit Nowald
"""

import numpy as np

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

def kappa(X,Y):
    if X == 0:    # singularity
        if Y>0:
            tan_kappa = 1e16
        else:
            tan_kappa = -1e16
    else:
        tan_kappa = Y/X
    cos_kappa = (-1)**(X>0)*1/np.sqrt(1+tan_kappa**2)
    sin_kappa = (-1)**(X>0)*tan_kappa/np.sqrt(1+tan_kappa**2)
    # kappa = np.arctan2(Y,X)
    # sin_kappa = - np.sin(kappa)
    # cos_kappa = - np.cos(kappa)
    if sin_kappa/(1+cos_kappa) < 0:
        piS = np.pi
    else:
        piS = 0   
    return sin_kappa, cos_kappa, piS

def  journal_bearing_short(eps,epsS,phiS):
# journal bearing forces short bearing theory
    if epsS == 0 and phiS == 0:    # case1: rotation only
        fr   = 2*eps**2/(1-eps**2)**2
        fphi = - 0.5*np.pi*eps/(1-eps**2)**1.5
    else:                      # case2: rotation + squeeze
        sin_kappa, cos_kappa, piS = kappa(eps*(1-2*phiS),2*epsS)
        # integrals
        I1 = 2*eps*cos_kappa**3/(1-eps**2*cos_kappa**2)**2
        I2 = - eps*sin_kappa*(1-(2-eps**2)*cos_kappa**2)/(1-eps**2)/(1-eps**2*cos_kappa**2)**2 + (piS + np.arctan(np.sqrt(1-eps**2)/eps/sin_kappa))/(1-eps**2)**1.5
        I3 = - eps*sin_kappa*(4-eps**2*(1+(2+eps**2)*cos_kappa**2))/(1-eps**2)**2/(1-eps**2*cos_kappa**2)**2 + (piS + np.arctan(np.sqrt(1-eps**2)/eps/sin_kappa))*(2+eps**2)/(1-eps**2)**2.5
        # dimensionless forces
        fr   = I2*eps*(1-2*phiS) - 2*epsS*I1
        fphi = I1*eps*(1-2*phiS) + 2*epsS*(I2-I3)
        # forces fr and fphi are switched! correct:
        # fr	--> fphi
        # fphi	-->	-fr
    return np.array([fr, fphi])

def bearing_journal_short(q,B,D,C,eta,omega0):
    DX, DY, VX, VY = q
    offset = 1e-10
    # kinematics
    omega0 = np.abs(omega0) + offset    # always positive
    y = np.array([DX/C, VX/(C*omega0), DY/C, VY/(C*omega0)])
    eps  = np.sqrt( y[0]**2 + y[2]**2 ) + offset    # no singularity
    epsS = (y[0]*y[1] + y[2]*y[3])/eps
    phiS = (y[0]*y[3] - y[2]*y[1])/eps**2
    # dimensional forces transformed into absolute coordinates
    sin_gamma = y[2]/eps
    cos_gamma = y[0]/eps
    return 0.25*D**3*B*eta/C**2*omega0*(B/D)**2*np.array([
        [-sin_gamma, cos_gamma],
        [ cos_gamma, sin_gamma]
        ]) @ journal_bearing_short(eps,epsS,phiS)