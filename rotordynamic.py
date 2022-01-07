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

def alpha(X,Y):
    if X == 0:    # singularity
        if Y>0:
            tan_alpha = 1e16
        else:
            tan_alpha = -1e16
    else:
        tan_alpha = Y/X
    cos_alpha = (-1)**(X>0)*1/np.sqrt(1+tan_alpha**2)
    sin_alpha = (-1)**(X>0)*tan_alpha/np.sqrt(1+tan_alpha**2)
    # alpha = np.arctan2(Y,X)
    # sin_alpha = - np.sin(alpha)
    # cos_alpha = - np.cos(alpha)
    if sin_alpha/(1+cos_alpha) < 0:
        piS = np.pi
    else:
        piS = 0   
    return sin_alpha, cos_alpha, piS

def short_bearing_forces(eps,epsS,phiS):
# journal bearing forces short bearing theory
    if epsS == 0 and phiS == 0:    # case1: rotation only
        fr   = 2*eps**2/(1-eps**2)**2
        fphi = - 0.5*np.pi*eps/(1-eps**2)**1.5
    else:                      # case2: rotation + squeeze
        sin_alpha, cos_alpha, piS = alpha(eps*(1-2*phiS),2*epsS)
        # integrals
        I1 = 2*eps*cos_alpha**3/(1-eps**2*cos_alpha**2)**2
        I2 = - eps*sin_alpha*(1-(2-eps**2)*cos_alpha**2)/(1-eps**2)/(1-eps**2*cos_alpha**2)**2 + (piS + np.arctan(np.sqrt(1-eps**2)/eps/sin_alpha))/(1-eps**2)**1.5
        I3 = - eps*sin_alpha*(4-eps**2*(1+(2+eps**2)*cos_alpha**2))/(1-eps**2)**2/(1-eps**2*cos_alpha**2)**2 + (piS + np.arctan(np.sqrt(1-eps**2)/eps/sin_alpha))*(2+eps**2)/(1-eps**2)**2.5
        # dimensionless forces
        fr   = I2*eps*(1-2*phiS) - 2*epsS*I1
        fphi = I1*eps*(1-2*phiS) + 2*epsS*(I2-I3)
        # forces fr and fphi are switched! correct:
        # fr	--> fphi
        # fphi	-->	-fr
    return np.array([fr, fphi])

def bearing_journal_short(q,B,D,C,eta,omega0):
    # q = [x, y, xd, yd]
    # kinematics
    offset = 1e-10  # against singularities
    omega0 = np.abs(omega0) + offset
    eps    = np.sqrt( np.sum(q[0:2]**2) )/C + offset
    epsS   = q[0:2] @ q[2:]/C**2/omega0/eps
    phiS   = (q[0]*q[3] - q[1]*q[2])/C**2/omega0/eps**2
    cos_theta, sin_theta = q[0:2]/C/eps
    # dimensional forces transformed into absolute coordinates
    return 0.25*D**3*B*eta/C**2*omega0*np.array([
        [-sin_theta, cos_theta],
        [ cos_theta, sin_theta]
        ]) @ short_bearing_forces(eps,epsS,phiS)*(B/D)**2