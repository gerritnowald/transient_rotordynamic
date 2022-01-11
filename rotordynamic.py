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
    vs = - np.sqrt(epsS**2+(eps*(phiS-0.5))**2)
    cos_alpha, sin_alpha = cos_sin(np.array([epsS, eps*(0.5-phiS)]))
    delta = (-1)**(cos_alpha<0)
    A = (eps+sin_alpha)/(1+eps*sin_alpha)
    B = (eps-sin_alpha)/(1-eps*sin_alpha)
    # integrals
    I001 = (np.arccos(-delta*A) + np.arccos(-delta*B))/np.sqrt(1-eps**2)
    I023 = 1/(2*(1-eps**2)**2)*((1+2*eps**2)*I001 + 2*eps*cos_alpha*(3+(2-5*eps**2)*sin_alpha**2)/(1-eps**2*sin_alpha**2)**2)
    I113 = -2*eps*sin_alpha**3/(1-eps**2*sin_alpha**2)**2
    I203 = 1/(2*(1-eps**2))*(I001 + 2*eps*cos_alpha*(1-(2-eps**2)*sin_alpha**2)/(1-eps**2*sin_alpha**2)**2)
    # dimensionless forces
    fr   = 2*vs*(I023*cos_alpha - I113*sin_alpha)
    fphi = 2*vs*(I113*cos_alpha - I203*sin_alpha)
    return np.array([fr, fphi])

def bearing_journal_short(q,B,D,C,eta,omega0):
    # state vector q = [x, y, xd, yd]
    offset = 1e-10      # against singularities
    omega0 = np.abs(omega0) + offset
    d = q[0:2]/C        # journal displacements [x, y]
    v = q[2:]/C/omega0  # journal speeds [xd, yd]
    eps  = np.sqrt(np.sum(d**2)) + offset
    epsS = d @ v/eps
    phiS = np.cross(d,v)/eps**2
    cos_delta, sin_delta = d/eps
    # dimensional forces transformed into absolute coordinates
    return 0.25*D**3*B*eta/C**2*omega0*np.array([
        [ cos_delta, -sin_delta ],
        [ sin_delta,  cos_delta ]
        ]) @ short_bearing_forces(eps,epsS,phiS)*(B/D)**2