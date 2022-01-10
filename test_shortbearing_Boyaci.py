# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:45:21 2022

@author: uia52592
"""

import numpy as np

def cos_sin(q):                       # q = [x,y]
    return q/np.sqrt(np.sum(q**2))    # [cos, sin]

def short_bearing_forces(eps,epsS,phiS):
# journal bearing forces short bearing theory
    if epsS == 0 and phiS == 0:    # rotation only
        fr   = 2*eps**2/(1-eps**2)**2
        fphi = - 0.5*np.pi*eps/(1-eps**2)**1.5
    else:                      # rotation + squeeze
        cos_alpha, sin_alpha = cos_sin(np.array([2*epsS, eps*(1-2*phiS)]))
        cos_alpha, sin_alpha = - sin_alpha, - cos_alpha     # needs to be corrected
        # integrals
        angle = np.arctan2(np.sqrt(1-eps**2),eps*sin_alpha)     # old correct version
        # angle = np.arctan2(np.sqrt(1-eps**2),-eps*cos_alpha)
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

epsilon  = 0.9
epsilonS = 1.1
phiS     = 0.75

BzuD = 0.5

print(short_bearing_forces(epsilon,epsilonS,phiS)*BzuD**2)