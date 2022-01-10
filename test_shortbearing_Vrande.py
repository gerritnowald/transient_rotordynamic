# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:33:04 2022

@author: Gerrit Nowald
"""

import numpy as np

def cos_sin(q):                       # q = [x,y]
    return q/np.sqrt(np.sum(q**2))    # [cos, sin]

def short_bearing_forces(eps,epsS,phiS):
# journal bearing forces short bearing theory
    vs = np.sqrt(epsilonS**2+(epsilon*(phiS-0.5))**2)
    alpha = np.arctan2(-epsilon*(phiS-0.5), epsilonS)
    if np.cos(alpha)>=0:
        delta = 1
    elif np.cos(alpha)<0:
        delta = -1
    A = (epsilon+np.sin(alpha))/(1+epsilon*np.sin(alpha))
    B = (epsilon-np.sin(alpha))/(1-epsilon*np.sin(alpha))
    # integrals
    I001 = (np.arccos(-delta*A)+np.arccos(-delta*B))/np.sqrt(1-epsilon**2)
    I023 = 1/(2*(1-epsilon**2)**2)*((1+2*epsilon**2)*I001+2*epsilon*np.cos(alpha)*(3+(2-5*epsilon**2)*(np.sin(alpha))**2)/(1-epsilon**2*(np.sin(alpha))**2)**2)
    I113 = -2*epsilon*(np.sin(alpha))**3/(1-epsilon**2*(np.sin(alpha))**2)**2
    I203 = 1/(2*(1-epsilon**2))*(I001+2*epsilon*np.cos(alpha)*(1-(2-epsilon**2)*(np.sin(alpha))**2)/(1-epsilon**2*(np.sin(alpha))**2)**2 )
    # Impedanz-Komponenten (Kurzlager)
    Wr   = 2*(I023*np.cos(alpha)-I113*np.sin(alpha))
    Wphi = 2*(I113*np.cos(alpha)-I203*np.sin(alpha))
    fr   =  vs*Wr        # Vorzeichen aus Definition des Koordinatensystems
    fphi = -vs*Wphi
    return np.array([fr, fphi])

epsilon  = 0.9
epsilonS = 1.1
phiS     = 0.75

BzuD = 0.5

print(short_bearing_forces(epsilon,epsilonS,phiS)*BzuD**2)