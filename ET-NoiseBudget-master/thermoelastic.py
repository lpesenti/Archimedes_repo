# D Brown, T.Zhang

import os
import numpy as np
from numpy import pi, sqrt, arctan, sin, cos, roots, size, real, imag, sqrt, exp
import const
from scipy import integrate
from const import BESSEL_ZEROS as zeta
from const import J0M as j0m
from gwinc.noise.substratethermal import substrate_thermoelastic_FiniteCorr

def J(Ome):
    A=np.sqrt(1/Ome**4*np.sqrt(2))
    B=np.sqrt(1/Ome/10)
    Result=(1/(1/np.sqrt(A)+1/np.sqrt(B)))**2
    return Result
    
#def integrand(v, u, Omega):
#    """ Integrand fromhttps://link.aps.org/doi/10.1103/PhysRevLett.91.260602
#    """
#    a = u**2 + v**2
#    return np.sqrt(2/np.pi)/np.pi * u**3 * np.exp(-u**2/2) / (a * (a**2 + Omega**2))

#J = np.vectorize(
    # Obviously we can't numerically integrate to infinity.
    # So the bounds here are hand picked so it operates
    # reasonably well over 1e-4 to 1e4 range.
#    lambda Omega: integrate.dblquad(integrand, 0, 10, lambda x: -4, lambda x: 4, args=(Omega,))[0]
#)

def substratethermoelastic(f, materials, wBeam):

    sigma = materials.Substrate.MirrorSigma
    rho = materials.Substrate.MassDensity
    kappa = materials.Substrate.MassKappa # thermal conductivity
    alpha = materials.Substrate.MassAlpha # thermal expansion
    CM = materials.Substrate.MassCM # heat capacity @ constant mass
    Temp = materials.Substrate.Temp # temperature
    kBT = const.kB * materials.Substrate.Temp
    
    wc=kappa/(rho*CM*wBeam**2)
    
    Ome=2*pi*f/wc

    S = 8*(1+sigma)**2*kappa*alpha**2*Temp*kBT*Ome**2*J(Ome) # note kBT has factor Temp
    S /= (sqrt(2*pi)*(CM*rho)**2)
    S /= (wBeam/sqrt(2))**3 # LT 18 less factor 1/omega^2

    # Corrections for finite test masses:
    S *= substrate_thermoelastic_FiniteCorr(materials, wBeam)

    return S/(2*pi*f)**2
    
