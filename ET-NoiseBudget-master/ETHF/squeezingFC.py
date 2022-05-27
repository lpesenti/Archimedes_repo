import os
from numpy import pi, sqrt, arctan, sin, cos, roots, size, real, imag
import const
from gwinc.ifo.noises import ifo_power

def sql(ifo):
    """Computer standard quantum limit (SQL) for IFO"""
    c = const.c
    power = ifo_power(ifo)
    w0 = 2 * pi * c / ifo.Laser.Wavelength
    rho = ifo.Materials.Substrate.MassDensity
    m = ifo.Suspension.Stage[0].Mass
    Titm = ifo.Optics.ITM.Transmittance
    Tsrm = ifo.Optics.SRM.Transmittance
    tSR = sqrt(Tsrm)
    rSR = sqrt(1 - Tsrm)
    fSQL = (1/(2*pi))*(8/c)*sqrt((power.parm*w0)/(m*Titm))*(tSR/(1+rSR))
    return fSQL


def computeFCParams(ifo):
    """Compute ideal filter cavity Tin, detuning [Hz] and bandwidth [Hz]

    """
    # FC parameters
    fcParams = ifo.Squeezer.FilterCavity
    c = const.c
    fsrFC = c / (2 * fcParams.L)
    lossFC = fcParams.Lrt + fcParams.Te

    fSQL = sql(ifo)

    # detuning and cavity bandwidth (D&D paper P1400018 and/or PRD)
    eps = 4 / (2 + sqrt(2 + 2 * sqrt(1 + (4 * pi * fSQL / (fsrFC * lossFC))**4)))
    s1eps = sqrt(1 - eps)

    # cavity bandwidth [Hz]
    gammaFC = fSQL / sqrt(s1eps + s1eps**3)
    # cavity detuning [Hz]
    detuneFC = s1eps * gammaFC

    # input mirror transmission
    TinFC = 4 * pi * gammaFC / fsrFC - lossFC
    if TinFC < lossFC:
        raise RuntimeError(
            'IFC: Losses are too high! {:0.1f} ppm max.'.format(1e6 * gammaFC / fsrFC))

    # Add to fcParams structure
    fcParams.Ti = TinFC
    fcParams.fdetune = -detuneFC
    fcParams.gammaFC = gammaFC
    fcParams.fsrFC = fsrFC

    return fcParams
    
    
def computeFCsParams(ifo, fcParams):
    c = const.c
    Parm = ifo_power(ifo).parm
    w0 = 2 * pi * c / ifo.Laser.Wavelength
    rho = ifo.Materials.Substrate.MassDensity
    m = pi * ifo.Materials.MassRadius**2 * ifo.Materials.MassThickness * rho
    Larm = ifo.Infrastructure.Length
    Titm = ifo.Optics.ITM.Transmittance
    Tsrm = ifo.Optics.SRM.Transmittance
    phiSR= -ifo.Optics.SRM.Tunephase/2+pi/2


    tSR = sqrt(Tsrm)
    rSR = sqrt(1-Tsrm)

    zeta1 = pi/2

    zeta=arctan((rSR*cos(phiSR-zeta1)+cos(phiSR+zeta1))**(-1)*((-1)*rSR*sin(phiSR+(-1)*zeta1)+sin(phiSR+zeta1)))
    
    Theta = 8/c/Larm/m*w0*Parm

    Lambda = c*Titm/(4*Larm)*2*rSR*sin(2*phiSR)/(1+rSR**2+2*rSR*cos(2*phiSR))

    epsilon = c*Titm/(4*Larm)*(1-rSR**2)/(1+rSR**2+2*rSR*cos(2*phiSR))

    poly = [1,0,(epsilon+ 1j*Lambda)**2,0,Theta*(Lambda-1j*epsilon-1j*epsilon*cos(2*zeta)-epsilon*sin(2*zeta))]
    
    #poly = [1,0,(epsilon+ 1j*Lambda)**2,0,Theta*(Lambda-1j*epsilon+1j*epsilon*(cos(2*zeta1)+1j*sin(2*zeta1)))]

    ropo = roots(poly)
    
    ns = 0
    if isinstance(fcParams, list):
        for n in range(size(ropo))[::-1]:
            if imag(ropo[n]) < 0:
                continue
            try:
                fcParams[ns].fdetune = -real(ropo[n])/pi/2
                fcParams[ns].gammaFC = imag(ropo[n])/pi/2
                fcParams[ns].Ti= 4*fcParams[ns].L*fcParams[ns].gammaFC*2*pi /c
                ns += 1
            except IndexError:
                print("More roots than filter cavities!")
                break
    else:
        for root in ropo[::-1]:
            if imag(root) < 0:
                continue
            fcParams.fdetune = -real(root)/pi/2
            fcParams.gammaFC = imag(root)/pi/2
            fcParams.Ti= 4*fcParams.L * fcParams.gammaFC * 2 * pi /c
            break


    return fcParams

