from numpy import sqrt, pi, sin, cos, arctan, real, imag, size, roots
from gwinc import const
from gwinc.ifo.noises import ifo_power

def computeFCParams(ifo):
    c = const.c
    Parm = ifo_power(ifo).parm
    w0 = 2 * pi * c / ifo.Laser.Wavelength
    m = ifo.Suspension.Stage[0].Mass
    Larm = ifo.Infrastructure.Length
    Titm = ifo.Optics.ITM.Transmittance
    Tsrm = ifo.Optics.SRM.Transmittance
    phiSR= -ifo.Optics.SRM.Tunephase/2+pi/2

    rSR = sqrt(1-Tsrm)
    zeta1 = pi/2
    zeta = arctan(
        (-1*rSR*sin(phiSR - 1*zeta1) + sin(phiSR + zeta1)) / 
        (rSR*cos(phiSR-zeta1) + cos(phiSR+zeta1))
    )
    
    Theta = 8/c/Larm/m*w0*Parm

    Lambda = c*Titm/(4*Larm)*2*rSR*sin(2*phiSR)/ \
        (1+rSR**2+2*rSR*cos(2*phiSR))

    epsilon = c*Titm/(4*Larm)*(1-rSR**2)/ \
        (1+rSR**2+2*rSR*cos(2*phiSR))

    poly = [1, 0, (epsilon+1j*Lambda)**2, 0,
            Theta*(Lambda-1j*epsilon-1j*epsilon*cos(2*zeta) - epsilon*sin(2*zeta))]
    #poly = [1,0,(epsilon+ 1j*Lambda)**2,0,Theta*(Lambda-1j*epsilon+1j*epsilon*(cos(2*zeta1)+1j*sin(2*zeta1)))]

    ropo = roots(poly)
    ropo = ropo[imag(ropo) >= 0]
    
    fcParams = ifo.Squeezer.FilterCavity
    if not isinstance(fcParams, list) or len(fcParams) != len(ropo):
        raise Exception(f'Exactly {len(ropo)} filter cavities must be defined.')
    
    for fc, root in zip(fcParams, ropo[::-1]):
        fc.fdetune = -real(root)/pi/2
        fc.gammaFC = imag(root)/pi/2
        fc.Ti= 4*fc.L*fc.gammaFC*2*pi /c
                
    return fcParams

if __name__ == "__main__":
    import gwinc
    budget = gwinc.load_budget('../ETLF')
    for fc in computeFCParams(budget.ifo):
        print(fc.to_yaml())