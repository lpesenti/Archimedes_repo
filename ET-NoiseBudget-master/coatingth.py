'''Functions to calculate coating thermal noise

'''
from __future__ import division, print_function
import numpy as np
from numpy import pi, exp, real, imag, sqrt, sin, cos, sinh, cosh, ceil, log

import const
from const import BESSEL_ZEROS as zeta
from const import J0M as j0m


def build_stacks(mirror, optic):
    """ Assemble coating properties into arrays

    This function interprets the CoatingLayers definition and creates new
    arrays in the mirror structure that contain the material properties
    and dimensions for each layer of the coating.
    """
    if not 'CoatingLayers' in optic:
        raise Exception('==FIXME== optic without explicit layer design not yet ported to multi-material code')
        # == LATER: auto-generate coating, but only if there are exactly two materials given,
        # == because auto-generation with multi-material coatings would need more design input
        #T = optic.Transmittance
        #dL = optic.CoatingThicknessLown
        #dCap = optic.CoatingThicknessCap
        #mirror.Coating.dOpt = noise.coatingthermal.getCoatDopt(mirror, T, dL, dCap=dCap)

    initialize_stack(mirror, optic)
    def getExpansionRatio(Y_C, sigC, Y_S, sigS):
        ##############################################
        # Y_C and sigC are for the coating material (can also be substrate)
        # Y_S and sigS are for the substrate material
        #
        ce = ((1 + sigS) / (1 - sigC)) \
             * ( ((1 + sigC) / (1 + sigS)) + (1 - 2 * sigS) * Y_C / Y_S )
        return ce
        
    Y_S=mirror.Substrate.MirrorY
    sigS=mirror.Substrate.MirrorSigma
       
       
    known_materials = {} # cache loss angle functions
    for idx, layer in enumerate(optic.CoatingLayers):
        # unpack Material and optThickness
        mat, d = list(layer.items())[0]
        if mat in known_materials:
            m = known_materials[mat]
        else:
            try:
                m = mirror.Coating[mat]
                m.lossB_f, m.lossS_f = interpretLossAngles(m)
                known_materials[mat] = m
            except KeyError as e:
                raise Exception(f'Undefined coating material {str(e)} at position {idx+1} in CoatingLayers.') from None
 
        mirror.Coating.dOpt[idx] = d
        mirror.Coating.nN[idx] = m.Index
        mirror.Coating.yN[idx] = m.Y
        mirror.Coating.CV[idx] = m.CV
        mirror.Coating.TC[idx] = m.ThermalConductivity
        mirror.Coating.pratN[idx] = m.Sigma
        mirror.Coating.lossB[idx] = m.lossB_f
        mirror.Coating.lossS[idx] = m.lossS_f
        mirror.Coating.Alpha[idx]= m.Alpha
        mirror.Coating.nA[idx] = m.Alpha* getExpansionRatio(m.Y, m.Sigma, Y_S, sigS)
        mirror.Coating.nS[idx] = m.Alpha * (1 + m.Sigma) / (1 - m.Sigma)
        mirror.Coating.nB[idx]=m.Beta


def initialize_stack(mirror, optic):
    """ Prepare empty arrays of material properties for the coating stack
    """
    nol = len(optic.CoatingLayers)
    mirror.Coating.dOpt = np.zeros(nol)
    mirror.Coating.dGeo = np.zeros(nol)
    mirror.Coating.nN = np.zeros(nol)
    mirror.Coating.yN = np.zeros(nol)
    mirror.Coating.Alpha = np.zeros(nol)
    mirror.Coating.pratN = np.zeros(nol)
    mirror.Coating.CPE = np.zeros(nol)
    mirror.Coating.lossB = [None]*nol
    mirror.Coating.lossS = [None]*nol
    
    mirror.Coating.nA = np.zeros(nol)
    mirror.Coating.nS = np.zeros(nol)
    mirror.Coating.nB = np.zeros(nol)
    mirror.Coating.CV = np.zeros(nol)
    mirror.Coating.TC = np.zeros(nol)
    # TODO: add CPE

def coating_brownian(f, mirror, wavelength, wBeam, power=None):
    """Coating brownian noise for a given collection of coating layers

    This function calculates Coating Brownian noise using
    Hong et al . PRD 87, 082001 (2013).
    All references to 'the paper', 'Eq' and 'Sec' are to this paper.

    ***Important Note***
    Inside this function phi is used for denoting the phase shift suffered
    by light in one way propagation through a layer. This is in conflict
    with present nomenclature everywhere else where it is used as loss angle.

    The layers are assumed to be alernating low-n high-n layers, with
    low-n first.

    Inputs:
             f = frequency vector in Hz
        mirror = mirror properties Struct
    wavelength = laser wavelength
         wBeam = beam radius (at 1 / e**2 power)
         power = laser power falling on the mirror (W)

    If the power argument is present and is not None, the amplitude noise due
    to coating brownian noise will be calculated and its effect on the phase
    noise will be added (assuming the susceptibility is that of a free mass)

    ***The following parameters are experimental and unsupported as yet***
    The following optional parameters are available in the Materials object
    to provide separate Bulk and Shear loss angles and to include the effect
    of photoelasticity:
    lossBlown = Coating Bulk Loss Angle of Low Refrac.Index layer @ 100Hz
    lossSlown = Coating Shear Loss Angle of Low Refrac. Index layer @ 100Hz
    lossBhighn = Coating Bulk Loss Angle of High Refrac. Index layer @ 100Hz
    lossShighn = Coating Shear Loss Angle of High Refrac. Index layer @ 100Hz
    lossBlown_slope = Coating Bulk Loss Angle Slope of Low Refrac. Index layer
    lossSlown_slope = Coating Shear Loss Angle Slope of Low Refrac. Index layer
    lossBhighn_slope = Coating Bulk Loss Angle Slope of High Refrac. Index layer
    lossShighn_slope = Coating Shear Loss Angle Slope of High Refrac. Index layer
    PETlown = Relevant component of Photoelastic Tensor of High n layer*
    PEThighn = Relevant component of Photoelastic Tensor of Low n layer*

    Returns:
      SbrZ = Brownian noise spectra for one mirror in m**2 / Hz

    *
    Choice of PETlown and PEThighn can be inspired from sec. A.1. of the paper.
    There, values are chosen to get the longitudnal coefficent of
    photoelasticity as -0.5 for Tantala and -0.27 for Silica.
    These values also need to be added in Materials object.
    *
    If the optional arguments are not present, Phihighn and Philown will be
    used as both Bulk and Shear loss angles and PET coefficients will be set
    to 0.

    """
    # extract substructures
    sub = mirror.Substrate

    # Constants
    kBT = const.kB * sub.Temp
    c = const.c

    # substrate properties
    Ysub = sub.MirrorY            # Young's Modulous
    pratsub = sub.MirrorSigma     # Poisson Ratio
    nsub = sub.RefractiveIndex    # Refractive Index

    # geometrical thickness of each layer and total
    nol = len(mirror.Coating.dOpt)
    mirror.Coating.dGeo = wavelength * np.asarray(mirror.Coating.dOpt) / mirror.Coating.nN

    # WaveLength of light in each layer
    mirror.Coating.lambdaN = wavelength / mirror.Coating.nN

    # Calculate rho and derivatives of rho
    # with respect to both phi_k and r_j
    rho, dLogRho_dPhik, dLogRho_dRk, r = getCoatReflAndDer(mirror.Coating.nN, nsub, mirror.Coating.dOpt)
    print('T={:.1f}'.format((1-abs(rho)**2)*1e6))

    # Define the function epsilon as per Eq (25)
    # Split epsilon function as:
    # Epsilon = Ep1 - Ep2 * cos(2k0n(z-zjp1)) - Ep3 * sin(2k0n(z-zjp1))

    # Part 1 of epsilon function
    Ep1 = (mirror.Coating.nN + mirror.Coating.CPE) * dLogRho_dPhik[:-1]
    # Part 2 of epsilon function (Prefactor of cosine term)
    Ep2 = mirror.Coating.CPE * (dLogRho_dPhik[:-1] * (1 - r[:-1]**2) / (2*r[:-1])
                     - (dLogRho_dPhik[1:] * (1 + r[:-1]**2) / (2 * r[:-1])))
    # Part 3 of epsilon function (Prefactor of sine term)
    Ep3 = (1 - r[:-1]**2) * mirror.Coating.CPE * dLogRho_dRk[:-1]

    # Define (1 - Im(epsilon)/2)
    Ip1 = 1 - imag(Ep1) / 2  # First part of (1 - Im(epsilon)/2)
    Ip2 = imag(Ep2) / 2      # Prefactor to cosine in (1 - Im(epsilon)/2)
    Ip3 = imag(Ep3) / 2      # Prefactor to sine in (1 - Im(epsilon)/2)

    # Define transfer functions from bulk and shear noise fields to layer
    # thickness and surface height as per Table I in paper
    C_B = np.sqrt(0.5*(1+mirror.Coating.pratN))
    C_SA = np.sqrt(1 - 2*mirror.Coating.pratN)
    D_B = ((1 - pratsub - 2*pratsub**2)*mirror.Coating.yN/(np.sqrt(2*(1+mirror.Coating.pratN))*Ysub))
    D_SA = -((1 - pratsub - 2*pratsub**2)*mirror.Coating.yN/(2*np.sqrt(1-2*mirror.Coating.pratN)*Ysub))
    D_SB = (np.sqrt(3)*(1-mirror.Coating.pratN)*(1 - pratsub - 2*pratsub**2)*mirror.Coating.yN
            / (2*np.sqrt(1-2*mirror.Coating.pratN)*(1+mirror.Coating.pratN)*Ysub))

    # Calculating effective beam area on each layer
    # Assuming the beam radius does not change significantly through the
    # depth of the mirror.
    Aeff = pi*(wBeam**2)

    # PSD at single layer with thickness equal to WaveLength
    # in the medium Eq (96)
    S_Bk = np.zeros((nol, len(f)))
    S_Sk = np.zeros((nol, len(f)))
    for idx in range(nol):
        factor = (4 * kBT * mirror.Coating.lambdaN[idx]
                  * (1 - mirror.Coating.pratN[idx] - 2 * mirror.Coating.pratN[idx]**2)
                  / (3 * pi * f * mirror.Coating.yN[idx] * ((1 - mirror.Coating.pratN[idx])**2) * Aeff))
        S_Bk[idx] = mirror.Coating.lossB[idx](f) * factor
        S_Sk[idx] = mirror.Coating.lossS[idx](f) * factor

    # Coefficients q_j from Eq (94
    # See https://dcc.ligo.org/T2000552 for derivation
    k0 = 2 * pi / wavelength
    q_Bk = (+ 8 * C_B * (D_B + C_B * Ip1) * Ip3
            + 2 * C_B**2 * Ip2 * Ip3
            + 4 * (2 * D_B**2 + 4 * C_B * D_B * Ip1
                   + C_B**2 * (2 * Ip1**2 + Ip2**2 + Ip3**2)
                   ) * k0 * mirror.Coating.nN * mirror.Coating.dGeo
            - 8 * C_B * (D_B + C_B * Ip1) * Ip3 * np.cos(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            - 2 * C_B**2 * Ip2 * Ip3 * np.cos(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            + 8 * C_B * (D_B + C_B * Ip1) * Ip2 * np.sin(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            + C_B**2 * (Ip2 - Ip3) * (Ip2 + Ip3) * np.sin(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            ) / (8 * k0 * mirror.Coating.lambdaN * mirror.Coating.nN)

    q_Sk = (D_SB**2 * 8 * k0 * mirror.Coating.nN * mirror.Coating.dGeo
            + 8 * C_SA * (D_SA + C_SA * Ip1) * Ip3
            + 2 * C_SA**2 * Ip2 * Ip3
            + 4 * (2 * D_SA**2 + 4 * C_SA * D_SA * Ip1
                   + C_SA**2 * (2 * Ip1**2 + Ip2**2 + Ip3**2)
                   ) * k0 * mirror.Coating.nN * mirror.Coating.dGeo
            - 8 * C_SA * (D_SA + C_SA * Ip1) * Ip3 * np.cos(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            - 2 * C_SA**2 * Ip2 * Ip3 * np.cos(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            + 8 * C_SA * (D_SA + C_SA * Ip1) * Ip2 * np.sin(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            + C_SA**2 * (Ip2 - Ip3) * (Ip2 + Ip3) * np.sin(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
            ) / (8 * k0 * mirror.Coating.lambdaN * mirror.Coating.nN)

    # S_Xi as per Eq(94)
    S_Xi = (np.tensordot(q_Bk, S_Bk, axes=1)
            + np.tensordot(q_Sk, S_Sk, axes=1))

    # From Sec II.E. Eq.(41)
    # Conversion of brownian amplitude noise to displacement noise
    if power is not None:

        # get/calculate optic transmittance
        mTi = mirror.get('Transmittance', 1-np.abs(rho)**2)

        # Define Re(epsilon)/2
        Rp1 = np.real(Ep1) / 2   # First part of Re(epsilon)/2
        Rp2 = -np.real(Ep2) / 2  # Prefactor to cosine in Re(epsilon)/2
        Rp3 = -np.real(Ep3) / 2  # Prefactor to sine in Re(epsilon)/2
        # Coefficients p_j from Eq (95)
        # See https://dcc.ligo.org/T2000552 for derivation
        p_BkbyC = (+ 8 * Rp1 * Rp3
                   + 2 * Rp2 * Rp3
                   + 4 * (2 * Rp1**2 + Rp2**2 + Rp3**2) * k0 * mirror.N * mirror.Coating.dGeo
                   - 8 * Rp1 * Rp3 * np.cos(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
                   - 2 * Rp2 * Rp3 * np.cos(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
                   + 8 * Rp1 * Rp2 * np.sin(2 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
                   + (Rp2 - Rp3) * (Rp2 + Rp3) * np.sin(4 * k0 * mirror.Coating.nN * mirror.Coating.dGeo)
                   ) / (8 * k0 * mirror.Coating.lambdaN * mirror.Coating.nN)
        p_Bk = p_BkbyC * C_B**2
        p_Sk = p_BkbyC * C_SA**2

        # S_Zeta as per Eq(95)
        S_Zeta = (np.tensordot(p_Bk, S_Bk, axes=1)
                  + np.tensordot(p_Sk, S_Sk, axes=1))

        AmpToDispConvFac = ((32 * power)
                            / (mirror.MirrorMass * wavelength
                               * f**2 * c * 2 * pi * sqrt(mTi)))
        # Adding the two pathways of noise contribution as correlated noise
        SbrZ = (sqrt(S_Xi) + AmpToDispConvFac * sqrt(S_Zeta))**2
    else:
        SbrZ = S_Xi

    return SbrZ


def coating_thermooptic(f, mirror, wavelength, wBeam):
    """Optical coating thermo-optic displacement noise spectrum

    :f: frequency array in Hz
    :mirror: mirror parameter Struct
    :wavelength: laser wavelength
    :wBeam: beam radius (at 1 / e**2 power)

    :returns: tuple of:
    StoZ = displacement noise power spectrum at :f:
    SteZ = thermo-optic component of StoZ
    StrZ = thermo-refractive component of StoZ
    T = coating power transmission

    """
    # compute coefficients
    dTO, dTR, dTE, T, junk = getCoatTOPos(mirror, wavelength, wBeam)

    # compute correction factors
    gTO = getCoatThickCorr(f, mirror, wavelength, dTE, dTR)
    gTE = getCoatThickCorr(f, mirror, wavelength, dTE, 0)
    gTR = getCoatThickCorr(f, mirror, wavelength, 0, dTR)

    # compute thermal source spectrum
    SsurfT, junk = getCoatThermal(f, mirror, wBeam)

    StoZ = SsurfT * gTO * dTO**2
    SteZ = SsurfT * gTE * dTE**2
    StrZ = SsurfT * gTR * dTR**2

    return (StoZ, SteZ, StrZ, T)


def getCoatTOPos(mirror, wavelength, wBeam):
    """Mirror position derivative wrt thermal fluctuations

    :mirror: mirror parameter Struct
    :wavelength: laser wavelength
    :wBeam: beam radius (at 1 / e**2 power)

    :returns: tuple of:
    dTO = total thermo-optic dz/dT
    dTR = thermo-refractive dz/dT
    dTE = thermo-elastic dz/dT
    T = coating power transmission
    R = coating power reflection

    Compute thermal fluctuations with getCoatThermal.

    See LIGO-T080101.

    """
    # parameters
    nS = mirror.Substrate.RefractiveIndex
    dOpt = mirror.Coating.dOpt

    # compute refractive index, effective alpha and beta
    nLayer, aLayer, bLayer, dLayer, sLayer = getCoatLayers(mirror, wavelength)

    # compute coating average parameters
    dc, Cc, Kc, aSub = getCoatAvg(mirror, wavelength)

    # compute reflectivity and parameters
    dphi_dT, dphi_TE, dphi_TR, rCoat = getCoatTOPhase(1, nS, nLayer, dOpt, aLayer, bLayer, sLayer)
    R = abs(rCoat)**2
    T = 1 - R

    # for debugging
    #disp(sprintf('R = %.3f, T = %.0f ppm', R, 1e6 * T))

    # convert from phase to meters, subtracting substrate
    dTR = dphi_TR * wavelength / (4 * pi)
    dTE = dphi_TE * wavelength / (4 * pi) - aSub * dc

    # mirror finite size correction
    Cfsm = getCoatFiniteCorr(mirror, wavelength, wBeam)
    dTE = dTE * Cfsm

    # add TE and TR effects (sign is already included)
    dTO = dTE + dTR

    return dTO, dTR, dTE, T, R


def getCoatThickCorr(f, mirror, wavelength, dTE, dTR):
    """Finite coating thickness correction

    :f: frequency array in Hz
    :mirror: gwinc optic mirror structure
    :wavelength: laser wavelength
    :wBeam: beam radius (at 1 / e**2 power)

    Uses correction factor from LIGO-T080101, "Thick Coating
    Correction" (Evans).

    See getCoatThermoOptic for example usage.

    """
    ##############################################
    # For comparison in the bTR = 0 limit, the
    # equation from Fejer (PRD70, 2004)
    # modified so that gFC -> 1 as xi -> 0
    #  gTC = (2 ./ (R * xi.**2)) .* (sh - s + R .* (ch - c)) ./ ...
    #    (ch + c + 2 * R * sh + R**2 * (ch - c));
    # which in the limit of xi << 1 becomes
    #  gTC = 1 - xi * (R - 1 / (3 * R));

    # parameter extraction
    pS = mirror.Substrate
    Cs = pS.MassCM * pS.MassDensity
    Ks = pS.MassKappa

    # compute coating average parameters
    dc, Cc, Kc, junk = getCoatAvg(mirror, wavelength)

    # R and xi (from T080101, Thick Coating Correction)
    w = 2 * pi * f
    R = sqrt(Cc * Kc / (Cs * Ks))
    xi = dc * sqrt(2 * w * Cc / Kc)

    # trig functions of xi
    s = sin(xi)
    c = cos(xi)
    sh = sinh(xi)
    ch = cosh(xi)

    # pR and pE (dTR = -\bar{\beta} lambda, dTE = \Delta \bar{\alpha} d)
    pR = dTR / (dTR + dTE)
    pE = dTE / (dTR + dTE)

    # various parts of gTC
    g0 = 2 * (sh - s) + 2 * R * (ch - c)
    g1 = 8 * sin(xi / 2) * (R * cosh(xi / 2) + sinh(xi / 2))
    g2 = (1 + R**2) * sh + (1 - R**2) * s + 2 * R * ch
    gD = (1 + R**2) * ch + (1 - R**2) * c + 2 * R * sh

    # and finally, the correction factor
    gTC = (pE**2 * g0 + pE * pR * xi * g1 + pR**2 * xi**2 * g2) / (R * xi**2 * gD)

    return gTC


def getCoatThermal(f, mirror, wBeam):
    """Thermal noise spectra for a surface layer

    :f: frequency array in Hz
    :mirror: mirror parameter Struct
    :wBeam: beam radius (at 1 / e**2 power)

    :returns: tuple of:
    SsurfT = power spectra of thermal fluctuations in K**2 / Hz
    rdel = thermal diffusion length at each frequency in m

    """
    pS = mirror.Substrate
    C_S = pS.MassCM * pS.MassDensity
    K_S = pS.MassKappa
    kBT2 = const.kB * pS.Temp**2

    # omega
    w = 2 * pi * f

    # thermal diffusion length
    rdel = sqrt(2 * K_S / (C_S * w))

    # noise equation
    SsurfT = 4 * kBT2 / (pi * w * C_S * rdel * wBeam**2)

    return SsurfT, rdel


def getCoatLayers(mirror, wavelength):
    """Layer vectors for refractive index, effective alpha and beta and geometrical thickness

    :mirror: mirror parameter Struct
    :wavelength: laser wavelength

    :returns: tuple of:
    nLayer = refractive index of each layer, ordered input to output (N x 1)
    aLayer = change in geometrical thickness with temperature
           = the effective thermal expansion coeffient of the coating layer
    bLayer = change in refractive index with temperature
           = dn/dT
    dLayer = geometrical thicness of each layer
    sLayer = Yamamoto thermo-refractive correction
           = alpha * (1 + sigma) / (1 - sigma)

    """
    dOpt = mirror.Coating.dOpt
    Nlayer = len(dOpt)
    # compute effective alpha
    #aLayer = np.zeros(Nlayer)
    aLayer = mirror.Coating.nA
    # and beta
    #bLayer = np.zeros(Nlayer)
    bLayer = mirror.Coating.nB
    # and refractive index
    #nLayer= np.zeros(Nlayer)
    nLayer = mirror.Coating.nN
    # and geometrical thickness
    dLayer = wavelength * np.asarray(mirror.Coating.dOpt) / mirror.Coating.nN
    
    # and sigma correction
    #sLayer = np.zeros(Nlayer)
    sLayer = mirror.Coating.nS
    
    return nLayer, aLayer, bLayer, dLayer, sLayer


def getCoatAvg(mirror, wavelength):
    """Coating average properties

    :mirror: gwinc optic mirror structure
    :wavelength: laser wavelength

    :returns: tuple of:
    dc = total thickness (meters)
    Cc = heat capacity
    Kc = thermal diffusivity
    aSub = effective substrate thermal expansion (weighted by heat capacity)

    """
    # coating parameters
    pS = mirror.Substrate
    pC = mirror.Coating
    dOpt = mirror.Coating.dOpt

    alphaS = pS.MassAlpha
    C_S = pS.MassCM * pS.MassDensity
    sigS = pS.MirrorSigma

    
    # compute refractive index, effective alpha and beta
    junk1, junk2, junk3, dLayer, junk4 = getCoatLayers(mirror, wavelength)
    dc = np.sum(dLayer)
    nLayer= len(dOpt)
    Cc=np.zeros(1)
    KD=np.zeros(1)
    for idx in range(nLayer):
        C_ = pC.CV[idx]
        K_ = pC.TC[idx]
        # heat capacity
        d = dLayer[idx]
        Cc += (C_ * d) / dc
        # thermal diffusivity
        Kinv = 1 / K_
        KD += Kinv * d
    Kc = dc / KD

    # effective substrate thermal expansion
    aSub = 2 * alphaS * (1 + sigS) * Cc / C_S

    return dc, Cc, Kc, aSub


def getCoatTOPhase(nIn, nOut, nLayer, dOpt, aLayer, bLayer, sLayer):
    """Coating reflection phase derivatives w.r.t. temperature

    :nIn: refractive index of input medium (e.g., vacuum = 1)
    :nOut: refractive index of output medium (e.g., SiO2 = 1.45231 @ 1064nm)
    :nLayer: refractive index of each layer, ordered input to output (N x 1)
    :dOpt: optical thickness / lambda of each layer
           = geometrical thickness * refractive index / lambda
    :aLayer: change in geometrical thickness with temperature
             = the effective thermal expansion coeffient of the coating layer
    :bLayer: change in refractive index with temperature
             = dn/dT
             = dd/dT - n * a

    :returns: tuple of:
    dphi_dT = total thermo-optic phase derivative with respect to temperature
            = dphi_TE + dphi_TR
    dphi_TE = thermo-elastic phase derivative (dphi / dT)
    dphi_TR = thermo-refractive phase derivative (dphi / dT)
    rCoat = amplitude reflectivity of coating (complex)

    Note about aLayer: on a SiO2 substrate,
    a_Ta2O5 ~ 3.5 * alpha_Ta2O5
    a_SiO2 ~ 2.3 * alpha_SiO2

    See :getCoatTOPos: for more information.

    See LIGO-T080101.

    """
    # coating reflectivity calc
    rCoat, dcdp = getCoatRefl2(nIn, nOut, nLayer, dOpt)[:2]

    # geometrical distances
    dGeo = np.asarray(dOpt) / nLayer

    # phase derivatives
    dphi_dd = 4 * pi * dcdp

    # thermo-refractive coupling
    dphi_TR = np.sum(dphi_dd * (bLayer + sLayer * nLayer) * dGeo)

    # thermo-elastic
    dphi_TE = 4 * pi * np.sum(aLayer * dGeo)

    # total
    dphi_dT = dphi_TR + dphi_TE

    return dphi_dT, dphi_TE, dphi_TR, rCoat


def getCoatFiniteCorr(mirror, wavelength, wBeam):
    """Finite mirror size correction

    :mirror: mirror parameter Struct
    :wavelength: laser wavelength
    :wBeam: beam radius (at 1 / e**2 power)

    Uses correction factor from PLA 2003 vol 312 pg 244-255
    "Thermodynamical fluctuations in optical mirror coatings"
    by V. B. Braginsky and S. P. Vyatchanin
    http://arxiv.org/abs/cond-mat/0302617

    (see getCoatTOPos for example usage)

    version 1 by Sam Wald, 2008

    """
    # parameter extraction
    R = mirror.MassRadius
    H = mirror.MassThickness
    dOpt = mirror.Coating.dOpt

    alphaS = mirror.Substrate.MassAlpha
    C_S = mirror.Substrate.MassCM * mirror.Substrate.MassDensity
    Y_S = mirror.Substrate.MirrorY
    sigS = mirror.Substrate.MirrorSigma

    #alphaL = mirror.Coating.Alphalown
    #C_L = mirror.Coating.CVlown
    #Y_L = mirror.Coating.Ylown
    #sigL = mirror.Coating.Sigmalown
    #nL = mirror.Coating.Indexlown

    #alphaH = mirror.Coating.Alphahighn
    #C_H = mirror.Coating.CVhighn
    #Y_H = mirror.Coating.Yhighn
    #sigH = mirror.Coating.Sigmahighn
    #nH = mirror.Coating.Indexhighn

    # coating sums
   # dL = wavelength * np.sum(dOpt[::2]) / nL
    #dH = wavelength * np.sum(dOpt[1::2]) / nH
    dc = wavelength * np.sum(np.asarray(mirror.Coating.dOpt) / mirror.Coating.nN)
    nLayer=len(mirror.Coating.dOpt)
    # AVERAGE SPECIFIC HEAT (simple volume average for coating)
    Cf=np.zeros(1)
    Xf=np.zeros(1)
    Yf=np.zeros(1)
    Zf=np.zeros(1)
    xx=np.zeros(nLayer)
    yy=np.zeros(nLayer)
    zz=np.zeros(nLayer)
    for idx in range(nLayer):
        Cf += (mirror.Coating.CV[idx] * wavelength * dOpt[idx] / mirror.Coating.nN[idx]) / dc
        xx[idx] = mirror.Coating.Alpha[idx]*(1+mirror.Coating.pratN[idx]) / (1-mirror.Coating.pratN[idx])
        Xf += xx[idx] * wavelength * dOpt[idx] / mirror.Coating.nN[idx] / dc
        yy[idx] = mirror.Coating.Alpha[idx] * mirror.Coating.yN[idx]/(1-mirror.Coating.pratN[idx])
        Yf += yy[idx] * wavelength * dOpt[idx] / mirror.Coating.nN[idx] / dc
        zz[idx] = 1 / (1 - mirror.Coating.pratN[idx])
        Zf += zz[idx] * wavelength * dOpt[idx] / mirror.Coating.nN[idx] / dc
    Cr = Cf / C_S
    Xr = Xf / alphaS
    Yr = Yf / (alphaS * Y_S)
    
    # COATING AVERAGE VALUE X = ALPHAF*(1+POISSONf)/(1-POISSONf) avg
    #xxL = alphaL * (1 + sigL) / (1 - sigL)
    #xxH = alphaH * (1 + sigH) / (1 - sigH)
    #Xf = (xxL * dL + xxH * dH) / dc
    

    # COATING AVERAGE VALUE Y = ALPHAF* YOUNGSF/(1-POISSONF) avg
    #yyL = alphaL * Y_L / (1 - sigL)
    #yyH = alphaH * Y_H / (1 - sigH)
    #Yf = (yyL * dL + yyH * dH) / dc
    #Yr = Yf / (alphaS * Y_S)

    # COATING AVERAGE VALUE Z = 1/(1-POISSONF) avg
    #zzL = 1 / (1 - sigL)
    #zzH = 1 / (1 - sigH)
    #Zf = (zzL * dL + zzH * dH) / dc

    #################################### FINITE SIZE CORRECTION CALCULATION

    # beam size parameter used by Braginsky
    r0 = wBeam / sqrt(2)

    # between eq 77 and 78
    km = zeta / R
    Qm = exp(-2 * km * H)
    pm = exp(-km**2 * r0**2 / 4) / j0m # left out factor of pi * R**2 in denominator

    # eq 88
    Lm = Xr - Zf * (1 + sigS) + (Yr * (1 - 2 * sigS) + Zf - 2 * Cr) * \
         (1 + sigS) * (1 - Qm)**2 / ((1 - Qm)**2 - 4 * km**2 * H**2 * Qm)

    # eq 90 and 91
    S1 = (12 * R**2 / H**2) * np.sum(pm / zeta**2)
    S2 = np.sum(pm**2 * Lm**2)
    P = (Xr - 2 * sigS * Yr - Cr + S1 * (Cr - Yr * (1 - sigS)))**2 + S2

    # eq 60 and 70
    LAMBDA = -Cr + (Xr / (1 + sigS) + Yr * (1 - 2 * sigS)) / 2

    # eq 92
    Cfsm = sqrt((r0**2 * P) / (2 * R**2 * (1 + sigS)**2 * LAMBDA**2))

    return Cfsm


def getCoatDopt(materials, T, dL, dCap=0.5):
    """Coating layer optical thicknesses to match desired transmission

    :materials: gwinc optic materials structure
    :T: power transmission of coating
    :dL: optical thickness of low-n layers (high-n layers have dH = 0.5 - dL)
    :dCap: first layer (low-n) thickness (default 0.5)

    :returns: optical thickness array Nlayer x 1 (dOpt)

    """
    ##############################################
    def getTrans(materials, Ndblt, dL, dH, dCap, dTweak):

        # the optical thickness vector
        dOpt = np.zeros(2 * Ndblt)
        dOpt[0] = dCap
        dOpt[1::2] = dH
        dOpt[2::2] = dL

        N = dTweak.size
        T = np.zeros(N)
        for n in range(N):
            dOpt[-1] = dTweak[n]
            r = getCoatRefl(materials, dOpt)[0]
            T[n] = 1 - abs(r**2)

        return T

    ##############################################
    def getTweak(materials, T, Ndblt, dL, dH, dCap, dScan, Nfit):

        # tweak bottom layer
        Tn = getTrans(materials, Ndblt, dL, dH, dCap, dScan)
        pf = np.polyfit(dScan, Tn - T, Nfit)
        rts = np.roots(pf)
        if not any((imag(rts) == 0) & (rts > 0)):
            dTweak = None
            Td = 0
            return dTweak, Td
        dTweak = real(np.min(rts[(imag(rts) == 0) & (rts > 0)]))

        # compute T for this dTweak
        Td = getTrans(materials, Ndblt, dL, dH, dCap, np.array([dTweak]))

        return dTweak, Td

        # plot for debugging
        #   plot(dScan, [Tn - T, polyval(pf, dScan)], dTweak, Td - T, 'ro')
        #   grid on
        #   legend('T exact', 'T fit', 'T at dTweak')
        #   title(sprintf('%d doublets', Ndblt))
        #   pause(1)

    # get IFO model stuff (or create it for other functions)
    pS = materials.Substrate
    pC = materials.Coating

    nS = pS.RefractiveIndex
    nL = pC.Indexlown
    nH = pC.Indexhighn

    ########################
    # find number of quarter-wave layers required, as first guess
    nR = nH / nL
    a1 = (2 - T + 2 * sqrt(1 - T)) / (nR * nH * T)
    Ndblt = int(ceil(log(a1) / (2 * log(nR))))

    # search through number of doublets to find Ndblt
    # which gives T lower than required
    dH = 0.5 - dL
    Tn = getTrans(materials, Ndblt, dL, dH, dCap, np.array([dH]))
    while Tn < T and Ndblt > 1:
        # strange, but T is too low... remove doublets
        Ndblt = Ndblt - 1
        Tn = getTrans(materials, Ndblt, dL, dH, dCap, np.array([dH]))
    while Tn > T and Ndblt < 1e3:
        # add doublets until T > tN
        Ndblt = Ndblt + 1
        Tn = getTrans(materials, Ndblt, dL, dH, dCap, np.array([dH]))

    ########################
    # tweak bottom layer
    delta = 0.01
    dScan = np.arange(0, 0.25+delta, delta)
    dTweak = getTweak(materials, T, Ndblt, dL, dH, dCap, dScan, 5)[0]

    if not dTweak:
        if nS > nL:
            raise Exception('Coating tweak layer not sufficient since nS > nL.')
        else:
            raise Exception('Coating tweak layer not found... very strange.')

    # now that we are close, get a better result with a linear fit
    delta = 0.001
    dScan = np.linspace(dTweak - 3*delta, dTweak + 3*delta, 7)
    dTweak, Td = getTweak(materials, T, Ndblt, dL, dH, dCap, dScan, 3)

    # negative values are bad
    if dTweak < 0.01:
        dTweak = 0.01

    # check the result
    if abs(log(Td / T)) > 1e-3:
        print('Exact coating tweak layer not found... %g%% error.' % abs(log(Td / T)))

    ########################
    # return dOpt vector
    dOpt = np.zeros(2 * Ndblt)
    dOpt[0] = dCap
    dOpt[1::2] = dH
    dOpt[2::2] = dL
    dOpt[-1] = dTweak

    return dOpt


def getCoatRefl(materials, dOpt):
    """Amplitude reflectivity, with phase, of a coating

    :materials: gwinc optic materials sturcutre
    :dOpt: coating layer thickness array (Nlayer x 1)

    :returns: see return value of :geteCoatRefl2:

    """
    pS = materials.Substrate
    pC = materials.Coating

    nS = pS.RefractiveIndex
    nC = pC.nN

    Nlayer = len(dOpt)

    # refractive index of input, coating, and output materials
    nAll = np.zeros(Nlayer + 2)
    nAll[0] = 1  # vacuum input
    nAll[1::-1] = nC
    nAll[-1] = nS # substrate output

    # backend calculation
    return getCoatRefl2(nAll[0], nAll[-1], nAll[1:-1], dOpt)


def getCoatRefl2(nIn, nOut, nLayer, dOpt):
    """Coating reflection and phase derivatives

    :nIn: refractive index of input medium (e.g., vacuum = 1)
    :nOut: refractive index of output medium (e.g., SiO2 = 1.45231 @ 1064nm)
    :nLayer: refractive index of each layer, ordered input to output (N x 1)
    :dOpt: optical thickness / lambda of each layer,
           geometrical thickness * refractive index / lambda

    :returns: tuple of:
    rCoat = amplitude reflectivity of coating (complex) = rbar(0)
    dcdp = d reflection phase / d round-trip layer phase
    rbar = amplitude reflectivity of coating from this layer down
    r = amplitude reflectivity of this interface (r(1) is nIn to nLayer(1))

    See LIGO-T080101.

    """
    # Z-dir (1 => away from the substrate, -1 => into the substrate)
    zdir = 1

    # vector of all refractive indexs
    nAll = np.concatenate(([nIn], nLayer, [nOut]))

    # reflectivity of each interface
    r = (nAll[:-1] - nAll[1:]) / (nAll[:-1] + nAll[1:])

    # combine reflectivities
    rbar = np.zeros(r.size, dtype=complex)
    ephi = np.zeros(r.size, dtype=complex)

    # round-trip phase in each layer
    ephi[0] = 1
    ephi[1:] = exp(4j * zdir * pi * np.asarray(dOpt))

    rbar[-1] = ephi[-1] * r[-1]
    for n in range(len(dOpt), 0, -1):
        # accumulate reflectivity
        rbar[n-1] = ephi[n-1] * (r[n-1] + rbar[n]) / (1 + r[n-1] * rbar[n])

    # reflectivity derivatives
    dr_dphi = ephi[:-1] * (1 - r[:-1]**2) / (1 + r[:-1] * rbar[1:])**2
    dr_dphi = (1j * zdir * rbar[1:]) * np.multiply.accumulate(dr_dphi)

    # shift rbar index
    rCoat = rbar[0]
    rbar = rbar[1:]

    # phase derivatives
    dcdp = -imag(dr_dphi / rCoat)  ### Where did this minus come from???

    return rCoat, dcdp, rbar, r


def getCoatReflAndDer(nN, nsub, dOpt):
    '''
    Helper function for coating_brownian_hong().
    Follows Hong et al . PRD 87, 082001 (2013) Sec V.A.
    This function calculates derivatives of complex reflectivity of Coating
    with respect to phase shifts through each layer and reflectivities of
    each interface
    Input:

      nN = Refractive indices of coatings layers
    nsub = Refractive Index of Substrate
    dOpt = optical thickness / lambda of each layer,
           geometrical thickness * refractive index / lambda

    Returns:
     delLogRho_delPhik = Partial derivative of log of total effective
                         reflectivity of coating with respect to phase shifts
                         in each layer.
    delLogRho_delReflk = Partial derivative of log of total effective
                         reflectivity of coating with respect to reflectivity
                         of each interface.
    '''
    nol = len(dOpt)  # Number of layers in coating
    # Reflectivities and transmitivities
    # r[j] is reflectivity from (j-1)th and (j)th layer interface
    # Here r[0] is reflectivity from air and 0th layer
    # and r[-1] is reflectivity between last layer and substrate
    Refl = np.zeros(nol+1)
    Refl[0] = (1 - nN[0]) / (1 + nN[0])
    Refl[1:-1] = (nN[:-1] - nN[1:]) / (nN[:-1] + nN[1:])
    Refl[-1] = (nN[-1] - nsub) / (nN[-1] + nsub)
    # Note the shift from nomenclature
    # Phi is reserved for denoting one-way phase shift suffered by light
    # during propagation through a layer
    Phi = np.asarray(dOpt) * 2 * pi

    # Define rho_k as reflectivity of
    # k layers starting from (N-k-1)th lyer to (N-1)th layer
    # So rhoN[-1] is reflectivity for  no layers but interface from
    #                                              last layer to substrate
    # rhoN[0] is total complex reflectivity of the coating stack.
    rhoN = np.zeros_like(Refl, np.complex128)

    phiNmkm1 = np.flipud(Phi)             # phi_{N-k-1}
    rNmkm1 = np.flipud(Refl[:-1])              # r_{N-k-1}
    exp2iphiNmkm1 = np.exp(2j*phiNmkm1)      # exp(2i phi_{N-k-1})

    # Recursion relation for complex reflectivity
    # See https://dcc.ligo.org/T2000552 for derivation
    rhoN[0] = Refl[-1]
    for k in range(len(Refl)-1):
        rhoN[k+1] = ((rNmkm1[k] + exp2iphiNmkm1[k] * rhoN[k])
                     / (1 + exp2iphiNmkm1[k] * rNmkm1[k] * rhoN[k]))

    denTerm = (1 + exp2iphiNmkm1 * rNmkm1 * rhoN[:-1])**2

    # Derivatives of rho_{k+1} wrt to rho_{k}, r_{N-k-1} and phi_{N-k-1}
    delRhokp1_delRhok = exp2iphiNmkm1 * (1 - rNmkm1**2) / denTerm
    delRhokp1_delRNmkm1 = np.append(1, ((1 - (exp2iphiNmkm1*rhoN[:-1])**2)
                                        / denTerm))
    delRhokp1_delPhiNmkm1 = np.append(0, -2j * rhoN[:-1] * delRhokp1_delRhok)

    # Derivative of rho_{N} wrt to rho_{N-j}
    delRhoN_delRhoNmj = np.append(1, np.cumprod(np.flipud(delRhokp1_delRhok)))

    # Derivative of rho_{N} wrt to r_k and phi_k
    delRho_delRk = - delRhoN_delRhoNmj * np.flipud(delRhokp1_delRNmkm1)
    delRho_delPhik = - delRhoN_delRhoNmj * np.flipud(delRhokp1_delPhiNmkm1)
    delLogRho_delReflk = delRho_delRk / rhoN[-1]
    delLogRho_delPhik = delRho_delPhik / rhoN[-1]
    delLogRho_delPhik[-1] = 0        # Define this as per Eq (26)

    return rhoN[-1], delLogRho_delPhik, delLogRho_delReflk, Refl


def interpretLossAngles(coat):
    '''
    Helper function for coating_brownian().

    Creates function from 100 Hz value of loss angle and is logarithmic
    slope.

    if separate bulk and shear loss angles are not provided as
    lossBhighn, lossShighn, lossBlown and lossSlown
    then earlier version names of Phihighn and Philown are searched for and
    used as Bulk loss angle while setting Shear loss angles to zero.

    Input argument:
    coat = Coating object containing loss angle values or expressions.
    Returns:
    lossB = Coating Bulk Loss Angle
    lossS = Coating Shear Loss Angle
    '''
    if 'lossB' in coat and 'lossS' in coat:
        if 'lossB_slope' in coat:
            def lossB(f):
                return coat.lossB * (f / 100)**coat.lossB_slope
        else:
            def lossB(f): return coat.lossB
        if 'lossS_slope' in coat:
            def lossS(f):
                return coat.lossS * (f / 100)**coat.lossS_slope
        else:
            def lossS(f): return coat.lossS
    else:
        # Use Phihighn if specific Bulk & Shear loss angles not provided
        if 'Phi_slope' in coat:
            def Phi(f):
                return coat.Phi * (f / 100)**coat.Phi
            lossB = lossS = Phi
        else:
            lossB = lossS = lambda f: coat.Phi

    return lossB, lossS
