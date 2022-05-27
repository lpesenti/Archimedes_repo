import numpy as np
import gwinc.noise.seismic
from scipy import constants
from scipy.interpolate import interp1d

# Interpolate spectrum data in log-space
def loginterp(x, y, f):
    return np.power(10, np.interp(np.log10(f),np.log10(x),np.log10(y)))
    #10**interp1d(np.log10(x), np.log10(y), kind="slinear", fill_value="extrapolate")(np.log10(f))

################################################################################
## Spectrum data
################################################################################
def spectrum_bodywave(f,Seismic):
    dataSosEnattos = np.loadtxt("acoustic_spectra/bodywave_spectrum_SosEnattos.txt")
    dataSosTerziet = np.loadtxt("acoustic_spectra/bodywave_spectrum_Terziet.txt")
    
    if Seismic.Site=='ET':
        bodywave=(5 * gwinc.noise.seismic.seismic_ground_NLNM(f))**2
    elif Seismic.Site =='SosEnattos':
        bodywave=loginterp(dataSosEnattos.T[0],dataSosEnattos.T[1],f)
    elif Seismic.Site =='Terziet':
        bodywave=loginterp(dataSosTerziet.T[0],dataSosTerziet.T[1],f)
    return bodywave

def spectrum_rayleigh_horizontal(f,Seismic):
    dataSosEnattos = np.loadtxt("acoustic_spectra/rwave_spectrum_SosEnattos.txt")
    dataTerziet = np.loadtxt("acoustic_spectra/rwave_spectrum_Terziet.txt")
    
    if Seismic.Site=='ET':
        rayleighwave=(10**(
            0.5 * (
                np.log10(gwinc.noise.seismic.seismic_ground_NLNM(f))
                + np.log10(gwinc.noise.seismic.seismic_ground_NHNM(f))
                )
            )
            )**2
    elif Seismic.Site=='SosEnattos':
        rayleighwave=loginterp(dataSosEnattos.T[0],dataSosEnattos.T[1],f)
        
    elif Seismic.Site=='Terziet':
        rayleighwave=loginterp(dataTerziet.T[0],dataTerziet.T[1],f)
    return rayleighwave


def spectrum_rayleigh_vertical(f,Seismic):
    return spectrum_rayleigh_horizontal(f,Seismic)

def spectrum_rayleigh_tilt(f,Seismic):
    return 2 * np.pi * f / spectrum_rayleigh_dispersion(f,Seismic) * spectrum_rayleigh_vertical(f,Seismic)


def spectrum_rayleigh_dispersion(f,Seismic):
    dataSosEnattos = np.loadtxt("acoustic_spectra/rwave_dispersion_SosEnattos.txt")
    dataTerziet = np.loadtxt("acoustic_spectra/rwave_dispersion_Terziet.txt")

    if Seismic.Site=='ET':
        rayleigh_dispersion=2000 * np.exp(-f/4) + 300
    elif Seismic.Site=='SosEnattos':
        rayleigh_dispersion=loginterp(dataSosEnattos.T[0],dataSosEnattos.T[1],f)
    elif Seismic.Site=='Terziet':
        rayleigh_dispersion=loginterp(dataTerziet.T[0],dataTerziet.T[1],f)

    return rayleigh_dispersion  # Rayleigh wave dispersion (eqn. 1)


def isolation_H2H(f):
    data = np.loadtxt("sus_tf.txt")
    return loginterp(data.T[0], data.T[1], f)

def isolation_V2H(f):
    data = np.loadtxt("sus_tf.txt")
    v2h = loginterp(data.T[0], data.T[2], f)
    return v2h * 0.0016

def isolation_tilt2H(f):
    data = np.loadtxt("sus_tf.txt")
    return loginterp(data.T[0], data.T[3], f)


def spectrum_pressure_atmosphere(f):
    data = np.loadtxt("acoustic_spectra/atmosphere.txt")
    return loginterp(data.T[0], data.T[1], f)

def spectrum_pressure_cavern(f):
    data = np.loadtxt("acoustic_spectra/cavern.txt")
    return loginterp(data.T[0], data.T[1], f)

################################################################################
## Noise sources
## All equation numbers refer to those in https://arxiv.org/abs/2003.03434
################################################################################
def body_wave(f,Seismic):
    p = 0.33  # Fraction of body wave spectral density caused by compressional waves
    rock_density = 3e3  # kg / m^3
    Sh =  (4/3 * np.pi * constants.G * rock_density)**2 * (3*p + 1) * spectrum_bodywave(f,Seismic) * 4 / (2*np.pi*f)**4  # Equation 7
    return np.sqrt(Sh)

def rayleigh_wave(f,Seismic):
    vr = spectrum_rayleigh_dispersion(f,Seismic)
    vs = 1.1 * vr  # Shear wave dispersion TODO: pulled from slide 16 of http://rses.anu.edu.au/~nick/teachdoc/lecture5.pdf, find better source
    vp = 2 * vr  # TODO: quick guess
    kr = 2 * np.pi * f / vr

    qp = 2 * np.pi * f * np.sqrt(1 / vr**2 - 1 / vp**2)
    qs = 2 * np.pi * f * np.sqrt(1 / vr**2 - 1 / vs**2)
    zeta = np.sqrt(qp / qs)

    h = -Seismic.Height  # Detector depth in m
    gamma = 0.8  # Factor quantifying cancellation of newtonian noise
    density_surface = 2e3  # Density of surface in kg / m^3

    r0 = kr * (1 - zeta)  # eq. 3
    sh = -kr * (1 + zeta) * np.exp(-kr * h)  # eq. 4
    bh = 2/3 * (2 * kr * np.exp(-qp * h) + zeta * qs * np.exp(-qs * h))  # eq. 5
    R = np.abs((sh + bh) / r0)**2  # eq. 6
    SR = (2 * np.pi / np.sqrt(2) * gamma * constants.G * density_surface)**2 * R * spectrum_rayleigh_vertical(f,Seismic) * 4 / (2 * np.pi * f)**4  # Equation 2

    return np.sqrt(SR)

def seismic_noise(f,Seismic):
    vr = spectrum_rayleigh_dispersion(f,Seismic)
    vs = 1.1 * vr  # Shear wave dispersion TODO: pulled from slide 16 of http://rses.anu.edu.au/~nick/teachdoc/lecture5.pdf, find better source
    vp = 2 * vr  # TODO: quick guess
    kr = 2 * np.pi * f / vr

    qp = 2 * np.pi * f * np.sqrt(1 / vr**2 - 1 / vp**2)
    qs = 2 * np.pi * f * np.sqrt(1 / vr**2 - 1 / vs**2)
    zeta = np.sqrt(qp / qs)

    h = -Seismic.Height  # Detector depth

    xi0_ver = qp - zeta * kr
    xi0_hor = kr - zeta * qs

    xi_hor = (kr * np.exp(-qp * h) - zeta * qs * np.exp(-qs * h)) / xi0_hor
    xi_ver = 1j*(qp * np.exp(-qp * h) - zeta * kr * np.exp(-qs * h)) / xi0_ver

    return np.sqrt(
        (
            (np.abs(xi_hor)**2 * spectrum_rayleigh_horizontal(f,Seismic) + spectrum_bodywave(f,Seismic)) * np.abs(isolation_H2H(f))**2
            + (np.abs(xi_ver)**2 * spectrum_rayleigh_vertical(f,Seismic) + spectrum_bodywave(f,Seismic)) * np.abs(isolation_V2H(f))**2
            + (np.abs(xi_ver)**2 * spectrum_rayleigh_tilt(f,Seismic)) * np.abs(isolation_tilt2H(f))**2
        )
        * 4
    )

def atmospheric_noise(f,Seismic):
    cs = 340  # Speed of sound in m/s
    rho0 = 1.225  # Density of air in kg/m^3
    p0 = 101325  # Air pressure in Pa
    gamma = 1.4  # Adiabatic coefficient of air
    h = -Seismic.Height  # Depth underground in m
    coupling = 3 / (4 * np.pi * f * h / cs)**4  # Approximate isotropically averaged coupling coefficient (eqn. 12)

    return np.sqrt((2 * cs * constants.G * rho0 * spectrum_pressure_atmosphere(f) / (p0 * gamma * f))**2 * coupling * 4 / (2 * np.pi * f)**4) # Eqn. 10

def cavern_noise(f,Seismic):
    cs = 340  # Speed of sound in m/s
    rho0 = 1.225  # Density of air in kg/m^3
    p0 = 101325  # Air pressure in Pa

    gamma = 1.4  # Adiabatic coefficient of air
    h = -Seismic.Height  # Depth underground in m
    R = Seismic.CavernR  # Cavern radius in m
    coupling = 1/3 * (1 - np.sinc(2 * f * R / cs))**2  # Eqn. 13 without the pi as numpy defines sinc(x) = sin(pi * x) / (pi * x)

    return np.sqrt((2 * cs * constants.G * rho0 * spectrum_pressure_cavern(f) / (p0 * gamma * f))**2 * coupling * 4 / (2 * np.pi * f)**4)  # Eqn. 13
