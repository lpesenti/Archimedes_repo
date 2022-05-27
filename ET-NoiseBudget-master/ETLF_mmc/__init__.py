from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE
from gwinc.ifo.noises import arm_cavity
from gwinc.ifo.noises import ifo_power
from susth import STNRmodal
from susth import STNViol
import coatingth
#from coatingth import coating_brownian
#from coatingth import coating_thermooptic
#from coatingth import coating_brownian

from thermoelastic import substratethermoelastic
from envnoise import (
        atmospheric_noise,
        cavern_noise,
        body_wave,
        rayleigh_wave,
        seismic_noise
        )

newtonian_mitigation_factor = 3


def mirror_struct(ifo, tm):
    """Create a "mirror" Struct for a LIGO core optic

    This is a copy of the ifo.Materials Struct, containing Substrate
    and Coating sub-Structs, as well as some basic geometrical
    properties of the optic.

    """
    # NOTE: we deepcopy this struct since we'll be modifying it (and
    # it would otherwise be stored by reference)
    mirror = copy.deepcopy(ifo.Materials)
    optic = ifo.Optics.get(tm)
    coatingth.build_stacks(mirror, optic)
    mirror.update(optic)
    mirror.MassVolume = pi * mirror.MassRadius**2 * mirror.MassThickness
    mirror.MirrorMass = mirror.MassVolume * mirror.Substrate.MassDensity
    return mirror
    

class QuantumVacuum(nb.Budget):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    noises = [
        QuantumVacuumAS,
        QuantumVacuumArm,
        QuantumVacuumSEC,
        QuantumVacuumFilterCavity,
        QuantumVacuumInjection,
        QuantumVacuumReadout,
        QuantumVacuumQuadraturePhase,
    ]
    
class CoatingBrownian(nb.Noise):
    """Coating Brownian

    """
    style = dict(
        label='Coating Brownian',
        color='#fe0002',
    )

    def calc(self):
        ITM = mirror_struct(self.ifo, 'ITM')
        ETM = mirror_struct(self.ifo, 'ETM')
        cavity = arm_cavity(self.ifo)
        wavelength = self.ifo.Laser.Wavelength
        nITM = coatingth.coating_brownian(
            self.freq, ITM, wavelength, cavity.wBeam_ITM
        )
        nETM = coatingth.coating_brownian(
            self.freq, ETM, wavelength, cavity.wBeam_ETM
        )
        return (nITM + nETM) * 2


class CoatingThermoOptic(nb.Noise):
    """Coating Thermo-Optic

    """
    style = dict(
        label='Coating Thermo-Optic',
        color='#02ccfe',
        linestyle='--',
    )

    def calc(self):
        wavelength = self.ifo.Laser.Wavelength
        materials = self.ifo.Materials
        ITM = mirror_struct(self.ifo, 'ITM')
        ETM = mirror_struct(self.ifo, 'ETM')
        cavity = arm_cavity(self.ifo)
        nITM, junk1, junk2, junk3 = coatingth.coating_thermooptic(
            self.freq, ITM, wavelength, cavity.wBeam_ITM,
        )
        nETM, junk1, junk2, junk3 = coatingth.coating_thermooptic(
            self.freq, ETM, wavelength, cavity.wBeam_ETM,
        )
        return (nITM + nETM) * 2


class SusThermal(nb.Noise):
    style = dict(
        label = 'Suspension Thermal',
        color='#0d75f8',
        )
    def calc(self):
        _,noise = STNRmodal(self.freq, self.ifo.Suspension, self.ifo)
        violin = STNViol(self.freq, self.ifo.Suspension, self.ifo)
        return (noise+violin).real

class SubThermalElastic(nb.Noise):
    style = dict(
        label = 'Substrate Thermal Elastic',
        color='#f5bf03',
        linestyle='--',
        )
    def calc(self):
        cavity = arm_cavity(self.ifo)
        nITM = substratethermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        nETM = substratethermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ETM)
        return (nITM + nETM) * 2
        
class Seismic(nb.Noise):
    style = dict(
        label = 'Seismic',
        color='#855700'
        )
    def calc(self):
        noise = seismic_noise(self.freq,self.ifo.Seismic)**2
        return noise

class NewtonianBodyWave(nb.Noise):
    style = dict(
        label = 'Body Wave',
        )
    def calc(self):
        noise = body_wave(self.freq,self.ifo.Seismic)**2
        return noise / newtonian_mitigation_factor**2

class NewtonianRayleighWave(nb.Noise):
    style = dict(
        label = 'Rayleigh Wave',
        )
    def calc(self):
        noise = rayleigh_wave(self.freq,self.ifo.Seismic)**2
        return noise / newtonian_mitigation_factor**2

class NewtonianCavern(nb.Noise):
    style = dict(
        label = 'Cavern',
        )
    def calc(self):
        noise = cavern_noise(self.freq,self.ifo.Seismic)**2
        return noise / newtonian_mitigation_factor**2

class NewtonianAtmospheric(nb.Noise):
    style = dict(
        label = 'Atmospheric',
        )
    def calc(self):
        noise = atmospheric_noise(self.freq,self.ifo.Seismic)**2
        return noise / newtonian_mitigation_factor**2

class NewtonianNoise(nb.Budget):
    """Newtonian"""
    style = dict(
        label = 'Newtonian Gravity',
        color='#15b01a'
        )
    noises = [
            NewtonianBodyWave,
            NewtonianRayleighWave,
            NewtonianCavern,
            NewtonianAtmospheric
            ]
            
#class ResidualGas(nb.Noise):
 #   style = dict(
 #       label = 'Excess Gas',
 #       color='#add00d',
 #       linestyle='--',
 #       )
 #   def calc(self):
  #      n = noise.residualgas.residual_gas_cavity(self.freq, self.ifo)
  #      dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
  #      dam=calc_x_noise(self.freq,S_F_cavalleri(self.ifo),self.ifo)
  #      return n * 2 / sinc_sqr + dam
class ITMThermoRefractive(nb.Noise):

    style = dict(
        label='ITM Thermo-Refractive',
        color='#448ee4',
        linestyle='--',
    )

    def calc(self):
        power = ifo_power(self.ifo)
        gPhase = power.finesse * 2/np.pi
        cavity = arm_cavity(self.ifo)
        n = noise.substratethermal.substrate_thermorefractive(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM, exact=True)
        return n * 2 / gPhase**2

class ETLF_mmc(nb.Budget):

    name = 'ETLF_mmc'

    noises = [
        QuantumVacuum,
        Seismic,
        NewtonianNoise,
        SusThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        SubstrateBrownian,
        SubThermalElastic,
        ITMThermoRefractive,
        ExcessGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE


    freq = '1:3000:4000'

