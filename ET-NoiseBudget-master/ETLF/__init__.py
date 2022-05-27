from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE
from gwinc.ifo.noises import arm_cavity
from gwinc.ifo.noises import ifo_power
from susth import STNRmodal
from susth import STNViol
from thermoelastic import substratethermoelastic
from envnoise import (
        atmospheric_noise,
        cavern_noise,
        body_wave,
        rayleigh_wave,
        seismic_noise
        )

#newtonian_mitigation_factor = 3

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
        noise = body_wave(self.freq,self.ifo.Seismic)**2/ self.ifo.Seismic.Omicron**2
        return noise

class NewtonianRayleighWave(nb.Noise):
    style = dict(
        label = 'Rayleigh Wave',
        )
    def calc(self):
        noise = rayleigh_wave(self.freq,self.ifo.Seismic)**2/ self.ifo.Seismic.Omicron**2
        return noise

class NewtonianCavern(nb.Noise):
    style = dict(
        label = 'Cavern',
        )
    def calc(self):
        noise = cavern_noise(self.freq,self.ifo.Seismic)**2/ self.ifo.Seismic.Omicron**2
        return noise

class NewtonianAtmospheric(nb.Noise):
    style = dict(
        label = 'Atmospheric',
        )
    def calc(self):
        noise = atmospheric_noise(self.freq,self.ifo.Seismic)**2/ self.ifo.Seismic.Omicron**2
        return noise

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

class ETLF(nb.Budget):

    name = 'ETLF'

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

