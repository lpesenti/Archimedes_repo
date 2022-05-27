import os
from gwinc import nb, const
from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE
from susth import STNRmodal
from susth import STNViol
from envnoise import (
        atmospheric_noise,
        cavern_noise,
        body_wave,
        rayleigh_wave,
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
        #STNpy return PSD
        _, noise = STNRmodal(self.freq,self.ifo.Suspension,self.ifo)
        violin = STNViol(self.freq, self.ifo.Suspension, self.ifo)
        #turn into displacement PSD
        return (noise+violin).real

class NewtonianBodyWave(nb.Noise):
    style = dict(
        label = 'Body Wave',
        )
    def calc(self):
        noise = body_wave(self.freq,self.ifo.Seismic)**2/self.ifo.Seismic.Omicron**2
        return noise

class NewtonianRayleighWave(nb.Noise):
    style = dict(
        label = 'Rayleigh Wave',
        )
    def calc(self):
        noise = rayleigh_wave(self.freq,self.ifo.Seismic)**2/self.ifo.Seismic.Omicron**2
        return noise

class NewtonianCavern(nb.Noise):
    style = dict(
        label = 'Cavern',
        )
    def calc(self):
        noise = cavern_noise(self.freq,self.ifo.Seismic)**2/self.ifo.Seismic.Omicron**2
        return noise

class NewtonianAtmospheric(nb.Noise):
    style = dict(
        label = 'Atmospheric',
        )
    def calc(self):
        noise = atmospheric_noise(self.freq,self.ifo.Seismic)**2/self.ifo.Seismic.Omicron**2
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
    #style = dict(
    #    label = 'Excess Gas',
    #    color='#add00d',
    #    linestyle='--',
    #    )
    #def calc(self):
     #   n = noise.residualgas.residual_gas_cavity(self.freq, self.ifo)
     #   dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
     #   dam=calc_x_noise(self.freq,S_F_cavalleri(self.ifo),self.ifo)#
      #  return n * 2 / sinc_sqr + dam


class ETHF(nb.Budget):

    name = 'ETHF'

    noises = [
        QuantumVacuum,
        Seismic,
        Newtonian,
        SusThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        SubstrateBrownian,
        SubstrateThermoElastic,
        ExcessGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
