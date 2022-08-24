__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Davide Rozza"]
__version__ = "1.0.0"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Production"

r"""
This script produces the plot of the Archimedes' official sensitivity curve compared to the main noise sources together 
with the results obtained at the Virgo site.
For a more details see 'Commissioning and data analysis of the Archimedes experiment and its prototype at the SAR-GRAV 
laboratory (Chapter 3 and 4)' (https://drive.google.com/file/d/1tyJ8PX4Giby3LttXn6AAxVaf7s0vkJkp/view?usp=sharing) and
'Picoradiant tiltmeter and direct ground tilt measurements at the Sos Enattos site'
(https://doi.org/10.1140/epjp/s13360-021-01993-w9.
"""

import Arc_common as ac
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

arc_x, arc_y = np.loadtxt(r'.\Results\Arc_article.txt',
                          unpack=True, usecols=[0, 1], skiprows=1)
virgo_x, virgo_y = np.loadtxt(
    r'.\Results\Arc_VirgoData_Jul2019.txt', unpack=True,
    usecols=[0, 1])
x_crio, y_crio = np.loadtxt(r'D:\Archimedes\Data\cRIO\ADCnoise_1kHz.txt', unpack=True, usecols=[0, 1])

# x_sn = np.arange(arc_x.min(), arc_x.max())

y_sn = ac.shot_noise()
y_rpn = ac.radiation_pressure_noise(freq=arc_x)
y_stn = ac.suspension_thermal_noise(freq=arc_x)
y_itn = ac.internal_thermal_noise(freq=arc_x)

fig = plt.figure(figsize=(19.2, 10.8))
ax = fig.add_subplot()

ax.plot(virgo_x, virgo_y, linestyle='-', color='tab:red', linewidth=3, label='Tiltmeter @ Virgo')
ax.plot(arc_x, arc_y, linestyle='-', color='tab:blue', linewidth=3, label='Tiltmeter @ SAR-GRAV')
ax.plot(x_crio, y_crio, linestyle='-', linewidth=2, label='cRIO noise')
ax.plot(arc_x, np.repeat(y_sn, arc_x.size), linestyle='-', linewidth=2.5, label='Shot noise')
ax.plot(arc_x, y_rpn, linestyle='-', linewidth=2.5, label='Radiation pressure noise')
ax.plot(arc_x, y_stn, linestyle='-', linewidth=2.5, label='Suspension thermal noise')
ax.plot(arc_x, y_itn, linestyle='-', linewidth=2.5, label='Internal thermal noise')

ax.set_xlabel("Frequency (Hz)", fontsize=24)
ax.set_ylabel(r"ASD [rad/$\sqrt{Hz}$]", fontsize=24)
ax.tick_params(axis='both', which='minor', labelsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
# ax.set_xlim([2, 20])
# ax.set_ylim([1e-13, 1e-8])
ax.set_xscale("log")
ax.set_yscale("log")
# ax.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', which='both', axis='both')
ax.legend(loc='best', shadow=True, fontsize=24)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.0f'))
fig.savefig(r'D:\Archimedes\Images\Article_sensitivity.svg')
fig.savefig(r'D:\Archimedes\Images\Article_sensitivity.png', dpi=1200)
fig.tight_layout()

plt.show()
