import Arc_common as ac
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

arc_x, arc_y = np.loadtxt(r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Archimedes\Results\Arc_article.txt',
                          unpack=True, usecols=[0, 1], skiprows=1)
virgo_x, virgo_y = np.loadtxt(
    r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Archimedes\Results\Arc_VirgoData_Jul2019.txt', unpack=True,
    usecols=[0, 1])
x_crio, y_crio = np.loadtxt(r'D:\Archimedes\Data\cRIO\ADCnoise_1kHz.txt', unpack=True, usecols=[0, 1])

# x_sn = np.arange(arc_x.min(), arc_x.max())

y_sn = ac.shot_noise()
y_rpn = ac.radiation_pressure_noise(freq=arc_x)
y_stn = ac.suspension_thermal_noise(freq=arc_x)
y_itn = ac.internal_thermal_noise(freq=arc_x)

fig = plt.figure(figsize=(19.2, 10.8))
ax = fig.add_subplot()

ax.plot(virgo_x, virgo_y, linestyle='-', color='tab:orange', linewidth=3, label='@ Virgo')
ax.plot(arc_x, arc_y, linestyle='-', color='tab:blue', linewidth=3, label='@ SAR-GRAV')
# ax.plot(x_crio, y_crio, linestyle='-', linewidth=2, label='cRIO noise')
# ax.plot(arc_x, np.repeat(y_sn, arc_x.size), linestyle='-', linewidth=2.5, label='Shot noise')
# ax.plot(arc_x, y_rpn, linestyle='-', linewidth=2.5, label='Radiation pressure noise')
# ax.plot(arc_x, y_stn, linestyle='-', linewidth=2.5, label='Suspension thermal noise')
# ax.plot(arc_x, y_itn, linestyle='-', linewidth=2.5, label='Internal thermal noise')

ax.set_xlabel("Frequency (Hz)", fontsize=24)
ax.set_ylabel(r"ASD [rad/$\sqrt{Hz}$]", fontsize=24)
ax.tick_params(axis='both', which='minor', labelsize=22)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.set_xlim([2, 20])
ax.set_ylim([1e-13, 1e-8])
# ax.set_xscale("log")
ax.set_yscale("log")
# ax.ticklabel_format(style='plain')
ax.grid(True, linestyle='--', which='both', axis='both')
ax.legend(loc='best', shadow=True, fontsize=24)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.0f'))

fig.tight_layout()

plt.show()
