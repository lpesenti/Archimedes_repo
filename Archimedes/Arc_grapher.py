import Arc_common as ac
import matplotlib.pyplot as plt
import numpy as np

arc_x, arc_y = np.loadtxt(r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Archimedes\Results\Arc_article.txt',
                          unpack=True, usecols=[0, 1], skiprows=1)
virgo_x, virgo_y = np.loadtxt(
    r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Archimedes\Results\Arc_VirgoData_Jul2019.txt', unpack=True,
    usecols=[0, 1])

# x_sn = np.arange(arc_x.min(), arc_x.max())

y_sn = ac.shot_noise()
y_rpn = ac.radiation_pressure_noise(freq=arc_x)
y_stn = ac.suspension_thermal_noise(freq=arc_x)
y_itn = ac.internal_thermal_noise(freq=arc_x)

plt.plot(arc_x, arc_y, linestyle='-', linewidth=2, label='@ SAR-GRAV')
plt.plot(virgo_x, virgo_y, linestyle='-', linewidth=2, label='@ Virgo')
plt.plot(arc_x, np.repeat(y_sn, arc_x.size), linestyle='-', linewidth=2, label='Shot noise')
plt.plot(arc_x, y_rpn, linestyle='-', linewidth=2, label='Radiation pressure noise')
plt.plot(arc_x, y_stn, linestyle='-', linewidth=2, label='Suspension thermal noise')
plt.plot(arc_x, y_itn, linestyle='-', linewidth=2, label='Internal thermal noise')

plt.xlabel("Frequency (Hz)", fontsize=22)
plt.ylabel(r"ASD [rad/$\sqrt{Hz}$]", fontsize=22)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle='--', which='both')
plt.legend(loc='best', shadow=True, fontsize='x-large')

manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show()
