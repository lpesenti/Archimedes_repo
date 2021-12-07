import pandas as pd
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt

# FREQUENCY AND PSD WINDOW
fs = 1000  # if using oscilloscope read first row and do "1 / 2.000000e-04"
Num = int(60 * fs)

# READING DATA
# If using cRIO uncomment next line

df = pd.read_table(r'C:\Users\Arc_MDC\Documents\Optical_Lever\Data\OL_Dark_EPZ.lvm', sep='\t', usecols=[0, 1, 2, 3],
                   header=None)
df1 = pd.read_table(r'C:\Users\Arc_MDC\Documents\Optical_Lever\Data\OL_Dark_GPZ_SoE.lvm', sep='\t', usecols=[0, 1, 2, 3],
                    header=None)

epz_delta_y = df[1].values.flatten()
epz_delta_x = df[2].values.flatten()
epz_sum_ch = df[3].values.flatten()

gpz_delta_y = df1[1].values.flatten()
gpz_delta_x = df1[2].values.flatten()
gpz_sum_ch = df1[3].values.flatten()

# If using oscilloscope uncomment next lines

# df_col = pd.read_csv(r'D:\Archimedes\Data\Optical_Lever\Noise\C1Noise00000.csv', nrows=1, header=None).columns
# data_df = pd.read_csv(r'D:\Archimedes\Data\Optical_Lever\Noise\C1Noise00000.csv', usecols=[df_col[-1:][0]], header=None)
# data_df1 = pd.read_csv(r'D:\Archimedes\Data\Optical_Lever\Noise\C2Noise00000.csv', usecols=[df_col[-1:][0]], header=None)
# data_df2 = pd.read_csv(r'D:\Archimedes\Data\Optical_Lever\Noise\C3Noise00000.csv', usecols=[df_col[-1:][0]], header=None)
# data = data_df.values.flatten()
# data1 = data_df1.values.flatten()
# data2 = data_df2.values.flatten()

# PSD EVALUATION
# Creating frequency array
_, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs)  # ,noverlap=NOL)
f = f[1:]  # remove first value that is 0

# If using cRIO uncomment next lines
# Delta y channel
epz_psd_y, _ = mlab.psd(epz_delta_y, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
epz_psd_y = epz_psd_y[1:]
epz_asd_y = np.sqrt(epz_psd_y)

gpz_psd_y, _ = mlab.psd(gpz_delta_y, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
gpz_psd_y = gpz_psd_y[1:]
gpz_asd_y = np.sqrt(gpz_psd_y)

# Delta x channel
epz_psd_x, _ = mlab.psd(epz_delta_x, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
epz_psd_x = epz_psd_x[1:]
epz_asd_x = np.sqrt(epz_psd_x)

gpz_psd_x, _ = mlab.psd(gpz_delta_x, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
gpz_psd_x = gpz_psd_x[1:]
gpz_asd_x = np.sqrt(gpz_psd_x)

# Sum channel
epz_psd_sum, _ = mlab.psd(epz_sum_ch, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
epz_psd_sum = epz_psd_sum[1:]
epz_asd_sum = np.sqrt(epz_psd_sum)

gpz_psd_sum, _ = mlab.psd(gpz_sum_ch, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
gpz_psd_sum = gpz_psd_sum[1:]
gpz_asd_sum = np.sqrt(gpz_psd_sum)

# If using oscilloscope uncomment next lines

# s, _ = mlab.psd(data, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
# s = s[1:]
# s = np.sqrt(s)
#
# s1, _ = mlab.psd(data1, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
# s1 = s1[1:]
# s1 = np.sqrt(s1)
#
# s2, _ = mlab.psd(data2, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
# s2 = s2[1:]
# s2 = np.sqrt(s2)

# DATA VISUALIZATION

fig, ax1 = plt.subplots()

# If using cRIO uncomment next lines

ax1.plot(f, gpz_asd_y, label=r"$\Delta$y - OP27GPZ")
ax1.plot(f, gpz_asd_x, label=r"$\Delta$x - OP27GPZ")
ax1.plot(f, gpz_asd_sum, label=r"$\Sigma$ - OP27GPZ")
ax1.plot(f, epz_asd_y, label=r"$\Delta$y - OP27EPZ")
ax1.plot(f, epz_asd_x, label=r"$\Delta$x - OP27EPZ")
ax1.plot(f, epz_asd_sum, label=r"$\Sigma$ - OP27EPZ")

# If using oscilloscope uncomment next lines

# ax1.plot(f, s, label="$\Delta$y")
# ax1.plot(f, s1, label="$\Delta$x")
# ax1.plot(f, s2, label="$\Sigma$")
legend = ax1.legend(loc='best', shadow=True)  # , fontsize='x-large')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True)
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel(r'ASD [V/$\sqrt{Hz}$]')

# RMS COMPARISON

# Delta y channel
fig_comp_y = plt.figure()
gs_y = fig_comp_y.add_gridspec(2, hspace=0.15)  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
axs_y = gs_y.subplots(sharex=True)

axs_y[0].set_xlabel('Frequency [Hz]', fontsize=20)
axs_y[0].set_xscale("log")
axs_y[0].set_yscale("log")
axs_y[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
axs_y[0].grid(True, linestyle='--')

axs_y[1].set_xlabel('Frequency [Hz]', fontsize=20)
axs_y[1].set_xscale("log")
# axs_y[1].set_yscale("log")
axs_y[1].set_ylabel(r'$\frac{|a-b|}{a+b}$ [V/$\sqrt{Hz}$]', fontsize=20)
axs_y[1].grid(True, linestyle='--')

axs_y[0].plot(f, gpz_asd_y, linestyle='-', label=r'$\Delta y$ - OP27GPZ')
axs_y[0].plot(f, epz_asd_y, linestyle='-', label=r'$\Delta y$ - OP27EPZ')
axs_y[1].plot(f, (gpz_asd_y - epz_asd_y) / (gpz_asd_y + epz_asd_y), linestyle='-', label=r'$\frac{|a-b|}{a+b}$')
axs_y[0].legend(loc='best', shadow=True, fontsize='medium')
axs_y[1].legend(loc='best', shadow=True, fontsize='medium')

# Delta x channel

fig_comp_x = plt.figure()
gs_x = fig_comp_x.add_gridspec(2, hspace=0.15)  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
axs_x = gs_x.subplots(sharex=True)

axs_x[0].set_xlabel('Frequency [Hz]', fontsize=20)
axs_x[0].set_xscale("log")
axs_x[0].set_yscale("log")
axs_x[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
axs_x[0].grid(True, linestyle='--')

axs_x[1].set_xlabel('Frequency [Hz]', fontsize=20)
axs_x[1].set_xscale("log")
# axs_x[1].set_yscale("log")
axs_x[1].set_ylabel(r'$\frac{|a-b|}{a+b}$ [V/$\sqrt{Hz}$]', fontsize=20)
axs_x[1].grid(True, linestyle='--')

axs_x[0].plot(f, gpz_asd_x, linestyle='-', label=r'$\Delta x$ - OP27GPZ')
axs_x[0].plot(f, epz_asd_x, linestyle='-', label=r'$\Delta x$ - OP27EPZ')
axs_x[1].plot(f, (gpz_asd_x - epz_asd_x) / (gpz_asd_x + epz_asd_x), linestyle='-', label=r'$\frac{|a-b|}{a+b}$')
axs_x[0].legend(loc='best', shadow=True, fontsize='medium')
axs_x[1].legend(loc='best', shadow=True, fontsize='medium')

# Sum channel

fig_comp_sum = plt.figure()
gs_sum = fig_comp_sum.add_gridspec(2, hspace=0.15)  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
axs_sum = gs_sum.subplots(sharex=True)

axs_sum[0].set_xlabel('Frequency [Hz]', fontsize=20)
axs_sum[0].set_xscale("log")
axs_sum[0].set_yscale("log")
axs_sum[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
axs_sum[0].grid(True, linestyle='--')

axs_sum[1].set_xlabel('Frequency [Hz]', fontsize=20)
axs_sum[1].set_xscale("log")
# axs_sum[1].set_yscale("log")
axs_sum[1].set_ylabel(r'$\frac{|a-b|}{a+b}$ [V/$\sqrt{Hz}$]', fontsize=20)
axs_sum[1].grid(True, linestyle='--')

axs_sum[0].plot(f, gpz_asd_sum, linestyle='-', label=r'$\Sigma$ - OP27GPZ')
axs_sum[0].plot(f, epz_asd_sum, linestyle='-', label=r'$\Sigma$ - OP27EPZ')
axs_sum[1].plot(f, (gpz_asd_sum - epz_asd_sum) / (gpz_asd_sum + epz_asd_sum), linestyle='-', label=r'$\frac{|a-b|}{a+b}$')
axs_sum[0].legend(loc='best', shadow=True, fontsize='medium')
axs_sum[1].legend(loc='best', shadow=True, fontsize='medium')
plt.show()
