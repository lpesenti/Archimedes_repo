import pandas as pd
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt

fs = 1 / 2.000000e-04
Num = int(500 * fs)
df_col = pd.read_csv(r'D:\Archimedes\Data\Optical_Level\C1Dy00002.csv', nrows=1, header=None).columns
data_df = pd.read_csv(r'D:\Archimedes\Data\Optical_Level\C1Dy00002.csv', usecols=[df_col[-1:][0]], header=None)
data_df1 = pd.read_csv(r'D:\Archimedes\Data\Optical_Level\C2Sum00002.csv', usecols=[df_col[-1:][0]], header=None)
data_df2 = pd.read_csv(r'D:\Archimedes\Data\Optical_Level\C3Dx00002.csv', usecols=[df_col[-1:][0]], header=None)
data = data_df.values.flatten()
data1 = data_df1.values.flatten()
data2 = data_df2.values.flatten()

_, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs)  # ,noverlap=NOL)
f = f[1:]  # remove first value that is 0

s, _ = mlab.psd(data, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
s = s[1:]
s = np.sqrt(s)

s1, _ = mlab.psd(data1, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
s1 = s1[1:]
s1 = np.sqrt(s1)

s2, _ = mlab.psd(data2, NFFT=Num, Fs=fs, detrend="linear")  # ,noverlap=NOL)
s2 = s2[1:]
s2 = np.sqrt(s2)

fig, ax1 = plt.subplots()
ax1.plot(f, s, label="$\Delta$y")
ax1.plot(f, s1, label="$\Sigma$")
ax1.plot(f, s2, label="$\Delta$x")
legend = ax1.legend(loc='best', shadow=True)  # , fontsize='x-large')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.grid(True)
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel(r'ASD [V/$\sqrt{Hz}$]')

plt.show()

