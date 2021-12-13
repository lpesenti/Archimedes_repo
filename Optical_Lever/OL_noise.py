__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.5.0"
__maintainer__ = "Luca Pesenti"
__email__ = "l.pesenti6@campus.unimib.it, drozza@uniss.it"
__status__ = "Prototype"

import pandas as pd
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec


def old():
    # FREQUENCY AND PSD WINDOW
    fs = 1000  # if using oscilloscope read first row and do "1 / 2.000000e-04"
    time_window = 60  # in sec
    Num = int(time_window * fs)

    # READING DATA
    # If using cRIO uncomment next line

    df = pd.read_table(
        r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz\OL_Dark_EPZ_SoE_noUPS_Open_1kHz.lvm',
        sep='\t',
        usecols=[0, 1, 2, 3],
        header=None)
    df1 = pd.read_table(
        r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz\OL_Dark_GPZ_SoE_noUPS_Open_1kHz.lvm',
        sep='\t',
        usecols=[0, 1, 2, 3],
        header=None)

    # DEFINING PLOT GRAPHIC VARIABLES
    first_file_label = 'OP27EPZ'
    second_file_label = 'OP27GPZ'
    x_limit_inf = 1 / time_window
    x_limit_sup = fs / 2
    y_limit_inf_asd = 1e-6
    y_limit_sup_asd = 3e-5
    y_limit_inf_diff = -1
    y_limit_sup_diff = 1

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

    ax1.plot(f, gpz_asd_y, label=r"$\Delta$y - {}".format(first_file_label))
    ax1.plot(f, gpz_asd_x, label=r"$\Delta$x - {}".format(first_file_label))
    ax1.plot(f, gpz_asd_sum, label=r"$\Sigma$ - {}".format(first_file_label))
    ax1.plot(f, epz_asd_y, label=r"$\Delta$y - {}".format(second_file_label))
    ax1.plot(f, epz_asd_x, label=r"$\Delta$x - {}".format(second_file_label))
    ax1.plot(f, epz_asd_sum, label=r"$\Sigma$ - {}".format(second_file_label))
    # ax1.plot(f, gpz_asd_y, label=r"$\Delta$y - {}".format(first_file_label))
    # ax1.plot(f, gpz_asd_x, label=r"$\Delta$x - {}".format(first_file_label))
    # ax1.plot(f, gpz_asd_sum, label=r"$\Sigma$ - {}".format(first_file_label))
    # ax1.plot(f, epz_asd_y, label=r"$\Delta$y - {}".format(second_file_label))
    # ax1.plot(f, epz_asd_x, label=r"$\Delta$x - {}".format(second_file_label))
    # ax1.plot(f, epz_asd_sum, label=r"$\Sigma$ - {}".format(second_file_label))

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
    gs_y = fig_comp_y.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[4,
                                                                                    1])  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
    axs_y = gs_y.subplots(sharex=True)

    # axs_y[0].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_y[0].set_xscale("log")
    axs_y[0].set_yscale("log")
    axs_y[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs_y[0].grid(True, linestyle='--', axis='both', which='both')
    axs_y[0].set_xlim([x_limit_inf, x_limit_sup])
    axs_y[0].set_ylim([y_limit_inf_asd, y_limit_sup_asd])

    axs_y[1].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_y[1].set_xscale("log")
    # axs_y[1].set_yscale("log")
    axs_y[1].set_ylabel(r'{0}/{1}'.format(second_file_label, first_file_label), fontsize=20)
    axs_y[1].grid(True, linestyle='--', axis='both', which='both')
    axs_y[1].set_xlim([x_limit_inf, x_limit_sup])
    axs_y[1].set_ylim([y_limit_inf_diff, y_limit_sup_diff])

    axs_y[0].plot(f, gpz_asd_y, linestyle='-', label=r'$\Delta y$ - {}'.format(first_file_label))
    axs_y[0].plot(f, epz_asd_y, linestyle='-', label=r'$\Delta y$ - {}'.format(second_file_label))
    axs_y[1].plot(f, (gpz_asd_y / epz_asd_y), linestyle='-', label=r'$\frac{a}{b}$')
    axs_y[0].legend(loc='best', shadow=True, fontsize='medium')
    axs_y[1].legend(loc='best', shadow=True, fontsize='medium')

    # Delta x channel

    fig_comp_x = plt.figure()
    gs_x = fig_comp_x.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[4,
                                                                                    1])  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
    axs_x = gs_x.subplots(sharex=True)

    # axs_x[0].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_x[0].set_xscale("log")
    axs_x[0].set_yscale("log")
    axs_x[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs_x[0].grid(True, linestyle='--', axis='both', which='both')
    axs_x[0].set_xlim([x_limit_inf, x_limit_sup])
    axs_x[0].set_ylim([y_limit_inf_asd, y_limit_sup_asd])

    axs_x[1].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_x[1].set_xscale("log")
    # axs_x[1].set_yscale("log")
    axs_x[1].set_ylabel(r'{0}/{1}'.format(second_file_label, first_file_label), fontsize=20)
    axs_x[1].grid(True, linestyle='--', axis='both', which='both')
    axs_x[1].set_xlim([x_limit_inf, x_limit_sup])
    axs_x[1].set_ylim([y_limit_inf_diff, y_limit_sup_diff])

    axs_x[0].plot(f, gpz_asd_x, linestyle='-', label=r'$\Delta x$ - {}'.format(first_file_label))
    axs_x[0].plot(f, epz_asd_x, linestyle='-', label=r'$\Delta x$ - {}'.format(second_file_label))
    axs_x[1].plot(f, (gpz_asd_x / epz_asd_x), linestyle='-', label=r'$\frac{a}{b}$')
    axs_x[0].legend(loc='best', shadow=True, fontsize='medium')
    axs_x[1].legend(loc='best', shadow=True, fontsize='medium')

    # Sum channel

    fig_comp_sum = plt.figure()
    gs_sum = fig_comp_sum.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[4,
                                                                                        1])  # set hspace=0.15 and uncomment titles to see them, or hspace=0 to not
    axs_sum = gs_sum.subplots(sharex=True)

    # axs_sum[0].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_sum[0].set_xscale("log")
    axs_sum[0].set_yscale("log")
    axs_sum[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs_sum[0].grid(True, linestyle='--', axis='both', which='both')
    axs_sum[0].set_xlim([x_limit_inf, x_limit_sup])
    axs_sum[0].set_ylim([y_limit_inf_asd, y_limit_sup_asd])

    axs_sum[1].set_xlabel('Frequency [Hz]', fontsize=20)
    axs_sum[1].set_xscale("log")
    # axs_sum[1].set_yscale("log")
    axs_sum[1].set_ylabel(r'{0}/{1}'.format(second_file_label, first_file_label), fontsize=20)
    axs_sum[1].grid(True, linestyle='--', axis='both', which='both')
    axs_sum[1].set_xlim([x_limit_inf, x_limit_sup])
    axs_sum[1].set_ylim([y_limit_inf_diff, y_limit_sup_diff])

    axs_sum[0].plot(f, gpz_asd_sum, linestyle='-', label=r'$\Sigma$ - {}'.format(first_file_label))
    axs_sum[0].plot(f, epz_asd_sum, linestyle='-', label=r'$\Sigma$ - {}'.format(second_file_label))
    axs_sum[1].plot(f, (gpz_asd_sum / epz_asd_sum), linestyle='-',
                    label=r'$\frac{a}{b}$')
    axs_sum[0].legend(loc='best', shadow=True, fontsize='medium')
    axs_sum[1].legend(loc='best', shadow=True, fontsize='medium')


def asd_extractor(file_directory, filename, samp_freq, psd_window, channel, means_number=1, final_df=pd.DataFrame(),
                  offset=1,
                  max_length=1200000):
    Num = samp_freq * psd_window
    df = pd.read_table(os.path.join(file_directory, filename),
                       sep='\t',
                       usecols=[0, 1, 2, 3],
                       header=None,
                       names=['NaN', 'Dy', 'Dx', 'Sum'])
    # print(df.head())
    vec_asd, freq_asd = np.array([]), np.array([])
    data = np.array_split(df[channel][:max_length].values.flatten() * offset, max_length / Num)
    nfft_used = int(Num / means_number)
    for i in range(len(data)):
        psd, freq = mlab.psd(data[i], NFFT=nfft_used, Fs=samp_freq, detrend="linear")  # ,noverlap=NOL)
        psd = psd[1:]
        freq = freq[1:]
        asd = np.sqrt(psd)
        freq_asd = np.append(freq_asd, freq)
        vec_asd = np.append(vec_asd, asd)
    op27_type = filename.replace('_SoE_noUPS_Open_1kHz.lvm', '').replace('OL_Dark_', '')
    ch = np.array('{0} ({1})'.format(channel, op27_type)).repeat(vec_asd.size)
    resulting_df = pd.DataFrame(
        {'asd_data': vec_asd, 'freq_data': freq_asd, 'Channels': ch},
        columns=['asd_data', 'freq_data', 'Channels'])
    final_df = final_df.append(resulting_df, ignore_index=True)
    return final_df, nfft_used, samp_freq  # freq_asd, vec_asd, final_df


def ratio(first_df, second_df):
    ch = np.array('{0}/{1}'.format(first_df['Channels'][0], second_df['Channels'][0])).repeat(len(first_df['Channels']))
    ratio_df = pd.DataFrame(
        {'ratio': first_df['asd_data'] / second_df['asd_data'], 'freq_data': first_df['freq_data'], 'Channels': ch},
        columns=['ratio', 'freq_data', 'Channels'])
    return ratio_df


df_1, t_w, f_samp = asd_extractor(
    file_directory=r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz',
    filename=r'OL_Dark_EPZ_SoE_noUPS_Open_1kHz.lvm',
    samp_freq=2000,
    psd_window=60,
    means_number=10,
    channel='Sum')
print(t_w)
print('first df')
df_2, _, _ = asd_extractor(
    file_directory=r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz',
    filename=r'OL_Dark_GPZ_SoE_noUPS_Open_1kHz.lvm',
    samp_freq=2000,
    psd_window=60,
    means_number=10,
    channel='Sum')
print('second df')
df_3 = ratio(first_df=df_1,
             second_df=df_2)
print('third df')
# MAKING PSD PLOT WITH 95% CONFIDENCE LEVEL
fig_psd_confidence = plt.figure(figsize=(10, 5))

outer_grid = gridspec.GridSpec(2, 1, width_ratios=[1],
                               height_ratios=[3.5, 1])  # gridspec with two adjacent horizontal cells
upper_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid

inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, upper_cell)

# From here we can plot usinginner_grid's SubplotSpecs
ax1 = plt.subplot(inner_grid[0, 0])
ax2 = plt.subplot(inner_grid[0, 1])
ax3 = plt.subplot(outer_grid[1, 0])
# gs = fig_psd_confidence.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[4, 1])
# heatmap plot
# ax = fig_psd_confidence.add_subplot()
sns.lineplot(x="freq_data", y="asd_data", hue='Channels', palette=['tab:blue'], ci='sd', data=df_1, ax=ax1)
sns.lineplot(x="freq_data", y="asd_data", hue='Channels', palette=['tab:orange'], ci='sd', data=df_2, ax=ax2)
sns.lineplot(x="freq_data", y="ratio", hue='Channels', palette=['tab:green'], ci='sd', data=df_3, ax=ax3)

x_limit_inf = 1 / t_w * f_samp
x_limit_sup = f_samp / 2
y_limit_inf_asd = 8e-7
y_limit_sup_asd = 8e-5
y_limit_inf_diff = -1
y_limit_sup_diff = 1

ax1.set_xlabel('Frequency [Hz]', fontsize=20)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim([x_limit_inf, x_limit_sup])
ax1.set_ylim([y_limit_inf_asd, y_limit_sup_asd])
ax1.set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
ax1.grid(True, linestyle='--', axis='both', which='both')

ax2.set_xlabel('Frequency [Hz]', fontsize=20)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim([x_limit_inf, x_limit_sup])
ax2.set_ylim([y_limit_inf_asd, y_limit_sup_asd])
ax2.set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
ax2.grid(True, linestyle='--', axis='both', which='both')

ax3.set_xlabel('Frequency [Hz]', fontsize=20)
ax3.set_xscale("log")
# ax3.set_yscale("log")
ax3.set_ylabel(r'$\frac{a}{b}$', fontsize=20)
ax3.grid(True, linestyle='--', axis='both', which='both')

plt.show()
