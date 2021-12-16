__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.6"
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
from win32com.makegw.makegwparse import error_not_supported


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


# TODO: Eventually add config for all the variables (?)


def asd_extractor(file_directory, filename, samp_freq, psd_window, channel, means_number=1, final_df=pd.DataFrame(),
                  offset=1,
                  max_length=600):
    """

    :param file_directory: The directory where the file is stored
    :type file_directory: str

    :param filename: The name of the file
    :type filename: str

    :param samp_freq: The sampling frequency used
    :type samp_freq: int

    :param psd_window: Represents the length, in seconds, of the psd
    :type psd_window: int

    :param channel: One of the three channel acquired by the OL-shield. It MUST BE one of the following: Dy, Dx or Sum
    :type channel: str

    :param means_number: Specify how many means for each psd evaluation you want to perform
    :type means_number: int

    :param final_df: Already existing DataFrame to which you want to append the new data
    :type final_df: pandas.DataFrame

    :param offset: If you want to apply a vertical translation to the data
    :type offset: float

    :param max_length: Specify the maximum length of the data, in seconds, to be used. It is used to uniform the length
     between the channels
    :type max_length: int

    :return: A tuple of: a DataFrame containing three columns (asd_data, freq_data and Channels) the ratio between the
     psd window and the means used (necessary to evaluate the minimum frequency achieved)
     and the sample frequency applied
    """
    # TODO: Add function description
    Num = samp_freq * psd_window
    last_index = max_length * samp_freq
    df = pd.read_table(os.path.join(file_directory, filename),
                       sep='\t',
                       usecols=[0, 1, 2, 3],
                       header=None,
                       names=['NaN', 'Dy', 'Dx', 'Sum'])
    vec_asd, freq_asd = np.array([]), np.array([])
    data = np.array_split(df[channel][:last_index].values.flatten() * offset, last_index / Num)
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
    return final_df, psd_window / means_number, samp_freq  # freq_asd, vec_asd, final_df


def ratio(first_df, second_df, lower_quantile, upper_quantile, estimator='mean'):
    # TODO: Add function description
    try:
        if estimator.lower() == 'mean':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].mean()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].mean()
        elif estimator.lower() == 'median':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].median()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].median()
        else:
            raise ValueError('The estimator must be one between median or mean')
    except ValueError:
        raise

    try:
        if not 0 <= lower_quantile and upper_quantile <= 1:
            raise ValueError('The quantile must have a value between 0 and 1')
    except ValueError:
        raise

    # Find the quantile value grouping by frequency data
    first_low_quantile = first_df.groupby(['freq_data'])['asd_data'].quantile(lower_quantile)
    first_up_quantile = first_df.groupby(['freq_data'])['asd_data'].quantile(upper_quantile)
    second_low_quantile = second_df.groupby(['freq_data'])['asd_data'].quantile(lower_quantile)
    second_up_quantile = second_df.groupby(['freq_data'])['asd_data'].quantile(upper_quantile)

    # Evaluate the ratio between the quantities
    ratio_estim = first_estim / second_estim
    ratio_low_quantile = first_low_quantile / second_low_quantile
    ratio_up_quantile = first_up_quantile / second_up_quantile

    # Frequencies are the indeces of the DataFrame because of groupby()
    frequency_array = ratio_estim.index.to_numpy()

    # Convert the DataFrame in numpy array
    ratio_estim_array = ratio_estim.to_numpy()
    ratio_low_quantile_array = ratio_low_quantile.to_numpy()
    ratio_up_quantile_array = ratio_up_quantile.to_numpy()

    label_1 = first_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    label_2 = second_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    ratio_label = label_1 + '/' + label_2

    return frequency_array, ratio_estim_array, ratio_low_quantile_array, ratio_up_quantile_array, ratio_label


def ratio_std(first_df, second_df, num_std=1, estimator='mean'):
    # TODO: Add function description
    # TODO: Create mean variable as default because in ratio it is performed (mean1 +- n*std)/(mean2 +- n*std)
    try:
        if estimator.lower() == 'mean':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].mean()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].mean()
        elif estimator.lower() == 'median':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].median()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].median()
        else:
            raise ValueError('The estimator must be one between median or mean')
    except ValueError:
        raise

    # Find the std value grouping by frequency data
    first_std = first_df.groupby(['freq_data'])['asd_data'].std()
    second_std = second_df.groupby(['freq_data'])['asd_data'].std()

    # Evaluate the ratio between the quantities
    std_estim = first_estim / second_estim
    std_low_quantile = (first_df.groupby(['freq_data'])['asd_data'].mean() + num_std * first_std) / (
                second_df.groupby(['freq_data'])['asd_data'].mean() + num_std * second_std)
    std_up_quantile = (first_df.groupby(['freq_data'])['asd_data'].mean() - num_std * first_std) / (
                second_df.groupby(['freq_data'])['asd_data'].mean() - num_std * second_std)

    # Frequencies are the indeces of the DataFrame because of groupby()
    frequency_array = std_estim.index.to_numpy()

    # Convert the DataFrame in numpy array
    std_estim_array = std_estim.to_numpy()
    std_low_quantile_array = std_low_quantile.to_numpy()
    std_up_quantile_array = std_up_quantile.to_numpy()

    label_1 = first_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    label_2 = second_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    ratio_label = label_1 + '/' + label_2

    return frequency_array, std_estim_array, std_low_quantile_array, std_up_quantile_array, ratio_label


def difference(first_df, second_df, lower_quantile, upper_quantile, estimator='mean'):
    # TODO: Add function description
    try:
        if estimator.lower() == 'mean':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].mean()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].mean()
        elif estimator.lower() == 'median':
            first_estim = first_df.groupby(['freq_data'])['asd_data'].median()
            second_estim = second_df.groupby(['freq_data'])['asd_data'].median()
        else:
            raise ValueError('The estimator must be one between median or mean')
    except ValueError:
        raise

    try:
        if not 0 <= lower_quantile and upper_quantile <= 1:
            raise ValueError('The quantile must have a value between 0 and 1')
    except ValueError:
        raise

    # Find the quantile value grouping by frequency data
    first_low_quantile = first_df.groupby(['freq_data'])['asd_data'].quantile(lower_quantile)
    first_up_quantile = first_df.groupby(['freq_data'])['asd_data'].quantile(upper_quantile)
    second_low_quantile = second_df.groupby(['freq_data'])['asd_data'].quantile(lower_quantile)
    second_up_quantile = second_df.groupby(['freq_data'])['asd_data'].quantile(upper_quantile)

    # Evaluate the ratio between the quantities
    diff_estim = first_estim - second_estim
    diff_low_quantile = first_low_quantile - second_low_quantile
    diff_up_quantile = first_up_quantile - second_up_quantile

    # Frequencies are the indeces of the DataFrame because of groupby()
    frequency_array = diff_estim.index.to_numpy()

    # Convert the DataFrame in numpy array
    diff_estim_array = diff_estim.to_numpy()
    diff_low_quantile_array = diff_low_quantile.to_numpy()
    diff_up_quantile_array = diff_up_quantile.to_numpy()

    label_1 = first_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    label_2 = second_df['Channels'][0].split(' ')[1].replace('(', '').replace(')', '')
    ratio_label = label_1 + ' - ' + label_2

    return frequency_array, diff_estim_array, diff_low_quantile_array, diff_up_quantile_array, ratio_label


df_1, t_w, f_samp = asd_extractor(
    file_directory=r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz',
    filename=r'OL_Dark_EPZ_SoE_noUPS_Open_1kHz.lvm',
    samp_freq=2000,
    psd_window=60,
    means_number=10,
    channel='Sum')

df_2, _, _ = asd_extractor(
    file_directory=r'C:\Users\lpese\PycharmProjects\Archimedes_repo\Characterization\Noise\1_kHz',
    filename=r'OL_Dark_GPZ_SoE_noUPS_Open_1kHz.lvm',
    samp_freq=2000,
    psd_window=60,
    means_number=10,
    channel='Sum')

f_data, ratio_data, ratio_low, ratio_up, ratio_lab = ratio(first_df=df_1,
                                                           second_df=df_2,
                                                           lower_quantile=0.1,
                                                           upper_quantile=0.9,
                                                           estimator='median')

# TO MAKE RATIO USING STANDARD DEVIATIONS
# _, std_data, std_low, std_up, std_lab = ratio_std(first_df=df_1,
#                                                   second_df=df_2,
#                                                   num_std=2,
#                                                   estimator='median')

# TO MAKE DIFFERENCE BETWEEN SIGNALS
# _, diff_data, diff_low, diff_up, diff_lab = difference(first_df=df_1,
#                                                        second_df=df_2,
#                                                        lower_quantile=0.1,
#                                                        upper_quantile=0.9,
#                                                        estimator='median')

# MAKING PSD PLOT WITH 95% CONFIDENCE LEVEL
fig_psd_confidence = plt.figure(figsize=(10, 5))

# TO USE PARTICULAR LAYOUTS UNCOMMENT FOLLOWING LINES
# outer_grid = gridspec.GridSpec(3, 1, width_ratios=[1],
#                                height_ratios=[3.5, 3.5, 1])  # gridspec with two adjacent horizontal cells
# upper_cell = outer_grid[0, 0]  # the left SubplotSpec within outer_grid
#
# inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, upper_cell)
#
# # From here we can plot usinginner_grid's SubplotSpecs
# ax1 = plt.subplot(outer_grid[0, 0])
# ax2 = plt.subplot(outer_grid[1, 0])
# ax3 = plt.subplot(outer_grid[2, 0])

# TO MAKE THREE VERTICAL PLOTS UNCOMMENT NEXT LINES
gs = fig_psd_confidence.add_gridspec(3, hspace=0.15, width_ratios=[1], height_ratios=[3, 3, 1.5])
ax = gs.subplots(sharex=True)

# DATA PLOT USING SEABORN
# ci: Size of the confidence interval to draw when aggregating with an estimator.
# “sd” means to draw the standard deviation of the data. Setting to None will skip bootstrapping.
# See: https://seaborn.pydata.org/generated/seaborn.lineplot.html
sns.lineplot(x="freq_data", y="asd_data", hue='Channels', palette=['tab:blue'], ci='sd', data=df_1, ax=ax[0])
sns.lineplot(x="freq_data", y="asd_data", hue='Channels', palette=['tab:orange'], ci='sd', data=df_2, ax=ax[1])

# PLOT RATIO
ax[2].fill_between(f_data, ratio_low, ratio_up, alpha=.5, color='tab:green', linewidth=0)
ax[2].plot(f_data, ratio_data, linewidth=2, color='tab:green', label=ratio_lab + ' (Median)')
# ax[2].scatter(f_data, std_up, linewidth=1, color='tab:red', label='90%')
# ax[2].scatter(f_data, std_low, linewidth=1, color='tab:blue', label='10%')

# PLOT DIFFERENCE
# ax[2].fill_between(f_data, diff_low, diff_up, alpha=.5, color='tab:green', linewidth=0)
# ax[2].plot(f_data, diff_data, linewidth=2, color='tab:green', label=diff_lab)

# sns.lineplot(x="freq_data", y="ratio", hue='Channels', palette=['tab:green'], ci='sd', data=df_3, ax=ax[2])

# DEFINING VARIABLES TO CONTROL X/Y LIMIT IN PLOTS
x_limit_inf = 1 / t_w  # minimum frequency achieved
x_limit_sup = f_samp / 2  # Nyquist's theorem
y_limit_inf_asd = 1e-6  # set by user
y_limit_sup_asd = 8e-5  # set by user
y_limit_inf_ratio = -1  # set by user
y_limit_sup_ratio = 1  # set by user

# IF NOT USING VERTICAL LAYOUTS CHANE ax[0], ax[1], ax[2] ---> ax1, ax2, ax3
ax[0].set_xlabel('Frequency [Hz]', fontsize=20)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlim([x_limit_inf, x_limit_sup])
ax[0].set_ylim([y_limit_inf_asd, y_limit_sup_asd])
ax[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
ax[0].grid(True, linestyle='--', axis='both', which='both')

ax[1].set_xlabel('Frequency [Hz]', fontsize=20)
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_xlim([x_limit_inf, x_limit_sup])
ax[1].set_ylim([y_limit_inf_asd, y_limit_sup_asd])
ax[1].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
ax[1].grid(True, linestyle='--', axis='both', which='both')

ax[2].set_xlabel('Frequency [Hz]', fontsize=20)
ax[2].set_xscale("log")
ax[2].set_xlim([x_limit_inf, x_limit_sup])
# ax3.set_xlim([y_limit_inf_ratio, y_limit_sup_ratio])
ax[2].set_ylabel(r'$\frac{a}{b}$', fontsize=20)
ax[2].grid(True, linestyle='--', axis='both', which='both')
ax[2].legend(loc='best', shadow=True, fontsize='medium')

plt.show()
