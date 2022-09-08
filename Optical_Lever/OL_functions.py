__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.1.0"
__maintainer__ = "Luca Pesenti (until September 30, 2022)"
__email__ = "lpesenti@uniss.it"
__status__ = "Prototype"

r"""
[LAST UPDATE: August 25, 2022 - Luca Pesenti]

This file contains several useful functions for the analysis of the Optical Lever (OL) system data. This device was built
at the University of Sassari as a control tool for the Archimedes experiment. For further information see 'Commissioning
and data analysis of the Archimedes experiment and its prototype at the SAR-GRAV  laboratory Chapter 4' at 
https://drive.google.com/file/d/1tyJ8PX4Giby3LttXn6AAxVaf7s0vkJkp/view?usp=sharing
 
The functions contained in this file can be used to reproduce the plot shown in the .pdf linked above but also can be
used to analyze the OL data taken.

NOTE: These function were made to work with specific data file and are neither optimized nor in a stable version 
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib import cm
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from matplotlib import mlab
from scipy.stats import stats


def plot_3d_oscilloscope(data_file, qty):
    r"""
    This method makes the 3d plot of data acquired with the Teledyne oscilloscope at University of Sassari

    :type data_file: str
    :param data_file: the path to the data file

    :type qty: str
    :param qty: the quantity on which perform the analysis
    """
    df = pd.read_table(data_file, names=['x', 'y', 'dy', 'sum', 'dx'])
    df = df.sort_values(by=['x', 'y'])
    x = np.array(df['x'])
    y = np.array(df['y'])
    dz = np.array(df[qty])

    print(df)

    # x, y = np.loadtxt(data_file, unpack=True, usecols=[0, 1])
    z = np.zeros(x.shape[0])  # * y.shape[0])
    dx = np.arange(x.shape[0])  # * y.shape[0])
    dy = np.arange(x.shape[0])  # * y.shape[0])
    dx.fill(50)
    dy.fill(50)

    heatmap_data = np.reshape(dz.transpose(), (19, 19))

    fig = plt.figure(figsize=(10, 5))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[0, 0], projection='3d')
    # ax2 = fig.add_subplot(spec[0, 2], projection='3d')
    ax1 = fig.add_subplot(spec[0, 1])

    # X, Y = np.meshgrid(np.arange(-450, 500, 50), np.arange(-450, 500, 50))

    dz_normed = dz / np.max(dz)
    normed_cbar = colors.Normalize(dz_normed.min(), dz_normed.max())
    color = cm.jet(normed_cbar(dz_normed))

    # ax2.plot_surface(X, Y, heatmap_data, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)
    ax.set_title('3D Plot')
    ax.set_xlabel(r'$\Delta x [\mu m]$')
    ax.set_ylabel(r'$\Delta y [\mu m]$')
    ax.set_zlabel(qty.capitalize() + ' [mV]', rotation=270)
    # ax.view_init(0, 270)

    im = ax1.imshow(heatmap_data.T, cmap=plt.get_cmap('jet'), aspect='auto')
    ax1.set_xticks(np.arange(heatmap_data.shape[0]))
    ax1.set_yticks(np.arange(heatmap_data.shape[0]))
    ax1.set_xticklabels(np.arange(-450, 500, 50))
    ax1.set_yticklabels(np.arange(-450, 500, 50))

    ax1.set_title('Heatmap')
    ax1.set_xlabel(r'$\Delta x [\mu m]$')
    ax1.set_ylabel(r'$\Delta y [\mu m]$')

    cbar = fig.colorbar(im)
    cbar.set_label(qty.capitalize() + ' [mV]', rotation=270, labelpad=15)
    plt.show()


def plot_3d_crio(file_path, file_prefix):

    r"""
    This function uses the data acquired with the cRIO device (produced by National Instruments) to make a 3D plot
    of a specified quantity.

    :type file_path: str
    :param file_path: path to the data

    :type file_prefix: str
    :param file_prefix: prefix of the data you want to analyze
    """

    # DEFINING VARIABLES
    file_number = len(glob.glob(file_path + '*'))
    x_labels = np.arange(-4.5, 5, 0.5)
    y_labels = np.arange(-4.5, 5, 0.5)
    freq_init = 50
    freq_final = 500
    f_samp = 2000
    delta_x = np.array([])
    delta_y = np.array([])
    indeces = np.array([])
    sum_ch = np.array([])
    vec_rms_x, vec_rms_y, vec_rms_sum = np.array([]), np.array([]), np.array([])
    vec_asd_x, vec_asd_y, vec_asd_sum = np.array([]), np.array([]), np.array([])
    vec_freq_x, vec_freq_y, vec_freq_sum = np.array([]), np.array([]), np.array([])
    vec_indeces = np.array([])
    cmap = 'icefire'  # sns.diverging_palette(500, 15, l=40, s=100, center="dark", as_cmap=True)
    cmap1 = 'viridis'
    sns.set(font_scale=2)

    # PSD COMPARISON
    fig_psd = plt.figure()
    gs = fig_psd.add_gridspec(3, hspace=0)  # set hspace=0.15 and uncomment titles to see them
    axs = gs.subplots(sharex=True, sharey=True)

    axs[0].set_xlabel('Frequency [Hz]', fontsize=20)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs[0].grid(True, linestyle='--')
    # axs[0].set_title(r'$\Delta y$')

    axs[1].set_xlabel('Frequency [Hz]', fontsize=20)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs[1].grid(True, linestyle='--')
    # axs[1].set_title(r'$\Delta x$')

    axs[2].set_xlabel('Frequency [Hz]', fontsize=20)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    axs[2].grid(True, linestyle='--')
    # axs[2].set_title(r'$\Sigma$')

    for ax in axs:
        ax.label_outer()

    # READ DATA
    for i in range(file_number):
        df = pd.read_table(file_path + file_prefix + '{0}.lvm'.format(i), sep='\t', header=None)
        # SOME VERBOSITY
        # print(file_prefix + '{0}.lvm'.format(i))
        # print('\ty', df[1].mean(), '+/-', df[1].std())
        # print('\tx', df[2].mean(), '+/-', df[2].std())
        # print('\tSum', df[3].mean(), '+/-', df[3].std())
        delta_y = np.append(delta_y, df[1].mean() / df[3].mean())
        delta_x = np.append(delta_x, df[2].mean() / df[3].mean())
        sum_ch = np.append(sum_ch, df[3].mean())
        indeces = np.append(indeces, i)

        # PSD EVALUATION
        # DELTA Y channel
        psd_y, f_y = mlab.psd(df[1][:5000], NFFT=1000, Fs=f_samp, detrend="linear", noverlap=0)
        psd_y = psd_y[1:]
        f_y = f_y[1:]
        start = np.where(f_y == freq_init)[0][0]
        stop = np.where(f_y == freq_final)[0][0]
        asd_y = np.sqrt(psd_y)
        vec_asd_y = np.append(vec_asd_y, asd_y)
        vec_freq_y = np.append(vec_freq_y, f_y)
        integral_y = sum(asd_y[start:stop] / len(asd_y[start:stop]))  # * (f_s[1]-f_s[0]))
        vec_rms_y = np.append(vec_rms_y, integral_y)

        # DELTA X channel
        psd_x, f_x = mlab.psd(df[2][:5000], NFFT=1000, Fs=f_samp, detrend="linear", noverlap=0)
        psd_x = psd_x[1:]
        f_x = f_x[1:]
        start = np.where(f_x == freq_init)[0][0]
        stop = np.where(f_x == freq_final)[0][0]
        asd_x = np.sqrt(psd_x)
        vec_asd_x = np.append(vec_asd_x, asd_x)
        vec_freq_x = np.append(vec_freq_x, f_x)
        integral_x = sum(asd_x[start:stop] / len(asd_x[start:stop]))  # * (f_s[1]-f_s[0]))
        vec_rms_x = np.append(vec_rms_x, integral_x)

        # Sum channel
        psd_sum, f_sum = mlab.psd(df[3][:5000], NFFT=1000, Fs=f_samp, detrend="linear", noverlap=0)
        psd_sum = psd_sum[1:]
        f_sum = f_sum[1:]
        start = np.where(f_sum == freq_init)[0][0]
        stop = np.where(f_sum == freq_final)[0][0]
        asd_sum = np.sqrt(psd_sum)
        vec_asd_sum = np.append(vec_asd_sum, asd_sum)
        vec_freq_sum = np.append(vec_freq_sum, f_sum)
        integral_sum = sum(asd_sum[start:stop] / len(asd_sum[start:stop]))  # * (f_s[1]-f_s[0]))
        vec_rms_sum = np.append(vec_rms_sum, integral_sum)

        # Indeces array for 95% comparison
        vec_indeces = np.append(vec_indeces, np.array(i).repeat(3 * len(f_y)))  # to consider the three channel

        if i == 66 or i == 174 or i == 180:
            axs[0].plot(f_y, asd_y, linestyle='-', label=r'$\Delta y$ - {0}'.format(i))
            axs[1].plot(f_x, asd_x, linestyle='-', label=r'$\Delta x$ - {0}'.format(i))
            axs[2].plot(f_sum, asd_sum, linestyle='-', label=r'$\Sigma$ - {0}'.format(i))
            axs[0].legend(loc='best', shadow=True, fontsize='medium')
            axs[1].legend(loc='best', shadow=True, fontsize='medium')
            axs[2].legend(loc='best', shadow=True, fontsize='medium')

    # ADDING DELTA TO MAKE ALL DATA POSITIVE FOR LOG SCALE
    # delta_x = delta_x + 2 * np.abs(delta_x.min())
    # delta_y = delta_y + 2 * np.abs(delta_y.min())

    # CREATING 2D-NUMPY ARRAY FOR HEATMAP RESHAPING 1D-ARRAY CONTAINING DATA
    heatmap_data_x = np.reshape(delta_x, (19, 19))
    heatmap_data_y = np.reshape(delta_y, (19, 19))
    heatmap_data_sum = np.reshape(sum_ch, (19, 19))
    indeces_map = np.reshape(indeces, (19, 19))
    heatmap_psd_x = np.reshape(vec_rms_x, (19, 19))
    heatmap_psd_y = np.reshape(vec_rms_y, (19, 19))
    heatmap_psd_sum = np.reshape(vec_rms_sum, (19, 19))

    # MAKING PSD PLOT WITH 95% CONFIDENCE LEVEL
    fig_psd_confidence = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax = fig_psd_confidence.add_subplot()
    all_psd_data = np.hstack((vec_asd_y, vec_asd_x, vec_asd_sum))
    all_freq_data = np.hstack((vec_freq_y, vec_freq_x, vec_freq_sum))
    dy_ch = np.array('Dy').repeat(vec_asd_y.size)
    dx_ch = np.array('Dx').repeat(vec_asd_x.size)
    dsum_ch = np.array('Sum').repeat(vec_asd_sum.size)
    channels = np.hstack((dy_ch, dx_ch, dsum_ch))
    # all_indeces = np.repeat(indeces, len(vec_asd_y)/180)  # It is assumed that every psd has the same length
    # print(all_psd_data.size)
    # print(all_freq_data.size)
    # print(vec_indeces.size)
    df = pd.DataFrame(
        {'psd_data': all_psd_data, 'freq_data': all_freq_data, 'Channels': channels},
        columns=['psd_data', 'freq_data', 'Channels'])
    sns.lineplot(x="freq_data", y="psd_data", hue='Channels', ci='sd', data=df, ax=ax)
    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, linestyle='--')

    # PLOT INDECES MAP
    fig_indeces = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax_indeces = fig_indeces.add_subplot()
    sns.heatmap(indeces_map, cmap=['white'],
                annot=False, square=True, cbar=None,
                linewidths=1, linecolor='k',
                fmt='0.0f',
                xticklabels=x_labels, yticklabels=y_labels,
                ax=ax_indeces)
    ax_indeces.set_xlabel(r'$x [mm]$', fontsize=24)
    ax_indeces.set_ylabel(r'$y [mm]$', fontsize=24)
    ax_indeces.tick_params(axis='both', labelsize=22, which='both')

    # PLOT THE LAST PSD
    # fig_psd = plt.figure()
    # ax_psd = fig_psd.add_subplot()
    # ax_psd.plot(f_y[1:], np.sqrt(psd_y), linestyle='-', label=r'$\Delta y$')
    # ax_psd.plot(f_x[1:], np.sqrt(psd_x), linestyle='-', label=r'$\Delta x$')
    # ax_psd.plot(f_sum[1:], np.sqrt(psd_sum), linestyle='-', label=r'$\Sigma$')
    # ax_psd.set_xlabel('Frequency [Hz]', fontsize=20)
    # ax_psd.set_xscale("log")
    # ax_psd.set_yscale("log")
    # ax_psd.set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    # ax_psd.grid(True, linestyle='--')
    # ax_psd.legend(loc='best', shadow=True, fontsize='medium')

    # PLOT FOR THE DELTA X CHANNEL
    fig_x = plt.figure(figsize=(19.2, 10.8))
    # heatmap plot
    ax1_x = fig_x.add_subplot()
    sns.heatmap(heatmap_data_x, ax=ax1_x, square=True, cbar=True, cmap=cmap,
                cbar_kws={'label': r'$\Delta x$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_x.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_x.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    # ax1_x.set_title(r'$\Delta x$')
    ax1_x.set_xlabel(r'$x [mm]$', fontsize=24)
    ax1_x.set_ylabel(r'$y [mm]$', fontsize=24)
    ax1_x.tick_params(axis='both', labelsize=22, which='both')

    # PLOT FOR THE DELTA Y CHANNEL
    fig_y = plt.figure(figsize=(19.2, 10.8))
    # heatmap plot
    ax1_y = fig_y.add_subplot()
    sns.heatmap(heatmap_data_y, ax=ax1_y, square=True, cbar=True, cmap=cmap,
                cbar_kws={'label': r'$\Delta y$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_y.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_y.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    # ax1_y.set_title(r'$\Delta y$')
    ax1_y.set_xlabel(r'$x [mm]$', fontsize=24)
    ax1_y.set_ylabel(r'$y [mm]$', fontsize=24)
    ax1_y.tick_params(axis='both', labelsize=22, which='both')

    # # PLOT FOR THE SUM CHANNEL
    fig_sum = plt.figure(figsize=(19.2, 10.8))
    # heatmap plot
    ax1_sum = fig_sum.add_subplot()
    sns.heatmap(heatmap_data_sum, ax=ax1_sum, square=True, cbar=True, cmap=cmap1,
                cbar_kws={'label': r'$\Sigma$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_sum.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_sum.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    # ax1_sum.set_title(r'$\Sigma$')
    ax1_sum.set_xlabel(r'$x [mm]$', fontsize=24)
    ax1_sum.set_ylabel(r'$y [mm]$', fontsize=24)
    ax1_sum.tick_params(axis='both', labelsize=22, which='both')

    """# PLOT FOR THE ASD INTEGRAL OF THE DELTA X CHANNEL
    fig_x = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_x_psd = fig_x.add_subplot()
    sns.heatmap(heatmap_psd_x, ax=ax1_x_psd, square=True, cbar=True, cmap='viridis',
                cbar_kws={'label': r'ASD integral'}, xticklabels=x_labels,
                yticklabels=y_labels, norm=LogNorm())
    ax1_x_psd.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x_psd.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x_psd.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_x_psd.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_x_psd.set_title(r'ASD integral [{0} Hz - {1} Hz] map of $\Delta x$'.format(freq_init, freq_final))
    ax1_x_psd.set_xlabel(r'$x [mm]$')
    ax1_x_psd.set_ylabel(r'$y [mm]$')

    # PLOT FOR THE ASD INTEGRAL OF THE DELTA Y CHANNEL
    fig_x = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_y_psd = fig_x.add_subplot()
    sns.heatmap(heatmap_psd_y, ax=ax1_y_psd, square=True, cbar=True, cmap='viridis',
                cbar_kws={'label': r'ASD integral'}, xticklabels=x_labels,
                yticklabels=y_labels, norm=LogNorm())
    ax1_y_psd.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y_psd.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y_psd.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_y_psd.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_y_psd.set_title(r'ASD integral [{0} Hz - {1} Hz] map of $\Delta y$'.format(freq_init, freq_final))
    ax1_y_psd.set_xlabel(r'$x [mm]$')
    ax1_y_psd.set_ylabel(r'$y [mm]$')

    # PLOT FOR THE ASD INTEGRAL OF THE SUM CHANNEL
    fig_x = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_sum_psd = fig_x.add_subplot()
    sns.heatmap(heatmap_psd_sum, ax=ax1_sum_psd, square=True, cbar=True, cmap='viridis',
                cbar_kws={'label': r'ASD integral'}, xticklabels=x_labels,
                yticklabels=y_labels, norm=LogNorm())
    ax1_sum_psd.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum_psd.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum_psd.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_sum_psd.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_sum_psd.set_title(r'ASD integral [{0} Hz - {1} Hz] map of $\Sigma$'.format(freq_init, freq_final))
    ax1_sum_psd.set_xlabel(r'$x [mm]$')
    ax1_sum_psd.set_ylabel(r'$y [mm]$')"""
    fig_x.tight_layout()
    fig_y.tight_layout()
    fig_sum.tight_layout()
    plt.show()


def eval_mean(data_file):
    r"""
    It is used to evaluate the mean of a list of .csv files

    See Also
    ---------
    * linearity(): in this method the user must pass an array with means and another one containing standard deviation.
    See OL.Analysis.py in which these are used together.

    :type data_file: str
    :param data_file: path to the data

    :return: a tuple of two unidimensional numpy.ndarray: the former with the evaluated means, while the latter with the
             relative standard deviation.
    """
    mean_val = np.array([])
    std_val = np.array([])
    for i in range(19):
        if i < 10:  # TODO: replace with zfill()
            data_file1 = data_file + '0{0}.csv'.format(i)  # + file_type
        else:
            data_file1 = data_file + '{0}.csv'.format(i)  # + file_type
        df = pd.read_csv(data_file1, header=None)
        # print(df.head())
        values = df[4].mean()
        std = df[4].std()
        mean_val = np.append(mean_val, values)
        std_val = np.append(std_val, std)
        print(data_file1, values, std)
    return mean_val, std_val


def linearity(mean, std):
    r"""
    This function is used to plot the mean values and their relative standard deviation to check the linearity between
    a known signal and the one read at the 'Sum' channel of the Optical Lever Shield (OLS).

    See Also
    --------
    * eval_mean(): in this method the user must pass a path to the data on which perform the means and standard
    deviation evaluation. See OL.Analysis.py in which these are used together.

    :type mean: numpy.ndarray
    :param mean: an array containing the mean values evaluated for the sum channel

    :type std: numpy.ndarray
    :param std: an array containing the standard deviation values evaluated for the sum channel
    """
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    # ORDERING THE DATA FROM -4.5 TO 4.5 (FILE 00 -> 0 [V], FILE 01 -> 0.5 [V], FILE 10 -> -0.5 [V], ...)
    firsts = mean[10:20]
    seconds = mean[0:10]
    data = np.hstack([firsts[::-1], seconds])

    std_firsts = std[10:20]
    std_seconds = std[0:10]
    std_data = np.hstack([std_firsts[::-1], std_seconds])

    # CREATING THE Vps DATA
    x = np.linspace(-4.5, 4.5, data.size)

    print('Error %: ', np.transpose(std_data * 100 / data))

    # TAKING ONLY DATA IN THE LINEAR RANGE
    data_reduced = data[2:17]
    std_reduced = std_data[2:17]
    x_reduced = np.linspace(-3.5, 3.5, data_reduced.size)

    # print(data_reduced)
    # print(x_reduced)

    # UNCOMMENT FOR FIT ON ALL DATA
    # coef = np.polyfit(x, data, 1)

    # UNCOMMENT FOR FIT ON REDUCED DATA
    coef, V = np.polyfit(x_reduced, data_reduced, 1, cov=True)

    print(np.polyfit(x_reduced, data_reduced, 1, cov=True))

    poly1d_fn = np.poly1d(coef)

    num_chi = poly1d_fn(x_reduced) - data_reduced

    chi_homemade = np.sum((num_chi / std_reduced) ** 2)

    print(chi_homemade)

    # UNCOMMENT FOR CHI-SQUARE ON ALL DATA
    # chi = stats.chisquare(f_obs=test, f_exp=data)

    # UNCOMMENT FOR CHI-SQUARE ON REDUCED DATA
    # chi = stats.chisquare(f_obs=poly1d_fn(x_reduced), f_exp=data_reduced)

    # print(chi)

    fit_label_1 = 'y = Ax + B\n'
    fit_label_2 = r'$\chi^2$ / $ndf$ = {0}/{1}'.format(round(float(chi_homemade), 2),
                                                       int(data_reduced.size - len(coef)))
    fit_label_3 = r'A = {:.3f} $\pm$ {:.3f}'.format(coef[0], np.sqrt(V[0][0]))
    fit_label_4 = r'B = {:.3f} $\pm$ {:.3f}'.format(coef[1], np.sqrt(V[1][1]))
    fit_label = fit_label_1 + fit_label_2 + '\n' + fit_label_3 + '\n' + fit_label_4

    print('y = {0}*x +{1}'.format(coef[0], coef[1]))

    # ax.plot(x, poly1d_fn(x), 'r', label=fit_label)

    ax.plot(x_reduced, poly1d_fn(x_reduced), 'r', label=fit_label)

    ax.errorbar(x, data, linestyle='', marker='o', linewidth=2, markersize=7, yerr=std_data, label='Data')

    ax.set_xlabel(r"$V_{ps}$ [V]", fontsize=24)
    ax.set_ylabel(r"$V_{\Sigma}$ [V]", fontsize=24)
    ax.tick_params(axis='both', labelsize=22, which='both')
    # ax.tick_params(axis='y', labelsize=20, which='both')
    ax.grid(True, linestyle='--', which='both')
    ax.legend(loc='best', shadow=True, fontsize=24)
    fig.tight_layout()
    plt.show()


def ols_saturation(colorregion=False, limitlines=False):
    r"""
    This method makes the plot of the voltage read at the sum channel of the Optical Lever Shield (OLS) against the
    supply current of the SLED used. However, this function is made to work with a set of measure taken in October 2021.
    This measurement was made without looking at the errors on the voltage values, then a random error factor is
    inserted to be reasonable.

    :type colorregion: bool
    :param colorregion: if True, two coloured region are added on the plot, one for the OP27 saturation region and the
                         other for the cRIO saturation region

    :type limitlines: bool
    :param limitlines: if True, two horizontal lines are added on the plot, one for the OP27 saturation limit and the
                        other for the cRIO saturation limit
    """
    v_sum = np.array([0, 30, 70, 140, 320, 760, 1500, 2920, 5600, 9300, 13800, 14100, 14100, 14100])  # mVolt
    v_sum = v_sum / 1000  # convert to Volt

    err_factor = np.array([100, 100, 80, 60, 50, 40, 30, 20, 10, 10, 5, 1, 1, 1])

    err_v = v_sum * err_factor / 100  # considering 10% error on each measure
    err_v[0] = 0.07

    print(err_v)
    i_sled = np.array([0, 10, 20, 30, 39, 50, 60, 70, 80, 90, 99, 110, 120, 130])

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    ax.errorbar(i_sled, v_sum, yerr=err_v, linestyle='', marker='o', linewidth=2, markersize=7)
    ax.set_xlabel(r"$I_{SLED}$ [mA]", fontsize=24)
    ax.set_ylabel(r"$V_{\Sigma}$ [V]", fontsize=24)
    ax.tick_params(axis='both', labelsize=22, which='both')
    ax.grid(True, linestyle='--', which='both')
    ax.set_ylim([-0.5, 16])
    # ax.legend(loc='best', shadow=True, fontsize='xx-large')

    if colorregion:
        limitlines = True
        ax.axhspan(ymin=10, ymax=20, alpha=0.2, facecolor='green')
        ax.axhspan(ymin=14, ymax=20, alpha=0.2, facecolor='red')

    if limitlines:
        ax.axhline(y=10, color='green', linestyle='--', linewidth=2.5)
        ax.text(5, 11, 'cRIO-ADC limit', fontsize=18, color='green', ha='center', va='center',
                bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'green', 'boxstyle': 'round'})

        ax.axhline(y=14, color='red', linestyle='--', linewidth=2.5)
        ax.text(6.5, 15, 'OP27 saturation limit', fontsize=18, color='r', ha='center', va='center',
                bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'red', 'boxstyle': 'round'})

    fig.tight_layout()
    plt.show()


def sled_p_to_i():
    r"""
    Similarly to the precious function, this one makes a plot relative to the power read on a power meter while varying
    the supply current of the SLED. As before, the data used are the ones taken in October 2021.
    """
    i_sled = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 105, 110, 120, 130, 140, 150])
    power_data1 = np.array(
        [0, 0.001, 0.004, 0.009, 0.019, 0.041, 0.076, 0.142, 0.243, 0.473, 0.564, 0.802, 1.079, 1.380, 1.720])

    power_data2 = np.array(
        [0, 0.001, 0.004, 0.009, 0.019, 0.04, 0.078, 0.142, 0.241, 0.476, 0.574, 0.806, 1.089, 1.38, 1.72])

    power_data = np.array([power_data1, power_data2])

    print(power_data1.size, i_sled.size)

    power_mean = np.mean(power_data, axis=0)
    power_std = np.std(power_data, axis=0)
    print(power_mean.size)
    print(power_std)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    ax.errorbar(i_sled, power_mean, yerr=power_std, linestyle='', marker='o', linewidth=3, markersize=7)
    ax.set_xlabel(r"$I_{SLED}$ [mA]", fontsize=24)
    ax.set_ylabel(r"$Power$ [mW]", fontsize=24)
    ax.tick_params(axis='both', labelsize=22, which='both')
    ax.grid(True, linestyle='--', which='both')
    ax.set_xlim([0, 160])
    ax.set_ylim([-0.2, 2])
    # ax.legend(loc='best', shadow=True, fontsize='xx-large')

    fig.tight_layout()
    plt.show()
