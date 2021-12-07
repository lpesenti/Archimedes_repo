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


def plot_3d_oscilloscope(data_file, qty):
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
    # DEFINING VARIABLES
    file_number = len(glob.glob(file_path + '*'))
    x_labels = np.arange(-4.5, 5, 0.5)
    y_labels = np.arange(-4.5, 5, 0.5)
    freq_init = 50
    freq_final = 500
    delta_x = np.array([])
    delta_y = np.array([])
    indeces = np.array([])
    sum_ch = np.array([])
    vec_rms_x, vec_rms_y, vec_rms_sum = np.array([]), np.array([]), np.array([])
    vec_asd_x, vec_asd_y, vec_asd_sum = np.array([]), np.array([]), np.array([])
    vec_freq_x, vec_freq_y, vec_freq_sum = np.array([]), np.array([]), np.array([])
    vec_indeces = np.array([])

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
        # print('\ty', df[1].mean())
        # print('\tx', df[2].mean())
        # print('\tSum', df[2].mean())
        delta_y = np.append(delta_y, df[1].mean() / df[3].mean())
        delta_x = np.append(delta_x, df[2].mean() / df[3].mean())
        sum_ch = np.append(sum_ch, df[3].mean())
        indeces = np.append(indeces, i)

        # PSD EVALUATION
        # DELTA Y channel
        psd_y, f_y = mlab.psd(df[1][:5000], NFFT=1000, Fs=1000, detrend="linear", noverlap=0)
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
        psd_x, f_x = mlab.psd(df[2][:5000], NFFT=1000, Fs=1000, detrend="linear", noverlap=0)
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
        psd_sum, f_sum = mlab.psd(df[3][:5000], NFFT=1000, Fs=1000, detrend="linear", noverlap=0)
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
    # all_indeces = np.repeat(indeces, len(vec_asd_y)/180)  # It is assumed that every psd as the same length
    print(all_psd_data.size)
    print(all_freq_data.size)
    print(vec_indeces.size)
    df = pd.DataFrame(
        {'psd_data': all_psd_data, 'freq_data': all_freq_data, 'Channels': channels},
        columns=['psd_data', 'freq_data', 'Channels'])
    sns.lineplot(x="freq_data", y="psd_data", hue='Channels', data=df, ax=ax)
    ax.set_xlabel('Frequency [Hz]', fontsize=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r'ASD [V/$\sqrt{Hz}$]', fontsize=20)
    ax.grid(True, linestyle='--')

    # PLOT INDECES MAP
    # sns.heatmap(indeces_map, cmap=['black'], annot=True, cbar=None, linewidths=0.2, fmt='0.0f')

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
    fig_x = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_x = fig_x.add_subplot()
    sns.heatmap(heatmap_data_x, ax=ax1_x, square=True, cbar=True, cmap='icefire',
                cbar_kws={'label': r'$\Delta x$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_x.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_x.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_x.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_x.set_title(r'$\Delta x$')
    ax1_x.set_xlabel(r'$x [mm]$')
    ax1_x.set_ylabel(r'$y [mm]$')

    # PLOT FOR THE DELTA Y CHANNEL
    fig_y = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_y = fig_y.add_subplot()
    sns.heatmap(heatmap_data_y, ax=ax1_y, square=True, cbar=True, cmap='icefire',
                cbar_kws={'label': r'$\Delta y$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_y.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_y.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_y.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_y.set_title(r'$\Delta y$')
    ax1_y.set_xlabel(r'$x [mm]$')
    ax1_y.set_ylabel(r'$y [mm]$')

    # # PLOT FOR THE SUM CHANNEL
    fig_sum = plt.figure(figsize=(10, 5))
    # heatmap plot
    ax1_sum = fig_sum.add_subplot()
    sns.heatmap(heatmap_data_sum, ax=ax1_sum, square=True, cbar=True, cmap='viridis',
                cbar_kws={'label': r'$\Sigma$ [V]'}, xticklabels=x_labels, yticklabels=y_labels)  # , norm=LogNorm())
    ax1_sum.hlines(y=4.8, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum.hlines(y=14.2, xmin=4.8, xmax=14.2, color='r', linewidth=2.5)
    ax1_sum.vlines(x=4.8, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_sum.vlines(x=14.2, ymin=4.8, ymax=14.2, color='r', linewidth=2.5)
    ax1_sum.set_title(r'$\Sigma$')
    ax1_sum.set_xlabel(r'$x [mm]$')
    ax1_sum.set_ylabel(r'$y [mm]$')

    # PLOT FOR THE PSD VALUE OF THE DELTA X CHANNEL
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

    # PLOT FOR THE PSD VALUE OF THE DELTA Y CHANNEL
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

    # PLOT FOR THE PSD VALUE OF THE SUM CHANNEL
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
    ax1_sum_psd.set_ylabel(r'$y [mm]$')

    plt.show()


def eval_mean(data_file):
    for i in range(19):
        if i < 10:
            data_file1 = data_file + '0{0}.csv'.format(i)  # + file_type
        else:
            data_file1 = data_file + '{0}.csv'.format(i)  # + file_type
        df = pd.read_csv(data_file1, header=None)
        # print(df)
        values = np.array(df[4]).mean()
        print(data_file1, values)
