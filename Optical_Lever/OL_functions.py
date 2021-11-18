import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
import numpy as np
import glob
import seaborn as sns


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
    file_number = len(glob.glob(file_path + '*'))
    delta_x = np.array([])
    delta_y = np.array([])
    sum = np.array([])
    for i in range(file_number):
        df = pd.read_table(file_path + file_prefix + '{0}.lvm'.format(i), sep='\t', header=None)
        delta_x = np.append(delta_x, df[1].mean())
        delta_y = np.append(delta_y, df[2].mean())
        sum = np.append(sum, df[3].mean())

    heatmap_data_x = np.reshape(delta_x, (19, 19))
    heatmap_data_y = np.reshape(delta_y, (19, 19))
    heatmap_data_sum = np.reshape(sum, (19, 19))

    heatmap_data_x[1::2, :] = heatmap_data_x[1::2, ::-1]

    # SETTING SEABORN PALETTE


    # PLOT FOR THE DELTA X CHANNEL
    fig_x = plt.figure(figsize=(10, 5))
    # fig_x.suptitle(r'$\Delta x$')
    # heatmap plot
    ax1_x = fig_x.add_subplot()
    sns.heatmap(heatmap_data_x, ax=ax1_x, square=True, cbar=True, cmap='viridis', cbar_kws={'label': r'$\Delta x [mV]$'})
    # im_x = ax1_x.imshow(heatmap_data_x.T, cmap=plt.get_cmap('jet'), aspect='auto')
    # cbar = fig_x.colorbar(im_x)
    # cbar.set_label(r'$\Delta x [mV]$', rotation=270, labelpad=15)
    ax1_x.set_xticks(np.arange(heatmap_data_x.shape[0]))
    ax1_x.set_yticks(np.arange(heatmap_data_x.shape[0]))
    ax1_x.set_xticklabels(np.arange(-450, 500, 50))
    ax1_x.set_yticklabels(np.arange(-450, 500, 50))
    ax1_x.set_title(r'$\Delta x$')
    ax1_x.set_xlabel(r'$x [\mu m]$')
    ax1_x.set_ylabel(r'$y [\mu m]$')

    # PLOT FOR THE DELTA Y CHANNEL
    fig_y = plt.figure(figsize=(10, 5))
    # fig_y.suptitle(r'$\Delta y$')
    # heatmap plot
    ax1_y = fig_y.add_subplot()
    sns.heatmap(heatmap_data_y, ax=ax1_y, square=True, cbar=True, cmap='viridis',  cbar_kws={'label': r'$\Delta y [mV]$'})
    # im_y = ax1_y.imshow(heatmap_data_y.T, cmap=plt.get_cmap('jet'), aspect='auto')
    # cbar = fig_y.colorbar(im_y)
    # cbar.set_label(r'$\Delta y [V]$', rotation=270, labelpad=15)
    ax1_y.set_xticks(np.arange(heatmap_data_y.shape[0]))
    ax1_y.set_yticks(np.arange(heatmap_data_y.shape[0]))
    ax1_y.set_xticklabels(np.arange(-450, 500, 50))
    ax1_y.set_yticklabels(np.arange(-450, 500, 50))
    ax1_y.set_title(r'$\Delta y$')
    ax1_y.set_xlabel(r'$x [\mu m]$')
    ax1_y.set_ylabel(r'$y [\mu m]$')

    # # PLOT FOR THE SUM CHANNEL
    fig_sum = plt.figure(figsize=(10, 5))
    # fig_sum.suptitle(r'$\Sigma$')
    # heatmap plot
    ax1_sum = fig_sum.add_subplot()
    sns.heatmap(heatmap_data_sum, ax=ax1_sum, square=True, cbar=True, cmap='viridis',  cbar_kws={'label': r'$\Sigma [mV]$'})
    # im_sum = ax1_sum.imshow(heatmap_data_sum.T, cmap=plt.get_cmap('jet'), aspect='auto')
    # cbar = fig_sum.colorbar(im_sum)
    # cbar.set_label(r'$\Sigma [V]$', rotation=270, labelpad=15)
    ax1_sum.set_xticks(np.arange(heatmap_data_sum.shape[0]))
    ax1_sum.set_yticks(np.arange(heatmap_data_sum.shape[0]))
    ax1_sum.set_xticklabels(np.arange(-450, 500, 50))
    ax1_sum.set_yticklabels(np.arange(-450, 500, 50))
    ax1_sum.set_title(r'$\Sigma$')
    ax1_sum.set_xlabel(r'$x [\mu m]$')
    ax1_sum.set_ylabel(r'$y [\mu m]$')

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
