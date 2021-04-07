import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import Arc_functions as arc
import numpy as np
import timeit
import os
import logging

mpl.rcParams['agg.path.chunksize'] = 5000000
path_to_img = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Images"
logging.basicConfig(filename='Arc_Data_Analysis_info.log',
                    level=logging.INFO,
                    filemode='w',
                    format='[%(asctime)s %(filename)s %(funcName)20s()] %(levelname)s   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(filename='Arc_Data_Analysis_debug.log',
                    level=logging.DEBUG,
                    filemode='w',
                    format='[%(asctime)s %(filename)s %(funcName)20s()] %(levelname)s   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def plot_3d(x, y, z, dx, dy, dz, heat_data, axes1, axes2):
    logging.debug('Data read completed')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    dz_normed = dz / np.amax(dz)
    normed_cbar = colors.Normalize(dz_normed.min(), dz_normed.max())
    color = cm.jet(normed_cbar(dz_normed))

    axes1.bar3d(x, y, z, dx, dy, dz, shade=True, color=color)
    axes1.grid(True, linestyle='-')
    axes1.set_title('Fraction of values rejected')
    axes1.set_xlabel('Threshold')
    axes1.set_ylabel('ndivision')
    axes1.set_zlabel('Fraction rejected')

    im = axes2.imshow(heat_data, cmap=plt.get_cmap('jet'), aspect='auto')

    axes2.set_xticks(np.arange(len(ndivision_val)))
    axes2.set_yticks(np.arange(len(threshold_val)))
    x_labels = ["".join(item) for item in np.round(ndivision_val, 4).astype(str)]
    y_labels = ["".join(item) for item in np.round(threshold_val, 4).astype(str)]
    y_labels = ['' if (i + 1) % 2 == 0 else y_labels[i] for i in range(len(threshold_val))]
    axes2.set_xticklabels(x_labels)
    axes2.set_yticklabels(y_labels)
    axes2.set_title('Heatmap')
    axes2.set_xlabel('ndivision')
    axes2.set_ylabel('Threshold')

    return axes1, im


if __name__ == '__main__':
    start = timeit.default_timer()
    logging.info('Started')
    data, _, _ = arc.read_data(day=19, month=2, year=2021, col_to_save='ITF', num_d=1, verbose=True)
    data1, _, _ = arc.read_data(day=20, month=2, year=2021, col_to_save='ITF', num_d=1, verbose=True)
    data2, _, _ = arc.read_data(day=28, month=2, year=2021, col_to_save='ITF', num_d=1, verbose=True)

    x, y, dz = np.array([]), np.array([]), np.array([])
    dz1, dz2 = np.array([]), np.array([])
    data_matrix, data_matrix1, data_matrix2 = [], [], []
    threshold_val = np.arange(0.001, 0.01, 0.0001)
    ndivision_val = np.arange(100, 2100, 100)

    for i in threshold_val:
        print(round(i / 0.01 * 100, 2), '%')
        data_to_heatmap = np.array([])
        data_to_heatmap1 = np.array([])
        data_to_heatmap2 = np.array([])
        for j in ndivision_val:
            _, _, val_rej = arc.th_comparison(data, threshold=i, ndivision=j, verbose=False)
            _, _, val_rej1 = arc.th_comparison(data1, threshold=i, ndivision=j, verbose=False)
            _, _, val_rej2 = arc.th_comparison(data2, threshold=i, ndivision=j, verbose=False)
            x = np.append(x, i)
            y = np.append(y, j)
            data_to_heatmap = np.append(data_to_heatmap, val_rej)
            data_to_heatmap1 = np.append(data_to_heatmap1, val_rej1)
            data_to_heatmap2 = np.append(data_to_heatmap2, val_rej2)
            dz = np.append(dz, val_rej)
            dz1 = np.append(dz1, val_rej1)
            dz2 = np.append(dz2, val_rej2)
        data_matrix.append(data_to_heatmap)
        data_matrix1.append(data_to_heatmap1)
        data_matrix2.append(data_to_heatmap2)

    heatmap_data = np.array(data_matrix)
    heatmap_data1 = np.array(data_matrix1)
    heatmap_data2 = np.array(data_matrix2)
    bottom = np.zeros(x.size)
    width = 0.0001
    depth = 100

    fig = plt.figure()
    fig.suptitle('Data from 19/02/2021')
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax0 = fig.add_subplot(spec[0, 0], projection='3d')
    ax1 = fig.add_subplot(spec[0, 1])
    ax0, im0 = plot_3d(x, y, bottom, width, depth, dz, heatmap_data, ax0, ax1)
    cbar = fig.colorbar(im0)
    cbar.set_label('Fraction rejected', rotation=90, labelpad=15)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig1 = plt.figure()
    fig1.suptitle('Data from 20/02/2021')
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
    ax2 = fig1.add_subplot(spec1[0, 0], projection='3d')
    ax3 = fig1.add_subplot(spec1[0, 1])
    ax2, im1 = plot_3d(x, y, bottom, width, depth, dz1, heatmap_data1, ax2, ax3)
    cbar1 = fig1.colorbar(im1)
    cbar1.set_label('Fraction rejected', rotation=90, labelpad=15)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig2 = plt.figure()
    fig2.suptitle('Data from 28/02/2021')
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig2)
    ax4 = fig2.add_subplot(spec1[0, 0], projection='3d')
    ax5 = fig2.add_subplot(spec1[0, 1])
    ax4, im2 = plot_3d(x, y, bottom, width, depth, dz2, heatmap_data2, ax4, ax5)
    cbar2 = fig2.colorbar(im2)
    cbar2.set_label('Fraction rejected', rotation=90, labelpad=15)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    # fig.savefig(os.path.join(path_to_img, r'2021219_3d_heat.png'))
    # fig1.savefig(os.path.join(path_to_img, r'2021220_3d_heat.png'))
    # fig2.savefig(os.path.join(path_to_img, r'2021228_3d_heat.png'))
    stop1 = timeit.default_timer()
    print('Time after saving png: ', (stop1 - start), 's')
    print('Time after saving png: ', (stop1 - start) / 60, 'min')
    plt.show()
