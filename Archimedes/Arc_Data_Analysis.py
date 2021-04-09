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
logging.basicConfig(filename=r'.\logs\debug.log',
                    level=logging.DEBUG,
                    filemode='w',
                    format='[%(asctime)s %(filename)s %(funcName)20s()] %(levelname)s   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def plot_3d(day, month, year, col_to_save, num_d, axes1, axes2, verbose=True):
    # In order to work properly, comment lines 250,251,252
    # in th_comparison function
    logging.debug('Started')
    data, _, _ = arc.read_data(day=day, month=month, year=year, col_to_save=col_to_save, num_d=num_d)

    x, y, dz = np.array([]), np.array([]), np.array([])
    data_matrix = []
    threshold_val = np.arange(0.001, 0.01, 0.0001)
    ndivision_val = np.arange(100, 2100, 100)

    for i in threshold_val:
        print(round(i / 0.01 * 100, 2), '%')
        data_to_heatmap = np.array([])
        for j in ndivision_val:
            _, _, val_rej = arc.th_comparison(data, threshold=i, ndivision=j, verbose=False)
            x = np.append(x, i)
            y = np.append(y, j)
            data_to_heatmap = np.append(data_to_heatmap, val_rej)
            dz = np.append(dz, val_rej)
        data_matrix.append(data_to_heatmap)

    heatmap_data = np.array(data_matrix)
    bottom = np.zeros(x.size)
    width = 0.0001
    depth = 100
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    dz_normed = dz / np.amax(dz)
    normed_cbar = colors.Normalize(dz_normed.min(), dz_normed.max())
    color = cm.jet(normed_cbar(dz_normed))

    axes1.bar3d(x, y, bottom, width, depth, dz, shade=True, color=color)
    axes1.grid(True, linestyle='-')
    axes1.set_title('Fraction of values rejected')
    axes1.set_xlabel('Threshold')
    axes1.set_ylabel('ndivision')
    axes1.set_zlabel('Fraction rejected')

    im = axes2.imshow(heatmap_data, cmap=plt.get_cmap('jet'), aspect='auto')

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


def make_plot_3d():
    start = timeit.default_timer()
    fig = plt.figure()
    fig.suptitle('Data from 19/02/2021')
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    ax0 = fig.add_subplot(spec[0, 0], projection='3d')
    ax1 = fig.add_subplot(spec[0, 1])
    ax0, im0 = plot_3d(day=19, month=2, year=2021, col_to_save='ITF', num_d=1, axes1=ax0, axes2=ax1)
    cbar = fig.colorbar(im0)
    cbar.set_label('Fraction rejected', rotation=90, labelpad=15)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # fig.savefig(os.path.join(path_to_img, r'2021219_3d_heat.png'))
    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    stop1 = timeit.default_timer()
    print('Time after saving png: ', (stop1 - start), 's')
    print('Time after saving png: ', (stop1 - start) / 60, 'min')
    logging.info('File .png saved')


def nine_der_plot(day, month, year, col_to_save, title):
    logging.info('Initialized')
    logging.debug('PARAMETERS: day=%i month=%i year=%i col_to_save=%s title=%s' % (
        day, month, year, col_to_save, title))
    fig1 = plt.figure()
    fig1.suptitle(title)
    logging.debug('fig1 created with title')
    fig2 = plt.figure()
    fig2.suptitle(title)
    logging.debug('fig2 created with title')
    fig3 = plt.figure()
    fig3.suptitle(title)
    logging.debug('fig3 created with title')
    fig4 = plt.figure()
    fig4.suptitle(title)
    logging.debug('fig4 created with title')
    fig5 = plt.figure()
    fig5.suptitle(title)
    logging.debug('fig5 created with title')
    fig6 = plt.figure()
    fig6.suptitle(title)
    logging.debug('fig6 created with title')
    fig7 = plt.figure()
    fig7.suptitle(title)
    logging.debug('fig7 created with title')
    fig8 = plt.figure()
    fig8.suptitle(title)
    logging.debug('fig8 created with title')
    fig9 = plt.figure()
    fig9.suptitle(title)
    logging.debug('fig9 created with title')
    # spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig1.add_subplot()
    logging.debug('ax1 added')
    ax2 = fig2.add_subplot()
    logging.debug('ax2 added')
    ax3 = fig3.add_subplot()
    logging.debug('ax3 added')
    ax4 = fig4.add_subplot()
    logging.debug('ax4 added')
    ax5 = fig5.add_subplot()
    logging.debug('ax5 added')
    ax6 = fig6.add_subplot()
    logging.debug('ax6 added')
    ax7 = fig7.add_subplot()
    logging.debug('ax7 added')
    ax8 = fig8.add_subplot()
    logging.debug('ax8 added')
    ax9 = fig9.add_subplot()
    logging.debug('ax9 added')
    logging.info('Starting with der_plot')
    ax1, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax1, threshold=0.001, ndivision=100)
    logging.debug('ax1 der_plot completed')
    ax2, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax2, threshold=0.0022,
                          ndivision=100)
    logging.debug('ax2 der_plot completed')
    ax3, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax3, threshold=0.0044,
                          ndivision=100)
    logging.debug('ax3 der_plot completed')
    ax4, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax4, threshold=0.001, ndivision=500)
    logging.debug('ax4 der_plot completed')
    ax5, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax5, threshold=0.0022,
                          ndivision=500)
    logging.debug('ax5 der_plot completed')
    ax6, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax6, threshold=0.0044,
                          ndivision=500)
    logging.debug('ax6 der_plot completed')
    ax7, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax7, threshold=0.001,
                          ndivision=1000)
    logging.debug('ax7 der_plot completed')
    ax8, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax8, threshold=0.0022,
                          ndivision=1000)
    logging.debug('ax8 der_plot completed')
    ax9, _ = arc.der_plot(day=day, month=month, year=year, quantity=col_to_save, ax=ax9, threshold=0.0044,
                          ndivision=1000)
    logging.debug('ax9 der_plot completed')
    logging.info('Finished der_plot evaluation')
    logging.info('Starting with time_evolution')
    ax1, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax1, verbose=False)
    logging.debug('ax1 time_evolution completed')
    ax2, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax2, verbose=False)
    logging.debug('ax2 time_evolution completed')
    ax3, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax3, verbose=False)
    logging.debug('ax3 time_evolution completed')
    ax4, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax4, verbose=False)
    logging.debug('ax4 time_evolution completed')
    ax5, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax5, verbose=False)
    logging.debug('ax5 time_evolution completed')
    ax6, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax6, verbose=False)
    logging.debug('ax6 time_evolution completed')
    ax7, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax7, verbose=False)
    logging.debug('ax7 time_evolution completed')
    ax8, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax8, verbose=False)
    logging.debug('ax8 time_evolution completed')
    ax9, _ = arc.time_evolution(day=day, month=month, year=year, quantity=col_to_save, ax=ax9, verbose=False)
    logging.debug('ax9 time_evolution completed')
    ax1.set_title('Th=0.001, ndiv=100')
    ax2.set_title('Th=0.0022, ndiv=100')
    ax3.set_title('Th=0.0044, ndiv=100')
    ax4.set_title('Th=0.001, ndiv=500')
    ax5.set_title('Th=0.0022, ndiv=500')
    ax6.set_title('Th=0.0044, ndiv=500')
    ax7.set_title('Th=0.001, ndiv=1000')
    ax8.set_title('Th=0.0022, ndiv=1000')
    ax9.set_title('Th=0.0044, ndiv=1000')
    logging.info('Titles set')
    fig1.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig1.png'.format(year, month, day)))
    fig2.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig2.png'.format(year, month, day)))
    fig3.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig3.png'.format(year, month, day)))
    fig4.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig4.png'.format(year, month, day)))
    fig5.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig5.png'.format(year, month, day)))
    fig6.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig6.png'.format(year, month, day)))
    fig7.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig7.png'.format(year, month, day)))
    fig8.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig8.png'.format(year, month, day)))
    fig9.savefig(os.path.join(path_to_img, r'{0}{1}{2}_fig9.png'.format(year, month, day)))
    logging.info('Images saved')


if __name__ == '__main__':
    logging.info('Started')
    start = timeit.default_timer()

    # nine_der_plot(day=20, month=2, year=2021, col_to_save='ITF', title='Data from 20/02/2021')
    # nine_der_plot(day=28, month=2, year=2021, col_to_save='ITF', title='Data from 28/02/2021')

    fig1 = plt.figure()
    fig1.suptitle('Data from 19/02/2021 - No derivative')

    ax1 = fig1.add_subplot()
    ax1, _ = arc.der_plot(day=19, month=2, year=2021, quantity='ITF', ax=ax1, threshold=0.03, ndivision=10000,
                          verbose=True)
    ax1, _ = arc.time_evolution(day=19, month=2, year=2021, quantity='ITF', ax=ax1, verbose=False)
    ax1.set_title('Th=0.03, ndiv=10000')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig2 = plt.figure()
    fig2.suptitle('Data from 20/02/2021 - No derivative')

    ax2 = fig2.add_subplot()
    ax2, _ = arc.der_plot(day=20, month=2, year=2021, quantity='ITF', ax=ax2, threshold=0.03, ndivision=10000,
                          verbose=True)
    ax2, _ = arc.time_evolution(day=20, month=2, year=2021, quantity='ITF', ax=ax2, verbose=False)
    ax2.set_title('Th=0.03, ndiv=10000')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig3 = plt.figure()
    fig3.suptitle('Data from 20/02/2021 - No derivative')

    ax3 = fig3.add_subplot()
    ax3, _ = arc.der_plot(day=28, month=2, year=2021, quantity='ITF', ax=ax3, threshold=0.03, ndivision=10000,
                          verbose=True)
    ax3, _ = arc.time_evolution(day=28, month=2, year=2021, quantity='ITF', ax=ax3, verbose=False)
    ax3.set_title('Th=0.03, ndiv=10000')
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')

    fig1.savefig(os.path.join(path_to_img, r'2021219_th_003_nd_10000_noDerivative_FULL.png'))
    fig2.savefig(os.path.join(path_to_img, r'2021220_th_003_nd_10000_noDerivative_FULL.png'))
    fig3.savefig(os.path.join(path_to_img, r'2021228_th_003_nd_10000_noDerivative_FULL.png'))

    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    plt.show()
    logging.info('Finished')
