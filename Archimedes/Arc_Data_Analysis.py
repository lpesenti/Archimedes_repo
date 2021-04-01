import matplotlib.pyplot as plt
import matplotlib as mpl
import Arc_functions as af
import timeit
import os

mpl.rcParams['agg.path.chunksize'] = 5000000
path_to_img = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Images"

if __name__ == '__main__':
    start = timeit.default_timer()
    # af.abs_derivative(day=19, month=2, year=2021, quantity='ITF')
    fig, ax = plt.subplots()
    ax, fname = af.time_evolution(day=19, month=2, year=2021, quantity='ITF', ax=ax)
    # ax, fname1 = af.time_evolution(day=19, month=2, year=2021, quantity='Pick Off', ax=ax)
    ax, fname2 = af.der_plot(day=19, month=2, year=2021, quantity='ITF', ax=ax, threshold=0.002)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    #
    # fig1, ax1 = plt.subplots()
    # ax1, fname2 = af.coherence(sign1='ITF', sign2='Pick Off', day=19, month=2, year=2021, ax=ax1)
    # # ax1, fname3 = af.time_evolution(day=19, month=2, year=2021, quantity='Pick Off', ax=ax1, ndays=2)
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')

    stop = timeit.default_timer()
    print('Time before plt.show(): ', (stop - start), 's')
    print('Time before plt.show(): ', (stop - start) / 60, 'min')
    # fig.savefig(os.path.join(path_to_img, fname + fname1 + fname2 + '_DER.png'))
    # fig1.savefig(os.path.join(path_to_img, fname2 + '.png'))
    stop1 = timeit.default_timer()
    print('Time after saving png: ', (stop1 - start), 's')
    print('Time after saving png: ', (stop1 - start) / 60, 'min')
    plt.show()
