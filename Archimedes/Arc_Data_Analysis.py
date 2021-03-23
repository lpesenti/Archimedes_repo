from matplotlib.mlab import cohere
from math import *
from matplotlib import mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
import timeit
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import os
import glob
import re
import matplotlib.ticker as mticker

mpl.rcParams['agg.path.chunksize'] = 5000000
path_to_data = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Data"


def read_data(day, month, year, col_to_save):
    month = '%02d' % month  # It transforms 2 -> 02
    cols = np.array(
        ['ITF', 'Pick Off', 'Signal injected', 'Error', 'Correction', 'Actuator 1', 'Actuator 2', 'After Noise',
         'Time'])
    index = np.where(cols == col_to_save)[0][0]
    data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day))
    final_df = pd.DataFrame()
    all_data = glob.glob(os.path.join(data_folder, "*.lvm"))
    all_data.sort(key=lambda f: int(re.sub('\D', '', f)))
    i = 0
    for data in all_data:
        print(round(i / len(all_data) * 100, 1), '%')
        a = pd.read_table(data, sep='\t', usecols=[index + 1], header=None)
        final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
        i += 1
    return final_df


def time_evolution(day, month, year, quantity, ax):
    df = read_data(day, month, year, quantity)
    ax.plot(df.index, df[1], label=quantity)
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    return ax


if __name__ == '__main__':
    start = timeit.default_timer()
    fig, ax = plt.subplots()
    ax = time_evolution(day=19, month=2, year=2021, quantity='ITF', ax=ax)
    ax.legend(loc='best', shadow=True, fontsize='medium')
    stop = timeit.default_timer()
    print('Time: ', (stop - start) / 60)
    plt.show()
