import datetime
from matplotlib.mlab import cohere
import numpy as np
import pandas as pd
import os
import glob
import re

unix_time = 0
path_to_data = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Data"
freq = 1000  # Hz


def read_data(day, month, year, col_to_save, num_d):
    print('--------------- Reading', col_to_save, 'data ---------------')
    month = '%02d' % month  # It transforms 2 -> 02
    cols = np.array(
        ['ITF', 'Pick Off', 'Signal injected', 'Error', 'Correction', 'Actuator 1', 'Actuator 2', 'After Noise',
         'Time'])
    index = np.where(cols == col_to_save)[0][0] + 1
    all_data = []
    final_df = pd.DataFrame()
    for i in range(num_d):
        data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day + i))
        temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
        temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_data += temp_data
    time = pd.read_table(all_data[0], sep='\t', usecols=[9], header=None)
    start_t = time[9][0].replace("\\", '')
    timestamp = datetime.datetime.timestamp(pd.to_datetime(start_t))
    i = 0
    for data in all_data:
        print(round(i / len(all_data) * 100, 1), '%')
        a = pd.read_table(data, sep='\t', usecols=[index], header=None)
        final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
        i += 1
    print('--------------- Reading completed! ---------------')
    return final_df, index, timestamp


def log_tick_formatter(val, pos=None):
    global unix_time
    if val % 1000 == 0:
        val = str(datetime.datetime.fromtimestamp(int(unix_time + val * 0.001)))
    else:
        val = ''
    return val


def time_evolution(day, month, year, quantity, ax, ndays=1):
    df, col_index, t = read_data(day, month, year, quantity, ndays)
    global unix_time
    unix_time = t
    print('--------------- Building Plot! ---------------')
    lab = quantity + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays)
    ax.plot(df.index, df[col_index], label=lab)
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    ax.xaxis.set_major_formatter(log_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='medium')
    # ax.xaxis.set_tick_params(rotation=30)
    return ax, filename


def coherence(sign1, sign2, day, month, year, ax, ndays=1, day2=None, month2=None, year2=None, samedate=True):
    if samedate:
        df, col_index, t = read_data(day, month, year, sign1, ndays)
        df1, col_index1, t1 = read_data(day, month, year, sign2, ndays)
    else:
        df, col_index, t = read_data(day, month, year, sign1, ndays)
        df1, col_index1, t1 = read_data(day2, month2, year2, sign2, ndays)
    global unix_time
    unix_time = t
    print()
    print('--------------- Building Plot! ---------------')
    lab = 'Coh: ' + sign1 + '-' + sign2 + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_Coh_' + sign1 + '-' + sign2 + '_nDays_' + str(ndays)
    c12, f12 = cohere(df[col_index], df1[col_index1], NFFT=10000, Fs=freq, detrend='linear')
    ax.plot(f12, c12, label=lab)
    ax.grid(True, linestyle='-')
    ax.set_xscale("log")
    ax.set_xlabel('Frequencies [Hz]')
    ax.set_ylabel('Coherence')
    ax.legend(loc='best', shadow=True, fontsize='medium')
    # ax.xaxis.set_major_formatter(log_tick_formatter)
    # ax.xaxis.set_tick_params(rotation=30)
    return ax, filename
