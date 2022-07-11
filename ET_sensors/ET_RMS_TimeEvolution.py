__author__ = "Luca Pesenti"
__credits__ = ["Sara Anzuinelli", "Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.1.0"
__maintainer__ = "Luca Pesenti"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

import concurrent.futures
import configparser
import datetime
import functools
import glob
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ET_common as Ec

# TODO: add check on python packages and add option for installing them
# TODO: add sensibility of the seismometer (e.g. x_lim = 1/240s)

config = configparser.ConfigParser()
config.read('RMS_time_config.ini')

# TODO: add check for file already existing
df_path = config['Paths']['outDF_path']  # TODO: change outDF_path name in a more right one
i_min = float(config['Quantities']['integral_min'])
i_max = float(config['Quantities']['integral_max'])
out_error = config.getboolean('DEFAULT', 'save_error')
freq_df_path = Ec.check_dir(df_path, 'Freq_df')
npz_path = Ec.check_dir(df_path, 'npz_files')
daily_path = Ec.check_dir(df_path, 'daily_df')
t_log = time.strftime('%Y%m%d_%H%M')
f_len = 0

# Twindow = int(config['Quantities']['psd_window'])  # TODO: remove this line and make it automatically check on filename


def eval_rms(filename, freq_len, psd_len, Twindow, chunk_index):
    # global i_min, i_max
    temp_df = pd.read_parquet(filename)
    day_num = filename.replace(daily_path + '\\', '').split('.')[0].split('_')[1].split('-')[1]
    year = filename.replace(daily_path + '\\', '').split('.')[0].split('_')[1].split('-')[0]
    # evaluate the integral in the frequency region between i_min and i_max
    rms = temp_df.iloc[chunk_index * freq_len:chunk_index * freq_len + freq_len].loc[i_min:i_max].sum()[0] * freq_len
    date = datetime.datetime.strptime(year + "-" + day_num, "%Y-%j") + datetime.timedelta(seconds=psd_len * chunk_index)
    if out_error and rms == 0:
        print(temp_df.iloc[chunk_index * freq_len:chunk_index * freq_len + freq_len].loc[i_min:i_max])
        with open(df_path + fr'\RMS-Error_{t_log}.txt', 'a') as text_file:
            print(f'{filename}|{date}|{Twindow}', file=text_file)
    # date = date.timestamp()
    return rms, date


def to_rms():
    # global daily_path, freq_df_path, npz_path, f_len  # , Twindow
    filename_list = glob.glob(daily_path + r"\*.brotli")
    Twindow = filename_list[0].split('_')[2][-4:]  # TODO: find a better way
    data = np.load(npz_path + fr'\{Twindow}_Frequency.npz')
    freq_data = data['frequency']
    f_len = len(freq_data)
    temporary_df = pd.read_parquet(filename_list[0])  # assuming all the df have the same length (i.e. number of psd)
    for_length = len(temporary_df.loc[freq_data.min()])  # number of psd in the df by looking at the freqs repetitions
    psd_length = 24 * 3600 / for_length
    rms_date_lst = []
    num_asd = np.arange(for_length)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for f_index, file in enumerate(filename_list):
            j1 = (f_index + 1) / len(filename_list)
            print("\r[%-75s] %g%%" % ('=' * int(75 * j1), round(100 * j1, 3)), end='\n')
            results = executor.map(functools.partial(eval_rms, file, f_len, psd_length, Twindow), num_asd)
            for res in results:
                rms_date_lst.append(res)
    return rms_date_lst


def plot_from_df(x_array, y_array, ax, xlabel='', ylabel=r'RMS', label_size=24, xscale='linear', yscale='linear'):
    ax.plot(x_array, y_array, linestyle='--', marker='o', markersize=10, linewidth=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M:%S"))

    # ax.set_ylim([-200, -30])

    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    # ax.legend(loc='best', shadow=True, fontsize=24)
    ax.grid(True, linestyle='--', axis='both', which='both')
    fig.tight_layout()

    return ax


def print_on_screen(symbol1, message, quantity, symbol2=None):
    global df_path, t_log

    with open(df_path + fr'\{t_log}.txt', 'a') as text_file:
        if symbol2 is not None:
            if symbol1 == '+':
                print(f'{symbol1}', f'{symbol2}' * 98, f'{symbol1}', sep='', file=text_file)
                print(f'| {message}'.ljust(70, '.'), f"{round(quantity, 3)} seconds |".rjust(30, '.'), sep='',
                      file=text_file)
                print(f'| {message}'.ljust(70, '.'), f"{round(quantity / 60, 3)} minutes |".rjust(30, '.'), sep='',
                      file=text_file)
                print(f'| {message}'.ljust(70, '.'), f"{round(quantity / 3600, 3)} hours |".rjust(30, '.'), sep='',
                      file=text_file)
                print(f'{symbol1}', f'{symbol2}' * 98, f'{symbol1}', sep='', file=text_file)
        else:
            print(f'{symbol1}' * 100, file=text_file)
            print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity, 3)} seconds {symbol1}".rjust(30, '.'),
                  sep='', file=text_file)
            print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity / 60, 3)} minutes {symbol1}".rjust(30, '.'),
                  sep='', file=text_file)
            print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity / 3600, 3)} hours {symbol1}".rjust(30, '.'),
                  sep='', file=text_file)
            print(f'{symbol1}' * 100, file=text_file)
    if symbol2 is not None:
        if symbol1 == '+':
            print(f'{symbol1}', f'{symbol2}' * 98, f'{symbol1}', sep='')
            print(f'| {message}'.ljust(70, '.'), f"{round(quantity, 3)} seconds |".rjust(30, '.'), sep='')
            print(f'| {message}'.ljust(70, '.'), f"{round(quantity / 60, 3)} minutes |".rjust(30, '.'), sep='')
            print(f'| {message}'.ljust(70, '.'), f"{round(quantity / 3600, 3)} hours |".rjust(30, '.'), sep='')
            print(f'{symbol1}', f'{symbol2}' * 98, f'{symbol1}', sep='')
    else:
        print(f'{symbol1}' * 100)
        print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity, 3)} seconds {symbol1}".rjust(30, '.'), sep='')
        print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity / 60, 3)} minutes {symbol1}".rjust(30, '.'),
              sep='')
        print(f'{symbol1} {message}'.ljust(70, '.'), f"{round(quantity / 3600, 3)} hours {symbol1}".rjust(30, '.'),
              sep='')
        print(f'{symbol1}' * 100)


if __name__ == '__main__':
    t0 = time.perf_counter()
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    data = to_rms()
    plot_from_df([x[1] for x in data], [x[0] for x in data], ax=ax)
    t1 = time.perf_counter()
    print_on_screen(symbol1='+', symbol2='-', message='RMS evaluation', quantity=t1 - t0)

    # print_on_screen(symbol1='*', message=f'Total time elapsed', quantity=t1 - t0)
    plt.show()
