__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.2.0"
__maintainer__ = "Luca Pesenti"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

import concurrent.futures
import configparser
import functools
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import mlab
from obspy import UTCDateTime
from obspy import read
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm

import ET_common as Ec
import ET_functions as Ef

r"""
[LAST UPDATE: 25 May 2022 - Luca Pesenti]

!!! BEFORE CHANGING ANY PART OF THE CODE, PLEASE CONTACT THE MAINTAINER !!!

This is a self-contained script and to use it, please change only the relative config file (Quantile_config.ini).
For further information see the README file in this repository

The following functions have been built to work with the data obtained by the seismometer used at the Sos Enattos site
and uploaded to the et-repo.
The data are stored in daily file with the name-format: 

                {NETWORK}.{SENSOR}.{LOCATION}.{CHANNEL}.D.{YEAR}.{FILE_NUMBER}      
                
The logic of the code is:
    1. Read file ---> Perform PSD [x 24_hours/(psd_time_window)] ---> Create a dataframe (daily_df)
           ^                                                                             |            
           |                                                                             | x number of files          
           |                                                                             |             
           +-----------------------------------------------------------------------------+
           
    2. Split frequency into sub-arrays of len=100 ---> Open daily_df ---> Group by frequency ---> Save frequency_df
                                                            ^                                            |            
                                                            |                                            |             
                                                            |            x number sub-arrays             |             
                                                            +--------------------------------------------+
           
    3. Open frequency_df ---> Evaluate quantile
           ^                      |            
           |                      | x number of quantiles            
           |                      |             
           +----------------------+  
           
    4. Plot results         
"""

# TODO: add check on python packages and add option for installing them
# TODO: add sensibility of the seismometer (e.g. x_lim = 1/240s)

config = configparser.ConfigParser()
config.read('Quantile_config.ini')

# Paths
df_path = config['Paths']['sensor_path']  # TODO: add check for file already existing
filexml = config['Paths']['xml_path']
Data_path = config['Paths']['data_path']

# Quantities
Twindow = int(config['Quantities']['psd_window'])
TLong = int(config['Quantities']['TLong'])
Overlap = float(config['Quantities']['psd_overlap'])  # TODO: check if it is the right way to perform overlap
quantiles = [float(x) for x in config['Quantities']['quantiles'].split(',')]

# Instrument
network = config['Instrument']['network']
sensor = config['Instrument']['sensor']
location = config['Instrument']['location']
channel = config['Instrument']['channel']

# Booleans
verbose = config.getboolean('DEFAULT', 'verbose')
unit = config['DEFAULT']['unit']  # TODO: enable the VEL option
only_daily = config.getboolean('DEFAULT', 'only_daily')
skip_daily = config.getboolean('DEFAULT', 'skip_daily')
skip_freq_df = config.getboolean('DEFAULT', 'skip_freq_df')
skip_quant_eval = config.getboolean('DEFAULT', 'skip_quant_eval')

# Directory check and creation
freq_df_path = Ec.check_dir(df_path, 'Freq_df')
npz_path = Ec.check_dir(df_path, 'npz_files')
daily_path = Ec.check_dir(df_path, 'daily_df')
log_path = Ec.check_dir(df_path, 'logs')

t_log = time.strftime('%Y%m%d_%H%M')


def make_daily():
    seed_id = network + '.' + sensor + '.' + location + '.' + channel

    filename_list = glob.glob(Data_path + seed_id + "*")
    filename_list.sort()

    st1 = read(filename_list[0])
    st1 = st1.sort()
    startdate = st1[-1:][0].times('timestamp')[0]
    startdate = UTCDateTime(startdate)
    st2 = read(filename_list[len(filename_list) - 1])
    st2 = st2.sort()
    stopdate = st2[0].times('timestamp')[-1:][0]
    stopdate = UTCDateTime(stopdate)

    # TODO: thread or pool? Read: https://superfastpython.com/threadpoolexecutorv-vs-processpoolexecutor/
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:  # os.cpu_count()//2
        # for f_index, file in enumerate(filename_list):
        #     j1 = (f_index + 1) / len(filename_list)
        #     print("\r[%-75s] %g%%" % ('=' * int(75 * j1), round(100 * j1, 3)), end='\n')
        executor.map(functools.partial(daily_df, startdate, stopdate), filename_list)


def daily_df(startdate, stopdate, file):
    dT = TLong + Twindow * Overlap
    M = int((dT - Twindow) / (Twindow * (1 - Overlap)) + 1)
    K = int((stopdate - startdate) / (Twindow * (1 - Overlap)) + 1)
    v = np.empty(K)
    fsxml = 100
    Num = int(Twindow * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    freq_path = npz_path + fr'\{Twindow}_Frequency.npz'
    np.savez(freq_path, frequency=f) if not os.path.exists(freq_path) else ''
    w = 2.0 * np.pi * f

    temp_array = np.array([])

    if verbose:
        print('+', '-' * 84, '+', sep='')
        print('| The analysis start from', startdate, 'to', stopdate, '|')
        print('| K:'.ljust(30, '.'), '{0} |'.format(K).rjust(56, '.'), sep='')
        print('| M:'.ljust(30, '.'), '{0} |'.format(M).rjust(56, '.'), sep='')
        print('| Minimum Frequency:'.ljust(30, '.'), '{0} |'.format(f.min()).rjust(56, '.'), sep='')
        print('| Maximum Frequency:'.ljust(30, '.'), '{0} |'.format(f.max()).rjust(56, '.'), sep='')
        print('| Frequency Length'.ljust(30, '.'), '{0} |'.format(len(f)).rjust(56, '.'), sep='')
        print('| PSD time window'.ljust(30, '.'), '{0} |'.format(Twindow).rjust(56, '.'), sep='')
        print('| Overlap'.ljust(30, '.'), '{0} |'.format(Overlap).rjust(56, '.'), sep='')
        print('| Slice length'.ljust(30, '.'), '{0} |'.format(TLong).rjust(56, '.'), sep='')
        print('+', '-' * 84, '+', sep='')

    k = 0

    st = read(file)
    st = st.sort()

    Tstop = st[-1:][0].times('timestamp')[-1:][0]
    Tstop = UTCDateTime(Tstop)
    time = st[0].times('timestamp')[0]
    time = UTCDateTime(time)

    fxml, respamp, fsxml, gain = Ef.read_Inv(filexml, network, sensor, location, channel, time, Twindow,
                                             verbose=verbose)
    if unit.upper() == 'VEL':
        w1 = 1 / respamp
    elif unit.upper() == 'ACC':
        w1 = w ** 2 / respamp
    else:
        import warnings
        msg = "Invalid data unit chosen [VEl or ACC], using VEL"
        warnings.warn(msg)
        w1 = 1 / respamp

    while time < Tstop:
        tstart = time
        tstop = time + dT - 1 / fsxml
        st = read(file, starttime=tstart, endtime=tstop)
        st = st.sort()
        t1 = time

        # print('\t({0}{1}) Evaluating from\t'.format(sensor, location), time, '\tto\t', tstop) if verbose else ''

        for n in range(0, M):
            v[k] = np.nan
            tr = st.slice(t1, t1 + Twindow - 1 / fsxml)
            if tr.get_gaps() == [] and len(tr) > 0:
                tr1 = tr[0]
                if tr1.stats.npts == Num:
                    s, _ = mlab.psd(tr1.data, NFFT=Num, Fs=fsxml, detrend="linear")
                    s = s[1:] * w1
                    psd_values_db = 10 * np.log10(s)
                    v[k] = 0

            if np.isnan(v[k]):
                temp_array = np.append(temp_array, np.repeat(np.nan, f.size))
            else:
                temp_array = np.append(temp_array, psd_values_db)
            t1 = t1 + Twindow * Overlap
            k += 1
        time = time + TLong
    # print('\t*** Saving data to file ***')
    filename = file.split('.')[-2] + '-' + file.split('.')[-1]
    print(f'\t*** Saving {filename} to file ***')

    temp_df = pd.DataFrame(temp_array, columns=['psd'])
    temp_df = temp_df.set_index(np.tile(f, int(len(temp_array) / len(f))))

    temp_df.to_parquet(daily_path + fr'\{Twindow}_{channel}_{filename}.parquet.brotli', compression='brotli',
                       compression_level=9)

    print(f'\t*** {filename} correctly saved ***')


def read_daily_df(freq_indeces, filename):
    # print(filename)
    temp_df = pd.read_parquet(filename).dropna()
    return temp_df.loc[freq_indeces]


def to_frequency():
    filename_list = glob.glob(daily_path + r"\*.brotli")
    data = np.load(npz_path + fr'\{Twindow}_Frequency.npz')
    freq_data = data['frequency']
    num_chunk = int(freq_data.size / 100)
    f_lst = np.split(freq_data, num_chunk)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for index, freq in enumerate(f_lst):
            j1 = (index + 1) / len(f_lst)
            print("\r[%-75s] %g%%" % ('=' * int(75 * j1), round(100 * j1, 3)),
                  f"\t{round(freq[0], 2)} Hz - {round(freq[-1], 2)} Hz", end='\n')
            results = executor.map(functools.partial(read_daily_df, freq), filename_list)
            df = pd.DataFrame()
            for res in results:
                df = pd.concat([df, res])
            df.to_parquet(freq_df_path + fr'\{str(index).zfill(len(str(num_chunk)))}_{channel}.parquet.brotli',
                          compression='brotli', compression_level=9)
    return freq_data


def read_freq_df(q, filename):
    # print(filename)
    temp_df = pd.read_parquet(filename)
    temp_df = temp_df.sort_index()
    return temp_df.groupby(temp_df.index)['psd'].quantile(q).values.flatten()


def from_freq_to_quantile(q):
    filename_list = glob.glob(freq_df_path + r"\*.brotli")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(functools.partial(read_freq_df, q), filename_list)
        quantile_array = np.array([])
        for res in results:
            quantile_array = np.append(quantile_array, res)

    np.savez(npz_path + fr"\{str(q).replace('.', '')}_{channel}.npz", q_array=quantile_array)
    return quantile_array


def plot_from_df(x_array, y_array, quant, ax, xlabel='Frequency [Hz]', ylabel=r'ASD $\frac{m^2/s^4}{Hz}$ [dB]',
                 label_size=24, xscale='log', yscale='linear'):
    y_min = float(config['Quantities']['range_y_min'])
    y_max = float(config['Quantities']['range_y_max'])

    if config['Quantities']['range_x_min'] != '':
        x_min = float(config['Quantities']['range_x_min'])
    else:
        x_min = x_array.min()
    if config['Quantities']['x_max'] != '':
        x_max = float(config['Quantities']['x_max'])
    else:
        x_max = 20

    ax.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')
    ax.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')
    ax.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
    ax.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)
    ax.plot(x_array, y_array, linewidth=2, label='{0}%'.format(quant * 100))
    ax.set_title(f'{channel}', fontsize=22)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.legend(loc='best', shadow=True, fontsize=24)
    ax.grid(True, linestyle='--', axis='both', which='both')
    fig.tight_layout()

    return ax


def print_on_screen(symbol1, message, quantity, symbol2=None):
    with open(log_path + fr'\{t_log}.txt', 'a') as text_file:
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
    if only_daily:
        make_daily()
        t2 = time.perf_counter()
        print_on_screen(symbol1='*', message=f'Total time elapsed', quantity=t2 - t0)
    else:
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot()

        t1 = time.perf_counter()

        make_daily() if not skip_daily else ''

        t2 = time.perf_counter()

        print_on_screen(symbol1='+', symbol2='-', message='Daily DataFrame creation finished in', quantity=t2 - t1)

        t1 = time.perf_counter()
        if not skip_freq_df:
            f_data = to_frequency()
        else:
            f_data = np.load(npz_path + fr'\{Twindow}_Frequency.npz')['frequency']
        f_data.sort()
        t2 = time.perf_counter()

        print_on_screen(symbol1='+', symbol2='-', message='Conversion to frequency DataFrame finished in',
                        quantity=t2 - t1)

        lst_array = []
        if not skip_quant_eval:
            for quant in quantiles:
                t1 = time.perf_counter()
                lst_array.append(from_freq_to_quantile(q=quant))
                t2 = time.perf_counter()
                print_on_screen(symbol1='+', message=f'Search for the {quant} quantile array finished in',
                                quantity=t2 - t1)
        else:
            for quant in quantiles:
                t1 = time.perf_counter()
                q = np.load(npz_path + fr"\{str(quant).replace('.', '')}.npz")['q_array']
                lst_array.append(q)
                t2 = time.perf_counter()
                print_on_screen(symbol1='+', message=f'Reading {quant} quantile array finished in', quantity=t2 - t1)

        t1 = time.perf_counter()
        for index, q in enumerate(quantiles):
            plot_from_df(x_array=f_data, y_array=lst_array[index], quant=q, ax=ax)
        t2 = time.perf_counter()

        print_on_screen(symbol1='+', symbol2='-', message='Plot building finished in', quantity=t2 - t1)
        print_on_screen(symbol1='*', message='Total time elapsed', quantity=t2 - t0)
        plt.show()
