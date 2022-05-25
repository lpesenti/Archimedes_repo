__author__ = "Luca Pesenti"
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.1.0"
__maintainer__ = "Luca Pesenti"
__email__ = "lpesenti@uniss.it"
__status__ = "Development"

import glob
import time
import functools
import numpy as np
import configparser
import pandas as pd
import concurrent.futures
import matplotlib.pyplot as plt
from obspy import read
from matplotlib import mlab
from obspy import UTCDateTime
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
The data are saved stored in daily file with the name-format: 

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

config = configparser.ConfigParser()
config.read('Quantile_config.ini')

savedata = config.getboolean('Quantities', 'save_data')
df_path = config['Paths']['outDF_path']  # TODO: add check for file already existing
skip_daily = config.getboolean('DEFAULT', 'skip_daily')
quantiles = [float(x) for x in config['Quantities']['quantiles'].split(',')]
freq_df_path = Ec.check_dir(df_path, 'Freq_df')
npz_path = Ec.check_dir(df_path, 'npz_files')


def daily_df():
    global df_path, config

    filexml = config['Paths']['xml_path']
    Data_path = config['Paths']['data_path']
    network = config['Instrument']['network']
    sensor = config['Instrument']['sensor']
    location = config['Instrument']['location']
    channel = config['Instrument']['channel']
    Twindow = int(config['Quantities']['psd_window'])
    TLong = int(config['Quantities']['TLong'])
    Overlap = float(config['Quantities']['psd_overlap'])  # TODO: check if it is the right way to perform overlap
    verbose = config.getboolean('DEFAULT', 'verbose')
    unit = config['DEFAULT']['unit']  # TODO: enable the VEL option

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

    dT = TLong + Twindow * Overlap
    M = int((dT - Twindow) / (Twindow * (1 - Overlap)) + 1)

    K = int((stopdate - startdate) / (Twindow * (1 - Overlap)) + 1)

    v = np.empty(K)

    fsxml = 100
    Num = int(Twindow * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    np.savez(npz_path + fr'\Frequency.npz', frequency=f)
    w = 2.0 * np.pi * f
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

    for file_index, file in enumerate(filename_list):
        temp_array = np.array([])
        file_perc = round(100 * (file_index + 1) / len(filename_list), 2)
        print('({0}%)'.format(file_perc), file)
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
            t1 = time

            print('\t({0}{1}) Evaluating from\t'.format(sensor, location), time, '\tto\t', tstop) if verbose else ''

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
        print('\t*** Saving data to file ***')
        filename = file.split('.')[-1]

        temp_df = pd.DataFrame(temp_array, columns=['psd'])
        temp_df = temp_df.set_index(np.tile(f, int(len(temp_array) / len(f))))
        # TODO: Change the format of the filename so that it contains the psd_window information
        temp_df.to_parquet(df_path + fr'{filename}.parquet.brotli', compression='brotli', compression_level=9)

        print('\t*** Data correctly saved ***')


def read_daily_df(freq_indeces, filename):
    # print(filename)
    temp_df = pd.read_parquet(filename).dropna()
    return temp_df.loc[freq_indeces]


def to_frequency():
    global df_path, freq_df_path, npz_path
    filename_list = glob.glob(df_path + "*.brotli")
    data = np.load(npz_path + r'\Frequency.npz')
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
            df.to_parquet(freq_df_path + fr'\{str(index).zfill(len(str(num_chunk)))}.parquet.brotli',
                          compression='brotli', compression_level=9)
    return freq_data


def read_freq_df(q, filename):
    # print(filename)
    temp_df = pd.read_parquet(filename)
    return temp_df.groupby(temp_df.index)['psd'].quantile(q).values.flatten()


def from_freq_to_quantile(q):
    global freq_df_path, npz_path
    filename_list = glob.glob(freq_df_path + r"\*.brotli")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(functools.partial(read_freq_df, q), filename_list)
        quantile_array = np.array([])
        for res in results:
            quantile_array = np.append(quantile_array, res)

    np.savez(npz_path + fr"\{str(q).replace('.', '')}.npz", q_array=quantile_array)
    return quantile_array


def plot_from_df(x_array, y_array, quant, ax, xlabel='Frequency [Hz]', ylabel=r'ASD $\frac{m^2/s^4}{Hz}$ [dB]',
                 label_size=24, xscale='log', yscale='linear'):
    ax.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')
    ax.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')
    ax.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
    ax.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)
    ax.plot(x_array, y_array, linewidth=2, label='{0}%'.format(quant * 100))

    ax.set_xlim([x_array.min(), 20])
    ax.set_ylim([-200, -30])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.legend(loc='best', shadow=True, fontsize=24)
    ax.grid(True, linestyle='--', axis='both', which='both')
    fig.tight_layout()

    return ax


if __name__ == '__main__':
    t0 = time.perf_counter()

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    t1 = time.perf_counter()
    daily_df() if not skip_daily else ''
    t2 = time.perf_counter()

    # TODO : add function for printing and make log

    print('+', '-' * 98, '+', sep='')
    print('| Daily DataFrame creation finished in'.ljust(70, '.'), f"{round(t2 - t1, 3)} seconds |".rjust(30, '.'),
          sep='')
    print('| Daily DataFrame creation finished in'.ljust(70, '.'),
          f"{round((t2 - t1) / 60, 3)} minutes |".rjust(30, '.'), sep='')
    print('| Daily DataFrame creation finished in'.ljust(70, '.'),
          f"{round((t2 - t1) / 3600, 3)} hours |".rjust(30, '.'), sep='')
    print('+', '-' * 98, '+', sep='')
    t1 = time.perf_counter()
    f_data = to_frequency()
    t2 = time.perf_counter()
    print('+', '-' * 98, '+', sep='')
    print('| Conversion to frequency DataFrame finished in'.ljust(70, '.'),
          f"{round(t2 - t1, 3)} seconds |".rjust(30, '.'), sep='')
    print('| Conversion to frequency DataFrame finished in'.ljust(70, '.'),
          f"{round((t2 - t1) / 60, 3)} minutes |".rjust(30, '.'), sep='')
    print('| Conversion to frequency DataFrame finished in'.ljust(70, '.'),
          f"{round((t2 - t1) / 3600, 3)} hours |".rjust(30, '.'), sep='')
    print('+', '-' * 98, '+', sep='')
    lst_array = []
    for quant in quantiles:
        t1 = time.perf_counter()
        lst_array.append(from_freq_to_quantile(q=quant))
        t2 = time.perf_counter()
        print('\t+', '-' * 98, '+', sep='')
        print(f'\t| Search for the {quant} quantile array finished in'.ljust(70, '.'),
              f"{round(t2 - t1, 3)} seconds |".rjust(30, '.'), sep='')
        print(f'\t| Search for the {quant} quantile array finished in'.ljust(70, '.'),
              f"{round((t2 - t1) / 60, 3)} minutes |".rjust(30, '.'), sep='')
        print(f'\t| Search for the {quant} quantile array finished in'.ljust(70, '.'),
              f"{round((t2 - t1) / 3600, 3)} hours |".rjust(30, '.'), sep='')
        print('\t+', '-' * 98, '+', sep='')
    t1 = time.perf_counter()
    for index, q in enumerate(quantiles):
        plot_from_df(x_array=f_data, y_array=lst_array[index], quant=q, ax=ax)
    t2 = time.perf_counter()

    print('+', '-' * 98, '+', sep='')
    print('| Plot building finished in'.ljust(70, '.'), f"{round(t2 - t1, 3)} seconds |".rjust(30, '.'), sep='')
    print('| Plot building finished in'.ljust(70, '.'), f"{round((t2 - t1) / 60, 3)} minutes |".rjust(30, '.'), sep='')
    print('| Plot building finished in'.ljust(70, '.'), f"{round((t2 - t1) / 3600, 3)} hours |".rjust(30, '.'), sep='')
    print('+', '-' * 98, '+', sep='')
    print('*' * 100)
    print('* Total time elapsed '.ljust(70, '.'), f"{round(t2 - t1, 3)} seconds *".rjust(30, '.'), sep='')
    print('* Total time elapsed '.ljust(70, '.'), f"{round((t2 - t1) / 60, 3)} minutes *".rjust(30, '.'), sep='')
    print('* Total time elapsed '.ljust(70, '.'), f"{round((t2 - t1) / 3600, 3)} hours *".rjust(30, '.'), sep='')
    print('*' * 100)
    plt.show()
