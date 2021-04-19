r"""
[LAST UPDATE: 7 April 2021 - Luca Pesenti]

The following functions have been built to work with the data obtained by the Archimedes experiment.
The experiment save the data in a file .lvm containing 9 columns,

| ITF | Pick Off | Signal injected | Error | Correction | Actuator 1 | Actuator 2 | After Noise | Time |
|-----+----------+-----------------+-------+------------+------------+------------+-------------+------|
| ... | ........ | ............... | ..... | .......... | .......... | .......... | ........... | .... |
|-----+----------+-----------------+-------+------------+------------+------------+-------------+------|
| ... | ........ | ............... | ..... | .......... | .......... | .......... | ........... | .... |

ITF [V]            : signal from the interferometer
Pick Off [V]       : signal from the laser before the filters
Signal injected [?]: it has the shape of a sinusoid and it is necessary to make some operations with the signals
Error [-]          : it is given by the ratio between the ITF and the Pick Off minus some constant in order to translate
                     the result around the zero
Correction [-]     :
Actuator 1/2 [V]   : are the voltage coming from the two actuators before the amplification -> if one works, the other
                     should be off
After Noise [-]    :
Time [-]           : every 10 millisecond (100 rows) the system print the datetime in the format
                     -> 02/19/2021 20:01:11.097734\09

In the current configuration the sampling rate is 1 KHz but can be increase.
"""

from matplotlib.mlab import cohere
from matplotlib import mlab
import matplotlib.dates as mdates
from itertools import groupby
import datetime
import numpy as np
import pandas as pd
import os
import glob
import re
import logging
import math
import timeit

cols = np.array(
    ['ITF', 'Pick Off', 'Signal injected', 'Error', 'Correction', 'Actuator 1', 'Actuator 2', 'After Noise',
     'Time'])
unix_time = 0  # It is used only to make right conversion in time for time evolution analysis
path_to_data = r"D:\Archimedes\Data"
freq = 1000  # Hz
lambda_laser = 532.e-9  # Meter --> 532 nanometer
distance_mirrors = 0.1  # Length between mirrors
first_coef = lambda_laser / (2. * np.pi * distance_mirrors)


def find_rk(seq, subseq):
    """
    Additional help can be found in the answer of norok2 `in this discussion
    <https://stackoverflow.com/questions/7100242/python-numpy-first-occurrence-of-subarray>`_.

    """
    n = len(seq)
    m = len(subseq)
    if np.all(seq[:m] == subseq):
        yield 0
    hash_subseq = sum(hash(x) for x in subseq)  # compute hash
    curr_hash = sum(hash(x) for x in seq[:m])  # compute hash
    for i in range(1, n - m + 1):
        curr_hash += hash(seq[i + m - 1]) - hash(seq[i - 1])  # update hash
        if hash_subseq == curr_hash and np.all(seq[i:i + m] == subseq):
            yield i


def find_factors(n):
    factor_lst = np.array([])
    # Note that this loop runs till square root
    i = 1
    while i <= math.sqrt(n):
        if n % i == 0:
            # If divisors are equal, print only one
            if n / i == i:
                factor_lst = np.append(factor_lst, i)
            else:
                # Otherwise print both
                factor_lst = np.append(factor_lst, [i, n / i])
        i = i + 1
    return factor_lst


def time_tick_formatter(val, pos=None):
    """
    Return val reformatted as a human readable date

    See Also
    --------
    time_evolution : it is used to rewrite x-axis
    """
    global unix_time  # It is the method to access the global variable unix_time in order to read it
    # The following statement is used to change the label only every 1000 point
    # because of time consuming.
    if val % 1000 == 0:
        val = str(datetime.datetime.fromtimestamp(int(unix_time + val * 0.001)))
    else:
        val = ''
    return val


def myfromtimestampfunction(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)


def myvectorizer(input_func):
    def output_func(array_of_numbers):
        return [input_func(a) for a in array_of_numbers]

    return output_func


def read_data(day, month, year, col_to_save, num_d=1, file_start=None, file_stop=None, verbose=True):
    """
    Search data present in a specific folder and read only the column associated with the quantity you are interested in

    Parameters
    ----------
    day : int
        It refers to the first day of the data to be read

    month : int
        It refers to the first month of the data to be read

    year : int
        It refers to the first year of the data to be read

    col_to_save : str
        The quantity to be read.

    num_d : int
        How many days of data you want to analyze.

    file_start : any
        The first file to be read.

    file_stop : any
        The last file to be read.

    verbose : bool
        If True the verbosity is enabled.

    Notes
    -----
    *col_to_save* takes only one of the following parameter:
        - ITF : the signal from the interferometer expressed in V
        - Pick Off : the signal from the pick-off expressed in V
        - Signal injected : the signal from the waveform used expressed in V
        - Error :
        - Correction :
        - Actuator 1 : the output of the actuator 1 before amplification expressed in V
        - Actuator 2 : the output of the actuator 2 before amplification expressed in V
        - After Noise :
        - Time : the timestamp of the data saved every milli second in human-readable format

    Returns
    -------
    A tuple of a pandas DataFrame [n-rows x 1-column] containing the data, the index of the column and the timestamp
    of the first data expressed in UNIX
    """
    logging.info('Data read started')
    logging.debug('PARAMETERS: day=%i month=%i year=%i col_to_save=%s num_d=%i verbose=%s' % (
        day, month, year, col_to_save, num_d, verbose))
    if verbose:
        print('--------------- Reading', day, '/', month, '/', year, '-', col_to_save, 'data ---------------')
    month = '%02d' % month  # It transforms 1,2,3,... -> 01,02,03,...
    index = np.where(cols == col_to_save)[0][0] + 1  # Find the index corresponding to the the col_to_save
    all_data = []
    final_df = pd.DataFrame()
    time_array = []
    for i in range(num_d):  # Loop over number of days
        data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day + i))
        temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
        temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_data += temp_data
    i = 1
    logging.debug('Number of *.lvm: %i' % len(all_data))
    if file_start and file_stop:
        for data in all_data:
            if verbose:
                print(round(i / len(all_data) * 100, 1), '%')
            if file_start <= i <= file_stop:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', usecols=[index], header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
                time = pd.read_table(data, sep='\t', usecols=[9], header=None).replace(r'\\09', '',
                                                                                       regex=True).values.flatten().tolist()
                timestamp = datetime.datetime.timestamp(pd.to_datetime(time[0]))
                for j in range(1, len(time) + 1):
                    time_array.append(timestamp + j / freq)
            i += 1
    elif file_start and not file_stop:
        for data in all_data:
            if verbose:
                print(round(i / len(all_data) * 100, 1), '%')
            if i >= file_start:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', usecols=[index], header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
                time = pd.read_table(data, sep='\t', usecols=[9], header=None).replace(r'\\09', '',
                                                                                       regex=True).values.flatten().tolist()
                timestamp = datetime.datetime.timestamp(pd.to_datetime(time[0]))
                for j in range(1, len(time) + 1):
                    time_array.append(timestamp + j / freq)
            i += 1
    elif file_stop and not file_start:
        for data in all_data:
            if verbose:
                print(round(i / len(all_data) * 100, 1), '%')
            if i <= file_stop:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', usecols=[index], header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
                time = pd.read_table(data, sep='\t', usecols=[9], header=None).replace(r'\\09', '',
                                                                                       regex=True).values.flatten().tolist()
                timestamp = datetime.datetime.timestamp(pd.to_datetime(time[0]))
                for j in range(1, len(time) + 1):
                    time_array.append(timestamp + j / freq)
            i += 1
    else:
        for data in all_data:
            if verbose:
                print(round(i / len(all_data) * 100, 1), '%')
            # Read only the column of interest -> [index]
            a = pd.read_table(data, sep='\t', usecols=[index], header=None)
            # At the end we have a long column with all data
            final_df = pd.concat([final_df, a], axis=0, ignore_index=True)
            time = pd.read_table(data, sep='\t', usecols=[9], header=None).replace(r'\\09', '',
                                                                                   regex=True).values.flatten().tolist()
            timestamp = datetime.datetime.timestamp(pd.to_datetime(time[0]))
            for j in range(1, len(time) + 1):
                time_array.append(timestamp + j / freq)
    if verbose:
        print('--------------- Reading completed! ---------------')
    logging.info('Data read completed')
    return final_df, index, time_array


def time_evolution(day, month, year, quantity, ax, ndays=1, file_start=None, file_stop=None, verbose=True):
    """
    Make the plot of time evolution

    Parameters
    ----------
        day : int
            It refers to the first day of the data to be read

        month : int
            It refers to the first month of the data to be read

        year : int
            It refers to the first year of the data to be read

        quantity : str
            The quantity to be read. It must be one of the following:

            - 'ITF' : the signal from the interferometer expressed in V.
            - 'Pick Off' : the signal from the pick-off expressed in V.
            - 'Signal injected' : the signal from the waveform used expressed in V.
            - 'Error' : represent the ratio between ITF and the pick off signals.
            - 'Correction' :
            - 'Actuator 1' : the output of the actuator 1 before amplification expressed in V.
            - 'Actuator 2' : the output of the actuator 2 before amplification expressed in V.
            - 'After Noise' :
            - 'Time' : the timestamp of the data saved every milli second in human-readable format

        ax: ax
            The ax to be given in order to have a plot

        ndays : int
            How many days of data you want to analyze.

        verbose : bool
            If True the verbosity is enabled.

    Returns
    -------
    A tuple of an axes and the relative filename
    """
    df, col_index, t = read_data(day, month, year, quantity, ndays, file_start=file_start, file_stop=file_stop,
                                 verbose=verbose)
    # global unix_time  # Access the global variable and
    # unix_time = t  # change its value to the timestamp of the data
    if verbose:
        print('Building Time Evolution Plot...')
    date_convert = myvectorizer(myfromtimestampfunction)
    xtimex = date_convert(t)
    datenums = mdates.date2num(xtimex)
    lab = quantity + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays) + 'tEvo'
    ax.plot(datenums, df[col_index], color='tab:red', label=lab)
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    xfmt = mdates.DateFormatter('%b %d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    # ax.xaxis.set_major_formatter(time_tick_formatter)
    # print(pd.to_datetime(t[0]))
    # ax.set_xticks(np.arange(0, df.size, int(df.size / len(t))))
    # ax.set_xticklabels(t)
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc='best', shadow=True, fontsize='medium')
    return ax, filename


def th_comparison(data_frame, threshold=0.03, length=10000, verbose=True):
    """
    It performs the derivative of the data and compares it with a fixed threshold.
    To perform this comparison it divides the data in given number of slice and check if the maximum of the slice is
    greater than threshold or not.
    After the comparison it removes all the data above the threshold from the dataframe.

    Parameters
    ----------
        data_frame : pandas.DataFrame
            The dataframe containing the data in which perform the comparison

        threshold : float
            The threshold used to compare the data

        length : int
            Represent the length of each slice in whic the data will be divided into.
            The greater it is, the smaller will be the number of slice and so the comparison will be less
            accurate but will require less time to perform it.
            WARNING: a big number of length can cause a bad comparison due to the fluctuations in the signal.

        verbose : bool
            If True the verbosity is enabled.
    Notes
    -----
    Choose a multiple of 300000 for the value of ndivsion. This will ensure to split the data in equal size.

    See Also
    --------
    der_plot : it is used to perform the derivative and to plot the data cleared

    Returns
    -------
    A tuple of a numpy array containing the data cleared, a numpy array of the data derived and
    the rejected fraction of the data
    """
    logging.info('Comparison started')
    logging.debug('PARAMETERS: rows_in_df=%s threshold=%f length=%i verbose=%s' % (
        len(data_frame.index), threshold, length, verbose))
    if verbose:
        print('Starting the comparison...')
        print('\tThreshold:', threshold)
    # print(data_frame.columns)
    # print(data_frame.values.flatten())
    data_to_check = data_frame.values.flatten()
    # print(type(data_to_check))
    # data_deriv = np.array([])
    factors = find_factors(data_to_check.size)  # The following lines are needed in the case of length is not
    indx_len = np.argmin(np.abs(factors - length))  # a factor of the data_frame size
    # print(data_to_check.size)
    # for i in range(data_to_check.size):
    #     if i % factors[indx_len] == 0:
    #         data_deriv = np.append(data_deriv, np.abs((data_to_check[i] - data_to_check[i - int(factors[indx_len])])))
    # data_deriv = np.abs(np.diff(data_to_check, axis=0))
    # data_deriv = np.append(data_deriv, 0)
    num_slice = int(data_to_check.size / factors[indx_len])  # It must be an integer
    data_split = np.array_split(data_to_check, num_slice)
    lst = []
    # test = np.array([])
    i = 0
    for sub_arr in data_split:
        # mean = np.mean(sub_arr)
        # test = np.append(test, mean)
        # for element in sub_arr:
        # if np.amax(sub_arr) > threshold:
        #     lst.append(i)
        # print(sub_arr)
        # print(round(i / len(data_split) * 100, 1), '%')
        if np.abs(np.amax(sub_arr) - np.amin(sub_arr)) > threshold:
            lst.append(i)
        else:
            pass
        i += 1
    # test = test.repeat(int(factors[indx_len]))
    # print(test.size)
    logging.info('Comparison completed')
    for index in lst:
        # print(index)
        start = index * int(factors[indx_len])
        data_to_check[start:start + int(factors[indx_len])] = np.nan
    frac_rejected = len(lst) * factors[indx_len] / len(data_to_check)
    if verbose:
        print('\tData rejected:', frac_rejected * 100, '%')
        print('Comparison completed!')
    logging.debug('Data rejected: %f ' % frac_rejected)
    logging.info('Replacement completed')
    return data_to_check, frac_rejected


def psd(day, month, year, quantity, ax, interval, mode, low_freq=2, high_freq=10, threshold=0.03, ndays=1,
        length=10000, ax1=None, file_start=None, file_stop=None, verbose=True):
    logging.info('PSD evaluator started')
    logging.debug(
        'PARAMETERS: day=%i month=%i year=%i quantity=%s ax=%s interval=%i'
        ' mode=%s low_freq=%i high_freq=%i threshold=%f ndays=%i length=%i verbose=%s' % (
            day, month, year, quantity, ax, interval, mode, low_freq, high_freq, threshold, ndays, length, verbose))

    df_qty, col_index, t = read_data(day, month, year, quantity, num_d=ndays, file_start=file_start,
                                     file_stop=file_stop,
                                     verbose=verbose)
    df_itf, _, _ = read_data(day=day, month=month, year=year, col_to_save='ITF', num_d=ndays, file_start=None,
                             file_stop=None, verbose=verbose)
    df_po, _, _ = read_data(day=day, month=month, year=year, col_to_save='Pick Off', num_d=ndays, file_start=None,
                            file_stop=None, verbose=verbose)

    data_cleared, _ = th_comparison(df_qty, threshold=threshold, length=length, verbose=verbose)

    print('Removing NaN values...') if verbose else print()
    data_first_check = [list(group) for key, group in groupby(data_cleared, lambda x: not np.isnan(x)) if key]
    print('NaN values successfully removed') if verbose else print()

    logging.debug('Data cleared from NaN values')

    num = int(60 * freq)
    _, psd_f = mlab.psd(np.ones(num), NFFT=num, Fs=freq, detrend="linear")  # , noverlap=int(num / 2))
    psd_f = psd_f[1:]

    outdata, file_index = [], []
    len_max, pick_off_mean = 0, 0
    length_data_used = 0
    integral_min = np.inf

    start = np.where(psd_f == low_freq)[0][0]
    stop = np.where(psd_f == high_freq)[0][0]

    print('Number of usable array:', len(data_first_check)) if verbose else print()
    print('Starting evaluation of', mode, 'PSD...') if verbose else print()

    if mode == 'max interval':
        for el in data_first_check:
            if len(el) >= interval * freq and len(el) > len_max:
                len_max = len(el)
                outdata = el
        psd_s, _ = mlab.psd(outdata, NFFT=num, Fs=freq, detrend="linear", noverlap=int(num / 2))
        outdata = psd_s[1:]
    elif mode == 'low noise':
        i = 0
        for el in data_first_check:
            print(round(i / len(data_first_check) * 100, 2), '%')
            if len(el) >= interval * freq:
                el_s, _ = mlab.psd(el, NFFT=num, Fs=freq, detrend="linear")  # , noverlap=int(num / 2))
                el_s = el_s[1:]
                integral = sum(
                    el_s[start:stop] / len(el_s[start:stop]))  # len(el) differential in the integral to evaluate RMS
                if integral < integral_min:
                    integral_min = integral
                    file_index = list(find_rk(df_qty.values.flatten(), el))
                    outdata = el_s
                    length_data_used = len(el)
                    pick_off_mean = np.abs(
                        np.mean(df_po[file_index[0]:file_index[0] + len(el) + 1].values.flatten()))
            i += 1
        if file_index:
            file_number = file_index[0] / 300000
            print('File used:', file_number) if isinstance(file_number, int) else print('File used:', int(file_number),
                                                                                        'and',
                                                                                        int(file_number) + 1)
    print('Evaluation of', mode, 'PSD completed!') if verbose else print()
    # print('Length of data used for PSD:', len(outdata)) if verbose else print()
    v_max = df_itf.max().values.flatten()
    v_min = df_itf.min().values.flatten()
    print('Voltage max', v_max)
    print('Voltage min', v_min)
    alpha = first_coef / (v_max - v_min)
    print('Alpha', alpha)
    logging.info('V_max = %f, V_min = %f, alpha = %f' % (v_max, v_min, alpha))
    # if outdata:
    data_to_plot = np.sqrt(outdata) * alpha * pick_off_mean
    # else:
    #    data_to_plot = np.empty(len(psd_f)) * np.nan
    lab1 = r'$\sqrt{PSD}$ of ' + quantity + '(' + str(interval) + ' s)'

    x, y = np.loadtxt(os.path.join(path_to_data, 'VirgoData_Jul2019.txt'), unpack=True, usecols=[0, 1])
    data_used_x = df_qty.index[file_index[0]:file_index[0] + length_data_used + 1]
    data_used_y = df_qty[file_index[0]:file_index[0] + length_data_used + 1].values.flatten()
    ax.plot(x, y, linestyle='-', label='@ Virgo')
    ax.plot(psd_f, data_to_plot, linestyle='-', label='@ Sos-Enattos')
    ax.set_xscale("linear")
    ax.set_xlim([2, 20])
    ax.set_ylim([1.e-13, 1.e-8])
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=16)
    ax.set_ylabel(r"ASD [rad/$\sqrt{Hz}$]", fontsize=16)
    ax.grid(True, linestyle='-')
    ax.legend(loc='best', shadow=True, fontsize='medium')
    ax.set_title(mode)
    if ax1:
        ax1.plot(df_qty.index, df_qty[col_index], linestyle='dotted', label='All data')
        ax1.plot(data_used_x, data_used_y, linestyle='-', label='Data used')
        ax1.grid(True, linestyle='-')
        # ax1.xaxis.set_major_formatter(time_tick_formatter)
        ax1.set_ylabel(r"Voltage [V]", fontsize=16)
        ax1.legend(loc='best', shadow=True, fontsize='medium')
    return ax, ax1


def cleared_plot(day, month, year, quantity, ax, threshold=0.03, ndays=1, length=10000, verbose=True):
    """
    Make the derivative plot of a given quantity

    Parameters
    ----------
        day : int
            It refers to the first day of the data to be read

        month : int
            It refers to the first month of the data to be read

        year : int
            It refers to the first year of the data to be read

        quantity : str
            The quantity to be read.

        ax: ax
            The ax to be given in order to have a plot

        threshold : float
            The threshold used to compare the data

        ndays : int
            How many days of data you want to analyze.

        length : int
            Represent the length of each slice in whic the data will be divided into.
            The greater is, the smaller will be the number contained in each slice and so the comparison will be more
            accurate but will require more time to perform the comparison.
            WARNING: a big number of length can cause a bad comparison due to the fluctuations in the signal.

        verbose : bool
            If True the verbosity is enabled.

    Notes
    -----
    *quantity* takes only one of the following parameter:
        - ITF : the signal from the interferometer expressed in V
        - Pick Off : the signal from the pick-off expressed in V
        - Signal injected : the signal from the waveform used expressed in V
        - Error :
        - Correction :
        - Actuator 1 : the output of the actuator 1 before amplification expressed in V
        - Actuator 2 : the output of the actuator 2 before amplification expressed in V
        - After Noise :
        - Time : the timestamp of the data saved every milli second in human-readable format
    Choose a multiple of 300000 for the value of ndivsion. This will ensure to split the data in equal size.

    Returns
    -------
    A tuple of an axes and the relative filename
    """
    df, col_index, t = read_data(day, month, year, quantity, ndays, verbose=verbose)
    global unix_time
    unix_time = t
    data_und_th, val_rej = th_comparison(df, threshold=threshold, length=length, verbose=verbose)
    lab = r'$\partial_t$ ' + quantity + ' (n° days: ' + str(ndays) + ')'
    lab1 = quantity + r' cleared (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays) + 'der'
    # th_line = np.full(data_deriv.size, threshold)
    if verbose:
        print('--------------- Building Derivative and Data_Cleared Plot! ---------------')
    # ax.plot(df.index, data_deriv, color='tab:green', linestyle='-', label=lab)
    # ax.plot(np.arange(data_deriv.size), th_line, color='tab:red', linestyle='-', label='threshold')
    ax.plot(df.index, df[col_index], color='tab:red', label='Data')
    ax.plot(df.index, data_und_th + 5, color='tab:blue', linestyle='-', label='Under threshold')
    ax.grid(True, linestyle='-')
    ax.xaxis.set_major_formatter(time_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='medium')
    return ax, filename


def coherence(sign1, sign2, day, month, year, ax, ndays=1, day2=None, month2=None, year2=None, samedate=True,
              verbose=True):
    r"""
    Make the plot of the coherence between two signal, using the matplotlib built-in function.
    Coherence is the normalized cross spectral density:

    .. math::

        C_{xy} = \frac{|P_{xy}|^2}{P_{xx}P_{yy}}

    Parameters
    ----------
        sign1 : str
            It is the the first quantity used to evaluate the coherence.

        sign2 : str
            It is the the first quantity used to evaluate the coherence.

        day : int
            It refers to the first day of the data to be read

        month : int
            It refers to the first month of the data to be read

        year : int
            It refers to the first year of the data to be read

        ax: ax
            The ax to be given in order to have a plot

        ndays : int
            How many days of data you want to analyze.

        day2 : int
            It refers to the first day of the data to be read for the second signal
            if samedate variable is set to True

        month2 : int
            It refers to the first month of the data to be read for the second signal
            if samedate variable is set to True

        year2 : int
            It refers to the first year of the data to be read for the second signal
            if samedate variable is set to True

        samedate : bool
            It is needed if the two signal come from different date. By default it is set to False

        verbose : bool
            If True the verbosity is enabled.

    Notes
    -----
    *sign1* and *sign2* take only one of the following parameter:
        - ITF : the signal from the interferometer expressed in V
        - Pick Off : the signal from the pick-off expressed in V
        - Signal injected : the signal from the waveform used expressed in V
        - Error :
        - Correction :
        - Actuator 1 : the output of the actuator 1 before amplification expressed in V
        - Actuator 2 : the output of the actuator 2 before amplification expressed in V
        - After Noise :
        - Time : the timestamp of the data saved every milli second in human-readable format

    Returns
    -------
    A tuple of an axes and the relative filename
    """
    if samedate:
        df, col_index, t = read_data(day, month, year, sign1, ndays)
        df1, col_index1, t1 = read_data(day, month, year, sign2, ndays)
    else:
        df, col_index, t = read_data(day, month, year, sign1, ndays)
        df1, col_index1, t1 = read_data(day2, month2, year2, sign2, ndays)
    global unix_time
    unix_time = t
    if verbose:
        print('--------------- Building Coherence plot! ---------------')
    lab = 'Coh: ' + sign1 + '-' + sign2 + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_Coh_' + sign1 + '-' + sign2 + '_nDays_' + str(ndays)
    c12, f12 = cohere(df[col_index], df1[col_index1], NFFT=10000, Fs=freq, detrend='linear')
    ax.plot(f12, c12, label=lab)
    ax.grid(True, linestyle='-')
    ax.set_xscale("log")
    ax.set_xlabel('Frequencies [Hz]')
    ax.set_ylabel('Coherence')
    ax.legend(loc='best', shadow=True, fontsize='medium')
    ax.xaxis.set_major_formatter(time_tick_formatter)
    ax.xaxis.set_tick_params(rotation=30)
    return ax, filename
