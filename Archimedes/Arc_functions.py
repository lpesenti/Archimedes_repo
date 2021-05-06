__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti", "Davide Rozza"]
__version__ = "1.1.0"
__maintainer__ = "Luca Pesenti"
__email__ = "l.pesenti6@campus.unimib.it"
__status__ = "Developing"

r"""
[LAST UPDATE: 27 April 2021 - Luca Pesenti]

The following functions have been built to work with the data obtained by the Archimedes experiment.
The experiment save the data in a file .lvm containing 9 columns*,

| ITF | Pick Off | Signal injected | Error | Correction | Actuator 1 | Actuator 2 | After Noise | Time |
|-----+----------+-----------------+-------+------------+------------+------------+-------------+------|
| ... | ........ | ............... | ..... | .......... | .......... | .......... | ........... | .... |
| ... | ........ | ............... | ..... | .......... | .......... | .......... | ........... | .... |
| ... | ........ | ............... | ..... | .......... | .......... | .......... | ........... | .... |

*data up to February 2021 have only 8 columns, the injected signal has not been saved.

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

In the current configuration the sampling rate is 1 KHz but can be increase (data saved every millisecond).
Nevertheless, the acquisition tool made with LabVIEW run at 25 KHz.

Matplotlib color palette: https://matplotlib.org/stable/gallery/color/named_colors.html
"""

import configparser
import datetime
import glob
import logging
import os
import re
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from matplotlib import mlab
from matplotlib.mlab import cohere

import Arc_common as ac

logger = logging.getLogger('data_analysis.functions')

config = configparser.ConfigParser()
config.read('config.ini')
lvm_headers = [x for x in config['lvm_properties']['headers'].split(',')]
cols = np.array(lvm_headers)
path_to_data = config['Paths']['data_dir']
freq = int(config['Quantities']['sampling_rate'])
lambda_laser = float(config['Quantities']['laser_wavelength'])
distance_mirrors = float(config['Quantities']['distance_mirrors'])

first_coef = lambda_laser / (2. * np.pi * distance_mirrors)  # Used to calculate alpha


def read_data(day, month, year, quantity, num_d=1, tevo=False, file_start=None, file_stop=None, verbose=False):
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

    quantity : str
        The quantity to be read.

    num_d : int
        How many days of data you want to analyze.

    tevo : bool
        If True the time column will be read.

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
    out : tuple
        A tuple of a pandas DataFrame [n-rows x 1-column] containing the data, the index of the column corresponding to
        the quantity selected and the timestamp of the first data expressed in UNIX

    """
    try:
        if not 1 <= day <= 31:
            raise AttributeError('Day must be in range (1, 31), you inserted: {0}'.format(day))
        if not 1 <= month <= 12:
            raise AttributeError('Month must be in range (1, 12), you inserted: {0}'.format(month))
        if not quantity.lower() in cols:
            raise AttributeError("The quantity '{0}' does not exist.".format(quantity))
    except AttributeError as err:
        logger.error(err.args[0])
        raise
    if file_stop is not None and file_start is not None:
        if file_stop < file_start:
            logger.warning("file_stop must be greater or equal than file_start")
    logger.warning("num_d must be greater than 0") if num_d == 0 else ''
    logger.info("'{0}' data read started".format(quantity.lower()))
    logger.info("Data from {0}/{1}/{2} selected".format(day, month, year))
    logger.debug('PARAMETERS: day={0} month={1} year={2} quantity={3} num_d={4} tevo={5} '
                 'file_start={6} file_stop={7} verbose={8}'
                 .format(day, month, year, quantity, num_d, tevo, file_start, file_stop, verbose))

    print('#### Reading', day, '/', month, '/', year, '-', quantity, 'data ####') if verbose else ''
    month = '%02d' % month  # It transforms 1,2,3,... -> 01,02,03,...
    index = np.where(cols == quantity.lower())[0][0] + 1  # Find the index corresponding to the the col_to_save

    logger.debug('index={0}'.format(index))

    all_data = []
    final_df = pd.DataFrame()
    time_array = []
    for i in range(num_d):  # Loop over number of days
        data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day + i))
        temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
        temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_data += temp_data
    i = 1
    logger.info('Number of .lvm found: %i' % len(all_data))
    if file_start and file_stop:
        for data in all_data:
            print(round(i / len(all_data) * 100, 1), '%') if verbose else ''
            if file_start <= i <= file_stop:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', na_filter=False, low_memory=False, engine='c', usecols=[index],
                                  header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)

                if tevo:
                    df_col = pd.read_csv(data, sep='\t', nrows=1, header=None).columns
                    time = pd.read_csv(data, sep='\t', usecols=[df_col[-1:][0]], nrows=1, header=None) \
                        .replace(r'\\09', '', regex=True)
                    timestamp = datetime.datetime.timestamp(pd.to_datetime(time[df_col[-1:][0]][0]))
                    for j in range(1, len(a.index) + 1):
                        time_array.append(timestamp + j / freq)
            i += 1
    elif file_start and not file_stop:
        for data in all_data:
            print(round(i / len(all_data) * 100, 1), '%') if verbose else ''
            if i >= file_start:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', na_filter=False, low_memory=False, engine='c', usecols=[index],
                                  header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)

                if tevo:
                    df_col = pd.read_csv(data, sep='\t', nrows=1, header=None).columns
                    time = pd.read_csv(data, sep='\t', usecols=[df_col[-1:][0]], nrows=1, header=None) \
                        .replace(r'\\09', '', regex=True)
                    timestamp = datetime.datetime.timestamp(pd.to_datetime(time[df_col[-1:][0]][0]))
                    for j in range(1, len(a.index) + 1):
                        time_array.append(timestamp + j / freq)
            i += 1
    elif file_stop and not file_start:
        for data in all_data:
            print(round(i / len(all_data) * 100, 1), '%') if verbose else ''
            if i <= file_stop:
                # Read only the column of interest -> [index]
                a = pd.read_table(data, sep='\t', na_filter=False, low_memory=False, engine='c', usecols=[index],
                                  header=None)
                # At the end we have a long column with all data
                final_df = pd.concat([final_df, a], axis=0, ignore_index=True)

                if tevo:
                    df_col = pd.read_csv(data, sep='\t', nrows=1, header=None).columns
                    time = pd.read_csv(data, sep='\t', usecols=[df_col[-1:][0]], nrows=1, header=None) \
                        .replace(r'\\09', '', regex=True)
                    timestamp = datetime.datetime.timestamp(pd.to_datetime(time[df_col[-1:][0]][0]))
                    for j in range(1, len(a.index) + 1):
                        time_array.append(timestamp + j / freq)
            i += 1
    else:
        for data in all_data:
            print(round(i / len(all_data) * 100, 1), '%') if verbose else ''
            # Read only the column of interest -> [index]
            a = pd.read_table(data, sep='\t', na_filter=False, low_memory=False, engine='c', usecols=[index],
                              header=None)
            # At the end we have a long column with all data
            final_df = pd.concat([final_df, a], axis=0, ignore_index=True)

            if tevo:
                df_col = pd.read_csv(data, sep='\t', nrows=1, header=None).columns
                time = pd.read_csv(data, sep='\t', usecols=[df_col[-1:][0]], nrows=1, header=None) \
                    .replace(r'\\09', '', regex=True)
                timestamp = datetime.datetime.timestamp(pd.to_datetime(time[df_col[-1:][0]][0]))
                for j in range(1, len(a.index) + 1):
                    time_array.append(timestamp + j / freq)
            i += 1
    if not verbose:
        pass
    else:
        print('#### Reading completed! ####')
    logger.info("'{0}' data read successfully completed".format(quantity.lower()))
    logger.warning("The time array is empty!") if not time_array else ''
    try:
        if final_df.empty:
            raise NameError('The data frame is empty. No data found!')
    except NameError as err:
        logger.error(err.args[0])
        raise
    return final_df, index, time_array


def time_evolution(day, month, year, quantity, ax, ndays=1, show_extra=False, tevo=True, file_start=None,
                   file_stop=None, verbose=False):
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

        ax: ax
            The ax to be given in order to have a plot

        ndays : int
            How many days of data you want to analyze.

        show_extra : bool
            If True, data over threshold are displayed with a translation in the same plot.

        tevo : bool
            If True the time column will be read.

        file_start : any
            The first file to be read.

        file_stop : any
            The last file to be read.

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
    Returns
    -------
    out : tuple
        A tuple of an axes and the relative filename
    """
    try:
        if not ax:
            raise TypeError("Ax can not be a 'NoneType' object")
    except TypeError as err:
        logger.error(err.args[0])
        raise
    logger.debug('PARAMETERS: day={0} month={1} year={2} quantity={3} ax={4} ndays={5} show_extra={6} tevo={7} '
                 'file_start={8} file_stop={9} verbose={10}'
                 .format(day, month, year, quantity, ax, ndays, show_extra, tevo, file_start, file_stop, verbose))
    df, col_index, t = read_data(day, month, year, quantity, ndays, tevo=tevo, file_start=file_start,
                                 file_stop=file_stop, verbose=verbose)
    logger.info("Building plot")
    if not verbose:
        pass
    else:
        print('Building Time Evolution Plot...')
    lab = quantity
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays) + 'tEvo'
    ax.plot(t, df[col_index], linestyle='dotted', label=lab)
    if show_extra:
        logger.info("Building data-cleared plot")
        lab1 = quantity + ' cleared'
        data_und_th, _ = th_comparison(df, verbose=verbose)
        ax.plot(t, data_und_th + 5, linestyle='-', label=lab1)
        logger.info("Data-cleared plot successfully built")
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    ax.xaxis.set_major_formatter(ac.time_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='large')

    logger.info("Plot successfully built")
    return ax, filename


def th_comparison(data_frame, threshold=0.03, length=10000, verbose=True):
    """
    It performs the derivative of the data and compares it with a fixed threshold.
    To perform this comparison it divides the data in given number of slice and check if the delta between the maximum
    and the minimum of the slice is greater than threshold or not.
    After the comparison it transforms into np.nan values all the data above the threshold from the dataframe.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The dataframe containing the data in which perform the comparison

    threshold : float
        The threshold used to compare the data

    length : int
        Represent the length of each slice in which the data will be divided into.
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
    psd : it is used to find the best array

    Returns
    -------
    out : tuple
        A tuple of a numpy array containing the data cleared, a numpy array of the data derived and
        the rejected fraction of the data
    """
    try:
        if data_frame.empty:
            raise NameError('The data frame is empty. No data found!')
    except NameError as err:
        logger.error(err.args[0])
        raise
    logger.info('Comparison started')
    logger.debug('PARAMETERS: rows_in_df={0} threshold={1} length={2} verbose={3}'.format(
        len(data_frame.index), threshold, length, verbose))
    if not verbose:
        pass
    else:
        print('Starting the comparison...')
        print('\tThreshold:', threshold)
    data_to_check = data_frame.values.flatten()
    logger.debug("Size of data_to_check = {0}".format(data_to_check.size))
    factors = ac.find_factors(data_to_check.size)  # The following lines are needed in the case of length is not
    indx_len = np.argmin(np.abs(factors - length))  # a factor of the data_frame size
    logger.debug("Factor={0}".format(factors[indx_len]))
    num_slice = int(data_to_check.size / factors[indx_len])  # It must be an integer
    logger.debug("num_slice={0}".format(num_slice))
    logger.info("Number of slices = {0}".format(num_slice))

    data_split = np.array_split(data_to_check, num_slice)
    index_data_rej = []
    i = 0
    for sub_arr in data_split:
        if np.abs(np.amax(sub_arr) - np.amin(sub_arr)) > threshold:
            index_data_rej.append(i)
        else:
            pass
        i += 1
    logger.debug("len(index_data_rej)={0}".format(len(index_data_rej)))
    logger.info("Slices rejected = {0}".format(len(index_data_rej)))
    logger.info('Comparison successfully completed')
    for index in index_data_rej:
        start = index * int(factors[indx_len])
        data_to_check[start:start + int(factors[indx_len])] = np.nan
    frac_rejected = len(index_data_rej) * factors[indx_len] / len(data_to_check)
    logger.debug("frac_rejected={0}".format(frac_rejected))
    logger.info('Data rejected {0} %'.format(round(frac_rejected * 100, 3)))
    if not verbose:
        pass
    else:
        print('\tData rejected:', frac_rejected * 100, '%')
        print('Comparison completed!')
    logger.info('Replacement successfully completed')
    logger.warning("All data are above the threshold") if frac_rejected == 1 else ''
    return data_to_check, frac_rejected


def psd(day, month, year, quantity, ax, interval, mode, low_freq=2, high_freq=10, threshold=0.03, ndays=1,
        length=10000, ax1=None, file_start=None, file_stop=None, tevo=False, verbose=False):
    try:
        if not ax:
            raise TypeError("Ax can not be a 'NoneType' object")
        if mode not in ['low noise', 'max interval']:
            raise AttributeError("Mode must be either 'low noise' or 'max interval'. You insert {0}".format(mode))
        if not low_freq < high_freq:
            raise ValueError("high_freq must be greater or equal than low_freq")
    except TypeError as err:
        logger.error(err.args[0])
        raise
    except AttributeError as err:
        logger.error(err.args[0])
        raise
    except ValueError as err:
        logger.error(err.args[0])
        raise
    logger.debug('PARAMETERS: day={0} month={1} year={2} quantity={3} ax={4} interval={5} mode={6} low_freq={7} '
                 'high_freq={8} threshold={9} ndays={10} length={11} ax1={12} file_start={13} file_stop={14} tevo={15} '
                 'verbose={16}'.format(day, month, year, quantity, ax, interval, mode, low_freq, high_freq, threshold,
                                       ndays, length, ax1, file_start, file_stop, tevo, verbose))
    logger.info("Evaluation of the PSD on '{}' started".format(quantity))
    df_qty, col_index, t = read_data(day, month, year, quantity, num_d=ndays, tevo=tevo, file_start=file_start,
                                     file_stop=file_stop, verbose=verbose)
    df_itf, _, _ = read_data(day=day, month=month, year=year, quantity='ITF', num_d=ndays, file_start=None,
                             file_stop=None, verbose=verbose)
    df_po, _, _ = read_data(day=day, month=month, year=year, quantity='Pick Off', num_d=ndays, file_start=None,
                            file_stop=None, verbose=verbose)
    logger.debug('rows_in_df_{0}={1} rows_in_df_itf={2} rows_in_df_pickoff={3}'.format(quantity, len(df_qty.index),
                                                                                       len(df_itf.index),
                                                                                       len(df_po.index)))
    data_cleared, _ = th_comparison(df_qty, threshold=threshold, length=length, verbose=verbose)
    if not verbose:
        pass
    else:
        print('Removing NaN values...')
        print('NaN values successfully removed')
    logger.info('Start cleaning from NaN values')
    data_first_check = [list(group) for key, group in groupby(data_cleared, lambda x: not np.isnan(x)) if key]

    logger.info('Cleaning from NaN values successfully completed')
    num = int(60 * freq)  # Add checks for frequencies
    logger.debug('num={0}'.format(num))
    logger.info('Evaluating the PSD frequencies')
    _, psd_f = mlab.psd(np.ones(num), NFFT=num, Fs=freq, detrend="linear")  # , noverlap=int(num / 2))
    logger.debug('psd_f obtained with N={0}, NFFT={0}, Fs={1}, detrend=linear'.format(len(np.ones(num)), num, freq))
    psd_f = psd_f[1:]
    outdata, file_index = [], []
    len_max, pick_off_mean = 0, 0
    length_data_used = 0
    integral_min = np.inf

    data_to_plot = np.array([])
    opt_index_used = []
    data_size = 0

    start = np.where(psd_f == low_freq)[0][0]
    stop = np.where(psd_f == high_freq)[0][0]
    if not verbose:
        pass
    else:
        print('Number of usable array:', len(data_first_check))
        print('Starting evaluation of', mode, 'PSD...')
    logger.info("Searching the optimal array")

    if mode.lower() == 'max interval':
        for el in data_first_check:
            if len(el) >= interval * freq and len(el) > len_max:
                len_max = len(el)
                outdata = el
                file_index = list(ac.find_rk(df_qty.values.flatten(), el))

        data_size = len(outdata)
        logger.info('Length of the array chosen: {0}'.format(data_size))

        v_max = df_itf.max().values.flatten()
        v_min = df_itf.min().values.flatten()
        alpha = first_coef / (v_max - v_min)

        logger.info('Voltage min = {0}'.format(v_min))
        logger.info('Voltage max = {0}'.format(v_max))
        logger.info('Alpha = {0}'.format(alpha))

        opt_index_used = list(ac.find_rk(df_qty.values.flatten(), outdata))
        logger.debug('Size of the optimal data: {0}'.format(data_size))
        start_date = datetime.datetime.fromtimestamp(t[opt_index_used[0]]).strftime(
            '%d/%m/%y %H:%M:%S')
        end_date = datetime.datetime.fromtimestamp(
            t[opt_index_used[0]] + (data_size - 1) * 0.001).strftime(
            '%d/%m/%y %H:%M:%S')
        logger.info('Optimal data selected are from {0} to {1}'.format(start_date, end_date))
        psd_s, _ = mlab.psd(outdata, NFFT=num, Fs=freq, detrend="linear", noverlap=int(num / 2))
        psd_s = psd_s[1:]
        pick_off_mean = np.abs(
            np.mean(df_po[file_index[0]:file_index[0] + data_size + 1].values.flatten()))
        data_to_plot = np.sqrt(psd_s) * alpha * pick_off_mean

        ax.set_xscale("log")
        ax.set_xlim([psd_f[0], psd_f[-1]])
        ax.set_yscale("log")

    elif mode.lower() == 'low noise':
        i = 0
        for el in data_first_check:
            if not verbose:
                pass
            else:
                print(round(i / len(data_first_check) * 100, 2), '%')
            if len(el) >= interval * freq:
                el_s, _ = mlab.psd(el, NFFT=num, Fs=freq, detrend="linear")  # , noverlap=int(num / 2))
                logger.debug('el_s obtained with N={0}, NFFT={1}, Fs={2}, detrend=linear'.format(len(el), num, freq))
                el_s = el_s[1:]
                integral = sum(
                    el_s[start:stop] / len(el_s[start:stop]))  # len(el) differential in the integral to evaluate RMS
                logger.debug('RMS integral: {0}'.format(integral))
                if integral < integral_min:
                    integral_min = integral
                    file_index = list(ac.find_rk(df_qty.values.flatten(), el))
                    outdata = el
                    length_data_used = len(el)
            i += 1
        logger.debug("Size of the data used: {0}".format(length_data_used))
        logger.info('Value of the integral: {0}'.format(integral_min))
        logger.info('Mean of the pick-off: {0}'.format(pick_off_mean))
        if not file_index:
            logger.warning('Data chosen is not a subset of the general data!')
        else:
            start_date = datetime.datetime.fromtimestamp(t[file_index[0]]).strftime('%d/%m/%y %H:%M:%S')
            end_date = datetime.datetime.fromtimestamp(t[file_index[0]] + (length_data_used - 1) * 0.001).strftime(
                '%d/%m/%y %H:%M:%S')
            logger.info('First index of the data chosen: {0}'.format(file_index))
            logger.info('Data selected are from {0} to {1}'.format(start_date, end_date))
        logger.info("Optimal array successfully found")
        if not verbose:
            pass
        else:
            print('Evaluation of', mode, 'PSD completed!')
        v_max = df_itf.max().values.flatten()
        v_min = df_itf.min().values.flatten()
        alpha = first_coef / (v_max - v_min)

        logger.info('Voltage min = {0}'.format(v_min))
        logger.info('Voltage max = {0}'.format(v_max))
        logger.info('Alpha = {0}'.format(alpha))

        num_slice = int(len(outdata) / 300000)  # It must be an integer

        logger.debug('Number of slice for the subset: {0}'.format(num_slice))

        data_split = np.array_split(outdata, num_slice)
        integral_list, indeces_lst = [], []
        optimal_data = np.array([])

        for index, chunk in enumerate(data_split):
            chunk_s, _ = mlab.psd(chunk, NFFT=num, Fs=freq, detrend="linear")  # , noverlap=int(num / 2))
            logger.debug('chunk_s obtained with N={0}, NFFT={1}, Fs={2}, detrend=linear'.format(len(chunk), num, freq))
            chunk_s = chunk_s[1:]
            integral = sum(chunk_s[start:stop] / len(chunk_s[start:stop]))
            logger.debug('RMS integral: {0}'.format(integral))
            integral_list.append(integral)
            if not integral <= 1e-11:  # Add a variable to express the quantity
                pass
            else:
                indeces_lst.append(index)

        logger.debug('Chunks candidates {0}'.format(indeces_lst))

        index_optimal_data = [list(map(itemgetter(1), group)) for key, group in
                              groupby(enumerate(indeces_lst), lambda ix: ix[0] - ix[1])]
        max_optimal_index = max(index_optimal_data, key=lambda elem: len(elem))

        logger.debug('Consecutive indices: {0}'.format(index_optimal_data))
        logger.info('Optimal index chosen: {0}'.format(max_optimal_index))

        for inx in max_optimal_index:
            optimal_data = np.append(optimal_data, data_split[inx])

        data_size = len(optimal_data)
        opt_index_used = list(ac.find_rk(df_qty.values.flatten(), optimal_data))
        pick_off_mean = np.abs(
            np.mean(df_po[file_index[0]:file_index[0] + data_size + 1].values.flatten()))

        logger.debug('Size of the optimal data: {0}'.format(data_size))
        start_date = datetime.datetime.fromtimestamp(t[opt_index_used[0]]).strftime(
            '%d/%m/%y %H:%M:%S')
        end_date = datetime.datetime.fromtimestamp(
            t[opt_index_used[0]] + (data_size - 1) * 0.001).strftime(
            '%d/%m/%y %H:%M:%S')
        logger.info('Optimal data selected are from {0} to {1}'.format(start_date, end_date))

        opt_psd, _ = mlab.psd(optimal_data, NFFT=num, Fs=freq, detrend="linear", noverlap=int(num / 2))
        logger.debug(
            'opt_psd obtained with N={0}, NFFT={1}, Fs={2}, detrend=linear, noverlap = {3}'.format(data_size, num,
                                                                                                   freq, int(num / 2)))
        opt_psd = opt_psd[1:]
        data_to_plot = np.sqrt(opt_psd) * alpha * pick_off_mean

        ax.set_xscale("linear")
        ax.set_xlim([2, 20])
        ax.set_ylim([1.e-13, 1.e-8])
        ax.set_yscale("log")

        logger.info("Building plot")

    x, y = np.loadtxt(os.path.join(path_to_data, 'VirgoData_Jul2019.txt'), unpack=True, usecols=[0, 1])
    x_davide, y_davide = np.loadtxt(os.path.join(path_to_data, 'psd_52_57.txt'), unpack=True, usecols=[0, 1])

    ax.plot(x, y, linestyle='-', color='red', label='@ Virgo')
    ax.plot(x_davide, y_davide, linestyle='-', color='tab:blue', label='@ Sos-Enattos Davide')
    ax.plot(psd_f, data_to_plot, linestyle='-', color='tab:orange', label='@ Sos-Enattos Luca')

    # diff = (y-data_to_plot)/(y+data_to_plot)
    # ax.plot(psd_f, diff, linestyle='-', label='(a-b)/(a+b)')

    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel(r"ASD [rad/$\sqrt{Hz}$]", fontsize=20)
    ax.tick_params(axis='x', labelsize=20, which='both')
    ax.tick_params(axis='y', labelsize=20, which='both')
    ax.grid(True, linestyle='--', which='both')
    ax.legend(loc='best', shadow=True, fontsize='large')
    ax.set_title(mode)
    logger.info("Plot succesfully built")
    if ax1:
        logger.info("Building second plot")

        opt_data_used_x = t[opt_index_used[0]:opt_index_used[0] + data_size + 1]
        opt_data_used_y = df_qty[opt_index_used[0]:opt_index_used[0] + data_size + 1].values.flatten()
        ax1.plot(t, df_qty[col_index], linestyle='dotted', label='All data')
        ax1.plot(opt_data_used_x, opt_data_used_y, linestyle='-', label='Data used')
        ax1.grid(True, linestyle='-')
        ax1.xaxis.set_major_formatter(ac.time_tick_formatter)
        ax1.tick_params(axis='x', labelsize=16, which='both')
        ax1.tick_params(axis='y', labelsize=16, which='both')
        ax1.set_ylabel(r"Voltage [V]", fontsize=16)
        ax1.legend(loc='best', shadow=True, fontsize='large')
        logger.info("Second plot succesfully built")
    return ax, ax1


# OLD VERSION (TO BE REVIEWED...)
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
    ax.xaxis.set_major_formatter(ac.time_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='large')
    return ax, filename


# WORK IN PROGRESS...
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
    ax.legend(loc='best', shadow=True, fontsize='large')
    ax.xaxis.set_major_formatter(ac.time_tick_formatter)
    ax.xaxis.set_tick_params(rotation=30)
    return ax, filename
