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

import datetime
from matplotlib.mlab import cohere
import numpy as np
import pandas as pd
import os
import glob
import re
import logging

unix_time = 0  # It is used only to make right conversion in time for time evolution analysis
path_to_data = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Data"
freq = 1000  # Hz


def read_data(day, month, year, col_to_save, num_d, verbose=True):
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
    expressed in UNIX
    """
    logging.info('Data read started')
    logging.debug('PARAMETERS: day=%i month=%i year=%i col_to_save=%s num_d=%i verbose=%s' % (
        day, month, year, col_to_save, num_d, verbose))
    if verbose:
        print('--------------- Reading', day, '/', month, '/', year, '-', col_to_save, 'data ---------------')
    month = '%02d' % month  # It transforms 1,2,3,... -> 01,02,03,...
    cols = np.array(
        ['ITF', 'Pick Off', 'Signal injected', 'Error', 'Correction', 'Actuator 1', 'Actuator 2', 'After Noise',
         'Time'])
    index = np.where(cols == col_to_save)[0][0] + 1  # Find the index corresponding to the the col_to_save
    all_data = []
    final_df = pd.DataFrame()
    for i in range(num_d):  # Loop over number of days
        data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day + i))
        temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
        temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_data += temp_data
    time = pd.read_table(all_data[0], sep='\t', usecols=[9], header=None)  # Read the time from first data
    start_t = time[9][0].replace("\\", '')  # Remove \\ in the data format -> it allows to be converted
    timestamp = datetime.datetime.timestamp(pd.to_datetime(start_t))  # Conversion of the time to timestamp
    i = 0
    logging.debug('Data length: %f' % len(all_data))
    for data in all_data:
        if verbose:
            print(round(i / len(all_data) * 100, 1), '%')
        # if i == 0:
        a = pd.read_table(data, sep='\t', usecols=[index], header=None)  # Read only the column of interest -> [index]
        final_df = pd.concat([final_df, a], axis=0, ignore_index=True)  # At the end we have a long column with all data
        i += 1
    if verbose:
        print('--------------- Reading completed! ---------------')
    logging.info('Data read completed')
    return final_df, index, timestamp


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


def time_evolution(day, month, year, quantity, ax, ndays=1, verbose=True):
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
    df, col_index, t = read_data(day, month, year, quantity, ndays)
    global unix_time  # Access the global variable and
    unix_time = t  # change its value to the timestamp of the data
    if verbose:
        print('Building Time Evolution Plot...')
    lab = quantity + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays) + 'tEvo'
    ax.plot(df.index, df[col_index], color='tab:red', label=lab)
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    ax.xaxis.set_major_formatter(time_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='medium')
    return ax, filename


def th_comparison(data_frame, threshold, ndivision, verbose=True):
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

        ndivision : int
            Represent the length of each slice in whic the data will be divided into.
            The greater it is, the smaller will be the number of slice and so the comparison will be less
            accurate but will require less time to perform it.
            WARNING: a big number of ndivision can cause a bad comparison due to the fluctuations in the signal.

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
    logging.debug('PARAMETERS: rows_in_df=%s threshold=%i ndivision=%i verbose=%s' % (
        len(data_frame.index), threshold, ndivision, verbose))
    if verbose:
        print('Starting the comparison...')
        print('\tThreshold:', threshold)
    data_to_check = data_frame.to_numpy()
    data_deriv = np.abs(np.diff(data_to_check, axis=0))
    data_deriv = np.append(data_deriv, 0)
    num_slice = int(data_deriv.size / ndivision)  # In the case of data_to_check is not a multiple of ndivision
    data_split = np.array_split(data_deriv, num_slice)
    lst = []
    i = 0
    for sub_arr in data_split:
        # print(round(i / len(data_split) * 100, 1), '%')
        if np.amax(sub_arr) > threshold:
            lst.append(i)
        else:
            pass
        i += 1
    logging.info('Comparison completed')
    # for index in lst:
    #     start = index * ndivision
    #     data_to_check[start:start + ndivision + 1] = np.nan
    frac_rejected = len(lst) * ndivision / len(data_to_check)
    if verbose:
        print('\tData rejected:', frac_rejected * 100, '%')
        print('Comparison completed!')
    logging.debug('Data rejected: %f ' % frac_rejected)
    logging.info('Replacement completed')
    return data_to_check, data_deriv, frac_rejected


def der_plot(day, month, year, quantity, ax, threshold, ndays=1, ndivision=500, verbose=True):
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

        ndivision : int
            The number of division in which the data will be divided into.
            The greater is, the smaller will be the number contained in each slice and so the comparison will be more
            accurate but will require more time to perform the comparison.
            WARNING: a big number of ndivision can cause a bad comparison due to the fluctuations in the signal.

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
    df, _, t = read_data(day, month, year, quantity, ndays)
    global unix_time
    unix_time = t
    data_und_th, data_deriv, val_rej = th_comparison(df, threshold, ndivision)
    lab = r'$\partial_t$ ' + quantity + ' (n° days: ' + str(ndays) + ')'
    lab1 = quantity + r' cleared (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays) + 'der'
    th_line = np.full((len(df.index, )), threshold)
    if verbose:
        print('--------------- Building Derivative and Data_Cleared Plot! ---------------')
    ax.plot(df.index, data_deriv, color='tab:green', linestyle='-', label=lab)
    ax.plot(df.index, th_line, color='tab:red', linestyle='-', label='threshold')
    ax.plot(df.index, data_und_th, color='tab:blue', linestyle='-', label=lab1)
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
