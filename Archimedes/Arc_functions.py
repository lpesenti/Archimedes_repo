import datetime
from matplotlib.mlab import cohere
import numpy as np
import pandas as pd
import os
import glob
import re

unix_time = 0  # It is used only to make right conversion in time for time evolution analysis
path_to_data = r"C:\Users\lpese\PycharmProjects\Archimedes-Sassari\Archimedes\Data"
freq = 1000  # Hz


def read_data(day, month, year, col_to_save, num_d):
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
        The quantity to be read. It must be one of the following

        - 'ITF' : the signal from the interferometer expressed in V
        - 'Pick Off' : the signal from the pick-off expressed in V
        - 'Signal injected' : the signal from the waveform used expressed in V
        - 'Error' :
        - 'Correction' :
        - 'Actuator 1' : the output of the actuator 1 before amplification expressed in V
        - 'Actuator 2' : the output of the actuator 2 before amplification expressed in V
        - 'After Noise' :
        - 'Time' : the timestamp of the data saved every milli second in human-readable format

    num_d : int
        How many days of data you want to analyze.

    Returns
    -------
    A tuple of a pandas DataFrame [n-rows x 1-column] containing the data, the index of the column and the timestamp
    expressed in UNIX
    """

    print('--------------- Reading', day, '/', month, '/', year, '-', col_to_save, 'data ---------------')
    month = '%02d' % month  # It transforms 1,2,3,... -> 01,02,03,...
    cols = np.array(
        ['ITF', 'Pick Off', 'Signal injected', 'Error', 'Correction', 'Actuator 1', 'Actuator 2', 'After Noise',
         'Time'])
    index = np.where(cols == col_to_save)[0][0] + 1
    all_data = []
    final_df = pd.DataFrame()
    for i in range(num_d):  # Loop over number of days
        data_folder = os.path.join(path_to_data, "SosEnattos_Data_{0}{1}{2}".format(year, month, day + i))
        temp_data = glob.glob(os.path.join(data_folder, "*.lvm"))
        temp_data.sort(key=lambda f: int(re.sub('\D', '', f)))
        all_data += temp_data
    time = pd.read_table(all_data[0], sep='\t', usecols=[9], header=None)  # Read the time from first data
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


def time_tick_formatter(val, pos=None):
    """
    Return val reformatted as a human readable date

    See Also
    --------
    time_evolution : it is used to rewrite x-axis
    """
    global unix_time
    if val % 1000 == 0:
        val = str(datetime.datetime.fromtimestamp(int(unix_time + val * 0.001)))
    else:
        val = ''
    return val


def time_evolution(day, month, year, quantity, ax, ndays=1):
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
            - 'Error' :
            - 'Correction' :
            - 'Actuator 1' : the output of the actuator 1 before amplification expressed in V.
            - 'Actuator 2' : the output of the actuator 2 before amplification expressed in V.
            - 'After Noise' :
            - 'Time' : the timestamp of the data saved every milli second in human-readable format

        ax: ax
            The ax to be given in order to have a plot

        ndays : int
            How many days of data you want to analyze.

        Returns
        -------
        A tuple of an axes and the relative filename
    """
    df, col_index, t = read_data(day, month, year, quantity, ndays)
    global unix_time
    unix_time = t
    print('--------------- Building Plot! ---------------')
    lab = quantity + ' (n° days: ' + str(ndays) + ')'
    filename = str(year) + str(month) + str(day) + '_' + quantity + '_nDays_' + str(ndays)
    ax.plot(df.index, df[col_index], label=lab)
    ax.grid(True, linestyle='-')
    ax.set_ylabel('Voltage [V]')
    ax.xaxis.set_major_formatter(time_tick_formatter)
    ax.legend(loc='best', shadow=True, fontsize='medium')
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
