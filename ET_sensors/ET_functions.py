__author__ = "Luca Pesenti and Davide Rozza "
__credits__ = ["Domenico D'Urso", "Luca Pesenti", "Davide Rozza"]
__version__ = "0.8.5"
__maintainer__ = "Luca Pesenti and Davide Rozza"
__email__ = "l.pesenti6@campus.unimib.it, drozza@uniss.it"
__status__ = "Prototype"

import numpy as np
from obspy import UTCDateTime
from obspy import read, read_inventory, Stream
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm
import glob
from matplotlib import mlab
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.dates import date2num, num2date, DateFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter

import datetime


def NLNM(unit):
    """
    The Peterson model represents an ensemble of seismic spectra measured in a worldwide network (Peterson 1993,
    U.S. Geol. Surv. Rept. 2 93–322). In this way it is possible to define a low noise model (NLNM) and an high noise
    model (NHNM) representing respectively the minimum and the maximum natural seismic background that can be found
    on Earth. Here we define these two curves.

    :param unit:
        unit = 1 displacement, = 2 spped

    :return:
        A tuple for frequency, NLNM, freq, NHNM
    """
    PL = np.array([0.1, 0.17, 0.4, 0.8, 1.24, 2.4, 4.3, 5, 6, 10, 12, 15.6, 21.9, 31.6, 45, 70,
                   101, 154, 328, 600, 10000])
    AL = np.array([-162.36, -166.7, -170, -166.4, -168.6, -159.98, -141.1, -71.36, -97.26,
                   -132.18, -205.27, -37.65, -114.37, -160.58, -187.5, -216.47, -185,
                   -168.34, -217.43, -258.28, -346.88])
    BL = np.array([5.64, 0, -8.3, 28.9, 52.48, 29.81, 0, -99.77, -66.49, -31.57, 36.16,
                   -104.33, -47.1, -16.28, 0, 15.7, 0, -7.61, 11.9, 26.6, 48.75])

    PH = np.array([0.1, 0.22, 0.32, 0.8, 3.8, 4.6, 6.3, 7.9, 15.4, 20, 354.8, 10000])
    AH = np.array([-108.73, -150.34, -122.31, -116.85, -108.48, -74.66, 0.66, -93.37, 73.54,
                   -151.52, -206.66, -206.66])
    BH = np.array([-17.23, -80.5, -23.87, 32.51, 18.08, -32.95, -127.18, -22.42, -162.98,
                   10.01, 31.63, 31.63])

    fl = 1 / PL
    fh = 1 / PH
    lownoise = 10 ** ((AL + BL * np.log10(PL)) / 20)
    highnoise = 10 ** ((AH + BH * np.log10(PH)) / 20)

    if unit == 1:  # displacement
        lownoise = lownoise * (PL / (2 * np.pi)) ** 2
        highnoise = highnoise * (PH / (2 * np.pi)) ** 2

    if unit == 2:  # speed
        lownoise = lownoise * (PL / (2 * np.pi))
        highnoise = highnoise * (PH / (2 * np.pi))

    return fl, lownoise, fh, highnoise


def read_Inv(filexml, network, sensor, location, channel, t, Twindow, verbose):
    """
    Read Inventory (xml file) of the sensor

    :param filexml:
        path and name of the xml file to be read
    :param network:
        sensor network
    :param sensor:
        name of the sensor
    :param location:
        location of the sensor
    :param channel:
        channel to be analysed
    :param t:
        UTC time of the analysis
    :param Twindow:
        time range for PSD
    :param verbose:
        If True the verbosity is enabled

    :return:
        A tuple for frequency and sensor's response, the sample frequency
    """

    # read inventory
    invxml = read_inventory(filexml)
    if verbose:
        print(invxml)
        # print sensor information
        for inet in range(0, 1):
            for ista in range(0, 12):
                if invxml[inet].code == network and invxml[inet][ista].code == sensor:
                    print(invxml[inet][ista][0])
    # select sensor and channel
    invxmls = invxml.select(network=network, station=sensor, channel=channel)
    # print(invxmls)
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    resp = invxmls.get_response(seed_id, t)  # TODO: automatically set the correct time based on first data
    gain = invxmls.get_response(seed_id, t).instrument_sensitivity.value
    if verbose:
        invxmls.plot_response(1 / 120, label_epoch_dates=True)
        print(resp)
        print(gain)
    # return a nested dictionary detailing the sampling rates of each response stage
    sa = resp.get_sampling_rates()
    # TODO: look at the Nikita's code to see how retrieve correctly the information on the sampling rate
    # take the last output_sampling_rate
    fsxml = sa[len(sa)]['output_sampling_rate']
    if verbose:
        print('Sampling rate:', fsxml)
    Num = int(Twindow * 100)  # TODO: Why fsxml is different from the real sampling rate? See previous TODO
    # Returns frequency response and corresponding frequencies using evalresp
    #                                        time_res,       Npoints,  output
    sresp, fxml = resp.get_evalresp_response(t_samp=1 / 100, nfft=Num, output="VEL")
    fxml = fxml[1:]  # remove first value that is 0
    sresp = sresp[1:]  # remove first value that is 0
    # amplitude -> absolute frequency value
    respamp = np.absolute(sresp * np.conjugate(sresp))
    if verbose:
        print('Response amplitude length:', len(respamp))

    return fxml, respamp, fsxml, gain


def extract_stream(filexml, Data_path, network, sensor, location, channel, tstart, tstop, Twindow, verbose):
    """
    Extract the stream from data file

    :param filexml:
        path and name of the xml file to be read
    :type filexml:
        str
    :param Data_path:
        path of the data file to be read
    :param network:
        sensor network
    :param sensor:
        name of the sensor
    :param location:
        location of the sensor
    :param channel:
        channel to be analysed
    :param tstart:
        UTC start
    :param tstop:
        UTC stop
    :param Twindow:
        time range for PSD
    :param verbose:
        If True the verbosity is enabled

    :return:
        A tuple for frequency and sensor's PSD
    """

    # Time interval
    yi = str(tstart.year)
    mi = str(tstart.month)
    doyi = tstart.strftime('%j')
    yf = str(tstop.year)
    mf = str(tstop.month)
    doyf = tstop.strftime('%j')
    if verbose:
        print("Analysis from: ", yi, mi, doyi, " to: ", yf, mf, doyf)
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=verbose)
    filename_list = glob.glob(Data_path + seed_id + "*")
    filename_list.sort()
    if verbose:
        print('Response amplitude length:', len(respamp))
        print(filename_list)
    # read filename
    st_tot = Stream()
    for file in filename_list:
        st = read(file)  # , starttime=tstart, endtime=tf)
        st_tot += st
    if verbose:
        print(st_tot)
        print(len(st_tot[0].times("timestamp")))

    return st_tot


def ppsd(stream, filexml, sensor, Twindow, Overlap, temporal=False):
    """
    Make PPSD plot

    :param stream:
        stream of data
    :param filexml:
        path and name of the xml file to be read
    :param sensor:
        name of the sensor
    :param Twindow:
        time range for PSD
    :param Overlap:
        PSD overlap should be lower than 50 %

    :return:
        PPSD plot with NoiseModel
    """

    invxml = read_inventory(filexml)
    fs = stream[0].stats.sampling_rate
    Num = Twindow * fs
    seism_ppsd = PPSD(stream.select(station=sensor)[0].stats, invxml, ppsd_length=Twindow,
                      overlap=Overlap)  # (Overlap / 100 * Num))
    for itrace in range(len(stream)):
        seism_ppsd.add(stream.select(station=sensor)[itrace])

    seism_ppsd.plot(cmap=pqlx, xaxis_frequency=True, period_lim=(1 / 120, 20))  # fs / 2))
    if temporal:
        seism_ppsd.plot_temporal(period=1)  # Period of PSD values to plot. The period bin with the central period that
        # is closest to the specified value is selected. Multiple values can be specified in a list
        sps = int(stream[0].stats.sampling_rate)
        stream.merge()
        stream.spectrogram(wlen=.1 * sps, per_lap=0.90, dbscale=True, log=True)  # , cmap='YlOrRd')


def psd_rms_finder(stream, filexml, network, sensor, location, channel, tstart, Twindow, Overlap, mean_number,
                   verbose, out):  # , ax, ax1):
    """
    Best PSD and RMS finder function

    :param stream:
        data stream
    :param filexml:
        path and name of the xml file to be read
    :param network:
        sensor network
    :param sensor:
        name of the sensor
    :param location:
        location of the sensor
    :param channel:
        channel to be analysed
    :param tstart:
        UTC start
    :param Twindow:
        time range for PSD
    :param Overlap:
        PSD overlap should be lower than 50 %

    :return:
        PSD and RMS plots
    """
    _, _, _, gain = read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=False)
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    data = np.array([])
    vec_rms = np.array([])
    """for itrace in range(len(stream)):
        data = np.append(data, np.array(stream[itrace]) / gain)"""
    fs = stream[0].stats.sampling_rate
    Num = int(Twindow * fs)
    # data_split = np.array_split(data, len(data) / Num)
    _, f_s = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs, noverlap=int(Overlap / 100 * Num))
    f_s = f_s[1:]
    start = np.where(f_s == 1)[0][0]
    stop = np.where(f_s == 10)[0][0]
    integral_min = np.inf
    best_time = []
    rms_time = np.array([])

    # Num = int(Twindow * fs)

    for itrace in range(len(stream)):
        data = np.array(stream[itrace]) / gain
        time = stream[itrace].times('timestamp')
        data_split = np.array_split(data, len(data) / Num)
        for index, chunk in enumerate(data_split):
            chunk_s, _ = mlab.psd(chunk, NFFT=Num, Fs=fs, detrend="linear", noverlap=Overlap)
            chunk_s = chunk_s[1:]
            integral = sum(chunk_s[start:stop] / len(chunk_s[start:stop]))  # * (f_s[1]-f_s[0]))
            vec_rms = np.append(vec_rms, integral)
            rms_time = np.append(rms_time, time[index * len(chunk):len(chunk) + index * len(chunk)][0])
            if integral < integral_min:
                integral_min = integral
                data = chunk
                best_time = time[index * len(chunk):len(chunk) + index * len(chunk)]
    # data = np.array([])
    """for index, chunk in enumerate(data_split):
        chunk_s, _ = mlab.psd(chunk, NFFT=Num, Fs=fs, detrend="linear", noverlap=Overlap)
        chunk_s = chunk_s[1:]
        integral = sum(chunk_s[start:stop] / len(chunk_s[start:stop]))  # * (f_s[1]-f_s[0]))
        vec_rms = np.append(vec_rms, integral)
        if integral < integral_min:
            integral_min = integral
            data = chunk
            t = tstart.datetime
            if verbose:
                print(time.strftime('%d/%m/%y %H:%M:%S',
                                    time.gmtime((t - datetime.datetime(1970, 1, 1)).total_seconds() + index * Twindow)))"""
    best_psd, f_best = mlab.psd(data, NFFT=int(Num / mean_number), Fs=fs, detrend="linear",
                                noverlap=int(Overlap / 100 * Num / mean_number))
    f_best = f_best[1:]
    best_psd = best_psd[1:]
    print('Data taken from:', datetime.fromtimestamp(best_time[0]), 'to:', datetime.fromtimestamp(best_time[-1:]))

    if out:
        output(freq_data=f_best, psd_data=best_psd, rms_data=vec_rms, sampling_rate=fs)

    plot_maker(frequency_data=f_best, psd_data=best_psd, rms_data=vec_rms, sampling_rate=fs, sensor_id=seed_id)
    rms_plotter(rms_time=rms_time, rms_data=vec_rms)

    return f_best, best_psd, fs, vec_rms, seed_id


def plot_maker(frequency_data, psd_data, rms_data, sampling_rate, sensor_id):
    fl, nlnm, fh, nhnm = NLNM(2)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    ax0.tick_params(axis='both', which='both', labelsize=15)
    ax0.plot(frequency_data, np.sqrt(psd_data), linestyle='-', color='tab:blue', label='Best PSD')
    ax0.plot(fl, nlnm, 'k--', label="Noise Model")
    ax0.plot(fh, nhnm, 'k--')
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim([0.005, sampling_rate / 2])
    ax0.axvline(x=0.01, color='r', linestyle='dotted')
    ax0.axvline(x=20, color='r', linestyle='dotted')
    ax0.text(0.01, 1.5e-5, '0.01 Hz', fontsize=12, color='r', ha='center', va='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'r', 'boxstyle': 'round'})
    ax0.text(20, 1.5e-5, '20 Hz', fontsize=12, color='r', ha='center', va='center',
             bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'r', 'boxstyle': 'round'})
    ax0.set_ylim([1e-11, 1e-5])
    ax0.grid(True, linestyle='--')
    ax0.set_xlabel('Frequency [Hz]', fontsize=20)
    ax0.set_ylabel(r'Seismic [(m/s)/$\sqrt{Hz}$]', fontsize=20)
    ax0.set_title(sensor_id, fontsize=20)
    ax0.legend(loc='best', shadow=True, fontsize='medium')

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.tick_params(axis='both', which='both', labelsize=15)
    # ax1.hist(rms_data, bins=100)
    # ax1.grid(True, linestyle='--')
    # ax1.set_yscale("log")
    # ax1.set_xlabel(r'Integral [$(m/s)^2$]', fontsize=20)
    # ax1.set_ylabel(r'Counts', fontsize=20)

    plt.show()


def rms_plotter(rms_time, rms_data):
    def myfromtimestampfunction(timestamp):
        return datetime.fromtimestamp(timestamp)

    def myvectorizer(input_func):
        def output_func(array_of_numbers):
            return [input_func(a) for a in array_of_numbers]

        return output_func

    import matplotlib.dates as mdates

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    date_convert = myvectorizer(myfromtimestampfunction)
    xtimex = date_convert(rms_time)
    xfmt = mdates.DateFormatter('%b %d %H:%M')  #:%S')
    ax0.xaxis.set_major_formatter(xfmt)
    ax0.plot(xtimex, rms_data, linestyle='-', color='tab:blue', label='RMS')
    ax0.grid(True, linestyle='--')
    ax0.set_xlabel('time', fontsize=20)
    ax0.set_ylabel(r'RMS', fontsize=20)
    ax0.legend(loc='best', shadow=True, fontsize='medium')

    import pandas as pd

    df = pd.DataFrame({'rms': rms_data}, index=pd.to_datetime(rms_time, unit='s'))
    data_0_7 = df.loc[(df.index.hour >= 0) & (df.index.hour <= 7)]
    data_8_15 = df.loc[(df.index.hour >= 8) & (df.index.hour <= 15)]
    data_16_23 = df.loc[(df.index.hour >= 16) & (df.index.hour <= 23)]
    print(data_0_7.head())
    print('% of rms data between 0 and 7:', round(len(data_0_7) * 100 / len(rms_data), 2))
    print('% of rms data between 8 and 15:', round(len(data_8_15) * 100 / len(rms_data), 2))
    print('% of rms data between 16 and 23:', round(len(data_16_23) * 100 / len(rms_data), 2))
    print('mean of rms data between 0 and 8:', data_0_7.mean())
    print('mean of rms data between 8 and 15:', data_8_15.mean())
    print('mean of rms data between 16 and 23:', data_16_23.mean())

    data_0_7.reset_index().plot(x='index', y='rms', title='Integral between 0 and 7')
    data_8_15.reset_index().plot(x='index', y='rms', title='Integral between 8 and 15')
    data_16_23.reset_index().plot(x='index', y='rms', title='Integral between 16 and 23')

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax1.hist(rms_data, bins=100)
    ax1.grid(True, linestyle='--')
    ax1.set_yscale("log")
    ax1.set_xlabel(r'Integral [$(m/s)^2$]', fontsize=20)
    ax1.set_ylabel(r'Counts', fontsize=20)

    plt.show()


def output(freq_data=np.array([]), psd_data=np.array([]), rms_data=np.array([]), sampling_rate=None):
    import configparser
    from obspy import UTCDateTime
    import datetime

    now = datetime.datetime.now()
    config = configparser.ConfigParser()
    config.read('config.ini')
    out_path = config['Paths']['outfile_path']

    outfile = out_path + 'Results_' + now.strftime('%y-%m-%d_%H-%M-%S') + '.txt'
    outfile_data = out_path + 'Results_' + now.strftime('%y-%m-%d_%H-%M-%S_Data') + '.txt'

    network = config['Instrument']['network']
    sensor = config['Instrument']['sensor']
    location = config['Instrument']['location']
    channel = config['Instrument']['channel']

    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    filename_list = glob.glob(config['Paths']['data_path'] + seed_id + "*")
    filename_list.sort()

    start_doy = int(filename_list[0].replace('.miniseed', '').split('.')[-1:][0])
    start_year = int(filename_list[0].replace('.miniseed', '').split('.')[-2:-1][0])
    start_date_read = datetime.datetime(start_year, 1, 1) + datetime.timedelta(start_doy - 1)
    start = start_date_read.strftime('%d/%m/%y')

    end_doy = int(filename_list[-1:][0].replace('.miniseed', '').split('.')[-1:][0])
    end_year = int(filename_list[-1:][0].replace('.miniseed', '').split('.')[-2:-1][0])
    end_date_read = datetime.datetime(end_year, 1, 1) + datetime.timedelta(end_doy - 1)
    stop = end_date_read.strftime('%d/%m/%y')

    output_str = f"""These are the results of the run performed at: {now.strftime('%H:%M:%S %d/%m/%y')}
{'#' * 20 : <20} {'Analysis information' : ^15} {'#' * 20: >20}
{'': <10}{'Data path:' : <20}{config['Paths']['data_path'] : <15}
{'': <10}{'XML file:' : <20}{config['Paths']['xml_filename'] : <15}
{'': <10}{'Network:' : <20}{network : <15}
{'': <10}{'Sensor:' : <20}{sensor : <15}
{'': <10}{'Location:' : <20}{location : <15}
{'': <10}{'Channel:' : <20}{channel : <15}
{'': <10}{'Start date:' : <20}{str(UTCDateTime(config['Quantities']['start_date'])) : <15}
{'': <10}{'PSD window:' : <20}{config['Quantities']['psd_window'] : <15}
{'': <10}{'Overlap:' : <20}{config['Quantities']['psd_overlap'] : <15}
{'': <10}{'Number of means:' : <20}{config['Quantities']['number_of_means'] : <15}
{'': <10}{'Verbose:' : <20}{config['Quantities']['verbose'] : <15}
{'': <10}{'Save data:' : <20}{config['Quantities']['save_data'] : <15}
{'#' * 20 : <20} {'Data information' : ^15} {'#' * 20: >20}
{len(filename_list)} files has been loaded
The data taken are from {start} to {stop}
The sampling rate of the seismometer analyzed is: {sampling_rate} Hz
Both frequency and PSD data has been saved in {outfile_data}
{len(rms_data)} integrals has been performed
    """

    print(output_str, file=open(outfile, "w"))

    np.savetxt(outfile_data, np.c_[freq_data, psd_data], header='Frequency | PSD values')


#  THE FOLLOWING FUNCTIONS WORK BUT ARE NEITHER IN A OPTIMIZE VERSION NOR IN A RELEASE STATE.

def spectrogram(filexml, Data_path, network, sensor, location, channel, tstart, Twindow, Overlap, verbose,
                save_csv=False, save_img=False, linearplot=True, xscale='linear', show_plot=True):
    r"""
    It performs the spectrogram of given data. The spectrogram is a two-dimensional plot with on the y-axis the
    frequencies, on the x-axis the dates and on virtual z-axis the ASD value expressed
    in :math:`ms^{-2}/\sqrt{Hz}\:[dB]`.

    :type filexml: str
    :param filexml: The .xml needed to read the seismometer response

    :type Data_path: str
    :param Data_path: Path to the data.

    :type network: str
    :param network: Sensor network

    :type sensor: str
    :param sensor: Name of the sensor

    :type location: str
    :param location: Location of the sensor

    :type channel: str
    :param channel: Channel to be analysed

    :type tstart: str, :class: 'obspy.UTCDateTime'
    :param tstart: Start time to get the response from the seismometer (?)

    :type Twindow: float
    :param Twindow: Time windows used to evaluate the PSD.

    :type Overlap: float
    :param Overlap: The overlap expressed as a number, i.e. 0.5 = 50%. The data read is translated of a given quantity
        which depends on this parameter. For example 10' of data are trasnlated by DEFAULT of 10', but with Overlap=0.5,
        the data will be translated by 5'. Therefore, it will achieve 5' of Overlap.

    :type verbose: bool
    :param verbose: Needed for verobsity

    :type save_csv: bool
    :param save_csv: If you want to save the data analyzed ina .csv format

    :type save_img: bool
    :param save_img: If you want to save the images produce. Please note that it is highly recommended setting the value
        on True if more than 5 days of data are considered.

    :type linearplot: bool
    :param linearplot: If you want to create the linear plot of the data with the distinction between daytime and
        nighttime. This type of plot shows the mean with one sigma of confidence interval

    :type xscale: str
    :param xscale: It represents the scale of the lineplot produced. It can be one of 'linear', 'log' or 'both'. Please
        note that setting the variable on 'both' it will produce both linear and logarithmic x-scale plots.

    :type show_plot: bool
    :param show_plot: If you want to show the plot produced. Please note that the spectrogram requires lot of memory to
        be shown especially if the analysis is done on more than 5 days.
    """
    # TODO: check tstart variable, may remove it. Seems useless.

    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=verbose)
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

    print('The analysis start from\t', startdate, '\tto\t', stopdate)

    df = pd.DataFrame(columns=['frequency', 'timestamp', 'psd_value'], dtype=float)

    T = Twindow
    Ovl = Overlap
    TLong = 6 * 3600
    dT = TLong + T * Ovl
    M = int((dT - T) / (T * (1 - Ovl)) + 1)

    K = int((stopdate - startdate) / (T * (1 - Ovl)) + 1)

    print('K: ', K)
    print('M: ', M)

    v = np.empty(K)

    fsxml = 100
    Num = int(T * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    print('MIN. FREQ.:\t', f.min())
    # print(f.size)
    print('GAIN:\t', gain)
    # f_str = ["%.2e" % n for n in f]
    f = np.round(f, 3)
    # fmin = 2
    # fmax = 20
    w = 2.0 * np.pi * f
    # print(np.where(w == 0))
    w1 = w ** 2 / respamp
    # imin = (np.abs(f - fmin)).argmin()
    # imax = (np.abs(f - fmax)).argmin()

    import timeit

    start = timeit.default_timer()
    for file in filename_list:
        k = 0
        print(file)

        st = read(file)
        # print(st.sort().__str__(extended=True))
        st = st.sort()
        Tstop = st[-1:][0].times('timestamp')[-1:][0]
        Tstop = UTCDateTime(Tstop)
        time = st[0].times('timestamp')[0]
        time = UTCDateTime(time)
        # Tstop = st[-1:][0].times('utcdatetime')[-1:][0]
        # time = st[0].times('utcdatetime')[0]
        # startdate = UTCDateTime('{0}-{1}T12:23:34.5'.format(year, day))
        # time = startdate
        while time < Tstop:
            print('Evaluating from\t', time, '\tto\t', Tstop)
            tstart = time
            tstop = time + dT - 1 / fsxml
            st = read(file, starttime=tstart, endtime=tstop)

            t1 = time
            for n in range(0, M):
                v[k] = np.nan
                tr = st.slice(t1, t1 + T - 1 / fsxml)
                if tr.get_gaps() == [] and len(tr) > 0:
                    tr1 = tr[0]
                    if tr1.stats.npts == Num:
                        s, _ = mlab.psd(tr1.data, NFFT=Num, Fs=fsxml, detrend="linear")
                        s = s[1:]
                        # spec = s / gain
                        v[k] = 0

                # d[k] = date2num(t1.datetime)
                # print('t1 =', t1)
                time_measure = np.repeat(t1.timestamp, np.size(f))

                if np.isnan(v[k]):
                    psd_values = np.tile(v[k], np.size(f))
                else:
                    psd_values = s * w1  # w1 --> to have acceleration

                data_to_save = np.concatenate((f, time_measure, psd_values))  # , w1))
                data_to_save = np.reshape(data_to_save, (3, np.size(f))).T
                df = df.append(pd.DataFrame(data_to_save, columns=df.columns), ignore_index=True)
                t1 = t1 + T * Ovl
                k += 1
            time = time + TLong
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # , format='%d-%b %H:%M')
    print(df.head())
    df['psd_value'] = 10 * np.log10(df['psd_value'])  # / gain)
    # df['psd_value'] = 10 * np.log10(
    #     np.sqrt(df['psd_value'] / gain))  # TODO: in the ppsd it is evaluated the psd not asd, try removing the sqrt
    # df['psd_value'] = 10 * np.log10(df['psd_value'] * df['omega'])
    # for freq in range(f.size):
    #     mask = df['frequency'] == f[freq]
    #     df.loc[mask, 'psd_acc'] = df.loc[mask, 'psd_value'] * w1[freq]
    print(df.head())
    print(df.info())
    df['frequency'] = df['frequency'].astype(float)
    # df.frequency = df.frequency.to_string(float_format='{:.2e}'.format)
    df['psd_value'] = df['psd_value'].astype(float)
    result = df.pivot_table(index='frequency', columns='timestamp', values='psd_value')

    if save_csv:
        # TODO: add option for .csv exportation and create an easy function to load and plot data.
        #  Remember to change the df loaded indeces with the freq data. Storage ASD values for every day?
        result.to_csv(
            r'D:\ET\{0}{1}-{2}_{3}-{4}_ACC_{5}.csv'.format(sensor, location, channel, startdate.strftime('%Y%m%d'),
                                                           stopdate.strftime('%Y%m%d'), Twindow))
        # result.to_parquet(r"D:\ET\test.parquet.brotli", compression='brotli', compression_level=9)

    result.columns = result.columns.strftime('%d %b %H:%M')
    # result.index.format('%.2e')
    # pd.set_option('display.float_format', lambda x: '%.2e' % x)
    print(result.info())
    print(result.head())
    # result.index = result.reindex(f_str)
    stop = timeit.default_timer()
    print('Elapsed time before plot:', (stop - start), 's')

    if linearplot:
        daytime_df = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour < 20)]
        nighttime_df = df[(df['timestamp'].dt.hour <= 5) | (df['timestamp'].dt.hour >= 21)]

        print(daytime_df)

        print('xscale =', xscale)

        if xscale == 'linear' or xscale == 'both':
            fig1 = plt.figure(figsize=(19.2, 10.8))
            ax1 = fig1.add_subplot()

            sns.lineplot(x='frequency', y='psd_value', palette=['tab:orange'], data=daytime_df, ci='sd', ax=ax1,
                         label='Daytime')
            sns.lineplot(x='frequency', y='psd_value', palette=['tab:blue'], data=nighttime_df, ci='sd', ax=ax1,
                         label='Nighttime')

            ax1.set_xlabel(r'Frequency [Hz]', fontsize=24)
            ax1.set_ylabel(r'ASD $[(m/s^2)/\sqrt{Hz}]$ [dB]', fontsize=24)
            ax1.set_xlim([1 / 240, 50])
            ax1.set_ylim([-200, -60])
            ax1.tick_params(axis='both', which='major', labelsize=22)
            ax1.grid(True, linestyle='--', axis='both', which='both')
            ax1.legend(loc='best', shadow=True, fontsize=24)
            fig1.tight_layout()

        if xscale == 'log' or xscale == 'both':
            fig2 = plt.figure(figsize=(19.2, 10.8))
            ax2 = fig2.add_subplot()
            ax2.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')
            ax2.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')
            ax2.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
            ax2.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)

            sns.lineplot(x='frequency', y='psd_value', palette=['tab:orange'], data=daytime_df, ci='sd', ax=ax2,
                         label='Daytime')
            sns.lineplot(x='frequency', y='psd_value', palette=['tab:blue'], data=nighttime_df, ci='sd', ax=ax2,
                         label='Nighttime')

            ax2.set_xlabel(r'Frequency [Hz]', fontsize=24)
            ax2.set_ylabel(r'ASD $[(m/s^2)/\sqrt{Hz}]$ [dB]', fontsize=24)
            ax2.set_xlim([1 / 240, 50])
            ax2.set_ylim([-200, -60])
            ax2.set_xscale("log")
            ax2.tick_params(axis='both', which='major', labelsize=22)
            ax2.grid(True, linestyle='--', axis='both', which='both')
            ax2.legend(loc='best', shadow=True, fontsize='xx-large')
            fig2.tight_layout()

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    sns.set(font_scale=2)
    g = sns.heatmap(result, cbar=True, ax=ax, cbar_kws={'label': r'ASD $[(m/s^2)/\sqrt{Hz}]$ [dB]'},
                    cmap='mako', yticklabels=0, xticklabels=130, vmin=-170, vmax=-60)

    # ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2e}'))
    ax.set_ylabel(r'Frequency [Hz]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70, ha="right")
    g.set(xlabel=None)

    end, start = ax.get_ylim()
    stepsize = 1999
    # print(start, end)
    ax.yaxis.set_ticks(np.arange(start, end, stepsize))
    y_labels = f[::stepsize]
    for pos, val in enumerate(y_labels):
        if val < 1:
            y_labels[pos] = '%.2e' % val
        else:
            y_labels[pos] = '%.1f' % val
    # y_labels = ['%.2e' % n for n in y_labels]
    ax.set_yticklabels(y_labels)

    fig.tight_layout()

    if save_img:
        # fig1.savefig(r'D:\ET\Images\{0}\SOE0_LinePlot_{1}.SVG'.format(sensor, channel))
        fig.savefig(r'D:\ET\Images\HD\{0}\{0}{2}_HEAT-{3}_{4}_ASD_{1}_ACC.png'.format(sensor, channel, location,
                                                                                      startdate.strftime('%d-%b-%Y'),
                                                                                      stopdate.strftime('%d-%b-%Y')),
                    dpi=1200)
        fig.savefig(r'D:\ET\Images\{0}\{0}{2}_HEAT-{3}_{4}_ASD_{1}_ACC.png'.format(sensor, channel, location,
                                                                                   startdate.strftime('%d-%b-%Y'),
                                                                                   stopdate.strftime('%d-%b-%Y')),
                    dpi=300)
        if linearplot:
            fig1.savefig(r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC.png'.format(sensor, channel, location,
                                                                                           startdate.strftime(
                                                                                               '%d-%b-%Y'),
                                                                                           stopdate.strftime(
                                                                                               '%d-%b-%Y')), dpi=1200)
            fig1.savefig(r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC.pdf'.format(sensor, channel, location,
                                                                                           startdate.strftime(
                                                                                               '%d-%b-%Y'),
                                                                                           stopdate.strftime(
                                                                                               '%d-%b-%Y')), dpi=1200)
            fig2.savefig(
                r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log.png'.format(sensor, channel, location,
                                                                                      startdate.strftime('%d-%b-%Y'),
                                                                                      stopdate.strftime('%d-%b-%Y')),
                dpi=1200)
            fig2.savefig(
                r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log.pdf'.format(sensor, channel, location,
                                                                                      startdate.strftime('%d-%b-%Y'),
                                                                                      stopdate.strftime('%d-%b-%Y')),
                dpi=1200)

            fig1.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC.png'.format(sensor, channel, location,
                                                                                        startdate.strftime('%d-%b-%Y'),
                                                                                        stopdate.strftime('%d-%b-%Y')),
                         dpi=300)
            fig1.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC.pdf'.format(sensor, channel, location,
                                                                                        startdate.strftime('%d-%b-%Y'),
                                                                                        stopdate.strftime('%d-%b-%Y')),
                         dpi=300)
            fig2.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log.png'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')), dpi=300)
            fig2.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log.pdf'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')), dpi=300)
            # fig.savefig(r'D:\ET\Images\{0}\{0}_14Days_ASD_{1}.pdf'.format(sensor, channel), dpi=300)

    plt.show() if show_plot else ''


def rms(filexml, Data_path, network, sensor, location, channel, tstart, Twindow, verbose):
    # TODO: add descriptions and comments. This is an alpha version of the function but already working

    seed_id = network + '.' + sensor + '.' + location + '.' + channel

    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=verbose)
    filename_list = glob.glob(Data_path + seed_id + "*")
    filename_list.sort()

    print('THIS IS THE GAIN!!!!!', seed_id, gain)

    df = pd.DataFrame(columns=['timestamp', 'rms_value', 'Station_Name'])

    T = 600
    Ovl = 0.5
    TLong = 6 * 3600
    dT = TLong + T * Ovl
    M = int((dT - T) / (T * (1 - Ovl)) + 1)

    print('M: ', M)

    rms_val = np.array([])
    dates = np.array([])

    fsxml = 100
    Num = int(T * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    w = 2.0 * np.pi * f
    w1 = w ** 2 / respamp
    f = np.round(f, 2)

    fmin = 2  # 1 / 240  # Trillium model sensitivity
    fmax = 20  # fsxml / 2  # Nyquist's theorem

    imin = (np.abs(f - fmin)).argmin()
    imax = (np.abs(f - fmax)).argmin()

    for file in filename_list:
        print(file)
        k = 0
        st = read(file)
        # print(st.sort().__str__(extended=True))
        st = st.sort()
        Tstop = st[-1:][0].times('timestamp')[-1:][0]
        Tstop = UTCDateTime(Tstop)
        time = st[0].times('timestamp')[0]
        time = UTCDateTime(time)

        while time < Tstop:
            print('Evaluating from\t', time, '\tto\t', Tstop)
            tstart = time
            tstop = time + dT - 1 / fsxml
            st = read(file, starttime=tstart, endtime=tstop)
            t1 = time
            for n in range(0, M):

                tr = st.slice(t1, t1 + T - 1 / fsxml)
                if tr.get_gaps() == [] and len(tr) > 0:
                    tr1 = tr[0]
                    if tr1.stats.npts == Num:
                        s, _ = mlab.psd(tr1.data, NFFT=Num, Fs=fsxml, detrend="linear")
                        s = s[1:]
                        spec = s / gain
                        rms_val = np.append(rms_val, sum(spec[imin:imax] / T))

                        dates = np.append(dates, date2num(t1.datetime))

                t1 = t1 + T * Ovl
                k += 1
            time = time + TLong
        # rms_val = 10 * np.log10(rms_val)  # to dB
        if sensor == 'P2' or sensor == 'P3':
            station = np.array(int(filename_list[0].split('.')[2])).repeat(rms_val.size)
        else:
            station = np.array(int(filename_list[0].split('.')[1][3])).repeat(rms_val.size)

        data_to_save = np.concatenate((dates, rms_val, station))
        data_to_save = np.reshape(data_to_save, (3, np.size(rms_val))).T
        df = df.append(pd.DataFrame(data_to_save, columns=df.columns), ignore_index=True)
    print(df.head())
    print(df.info())
    return df


def rms_comparison(filexml, Data_path1, Data_path2, network, sensor1, sensor2, location, channel, tstart, Twindow,
                   verbose, ratio=True, save_img=False, hline=False):
    # TODO: add descriptions and comments. This is an alpha version of the function but already working
    df = rms(filexml, Data_path1, network, sensor1, location, channel, tstart, Twindow, verbose)
    if sensor1 == 'P2' or sensor1 == 'P3' and location == '00':
        df1 = rms(filexml, Data_path2, network, sensor2, '01', channel, tstart, Twindow, verbose)
    elif sensor1 == 'P2' or sensor1 == 'P3' and location == '01':
        df1 = rms(filexml, Data_path2, network, sensor2, '00', channel, tstart, Twindow, verbose)
    else:
        df1 = rms(filexml, Data_path2, network, sensor2, location, channel, tstart, Twindow, verbose)

    df['ratio'] = df['rms_value'] / df1['rms_value']
    df['Quantity'] = 'Ratio'

    print('Mean ratio =', df['ratio'].mean(), '+/-', df['ratio'].std())
    nighttime_df = df[pd.to_datetime(num2date(df['timestamp'])).hour <= 5]
    daytime_df = df[pd.to_datetime(num2date(df['timestamp'])).hour > 5]

    print('Mean nighttime ratio =', nighttime_df['ratio'].mean(), '+/-', nighttime_df['ratio'].std())
    print('Mean daytime ratio =', daytime_df['ratio'].mean(), '+/-', daytime_df['ratio'].std())

    df = df.append(df1, ignore_index=True)

    def soe_col_rename(x):
        return 'SOE ' + str(int(x))

    def boreholes_col_rename(x):
        name = ''
        if int(x) == 0:
            name = 'Surface'
        elif int(x) == 1:
            name = 'Underground'
        return name

    if sensor1 == 'P2' or sensor2 == 'P2':
        df['Station_Name'] = df['Station_Name'].apply(boreholes_col_rename)
    elif sensor1 == 'P3' or sensor2 == 'P3':
        df['Station_Name'] = df['Station_Name'].apply(boreholes_col_rename)
    else:
        df['Station_Name'] = df['Station_Name'].apply(soe_col_rename)

    if ratio:
        fig = plt.figure(figsize=(19.2, 10.8))
        gs = fig.add_gridspec(2, hspace=0.15, width_ratios=[1], height_ratios=[3, 1.5])
        ax = gs.subplots(sharex=True)

        sns.lineplot(x="timestamp", y="rms_value", hue='Station_Name', palette=['tab:blue', 'tab:red'], ci='sd',
                     data=df,
                     ax=ax[0])
        g = sns.lineplot(x="timestamp", y='ratio', hue='Quantity', palette=['tab:green'], data=df, ax=ax[1])

        ax[0].set_ylabel(r'Integral Under the PSD', fontsize=24)
        ax[0].tick_params(axis='both', which='major', labelsize=22)
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].set_ylim([df['rms_value'].min() - df['rms_value'].min() / 2, df['rms_value'].max() + 5])
        ax[1].set_ylim([0, df['ratio'].max() + 5])
        ax[1].axhline(y=1, color='r', linestyle='dotted', linewidth=2) if hline else ''

        ax[1].set_ylabel(r'{0} surf / {1} under'.format(sensor1, sensor2), fontsize=24)
        ax[1].tick_params(axis='both', which='major', labelsize=22)
        ax[1].xaxis.set_major_locator(mdates.DayLocator())
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%a - %d %b'))
        g.set(xlabel=None)

        ax[0].grid(True, linestyle='--', axis='both', which='both')
        ax[1].grid(True, linestyle='--', axis='both', which='both')
        ax[0].legend(loc='best', shadow=True, fontsize='xx-large')
        ax[1].legend(loc='best', shadow=True, fontsize='xx-large')
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=70)
        gs.tight_layout(fig)
    else:
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot()

        g = sns.lineplot(x="timestamp", y="rms_value", hue='Station_Name', palette=['tab:blue', 'tab:red'], ci='sd',
                         data=df,
                         ax=ax)
        g.set(xlabel=None)
        ax.set_ylabel(r'Integral Under the PSD', fontsize=24)
        ax.set_yscale("log")
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.legend(loc='best', shadow=True, fontsize='xx-large')
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a - %d %b'))
        ax.grid(True, linestyle='--', axis='both', which='both')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        fig.tight_layout()

    if save_img:
        fig.savefig(r'D:\ET\Images\HD\Comparison\{0}-{1}_{2}_Days.png'.format(sensor1, sensor2, channel), dpi=1200)
        fig.savefig(r'D:\ET\Images\HD\Comparison\{0}-{1}_{2}_Days.pdf'.format(sensor1, sensor2, channel), dpi=1200)
        fig.savefig(r'D:\ET\Images\Comparison\{0}-{1}_{2}_Days.png'.format(sensor1, sensor2, channel), dpi=300)
        fig.savefig(r'D:\ET\Images\Comparison\{0}-{1}_{2}_Days.pdf'.format(sensor1, sensor2, channel), dpi=300)

    plt.show()


def asd_from_csv(path_to_csv):
    df = pd.read_csv(path_to_csv)
    # fl, nlnm, fh, nhnm = NLNM(1)
    # w_fl = 2.0 * np.pi * fl
    # w_fh = 2.0 * np.pi * fh
    # nlnm = nlnm * w_fl ** 2
    # nhnm = nhnm * w_fh ** 2

    # print(fl)
    # print(fh)
    # asd_nlnm = np.sqrt(10 ** (get_nlnm()[1] / 10))
    # asd_nhnm = np.sqrt(10 ** (get_nhnm()[1] / 10))
    # asd_nlnm_db = 10 * np.log10(asd_nlnm)
    # asd_nhnm_db = 10 * np.log10(asd_nhnm)
    # print(asd_nlnm_db)
    # print(asd_nhnm_db)

    print(df.info())
    print(df.head())
    # df['frequency'] = df['frequency']
    # df['frequency'] = df['frequency'].map(lambda x: '{:.2e}'.format(x))
    # df = df.set_index('frequency')
    # print(df.head())
    # print(df.tail())
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()
    label = pd.to_datetime(df.iloc[:, 42].name, unit='ns').strftime('%d %b %H:%M:%S')
    # datetime.datetime.strptime(df.iloc[:, 1].name, '%Y-%m-%d %H:%M:%S').strftime('%d %b %H:%M')
    # psd_vals = 10 ** (df.iloc[:, 42] / 5)
    # psd_vals_db = 10 * np.log10(psd_vals)
    ax.plot(df['frequency'], df.iloc[:, 42], label=label)
    ax.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')  # , label="NLNM")
    ax.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')  # , label="NHNM")

    ax.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
    ax.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)

    ax.set_xlabel(r'Frequency [Hz]', fontsize=24)
    ax.set_ylabel(r'ASD $[(m/s^2)/\sqrt{Hz}]$ [dB]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, linestyle='--', axis='both', which='both')
    ax.legend(loc='best', shadow=True, fontsize='xx-large')
    ax.set_xscale("log")
    ax.set_xlim([1 / 240, 50])
    ax.set_ylim([-200, -60])
    ax.axvline(x=2, color='r', linestyle='dotted', linewidth=2)
    ax.axvline(x=20, color='r', linestyle='dotted', linewidth=2)
    ax.text(2, -55, '2 Hz', fontsize=20, color='r', ha='center', va='center',
            bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'r', 'boxstyle': 'round'})
    ax.text(20, -55, '20 Hz', fontsize=20, color='r', ha='center', va='center',
            bbox={'facecolor': 'white', 'alpha': 1, 'edgecolor': 'r', 'boxstyle': 'round'})

    # sns.heatmap(df, cbar=True, cbar_kws={'label': r'ASD $[(m/s)/\sqrt{Hz}]$ [dB]'}, cmap='mako', yticklabels=500)
    fig.tight_layout()
    plt.show()


def comparison_from_csv(path_to_csv1, path_to_csv2):
    df1 = pd.read_csv(path_to_csv1)
    df2 = pd.read_csv(path_to_csv2)

    df1['mean'] = df1.mean(axis=1)
    df2['mean'] = df2.mean(axis=1)

    mean_val = np.mean(df1[(df1['frequency'] >= 2) & (df1['frequency'] <= 20)]['mean'] /
                       df2[(df2['frequency'] >= 2) & (df2['frequency'] <= 20)]['mean'])
    std_val = np.std(df1[(df1['frequency'] >= 2) & (df1['frequency'] <= 20)]['mean'] /
                     df2[(df2['frequency'] >= 2) & (df2['frequency'] <= 20)]['mean'])

    mean_val = 10 ** (mean_val / 10)
    std_val = 10 ** (std_val / 10)
    print('Result:', mean_val, '+/-', std_val)


def heatmap_from_csv(path_to_file=r'D:\ET\SOE0-HHZ_20210326-20210410_ACC_3600.csv', path_to_csvs=None, multi_csv=False,
                     save_img=False):
    if not multi_csv:
        df = pd.read_csv(path_to_file)
        freq_indeces = df['frequency']
    else:
        import os
        all_files = glob.glob(os.path.join(path_to_csvs, "*"))
        df_list = []
        for filename in all_files:
            df_prov = pd.read_csv(filename, index_col=None, header=0)
            df_list.append(df_prov)
            print(filename)
            freq_indeces = df_prov['frequency']
        # print(df_list)
        df = pd.concat(df_list, axis=1)
    df.drop(columns='frequency', inplace=True)
    df = df.set_index(freq_indeces)
    print(df)
    print(df.info())

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot()

    sns.heatmap(df, cbar=True, ax=ax, cbar_kws={'label': r'ASD $[(m/s^2)/\sqrt{Hz}]$ [dB]'},
                cmap='mako', vmin=-170, vmax=-60)

    if save_img:
        fig.savefig(
            r'D:\ET\Images\HD\Heatmap_{0}-{1}_{2}.png'.format(df.columns[0][:10], df.columns[-1:][0][:10], 'ACC_600'),
            dpi=1200)
        fig.savefig(
            r'D:\ET\Images\Heatmap_{0}-{1}_{2}.png'.format(df.columns[0][:10], df.columns[-1:][0][:10], 'ACC_600'),
            dpi=300)
        # fig.savefig(r'D:\ET\Images\{0}\{0}_14Days_ASD_{1}.pdf'.format(sensor, channel), dpi=300)
    # plt.show() # MemoryError plot


def csv_creators(filexml, Data_path, network, sensor, location, channel, tstart, T, Ovl, verbose):
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = read_Inv(filexml, network, sensor, location, channel, tstart, T, verbose=verbose)
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

    print('The analysis start from\t', startdate, '\tto\t', stopdate)

    # df = pd.DataFrame(columns=['frequency', 'timestamp', 'psd_value'], dtype=float)

    TLong = 6 * 3600
    dT = TLong + T * Ovl
    M = int((dT - T) / (T * (1 - Ovl)) + 1)

    K = int((stopdate - startdate) / (T * (1 - Ovl)) + 1)

    print('K: ', K)
    print('M: ', M)

    v = np.empty(K)

    fsxml = 100
    Num = int(T * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    print(f.min())
    print(f.size)

    f = np.round(f, 3)

    w = 2.0 * np.pi * f

    w1 = w ** 2 / respamp

    import timeit

    start = timeit.default_timer()
    for file in filename_list:
        df = pd.DataFrame(columns=['frequency', 'timestamp', 'psd_value'], dtype=float)
        k = 0
        print(file)
        st = read(file)
        st = st.sort()

        Tstop = st[-1:][0].times('timestamp')[-1:][0]
        Tstop = UTCDateTime(Tstop)

        time = st[0].times('timestamp')[0]
        time = UTCDateTime(time)
        Tstart = time

        while time < Tstop:
            print('Evaluating from\t', time, '\tto\t', Tstop)
            tstart = time
            tstop = time + dT - 1 / fsxml
            st = read(file, starttime=tstart, endtime=tstop)

            t1 = time
            for n in range(0, M):
                v[k] = np.nan
                tr = st.slice(t1, t1 + T - 1 / fsxml)
                if tr.get_gaps() == [] and len(tr) > 0:
                    tr1 = tr[0]
                    if tr1.stats.npts == Num:
                        s, _ = mlab.psd(tr1.data, NFFT=Num, Fs=fsxml, detrend="linear")
                        s = s[1:]
                        # spec = s / gain
                        v[k] = 0

                # d[k] = date2num(t1.datetime)
                # print('t1 =', t1)
                time_measure = np.repeat(t1.timestamp, np.size(f))

                if np.isnan(v[k]):
                    psd_values = np.tile(v[k], np.size(f))
                else:
                    psd_values = s * w1  # w1 --> to have acceleration

                data_to_save = np.concatenate((f, time_measure, psd_values))  # , w1))
                data_to_save = np.reshape(data_to_save, (3, np.size(f))).T
                df = df.append(pd.DataFrame(data_to_save, columns=df.columns), ignore_index=True)
                t1 = t1 + T * Ovl
                k += 1
            time = time + TLong
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # , format='%d-%b %H:%M')
        df['psd_value'] = 10 * np.log10(np.sqrt(df['psd_value'] / gain))
        # print(df.head())
        # print(df.info())
        df['frequency'] = df['frequency'].astype(float)
        df['psd_value'] = df['psd_value'].astype(float)
        result = df.pivot_table(index='frequency', columns='timestamp', values='psd_value')
        result.to_csv(
            r'D:\ET\2021\Heatmap\csv_files\{5}\{0}{1}-{2}_{3}-{4}_ACC_{5}.csv'.format(sensor, location, channel,
                                                                                      Tstart.strftime('%Y%m%d'),
                                                                                      Tstop.strftime('%Y%m%d'),
                                                                                      int(T)))
    stop = timeit.default_timer()
    print('Elapsed time:', (stop - start), 's')


def quantile_plot(filexml, Data_path, network, sensor, location, channel, tstart, Twindow, Overlap, verbose,
                save_img=False, xscale='linear', show_plot=True, save_txt=False, q1=0.1, q2=0.5, q3=0.9):
    r"""
    It performs the spectrogram of given data. The spectrogram is a two-dimensional plot with on the y-axis the
    frequencies, on the x-axis the dates and on virtual z-axis the ASD value expressed
    in :math:`ms^{-2}/\sqrt{Hz}\:[dB]`.

    :type filexml: str
    :param filexml: The .xml needed to read the seismometer response

    :type Data_path: str
    :param Data_path: Path to the data.

    :type network: str
    :param network: Sensor network

    :type sensor: str
    :param sensor: Name of the sensor

    :type location: str
    :param location: Location of the sensor

    :type channel: str
    :param channel: Channel to be analysed

    :type tstart: str, :class: 'obspy.UTCDateTime'
    :param tstart: Start time to get the response from the seismometer (?)

    :type Twindow: float
    :param Twindow: Time windows used to evaluate the PSD.

    :type Overlap: float
    :param Overlap: The overlap expressed as a number, i.e. 0.5 = 50%. The data read is translated of a given quantity
        which depends on this parameter. For example 10' of data are trasnlated by DEFAULT of 10', but with Overlap=0.5,
        the data will be translated by 5'. Therefore, it will achieve 5' of Overlap.

    :type verbose: bool
    :param verbose: Needed for verobsity

    :type save_csv: bool
    :param save_csv: If you want to save the data analyzed ina .csv format

    :type save_img: bool
    :param save_img: If you want to save the images produce. Please note that it is highly recommended setting the value
        on True if more than 5 days of data are considered.

    :type linearplot: bool
    :param linearplot: If you want to create the linear plot of the data with the distinction between daytime and
        nighttime. This type of plot shows the mean with one sigma of confidence interval

    :type xscale: str
    :param xscale: It represents the scale of the lineplot produced. It can be one of 'linear', 'log' or 'both'. Please
        note that setting the variable on 'both' it will produce both linear and logarithmic x-scale plots.

    :type show_plot: bool
    :param show_plot: If you want to show the plot produced. Please note that the spectrogram requires lot of memory to
        be shown especially if the analysis is done on more than 5 days.
    """
    # TODO: check tstart variable, may remove it. Seems useless.

    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=verbose)
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

    print('The analysis start from\t', startdate, '\tto\t', stopdate)

    df = pd.DataFrame(columns=['frequency', 'timestamp', 'psd_value'], dtype=float)

    T = Twindow
    Ovl = Overlap
    TLong = 6 * 3600
    dT = TLong + T * Ovl
    M = int((dT - T) / (T * (1 - Ovl)) + 1)

    K = int((stopdate - startdate) / (T * (1 - Ovl)) + 1)

    print('K: ', K)
    print('M: ', M)

    v = np.empty(K)

    fsxml = 100
    Num = int(T * fsxml)

    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fsxml)
    f = f[1:]
    print('MIN. FREQ.:\t', f.min())
    # print(f.size)
    print('GAIN:\t', gain)
    # f_str = ["%.2e" % n for n in f]
    f = np.round(f, 3)
    # fmin = 2
    # fmax = 20
    w = 2.0 * np.pi * f
    # print(np.where(w == 0))
    w1 = w ** 2 / respamp
    # imin = (np.abs(f - fmin)).argmin()
    # imax = (np.abs(f - fmax)).argmin()

    import timeit

    start = timeit.default_timer()
    for file in filename_list:
        k = 0
        print(file)

        st = read(file)
        # print(st.sort().__str__(extended=True))
        st = st.sort()
        Tstop = st[-1:][0].times('timestamp')[-1:][0]
        Tstop = UTCDateTime(Tstop)
        time = st[0].times('timestamp')[0]
        time = UTCDateTime(time)
        # Tstop = st[-1:][0].times('utcdatetime')[-1:][0]
        # time = st[0].times('utcdatetime')[0]
        # startdate = UTCDateTime('{0}-{1}T12:23:34.5'.format(year, day))
        # time = startdate
        while time < Tstop:
            print('Evaluating from\t', time, '\tto\t', Tstop)
            tstart = time
            tstop = time + dT - 1 / fsxml
            st = read(file, starttime=tstart, endtime=tstop)

            t1 = time
            for n in range(0, M):
                v[k] = np.nan
                tr = st.slice(t1, t1 + T - 1 / fsxml)
                if tr.get_gaps() == [] and len(tr) > 0:
                    tr1 = tr[0]
                    if tr1.stats.npts == Num:
                        s, _ = mlab.psd(tr1.data, NFFT=Num, Fs=fsxml, detrend="linear")
                        s = s[1:]
                        # spec = s / gain
                        v[k] = 0

                # d[k] = date2num(t1.datetime)
                # print('t1 =', t1)
                time_measure = np.repeat(t1.timestamp, np.size(f))

                if np.isnan(v[k]):
                    psd_values = np.tile(v[k], np.size(f))
                else:
                    psd_values = s * w1  # w1 --> to have acceleration

                data_to_save = np.concatenate((f, time_measure, psd_values))  # , w1))
                data_to_save = np.reshape(data_to_save, (3, np.size(f))).T
                df = df.append(pd.DataFrame(data_to_save, columns=df.columns), ignore_index=True)
                t1 = t1 + T * Ovl
                k += 1
            time = time + TLong
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # , format='%d-%b %H:%M')
    print(df.head())
    df['psd_value'] = 10 * np.log10(df['psd_value'])
    print(df.head())
    print(df.info())
    df['frequency'] = df['frequency'].astype(float)
    df['psd_value'] = df['psd_value'].astype(float)

    stop = timeit.default_timer()
    print('Elapsed time before plot:', (stop - start), 's')

    daytime_df = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 20)]
    nighttime_df = df[(df['timestamp'].dt.hour <= 5) | (df['timestamp'].dt.hour >= 21)]

    print('xscale =', xscale)

    # Find the data correspond to the quantiles
    # DAY
    estim_values_day = daytime_df.groupby(['frequency'])['psd_value'].quantile(q1)
    first_quantile_day = daytime_df.groupby(['frequency'])['psd_value'].quantile(q2)
    second_quantile_day = daytime_df.groupby(['frequency'])['psd_value'].quantile(q3)

    # NIGHT
    estim_values_night = nighttime_df.groupby(['frequency'])['psd_value'].quantile(q1)
    first_quantile_night = nighttime_df.groupby(['frequency'])['psd_value'].quantile(q2)
    second_quantile_night = nighttime_df.groupby(['frequency'])['psd_value'].quantile(q3)

    # Frequencies are the indeces of the DataFrame because of groupby()
    frequency_array = estim_values_day.index.to_numpy()

    # Convert the DataFrame in numpy array
    # DAY
    estim_values_day_array = estim_values_day.to_numpy()
    first_quantile_day_array = first_quantile_day.to_numpy()
    second_quantile_day_array = second_quantile_day.to_numpy()

    # NIGHT
    estim_values_night_array = estim_values_night.to_numpy()
    first_quantile_night_array = first_quantile_night.to_numpy()
    second_quantile_night_array = second_quantile_night.to_numpy()

    # TODO: add saving curve data
    # if save_txt:
    #     np.savetxt()

    if xscale == 'linear' or xscale == 'both':
        fig1 = plt.figure(figsize=(19.2, 10.8))
        fig1_1 = plt.figure(figsize=(19.2, 10.8))
        ax1 = fig1.add_subplot()
        ax1_1 = fig1_1.add_subplot()

        ax1.plot(frequency_array, estim_values_day_array, linewidth=2, label='Daytime (10%)')
        ax1.plot(frequency_array, first_quantile_day_array, linewidth=2, label='Daytime (50%)')
        ax1.plot(frequency_array, second_quantile_day_array, linewidth=2, label='Daytime (90%)')

        ax1_1.plot(frequency_array, estim_values_night_array, linewidth=2, label='Nighttime (10%)')
        ax1_1.plot(frequency_array, first_quantile_night_array, linewidth=2, label='Nighttime (50%)')
        ax1_1.plot(frequency_array, second_quantile_night_array, linewidth=2, label='Nighttime (90%)')

        ax1.set_xlabel(r'Frequency [Hz]', fontsize=24)
        ax1.set_ylabel(r'PSD $[(m^2/s^4)/Hz]$ [dB]', fontsize=24)
        ax1.set_xlim([1 / 240, 20])
        ax1.set_ylim([-210, -70])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True, linestyle='--', axis='both', which='both')
        ax1.legend(loc='best', shadow=True, fontsize=24)
        fig1.tight_layout()

        ax1_1.set_xlabel(r'Frequency [Hz]', fontsize=24)
        ax1_1.set_ylabel(r'PSD $[(m^2/s^4)/Hz]$ [dB]', fontsize=24)
        ax1_1.set_xlim([1 / 240, 20])
        ax1_1.set_ylim([-210, -70])
        ax1_1.tick_params(axis='both', which='major', labelsize=22)
        ax1_1.grid(True, linestyle='--', axis='both', which='both')
        ax1_1.legend(loc='best', shadow=True, fontsize=24)
        fig1_1.tight_layout()

    if xscale == 'log' or xscale == 'both':
        fig2 = plt.figure(figsize=(19.2, 10.8))
        fig2_2 = plt.figure(figsize=(19.2, 10.8))
        ax2 = fig2.add_subplot()
        ax2_2 = fig2_2.add_subplot()

        ax2.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')
        ax2.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')
        ax2.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
        ax2.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)

        ax2_2.plot(1 / get_nlnm()[0], get_nlnm()[1], 'k--')
        ax2_2.plot(1 / get_nhnm()[0], get_nhnm()[1], 'k--')
        ax2_2.annotate('NHNM', xy=(1.25, -112), ha='center', fontsize=20)
        ax2_2.annotate('NLNM', xy=(1.25, -176), ha='center', fontsize=20)

        ax2.plot(frequency_array, estim_values_day_array, linewidth=2, label='Daytime (10%)')
        ax2.plot(frequency_array, first_quantile_day_array, linewidth=2, label='Daytime (50%)')
        ax2.plot(frequency_array, second_quantile_day_array, linewidth=2, label='Daytime (90%)')

        ax2_2.plot(frequency_array, estim_values_night_array, linewidth=2, label='Nighttime (10%)')
        ax2_2.plot(frequency_array, first_quantile_night_array, linewidth=2, label='Nighttime (50%)')
        ax2_2.plot(frequency_array, second_quantile_night_array, linewidth=2, label='Nighttime (90%)')

        ax2.set_xlabel(r'Frequency [Hz]', fontsize=24)
        ax2.set_ylabel(r'PSD $[(m^2/s^4)/Hz]$ [dB]', fontsize=24)
        ax2.set_xlim([1 / 240, 20])
        ax2.set_ylim([-210, -70])
        ax2.set_xscale("log")
        ax2.tick_params(axis='both', which='major', labelsize=22)
        ax2.grid(True, linestyle='--', axis='both', which='both')
        ax2.legend(loc='best', shadow=True, fontsize='xx-large')
        fig2.tight_layout()

        ax2_2.set_xlabel(r'Frequency [Hz]', fontsize=24)
        ax2_2.set_ylabel(r'PSD $[(m^2/s^4)/Hz]$ [dB]', fontsize=24)
        ax2_2.set_xlim([1 / 240, 20])
        ax2_2.set_ylim([-210, -70])
        ax2_2.set_xscale("log")
        ax2_2.tick_params(axis='both', which='major', labelsize=22)
        ax2_2.grid(True, linestyle='--', axis='both', which='both')
        ax2_2.legend(loc='best', shadow=True, fontsize='xx-large')
        fig2_2.tight_layout()

    if save_img:
        print('Saving images...')
        fig1.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_DAYTIME.png'.format(sensor, channel, location,
                                                                                      startdate.strftime(
                                                                                          '%d-%b-%Y'),
                                                                                      stopdate.strftime(
                                                                                          '%d-%b-%Y')), dpi=1200)
        fig1.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_DAYTIME.pdf'.format(sensor, channel, location,
                                                                                      startdate.strftime(
                                                                                          '%d-%b-%Y'),
                                                                                      stopdate.strftime(
                                                                                          '%d-%b-%Y')), dpi=1200)
        fig1_1.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_NIGHTTIME.png'.format(sensor, channel, location,
                                                                                        startdate.strftime(
                                                                                            '%d-%b-%Y'),
                                                                                        stopdate.strftime(
                                                                                            '%d-%b-%Y')), dpi=1200)
        fig1_1.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_NIGHTTIME.pdf'.format(sensor, channel, location,
                                                                                        startdate.strftime(
                                                                                            '%d-%b-%Y'),
                                                                                        stopdate.strftime(
                                                                                            '%d-%b-%Y')), dpi=1200)
        fig2.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log_DAYTIME.png'.format(sensor, channel, location,
                                                                                          startdate.strftime(
                                                                                              '%d-%b-%Y'),
                                                                                          stopdate.strftime(
                                                                                              '%d-%b-%Y')),
            dpi=1200)
        fig2.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log_DAYTIME.pdf'.format(sensor, channel, location,
                                                                                          startdate.strftime(
                                                                                              '%d-%b-%Y'),
                                                                                          stopdate.strftime(
                                                                                              '%d-%b-%Y')),
            dpi=1200)
        fig2_2.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log_NIGHTTIME.png'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')),
            dpi=1200)
        fig2_2.savefig(
            r'D:\ET\Images\HD\{0}\{0}{2}_LinePlot-{3}_{4}_{1}_ACC_log_NIGHTTIME.pdf'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')),
            dpi=1200)

        fig1.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_DAYTIME.png'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')),
                     dpi=300)
        fig1.savefig(r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_DAYTIME.pdf'.format(sensor, channel, location,
                                                                                            startdate.strftime(
                                                                                                '%d-%b-%Y'),
                                                                                            stopdate.strftime(
                                                                                                '%d-%b-%Y')),
                     dpi=300)
        fig1_1.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_NIGHTTIME.png'.format(sensor, channel, location,
                                                                                     startdate.strftime('%d-%b-%Y'),
                                                                                     stopdate.strftime('%d-%b-%Y')),
            dpi=300)
        fig1_1.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_NIGHTTIME.pdf'.format(sensor, channel, location,
                                                                                     startdate.strftime('%d-%b-%Y'),
                                                                                     stopdate.strftime('%d-%b-%Y')),
            dpi=300)
        fig2.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log_DAYTIME.png'.format(sensor, channel, location,
                                                                                       startdate.strftime(
                                                                                           '%d-%b-%Y'),
                                                                                       stopdate.strftime(
                                                                                           '%d-%b-%Y')), dpi=300)
        fig2.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log_DAYTIME.pdf'.format(sensor, channel, location,
                                                                                       startdate.strftime(
                                                                                           '%d-%b-%Y'),
                                                                                       stopdate.strftime(
                                                                                           '%d-%b-%Y')), dpi=300)
        fig2_2.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log_NIGHTTIME.png'.format(sensor, channel, location,
                                                                                         startdate.strftime(
                                                                                             '%d-%b-%Y'),
                                                                                         stopdate.strftime(
                                                                                             '%d-%b-%Y')), dpi=300)
        fig2_2.savefig(
            r'D:\ET\Images\{0}\{0}{2}_LinePlot_{1}-{3}_{4}_ACC_log_NIGHTTIME.pdf'.format(sensor, channel, location,
                                                                                         startdate.strftime(
                                                                                             '%d-%b-%Y'),
                                                                                         stopdate.strftime(
                                                                                             '%d-%b-%Y')), dpi=300)
        print('Images correctly saved')
    plt.show() if show_plot else ''
