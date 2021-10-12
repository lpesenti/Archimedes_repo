import numpy as np
from obspy import read, read_inventory, Stream
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
import glob
from matplotlib import mlab
import matplotlib.pyplot as plt
import time
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
    # select sensor and channel
    invxmls = invxml.select(network=network, station=sensor, channel=channel)
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    resp = invxmls.get_response(seed_id, t)
    gain = invxmls.get_response(seed_id, t).instrument_sensitivity.value
    if verbose:
        invxmls.plot_response(1 / 120, label_epoch_dates=True)
        print(resp)
        print(gain)
    # return a nested dictionary detailing the sampling rates of each response stage
    sa = resp.get_sampling_rates()
    # take the last output_sampling_rate
    fsxml = sa[len(sa)]['output_sampling_rate']
    if verbose:
        print('Sampling rate:', fsxml)
    Num = int(Twindow * fsxml)
    # Returns frequency response and corresponding frequencies using evalresp
    #                                        time_res,       Npoints,  output
    sresp, fxml = resp.get_evalresp_response(t_samp=1 / fsxml, nfft=Num, output="VEL")
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

    return st_tot


def ppsd(stream, filexml, sensor, Twindow, Overlap):
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
                      overlap=int(Overlap / 100 * Num))
    for itrace in range(len(stream)):
        seism_ppsd.add(stream.select(station=sensor)[itrace])

    seism_ppsd.plot(cmap=pqlx, xaxis_frequency=True, period_lim=(1 / 120, fs/2))


def psd_rms_finder(stream, filexml, network, sensor, location, channel, tstart, Twindow, Overlap, mean_number,
                   verbose):  # , ax, ax1):
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
    data = np.array([])
    vec_rms = np.array([])
    for itrace in range(len(stream)):
        data = np.append(data, np.array(stream[itrace]) / gain)
    fs = stream[0].stats.sampling_rate
    Num = int(Twindow * fs)
    data_split = np.array_split(data, len(data) / Num)
    _, f_s = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs, noverlap=int(Overlap / 100 * Num))
    f_s = f_s[1:]
    start = np.where(f_s == 1)[0][0]
    stop = np.where(f_s == 10)[0][0]
    integral_min = np.inf

    data = np.array([])
    for index, chunk in enumerate(data_split):
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
                                    time.gmtime((t - datetime.datetime(1970, 1, 1)).total_seconds() + index * Twindow)))
    best_psd, f_best = mlab.psd(data, NFFT=int(Num / mean_number), Fs=fs, detrend="linear", noverlap=int(Overlap / 100 * Num/mean_number))
    f_best = f_best[1:]
    best_psd = best_psd[1:]
    fl, nlnm, fh, nhnm = NLNM(2)
    fig0 = plt.figure()
    fig1 = plt.figure()
    ax0 = fig0.add_subplot()
    ax1 = fig1.add_subplot()
    ax0.tick_params(axis='both', which='both', labelsize=15)
    ax1.tick_params(axis='both', which='both', labelsize=15)
    ax0.plot(f_best, np.sqrt(best_psd), linestyle='-', color='tab:blue', label='Best PSD')
    ax0.plot(fl, nlnm, 'k--', label="Noise Model")
    ax0.plot(fh, nhnm, 'k--')
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim([0.01, fs/2])
    ax0.set_ylim([1e-11, 1e-5])
    ax0.grid(True, linestyle='--')
    ax0.legend(loc='best', shadow=True, fontsize='medium')
    ax0.set_xlabel('Frequency [Hz]', fontsize=20)
    ax0.set_ylabel(r'Seismic [(m/s)/$\sqrt{Hz}$]', fontsize=20)
    ax1.hist(vec_rms, bins=200)
    ax1.grid(True, linestyle='--')
    ax1.set_yscale("log")
    ax1.set_xlabel(r'Integral [$(m/s)^2$]', fontsize=20)
    ax1.set_ylabel(r'Counts', fontsize=20)
    plt.show()
