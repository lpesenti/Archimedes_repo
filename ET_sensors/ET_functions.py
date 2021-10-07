import numpy as np
from obspy import read, read_inventory, Stream
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
import glob
from matplotlib import mlab
import matplotlib.pyplot as plt



def NLNM(unit):
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


def Read_Inv(filexml, network, sensor, location, channel, t, Twindow, verbose):
    """
    Read Inventory (xml file) of the sensor

    Parameters
    ----------
    filexml : str
	path and name of the xml file to be read
    ch : str
	sensor's channel to be read
    sensor : str
	sensor's name
    t : UTCDateTime
	time of the analysis
    verbose : bool
	If True the verbosity is enabled.

    Notes
    -----
    response output in VEL

    Returns
    -------
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
        invxmls.plot_response(1/120, label_epoch_dates=True)
        print(resp)
        print(gain)
    # return a nested dictionary detailing the sampling rates of each response stage
    sa = resp.get_sampling_rates()
    # take the last output_sampling_rate
    fsxml = sa[len(sa)]['output_sampling_rate']
    if verbose:
        print('fs xml', fsxml)
    Num = int(Twindow * fsxml)
    # Returns frequency response and corresponding frequencies using evalresp
    #                                        time_res,       Npoints,  output
    sresp, fxml = resp.get_evalresp_response(t_samp=1 / fsxml, nfft=Num, output="VEL")
    fxml = fxml[1:]  # remove first value that is 0
    sresp = sresp[1:]  # remove first value that is 0
    # amplitude -> absolute frequency value
    respamp = np.absolute(sresp * np.conjugate(sresp))
    if verbose:
        print('len respamp', len(respamp))

    return fxml, respamp, fsxml, gain


def Evaluate_PSD(filexml, Data_path, network, sensor, location, channel, tstart, tstop, Twindow, verbose):
    """
    Read sensor data and evaluate PSD

    Parameters
    ----------
    filexml : str
        path and name of the xml file to be read
    ch : str
        sensor's channel to be read
    sensor : str
        sensor's name
    tstart : any
        start time of the analysis
    tstop : any
        stop time of the analysis
    Twindow : float
	time window for the PSD
    Overlap : float
	overlap in percentage for the PSD windows
    verbose : bool
        If True the verbosity is enabled.

    Notes
    -----
    Overlap should be lower than 50 %

    Returns
    -------
        A tuple for frequency and sensor's PSD
    """

    # Time interval
    yi = str(tstart.year)
    mi = str(tstart.month)
    di = str(tstart.day)
    hi = str(tstart.hour)
    mii = str(tstart.minute)
    si = str(tstart.second)
    doyi = tstart.strftime('%j')
    yf = str(tstop.year)
    mf = str(tstop.month)
    doyf = tstop.strftime('%j')
    if verbose:
        print("Analysis from: ", yi, mi, doyi, " to: ", yf, mf, doyf)
    seed_id = network + '.' + sensor + '.' + location + '.' + channel
    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml, gain = Read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=verbose)
    print("res", len(respamp))
    # read filename
    filename_list = glob.glob(Data_path + seed_id+"*")
    print(filename_list)
    st_tot = Stream()
    for file in filename_list:
        st = read(file)#, starttime=tstart, endtime=tf)
        print(st)
        st_tot += st
    if verbose:
        print(st_tot)

    return st_tot


def ppsd(stream, filexml, sensor, Twindow, Overlap):
    invxml = read_inventory(filexml)
    seism_ppsd = PPSD(stream.select(station=sensor)[0].stats, invxml, ppsd_length=Twindow, overlap=Overlap)
    for itrace in range(len(stream)):
        seism_ppsd.add(stream.select(station=sensor)[itrace])

    seism_ppsd.plot(cmap=pqlx, xaxis_frequency=True, period_lim=(1 / 120, 50))

def psd_rms_finder(stream, filexml, network, sensor, location, channel, tstart, Twindow, Overlap):
    _, _, _, gain = Read_Inv(filexml, network, sensor, location, channel, tstart, Twindow, verbose=False)
    data = np.array([])
    vec_rms = np.array([])
    for itrace in range(len(stream)):
        data = np.append(data, np.array(stream[itrace])/gain)
    fs = stream[0].stats.sampling_rate
    Num = int(Twindow*fs)
    data_split = np.array_split(data, len(data)/Num)
    _, f_s = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs, noverlap=Overlap)
    f_s = f_s[1:]
    print(f_s)
    start = np.where(f_s == 1)[0][0]
    stop = np.where(f_s == 10)[0][0]
    for index, chunk in enumerate(data_split):
        chunk_s, _ = mlab.psd(chunk, NFFT=Num, Fs=fs, detrend="linear", noverlap=Overlap)
        chunk_s = chunk_s[1:]
        integral = sum(chunk_s[start:stop] / len(chunk_s[start:stop]))#* (f_s[1]-f_s[0]))
        vec_rms = np.append(vec_rms, integral)
        '''if integral < integral_min:
            integral_min = integral
            file_index = list(ac.find_rk(df_qty.values.flatten(), el))
            outdata = el
            length_data_used = len(el)'''
    plt.hist(vec_rms, bins=200)
    #plt.xscale('log')
    plt.yscale('log')
    plt.show()