from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import datetime as dt
import time
import numpy as np
import math
from scipy import signal
import scipy.io
import scipy.fftpack
from obspy import read, read_inventory
from obspy import UTCDateTime
from obspy import Stream


def Read_Inv(filexml, ch, sensor, t, Twindow, verbose):
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
    invxmls = invxml.select(station=sensor, channel=ch)
    if sensor == 'SOE0' or sensor == 'SOE1' or sensor == 'SOE2' or sensor == 'P2' or sensor == 'P3':
        seed_id = "ET." + sensor + ".." + ch
    else:
        seed_id = "DR." + sensor + ".." + ch
    # 
    resp = invxmls.get_response(seed_id, t)
    if verbose:
        print(resp)
    # return a nested dictionary detailing the sampling rates of each response stage
    sa = resp.get_sampling_rates()
    # take the last output_sampling_rate
    fsxml = sa[len(sa)]['output_sampling_rate']
    #######################
    fsxml = 100.
    #######################
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

    return fxml, respamp, fsxml


def Evaluate_PSD(filexml, ch, dirdata, sensor, tstart, tstop, Twindow, Overlap, verbose):
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
    tstart : str
        start time of the analysis
    tstop : str
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
    ti = UTCDateTime(tstart)
    yi = str(ti.year)
    mi = str(ti.month)
    di = str(ti.day)
    hi = str(ti.hour)
    mii = str(ti.minute)
    si = str(ti.second)
    doyi = ti.strftime('%j')
    tf = UTCDateTime(tstop)
    yf = str(tf.year)
    mf = str(tf.month)
    doyf = tf.strftime('%j')
    if verbose:
        print("Analysis from: ", yi, mi, doyi, " to: ", yf, mf, doyf)

    # Read Inventory and get freq array, response array, sample freq.
    fxml, respamp, fsxml = Read_Inv(filexml, ch, sensor, ti, Twindow, verbose=verbose)
    print("res", len(respamp))
    # read filename
    if sensor == 'SOE0' or sensor == 'SOE1' or sensor == 'SOE2':
        filename = dirdata + "/" + yi + "/" + sensor + "/ET." + sensor + ".." + ch + ".D." + yi + "." + doyi
    if sensor == 'SOE4':
        filename = dirdata + "/" + yi + "/" + sensor + "/ET." + sensor + ".." + ch + "_centaur-3_6653_" + yi + str(
            mi).zfill(2) + str(di).zfill(2) + "_" + str(hi).zfill(2) + str(mii).zfill(2) + str(si).zfill(
            2) + ".miniseed"
    if sensor == 'SOE5':
        filename = dirdata + "/" + yi + "/" + sensor + "/ET." + sensor + ".." + ch + "_centaur-3_5046_" + yi + str(
            mi).zfill(2) + str(di).zfill(2) + "_" + str(hi).zfill(2) + str(mii).zfill(2) + str(si).zfill(
            2) + ".miniseed"
    if sensor == 'SOE6':
        filename = dirdata + "/" + yi + "/" + sensor + "/ET." + sensor + ".." + ch + "_centaur-3_7322_" + yi + str(
            mi).zfill(2) + str(di).zfill(2) + "_" + str(hi).zfill(2) + str(mii).zfill(2) + str(si).zfill(
            2) + ".miniseed"
    # +str(23).zfill(3)+
    #  st0 = Stream()
    #  st0 += read(dirdata0+"/[yi-yf]/"+name0+"/ET."+name0+".."+ch0+".D.[yi-yf].[doyi-doyf]",starttime=ti, endtime=tf)
    st = read(filename, starttime=ti, endtime=tf)
    if verbose:
        print(st.__str__(extended=True))
    tr = st[0]
    #  tr0.plot()
    fs = tr.stats.sampling_rate
    if verbose:
        print('fs', fs)
    fmin = 1 / Twindow  # Hz
    Num = int(Twindow * fs)
    NOL = int(Overlap / 100. * Num)
    if verbose:
        print(Num, NOL)
    # f,PSD arrays
    _, f = mlab.psd(np.ones(Num), NFFT=Num, Fs=fs, noverlap=NOL)
    f = f[1:]  # remove first value that is 0
    print("f", len(f))
    s, _ = mlab.psd(tr.data, NFFT=Num, Fs=fs, detrend="linear", noverlap=NOL)
    s = s[1:]
    print("s", len(s))
    # omega
    w = 2.0 * math.pi * f
    # PSD in (m/s2)^2/Hz
    #    s=s*(w**2)/respamp
    s = s / respamp
    s = np.sqrt(s)
    #    v=tr.data
    #    if verbose:
    #       print(v)

    return f, s


def Plot_PSD(filexml0, ch0, dirdata0, name0, tstart0, tstop0,
             filexml1, ch1, dirdata1, name1, tstart1, tstop1,
             filexml2, ch2, dirdata2, name2, tstart2, tstop2,
             filexml3, ch3, dirdata3, name3, tstart3, tstop3,
             filexml4, ch4, dirdata4, name4, tstart4, tstop4,
             filexml5, ch5, dirdata5, name5, tstart5, tstop5,
             Twindow, Overlap, Fi, Ff, LogaX, LogaY, verbose):
    """
    Read sensor data and plot PSD

    Parameters
    ----------
    filexml : str
        path and name of the xml file to be read
    ch : str
        sensor's channel to be read
    sensor : str
        sensor's name
    tstart : str
        start time of the analysis
    tstop : str
        stop time of the analysis
    Twindow : float
        time window for the PSD
    Overlap : float
        overlap in percentage for the PSD windows
    Fi : float
        min frequency for plot
    Ff : float
        max frequency for plot
    verbose : bool
        If True the verbosity is enabled.

    Notes
    -----
    Overlap should be lower than 50 %

    Returns
    -------
    PSD plots
    """

    f0, s0 = Evaluate_PSD(filexml0, ch0, dirdata0, name0, tstart0, tstop0, Twindow, Overlap, verbose)
    # Plot
    fig0, ax0 = plt.subplots()
    ax0.plot(f0, s0, label=name0, linewidth=2)
    if name1:
        f1, s1 = Evaluate_PSD(filexml1, ch1, dirdata1, name1, tstart1, tstop1, Twindow, Overlap, verbose)
        ax0.plot(f1, s1, label=name1, linewidth=2)
    if name2:
        f2, s2 = Evaluate_PSD(filexml2, ch2, dirdata2, name2, tstart2, tstop2, Twindow, Overlap, verbose)
        ax0.plot(f2, s2, label=name2, linewidth=2)
    if name3:
        f3, s3 = Evaluate_PSD(filexml3, ch3, dirdata3, name3, tstart3, tstop3, Twindow, Overlap, verbose)
        ax0.plot(f3, s3, label=name3, linewidth=2)
    if name4:
        f4, s4 = Evaluate_PSD(filexml4, ch4, dirdata4, name4, tstart4, tstop4, Twindow, Overlap, verbose)
        ax0.plot(f4, s4, label=name4, linewidth=2)
    if name5:
        f5, s5 = Evaluate_PSD(filexml5, ch5, dirdata5, name5, tstart5, tstop5, Twindow, Overlap, verbose)
        ax0.plot(f5, s5, label=name5, linewidth=2)
    if LogaX == True:
        ax0.set_xscale("log")
    if LogaY == True:
        ax0.set_yscale("log")
    if Fi != 0 and Ff != 0:
        ax0.set_xlim([Fi, Ff])
    ax0.set_ylim([1.e-11, 1.e-5])
    ax0.grid(True)
    ax0.tick_params(axis='both', which='major', labelsize=15)
    ax0.set_xlabel('Frequency [Hz]', fontsize=20);
    #    ax0.set_ylabel(r'PSD [(m/s$^2$)$^2$/Hz]',fontsize=20)
    #    ax0.set_ylabel(r'PSD [(m/s$^2$)/$\sqrt{Hz}$]',fontsize=20)
    ax0.set_ylabel(r'ASD [(m/s)/$\sqrt{Hz}$]', fontsize=20)
    legend = ax0.legend(loc='upper right', shadow=True, fontsize=15)

    plt.show()
