from __future__ import division

import argparse
import sys
import time

import lal
import lalsimulation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from lal import MSUN_SI, C_SI, G_SI
from lal.antenna import AntennaResponse
from scipy import interpolate
# from six.moves import cPickle
from scipy.optimize import brentq

from audio_functions import reqshift
from audio_functions import write_wavfile
import configparser
from scipy.interpolate import interp1d

config = configparser.ConfigParser()
config.read('config.ini')
# path_to_ligo = config['Paths']['ligo']
# path_to_virgo = config['Paths']['virgo']
# path_to_et = config['Paths']['et']

if sys.version_info >= (3, 0):
    xrange = range

safe = 2  # define the safe multiplication scale for the desired time length


class bbhparams:
    def __init__(self, mc, M, eta, m1, m2, ra, dec, iota, phi, psi, idx, fmin, snr, SNR):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.phi = phi
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR


def tukey(M, alpha=0.5):
    """
    Tukey window code copied from scipy
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha * (M - 1) / 2.0))
    n1 = n[0:width + 1]
    n2 = n[width + 1:M - width - 1]
    n3 = n[M - width - 1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])


def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='data_prep.py',
                                     description='generates GW data for application of deep learning networks.')

    # arguments for reading in a data file
    parser.add_argument('-N', '--Nsamp', type=int, default=7000, help='the number of samples')
    # parser.add_argument('-Nv', '--Nvalid', type=int, default=1500, help='the number of validation samples')
    # parser.add_argument('-Nt', '--Ntest', type=int, default=1500, help='the number of testing samples')
    parser.add_argument('-Nn', '--Nnoise', type=int, default=25, help='the number of noise realisations per signal')
    parser.add_argument('-Nb', '--Nblock', type=int, default=10000,
                        help='the number of training samples per output file')
    parser.add_argument('-f', '--fsample', type=int, default=8192, help='the sampling frequency (Hz)')
    parser.add_argument('-T', '--Tobs', type=int, default=1, help='the observation duration (sec)')
    parser.add_argument('-s', '--snr', type=float, default=None, help='the signal integrated SNR')
    parser.add_argument('-I', '--detectors', type=str, nargs='+', default=['H1', 'L1'], help='the detectors to use')
    parser.add_argument('-b', '--basename', type=str, default='test', help='output file path and basename')
    parser.add_argument('-m', '--mdist', type=str, default='astro',
                        help='mass distribution for training (astro,gh,metric)')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')

    return parser.parse_args()


def convert_beta(beta, fs, T_obs):
    """
    Converts beta values (fractions defining a desired period of time in
    central output window) into indices for the full safe time window
    """
    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    newbeta = np.array([(beta[0] + 0.5 * safe - 0.5), (beta[1] + 0.5 * safe - 0.5)]) / safe
    low_idx = int(T_obs * fs * newbeta[0])
    high_idx = int(T_obs * fs * newbeta[1])

    return low_idx, high_idx


def gen_noise(fs, T_obs, psd):
    """
    Generates noise from a psd
    """

    N = T_obs * fs  # the total number of time samples
    Nf = N // 2 + 1
    dt = 1 / fs  # the sampling time (sec)
    df = 1 / T_obs

    amp = np.sqrt(0.25 * T_obs * psd)
    idx = np.argwhere(psd == 0.0)
    amp[idx] = 0.0
    re = amp * np.random.normal(0, 1, Nf)
    im = amp * np.random.normal(0, 1, Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N * np.fft.irfft(re + 1j * im) * df

    return x


def gen_psd(fs, T_obs, op='AdvDesign', det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs  # the total number of time samples
    dt = 1 / fs  # the sampling time (sec)
    df = 1 / T_obs  # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df, lal.HertzUnit, N // 2 + 1)
    freq_array = np.arange(0.0, (N // 2 + 1) * df, df)
    print(f'{N//2+1}')
    print(f'{len(psd.data.data)}')
    print(f'{freq_array.size}')

    print(f'Detector selected is {det} and the op is {op}')

    if det == 'H1' or det == 'L1' or det == 'ET' or det == 'VIRGO':
        if op == 'AdvDesign':
            lalsimulation.SimNoisePSDaLIGODesignSensitivityP1200087(psd, 10.0) 
            print(f'AdvDesign selected! This is the psd found: {psd}')
            print(f'epoch = {psd.epoch}')
            print(f'f0 = {psd.f0}')
            print(f'deltaF = {psd.deltaF}')
            print(f'sampleUnits = {psd.sampleUnits}')
            print(f'data = {psd.data}')
        elif op == 'AdvEarlyLow':
            lalsimulation.SimNoisePSDAdVEarlyLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyHigh':
            lalsimulation.SimNoisePSDAdVEarlyHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidLow':
            lalsimulation.SimNoisePSDAdVMidLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidHigh':
            lalsimulation.SimNoisePSDAdVMidHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateLow':
            lalsimulation.SimNoisePSDAdVLateLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateHigh':
            lalsimulation.SimNoisePSDAdVLateHighSensitivityP1200087(psd, 10.0)
        elif op == 'VIRGO':  # not working
            lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
            print(f'VIRGO selected! This is the psd found: {psd}')
        elif op == 'ET':
            # change to the required path
            et_psd = r'et_sensitivity.txt'
            lalsimulation.SimNoisePSDFromFile(psd, 2, et_psd)  # frequency series | freq_cut-off | path/to/file
            print(f'ET selected! This is the psd found: {psd}')
        else:
            print('unknown noise option')
            exit(1)
    else:
        print('unknown detector - will add Virgo soon')
        exit(1)

    plot = False
    if plot:
        psd1 = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df, lal.HertzUnit, N // 2 + 1)
        lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd1, 10.0)

        plt.loglog(psd.data.data, label=f'{op}-{det} PSD')
        plt.loglog(psd1.data.data, label=f'VIRGO-H1 PSD')
        # plt.loglog(np.sqrt(psd.data.data), label=f'{op}-{det} ASD')
        plt.legend()
        plt.show()

    return psd


def read_psd(fs, T_obs, f_cutoff, path):
    N = T_obs * fs  # the total number of time samples
    
    df = 1 / T_obs  # the frequency resolution
    lal_psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df, lal.HertzUnit, N // 2 + 1)
    
    freq_array = np.arange(0.0, (N // 2 + 1) * df, df)
    print(f'{time.asctime()}: N // 2 +1 -> {N//2+1}')
    print(f'{time.asctime()}: Length of empty frequency array -> {len(lal_psd.data.data)}')
    print(f'{time.asctime()}: Length of created frequency array -> {freq_array.size}')
    lalsimulation.SimNoisePSDFromFile(lal_psd, f_cutoff, path)
    # else:
    #     if det.lower() == 'ligo':
    #         lalsimulation.SimNoisePSDFromFile(lal_psd, 10.0, path_to_ligo)  # frequency series | freq_cut-off | path/to/file
    #     elif det.lower() == 'virgo':
    #         lalsimulation.SimNoisePSDFromFile(lal_psd, 10.0, path_to_virgo)  # frequency series | freq_cut-off | path/to/file
    #     elif det.lower() == 'et':
    #         lalsimulation.SimNoisePSDFromFile(lal_psd, 2, path_to_et)  # frequency series | freq_cut-off | path/to/file
    #     else:
    #         import warnings
    #         msg = f"{time.asctime()}: Invalid detector chosen [Ligo, Virgo and ET], using ET"
    #         warnings.warn(msg)
    #         lalsimulation.SimNoisePSDFromFile(lal_psd, 2, path_to_et)  # frequency series | freq_cut-off | path/to/file

    return freq_array, lal_psd
        

def gen_bbh_new(fs, T_obs, psd, dist, dets, beta=[0.75, 0.95], par=None):
    """
    generates a BBH timedomain signal
    """
    
    N = T_obs * fs  # the total number of time samples
    
    if dets.lower() == 'et':
        f_low = 2.0  # lowest frequency of waveform (Hz)
    elif dets.lower() == 'ligo':    
        f_low = 12.0  # lowest frequency of waveform (Hz)
    elif dets.lower() == 'virgo':    
        f_low = 12.0  # lowest frequency of waveform (Hz)
    approximant = lalsimulation.IMRPhenomD

    # make waveform
    # loop until we have a long enough waveform - slowly reduce flow as needed
    flag = False
    while not flag:
        hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
            par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
            0, 0, 0, 0, 0, 0,
            dist * lal.PC_SI,
            par.iota, par.phi, 0,
            0, 0,
            1 / fs,
            f_low, f_low,
            lal.CreateDict(),
            approximant)
        flag = True if hp.data.length > 2 * N else False
        f_low -= 1  # decrease by 1 Hz each time
    orig_hp = hp.data.data
    orig_hc = hc.data.data

    # compute reference idx
    ref_idx = np.argmax(orig_hp ** 2 + orig_hc ** 2)

    # make aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    win = np.zeros(N)
    tempwin = tukey(int((16.0 / 15.0) * N / safe), alpha=1.0 / 8.0)
    win[int((N - tempwin.size) / 2):int((N - tempwin.size) / 2) + tempwin.size] = tempwin

    # loop over detectors
    ts = np.zeros(N)
    hp = np.zeros(N)
    hc = np.zeros(N)
    intsnr = []

    # make signal - apply antenna and shifts
    ht_shift, hp_shift, hc_shift = make_bbh(orig_hp, orig_hc, fs, par.ra, par.dec, par.psi, det='H1')

    # place signal into timeseries - including shift
    ts_temp = ht_shift[int(ref_idx - par.idx):]
    hp_temp = hp_shift[int(ref_idx - par.idx):]
    hc_temp = hc_shift[int(ref_idx - par.idx):]
    if len(ts_temp) < N:
        ts[:len(ts_temp)] = ts_temp
        hp[:len(ts_temp)] = hp_temp
        hc[:len(ts_temp)] = hc_temp
    else:
        ts = ts_temp[:N]
        hp = hp_temp[:N]
        hc = hc_temp[:N]

    ts *= win
    hp *= win
    hc *= win
    
    # compute SNR of pre-whitened data
    # intsnr.append(get_snr(ts, T_obs, fs, psd.data.data, par.fmin))
    intsnr.append(get_snr(ts, T_obs, fs, psd, par.fmin))

    # normalise the waveform using either integrated or peak SNR
    intsnr = np.array(intsnr)
    # print('{}: computed the network SNR = {}'.format(time.asctime(), snr))

    return np.sqrt(intsnr)


def get_snr(data, T_obs, fs, psd, fmin):
    """
    computes the snr of a signal given a PSD starting from a particular frequency index
    """

    N = T_obs * fs
    df = 1.0 / T_obs
    dt = 1.0 / fs
    fidx = int(fmin / df)

    win = tukey(N, alpha=1.0 / 8.0)
    idx = np.argwhere(psd > 0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0 / psd[idx]

    xf = np.fft.rfft(data * win) * dt
    
    SNRsq = 4.0 * np.sum((np.abs(xf[fidx:]) ** 2) * invpsd[fidx:]) * df
    return np.sqrt(SNRsq)


def gen_masses(m_min=5.0, M_max=100.0, mdist='astro'):
    """
    function returns a pair of masses drawn from the appropriate distribution
    """
    flag = False
    if mdist == 'astro':
        print('{}: using astrophysical logarithmic mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        log_m_max = np.log(new_M_max - new_m_min)
        while not flag:
            m12 = np.exp(np.log(new_m_min) + np.random.uniform(0, 1, 2) * (log_m_max - np.log(new_m_min)))
            flag = True if (np.sum(m12) < new_M_max) and (np.all(m12 > new_m_min)) and (m12[0] >= m12[1]) else False
        eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
    elif mdist == 'gh':
        print('{}: using George & Huerta mass distribution'.format(time.asctime()))
        m12 = np.zeros(2)
        while not flag:
            q = np.random.uniform(1.0, 10.0, 1)
            m12[1] = np.random.uniform(5.0, 75.0, 1)
            m12[0] = m12[1] * q
            flag = True if (np.all(m12 < 75.0)) and (np.all(m12 > 5.0)) and (m12[0] >= m12[1]) else False
        eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
    elif mdist == 'metric':
        print('{}: using metric based mass distribution'.format(time.asctime()))
        new_m_min = m_min
        new_M_max = M_max
        new_M_min = 2.0 * new_m_min
        eta_min = m_min * (new_M_max - new_m_min) / new_M_max ** 2
        while not flag:
            M = (new_M_min ** (-7.0 / 3.0) - np.random.uniform(0, 1, 1) * (
                    new_M_min ** (-7.0 / 3.0) - new_M_max ** (-7.0 / 3.0))) ** (-3.0 / 7.0)
            eta = (eta_min ** (-2.0) - np.random.uniform(0, 1, 1) * (eta_min ** (-2.0) - 16.0)) ** (-1.0 / 2.0)
            m12 = np.zeros(2)
            m12[0] = 0.5 * M + M * np.sqrt(0.25 - eta)
            m12[1] = M - m12[0]
            flag = True if (np.sum(m12) < new_M_max) and (np.all(m12 > new_m_min)) and (m12[0] >= m12[1]) else False
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        return m12, mc, eta
    else:
        print('{}: ERROR, unknown mass distribution. Exiting.'.format(time.asctime()))
        exit(1)


def get_fmin(M, eta, dt):
    """
    Compute the instantaneous frequency given a time till merger
    """
    M_SI = M * MSUN_SI

    def dtchirp(f):
        """
        The chirp time to 2nd PN order
        """
        v = ((G_SI / C_SI ** 3) * M_SI * np.pi * f) ** (1.0 / 3.0)
        temp = (v ** (-8.0) + ((743.0 / 252.0) + 11.0 * eta / 3.0) * v ** (-6.0) -
                (32 * np.pi / 5.0) * v ** (-5.0) + ((3058673.0 / 508032.0) + 5429 * eta / 504.0 +
                                                    (617.0 / 72.0) * eta ** 2) * v ** (-4.0))
        return (5.0 / (256.0 * eta)) * (G_SI / C_SI ** 3) * M_SI * temp - dt

    # solve for the frequency between limits
    fmin = brentq(dtchirp, 1.0, 2000.0, xtol=1e-6)
    print('{}: signal enters segment at {} Hz'.format(time.asctime(), fmin))

    return fmin


def gen_par(fs, T_obs, mdist='astro', beta=[0.75, 0.95]):
    """
    Generates a random set of parameters
    """
    # define distribution params
    m_min = 5.0  # rest frame component masses
    M_max = 100.0  # rest frame total mass
    log_m_max = np.log(M_max - m_min)

    m12, mc, eta = gen_masses(m_min, M_max, mdist=mdist)
    M = np.sum(m12)
    print('{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(), m12[0], m12[1], mc))

    # generate iota
    iota = np.arccos(-1.0 + 2.0 * np.random.rand())
    print('{}: selected bbh cos(inclination) = {}'.format(time.asctime(), np.cos(iota)))

    # generate polarisation angle
    psi = 2.0 * np.pi * np.random.rand()
    print('{}: selected bbh polarisation = {}'.format(time.asctime(), psi))

    # generate reference phase
    phi = 2.0 * np.pi * np.random.rand()
    print('{}: selected bbh reference phase = {}'.format(time.asctime(), phi))

    # pick sky position - uniform on the 2-sphere
    ra = 2.0 * np.pi * np.random.rand()
    dec = np.arcsin(-1.0 + 2.0 * np.random.rand())
    print('{}: selected bbh sky position = {},{}'.format(time.asctime(), ra, dec))

    # pick new random max amplitude sample location - within beta fractions
    # and slide waveform to that location
    low_idx, high_idx = convert_beta(beta, fs, T_obs)
    if low_idx == high_idx:
        idx = low_idx
    else:
        idx = int(np.random.randint(low_idx, high_idx, 1)[0])
    print('{}: selected bbh peak amplitude time = {}'.format(time.asctime(), idx / fs))

    # the start index of the central region
    sidx = int(0.5 * fs * T_obs * (safe - 1.0) / safe)

    # compute SNR of pre-whitened data
    fmin = get_fmin(M, eta, int(idx - sidx) / fs)
    print('{}: computed starting frequency = {} Hz'.format(time.asctime(), fmin))

    # store params
    par = bbhparams(mc, M, eta, m12[0], m12[1], ra, dec, np.cos(iota), phi, psi, idx, fmin, None, None)

    return par


def make_bbh(hp, hc, fs, ra, dec, psi, det):
    """
    turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    """

    # make basic time vector
    tvec = np.arange(len(hp)) / float(fs)
    
    # compute antenna response and apply
    resp, Fp, Fc, rp, rc = 1, 1, 1, 1, 1
    antenna = config.getboolean('Bool', 'antenna_response')
    polarization = config.getboolean('Bool', 'random_polarization')
    
    
    print(f'{time.asctime()}: Antenna response is {antenna}')
    print(f'{time.asctime()}: Random Polarization is {polarization}')
    
    if antenna:
        resp = AntennaResponse(det, ra, dec, psi, scalar=True, vector=True, times=0.0)
        Fp = resp.plus
        Fc = resp.cross
    if polarization:
        # np.random.seed(None)
        # seed = np.random.get_state()
        # with open(r'.\logs\seed.txt')
        rp = np.random.uniform(-1,1,1)
        rc = np.random.uniform(-1,1,1)
        print(f'{time.asctime()}: Random factor on plus polarization {rp}')
        print(f'{time.asctime()}: Random factor on cross polarization {rc}')
        # print(f'{time.asctime()}: Seed used {np.random.get_state()}')

    ht = hp * Fp * rp + hc * Fc * rc  # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector = lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location, ra, dec, 0.0)
    print('{}: computed {} Earth centre time delay = {}'.format(time.asctime(), det, tdelay))

    # interpolate to get time shifted signal
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, ht_tck, der=0, ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0, ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0, ext=1)

    return new_ht, new_hp, new_hc


def sim_data(fs, T_obs, snr=1.0, dets=['H1'], Nnoise=25, size=1000, mdist='astro', beta=[0.75, 0.95]):
    """
    Simulates all of the test, validation and training data timeseries
    """

    yval = []  # initialise the param output
    ts = []  # initialise the timeseries output
    par = []  # initialise the parameter output
    nclass = 2  # the hardcoded number of classes
    npclass = int(size / float(nclass))
    ndet = len(dets)  # the number of detectors
    psds = [gen_psd(fs, T_obs, op='AdvDesign', det=d) for d in dets]

    # for the noise class
    for x in xrange(npclass):
        print('{}: making a noise only instance'.format(time.asctime()))
        ts_new = np.array([gen_noise(fs, T_obs, psd.data.data) for psd in psds]).reshape(ndet, -1)
        ts.append(
            np.array([whiten_data(t, T_obs, fs, psd.data.data) for t, psd in zip(ts_new, psds)]).reshape(ndet, -1))
        par.append(None)
        yval.append(0)
        print('{}: completed {}/{} noise samples'.format(time.asctime(), x + 1, npclass))

    # for the signal class - loop over random masses
    cnt = npclass
    while cnt < size:

        # generate a single new timeseries and chirpmass
        par_new = gen_par(fs, T_obs, mdist=mdist, beta=beta)
        ts_new, _, _ = gen_bbh(fs, T_obs, psds, snr=snr, dets=dets, beta=beta, par=par_new)

        # loop over noise realisations
        for j in xrange(Nnoise):
            ts_noise = np.array([gen_noise(fs, T_obs, psd.data.data) for psd in psds]).reshape(ndet, -1)
            ts.append(
                np.array([whiten_data(t, T_obs, fs, psd.data.data) for t, psd in zip(ts_noise + ts_new, psds)]).reshape(
                    ndet, -1))
            par.append(par_new)
            yval.append(1)
            cnt += 1
        print('{}: completed {}/{} signal samples'.format(time.asctime(), cnt - npclass, int(size / 2)))

    # trim the data down to desired length
    ts = np.array(ts)[:size]
    yval = np.array(yval)[:size]
    par = par[:size]

    # return randomised the data
    idx = np.random.permutation(size)
    temp = [par[i] for i in idx]
    return [ts[idx], yval[idx]], temp


# the main part of the code
def main_NEW():
    T_obs = 20
    safeTobs = safe * T_obs
    fs = 8192
    beta = [0.85, 0.85]
    # desired_snr = 8
    # m12, mc, eta = gen_masses(m_min=5.0,M_max=100.0,mdist='astro');
    
    dets = ['ET', 'LIGO', 'VIRGO']

    path_lst = [config['Paths'][path] for path in config['Paths']]
    freq_co_lst = [float(x) for x in config['Quantities']['cutoff_frequencies'].split(',')]

    if len(freq_co_lst) > len(path_lst):
        raise ValueError("The number frequencies cut off are greater than the number of path selected")
    elif len(freq_co_lst) < len(path_lst):
        raise ValueError("The number frequencies cut off are less than the number of path selected")
    # op = ['ET', 'AdvDesign', 'VIRGO']
    freqs = [read_psd(fs=fs, T_obs=safeTobs, path=path, f_cutoff=f_co)[0] for path, f_co in zip(path_lst, freq_co_lst)]
    psds = [read_psd(fs=fs, T_obs=safeTobs, path=path, f_cutoff=f_co)[1] for path, f_co in zip(path_lst, freq_co_lst)]
    # lal_psds = [read_psd(fs=fs, T_obs=T_obs, det=d)[2] for d in dets]
    """psds, freqs = [], []
    for d in dets:
        freqs.append(read_psd(det=d)[0])
        psds.append(read_psd(det=d)[1])"""
    
    print(f'{time.asctime()}: Number of psd evaluated: {len(psds)}')
    if config.getboolean('Bool', 'plot_sensitivities'):
        fig_0 = plt.figure(figsize=(19.2, 10.8))
        ax_psds = fig_0.add_subplot()

        for i in range(len(psds)):
            #  ax_psds.plot(psds[i].data.data, label=f'{op[i]}-{dets[i]} PSD')
            ax_psds.plot(freqs[i], psds[i].data.data, label=f'{dets[i]} PSD')
        ax_psds.set_xscale('log')
        ax_psds.set_yscale('log')
        ax_psds.tick_params(axis='both', which='both', labelsize=22)
        # ax_psds.set_xlabel(r'Frequency [Hz]', fontsize=24)
        ax_psds.set_ylabel(r'$\frac{1}{Hz}$', fontsize=24)
        # plt.loglog(np.sqrt(psd.data.data), label=f'{op}-{det} ASD')
        ax_psds.legend(loc='best', shadow=True, fontsize=24)
        ax_psds.grid(True, linestyle='--', axis='both', which='both')
        fig_0.tight_layout()
        plt.show()
    
    
    m12_list = [[80, 80]]  # , [80, 50], [80, 80]]
    distances = np.linspace(1e6, 1e9, 500)
    print(f'{time.asctime()}: Minimum distance {distances.min()}')
    print(f'{time.asctime()}: Maximum distance {distances.max()}')
    print(f'{time.asctime()}: Number of distances computed {distances.size}')
    
    for m12 in m12_list:
        eta = m12[0] * m12[1] / (m12[0] + m12[1]) ** 2
        mc = np.sum(m12) * eta ** (3.0 / 5.0)
        M = np.sum(m12)
        print('{}: selected bbh masses = {},{} (chirp mass = {})'.format(time.asctime(), m12[0], m12[1], mc))
        # generate iota
        iota = np.arccos(-1.0 + 2.0 * np.random.rand())  # remove randomization
        print('{}: selected bbh cos(inclination) = {}'.format(time.asctime(), np.cos(iota)))
        # generate polarisation angle
        psi = 2.0 * np.pi * np.random.rand()  # remove randomization
        print('{}: selected bbh polarisation = {}'.format(time.asctime(), psi))
        # generate reference phase
        phi = 2.0 * np.pi * np.random.rand()  # remove randomization
        print('{}: selected bbh reference phase = {}'.format(time.asctime(), phi))
        # pick sky position - uniform on the 2-sphere
        ra = 2.0 * np.pi * np.random.rand()
        dec = np.arcsin(-1.0 + 2.0 * np.random.rand())  # remove randomization
        print('{}: selected bbh sky position = {},{}'.format(time.asctime(), ra, dec))
        # the start index of the central region
        sidx = int(0.5 * fs * safeTobs * (safe - 1.0) / safe)
        # pick new random max amplitude sample location - within beta fractions
        # and slide waveform to that location
        low_idx, high_idx = convert_beta(beta, fs, safeTobs)
        if low_idx == high_idx:
            idx = low_idx
        else:
            idx = int(np.random.randint(low_idx, high_idx, 1)[0])
        print('{}: selected bbh peak amplitude time = {}'.format(time.asctime(), idx / fs))
        fmin = get_fmin(M, eta, int(idx - sidx) / fs)
        print('{}: computed starting frequency = {} Hz'.format(time.asctime(), fmin))
        ###
        parameters = bbhparams(mc, M, eta, m12[0], m12[1], ra, dec, np.cos(iota), phi, psi, idx, fmin, None, None)
        ###

        fig = plt.figure(figsize=(19.2, 10.8))
        
        snr_array = []
        for inx_psd, psd in enumerate(psds):
            print(f'{time.asctime()}: Evaluating the {inx_psd}th-psd')
            for index, dist in enumerate(distances):
                # j1 = (index + 1) / len(distances)
                # print("\r[%-75s] %g%%" % ('=' * int(75 * j1), round(100 * j1, 3)), end='\n')
                print(f'{time.asctime()}: {round(index/len(distances) * 100, 2)} %')
                snr_array.append(  # 'H1' fixed for the AntennaResponse
                    gen_bbh_new(fs, safeTobs, psd.data.data, dist=dist, dets=dets[inx_psd], beta=beta, par=parameters))
            # print(f'{inx_psd} - {snr_array}')
            plt.plot(distances, snr_array, label=f'{dets[inx_psd]}')
            plt.title(f'{time.asctime()}: BBH masses {m12[0]}-{m12[1]}', fontsize=30)
            snr_array = []
        plt.xlabel('Distance $[pc]$', fontsize=24)
        plt.ylabel('SNR', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(loc='best', shadow=True, fontsize=24)
        plt.grid(True, linestyle='--', axis='both', which='both')
        fig.tight_layout()

    # Extract data and sampling rate from file
    # signal_shifted = reqshift(signal, fshift=300, sample_rate=fs)
    # write_wavfile('signal.wav', fs, signal_shifted)
    # sound_data, fs = sf.read('signal.wav', dtype='float32')
    # sd.play(sound_data, fs)

    plt.show()
    sd.wait()  # Wait until file is done playing


if __name__ == "__main__":
    exit(main_NEW())
