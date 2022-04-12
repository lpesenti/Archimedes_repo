from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import numpy as np
import math
from scipy import signal
import scipy.io
import scipy.fftpack

import csv, time, datetime
from pipython import GCSDevice, pitools

def fz_Motor(M1, M2):
    """
    Read axis

    Parameters
    ----------
    M1 : bool
        Motor 1
    M2 : bool
        Motor 2

    Notes
    -----

    Returns
    -------
    which motor: CONTROLLERNAME,STAGES,REFMODE
    """

    CONTROLLERNAME = 'C-663.12'
    STAGES = ('M-228.10S')  # connect stages to axes
    REFMODE = ('FNL')  # reference the connected stages
    SN = '021550465'
    if M1 == True and M2 == False:
       CONTROLLERNAME = 'C-663.12'
       STAGES = ('M-228.10S')  # connect stages to axes
       REFMODE = ('FNL')  # reference the connected stages
       SN = '021550465'
    elif M1 == False and M2 == True:
       CONTROLLERNAME = 'C-663.12'
       STAGES = ('M-228.10S')  # connect stages to axes
       REFMODE = ('FNL')  # reference the connected stages
    else:
       print('Choose only one motor')

    return CONTROLLERNAME,STAGES,REFMODE,SN

def fz_ReadAxis(M1, M2, Axis):
    """
    Read axis

    Parameters
    ----------
    M1 : bool
        Motor 1
    M2 : bool
        Motor 2
    Axis : float
        Axis's number

    Notes
    -----

    Returns
    -------
    axis position
    """

#    CONTROLLERNAME = 'C-663.12'
#    STAGES = ('M-228.10S')  # connect stages to axes
#    REFMODE = ('FNL')  # reference the connected stages
#    SN = '021550465'
#    if M1 == True and M2 == False:
#       CONTROLLERNAME = 'C-663.12'
#       STAGES = ('M-228.10S')  # connect stages to axes
#       REFMODE = ('FNL')  # reference the connected stages
#       SN = '021550465'
#    elif M1 == False and M2 == True:
#       CONTROLLERNAME = 'C-663.12'
#       STAGES = ('M-228.10S')  # connect stages to axes
#       REFMODE = ('FNL')  # reference the connected stages
#    else:
#       print('Choose only one motor')
    CONTROLLERNAME, STAGES, REFMODE, SN = fz_Motor(M1, M2)
    print(CONTROLLERNAME,STAGES,REFMODE,SN)
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectUSB(serialnum=SN)
        #pidevice.InterfaceSetupDlg(key='sample')
        print('initialize connected stages...')
        pitools.startup(pidevice, stages=STAGES, refmodes=REFMODE, servostates=True)
        print('after startup')
        positions = pidevice.qPOS(pidevice.axes)
        #print('position of axis {} = {:.2f}'.format(Axis, positions[Axis]))
        for Axis in pidevice.axes:
            print('position of axis {} = {:.2f}'.format(Axis, positions[Axis]))

    return positions[Axis]