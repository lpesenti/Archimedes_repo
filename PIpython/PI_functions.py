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
    which motor: CONTROLLERNAME,STAGES,REFMODE,SN
    """
    # REFMODE:
    # FNL: Start a reference move to the negative limit switch.
    #      Moves all 'axes' synchronously to the negative physical limits
    #      of their travel ranges and sets the current positions to the negative
    #      range limit values
    # FPL: Start a reference move to the positive limit switch.
    #      Moves all 'axes' synchronously to the positive physical limits
    #      of their travel ranges and sets the current positions to the positive
    #      range limit values.
    # FRF: Start a reference move to the reference switch. (half range)
    #      Moves all 'axes' synchronously to the physical reference point
    #      and sets the current positions to the reference position.
    CONTROLLERNAME = 'C-663.12'
    STAGES = ('M-228.10S')  # connect stages to axes
    REFMODE = ('')  # reference the connected stages
    SN = '021550465'
    if M1 == True and M2 == False:
        CONTROLLERNAME = 'C-663.12'
        STAGES = ('M-228.10S')  # connect stages to axes
        REFMODE = ('')  # reference the connected stages
        SN = '021550465'  # 021550465 SN stage in UNISS
    elif M1 == False and M2 == True:
        CONTROLLERNAME = 'E-872.401'
        STAGES = ('N-480K111')  # connect stages to axes
        REFMODE = ('')  # reference the connected stages
        SN = '021550465'
    else:
        print('Choose only one motor')

    return CONTROLLERNAME, STAGES, REFMODE, SN


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

    CONTROLLERNAME, STAGES, REFMODE, SN = fz_Motor(M1, M2)
    print(CONTROLLERNAME, STAGES, REFMODE, SN)
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectUSB(serialnum=SN)
        # pidevice.InterfaceSetupDlg(key='sample')
        print('initialize connected stages...')
        pitools.startup(pidevice, stages=STAGES, refmodes=None, servostates=True)
        positions = pidevice.qPOS(Axis)
        print('position of axis', Axis, '=', positions[Axis])
        # for Axis in pidevice.axes:
        #     print('position of axis {} = {:.2f}'.format(Axis, positions[Axis]))

    return positions[Axis]


def fz_MoveAxis(M1, M2, Axis, target, Vel):
    """
    Move axis

    Parameters
    ----------
    M1 : bool
        Motor 1
    M2 : bool
        Motor 2
    Axis : int
        Axis's number
    target : float
        new target position
    Vel : float
        velocity
    StepSize : float
        Step size

    Notes
    -----

    Returns
    -------
    new axis position
    """

    CONTROLLERNAME, STAGES, REFMODE, SN = fz_Motor(M1, M2)
    print(CONTROLLERNAME, STAGES, REFMODE, SN)
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectUSB(serialnum=SN)
        print('initialize connected stages...')
        pitools.startup(pidevice, stages=STAGES, refmodes='FRF', servostates=True)
        pidevice.VEL(Axis, Vel)
        rangemin = list(pidevice.qTMN(Axis).values())
        rangemax = list(pidevice.qTMX(Axis).values())
        print(rangemin[0], rangemax[0])
        if target < rangemax[0] and target > rangemin[0]:
            print('move stages...')
            pidevice.MOV(Axis, target)
            # pidevice.MVR(Axis, target)
            pitools.waitontarget(pidevice)
        else:
            print('target value out of range', rangemin[0], '-', rangemax[0])
        positions = pidevice.qPOS(Axis)
        print('position of axis', Axis, '=', positions[Axis])
        # pidevice.POS(Axis, positions)
        # for Axis in pidevice.axes:
        #    print('position of axis {} = {:.2f}'.format(Axis, positions[Axis]))

    return positions[Axis]
