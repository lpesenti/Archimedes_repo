

import csv, time, datetime
from pipython import GCSDevice, pitools
from logging.handlers import TimedRotatingFileHandler
import logging

# --- LOGGERS SETUP --- #

# Human readable logger
LogLogger = logging.getLogger('log_motors')
LogLogger.setLevel(logging.INFO)

Log_formatter = logging.Formatter("[%(asctime)s | %(CN)s/%(STAGES)s/%(SN)s] %(message)s", '%d-%m-%y %H:%M:%S')
Log_fh = TimedRotatingFileHandler(r'.\logs\Log_PI', when='midnight')
Log_fh.setLevel(logging.INFO)
Log_fh.setFormatter(Log_formatter)
LogLogger.addHandler(Log_fh)

# Monitor logger
CHDLogger = logging.getLogger('chd_motors')
CHDLogger.setLevel(logging.DEBUG)

CHD_formatter = logging.Formatter("%(asctime)s|%(CN)s/%(STAGES)s/%(SN)s|%(message)s", '%d-%m-%y %H:%M:%S')
CHD_fh = TimedRotatingFileHandler(r'.\logs\CHD_PI', when='midnight')
CHD_fh.setLevel(logging.DEBUG)
CHD_fh.setFormatter(CHD_formatter)
CHDLogger.addHandler(CHD_fh)


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
    REFMODE = ('FNL')  # reference the connected stages
    SN = '021550465'
    if M1 == True and M2 == False:
        CONTROLLERNAME = 'C-663.12'
        STAGES = ('M-228.10S')  # connect stages to axes
        REFMODE = ('FNL')  # reference the connected stages
        SN = '021550449'  # 021550465 SN stage @ UNISS
    elif M1 == False and M2 == True:
        CONTROLLERNAME = 'E-872.401'
        STAGES = ('N-480.210CV', 'N-480.210CV', 'NOSTAGE', 'NOSTAGE')  # connect stages to axes
        REFMODE = ('FNL', 'FNL')  # reference the connected stages
        SN = '121081258'  # 121081258 SN stage @ Sos Enattos
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
        pitools.startup(pidevice, stages=STAGES, refmodes=REFMODE, servostates=True)
        print('pippo1')
        positions = pidevice.qPOS(Axis)
        print('pippo2')
        print(pidevice.qPOS())
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
    log_dict = {'CN': CONTROLLERNAME, 'STAGES': STAGES, 'SN': SN}
    print(CONTROLLERNAME, STAGES, REFMODE, SN)
    with GCSDevice(CONTROLLERNAME) as pidevice:
        pidevice.ConnectUSB(serialnum=SN)
        print('initialize connected stages...')
        pitools.startup(pidevice, stages=STAGES, refmodes=REFMODE, servostates=True)
        if Vel >= 1.5:
            print('velocity out of range: >= 1.5 mm/s')
            Vel = 0.5
            print('velocity set to 0.5 mm/s')
        pidevice.VEL(Axis, Vel)
        rangemin = list(pidevice.qTMN(Axis).values())
        rangemax = list(pidevice.qTMX(Axis).values())
        print(rangemin[0], rangemax[0])
        if rangemax[0] > target > rangemin[0]:
            print('move stages...')
            pidevice.MOV(Axis, target)
            LogLogger.info("CMD: move axis {0} to {1}".format(Axis, target), extra=log_dict)
            # pidevice.MVR(Axis, target)
            pitools.waitontarget(pidevice)
            positions = pidevice.qPOS(Axis)
            LogLogger.info("RPL: axis {0} moved to {1}".format(Axis, positions[Axis]), extra=log_dict)
            CHDLogger.debug('{0}'.format(positions[Axis]), extra=log_dict)
        else:
            print('target value out of range', rangemin[0], '-', rangemax[0])
            positions = pidevice.qPOS(Axis)
            LogLogger.info("RPL: target value out of range. Position of axis {0} is {1}".format(Axis, positions[Axis]),
                            extra=log_dict)
            CHDLogger.debug('{0}'.format(positions[Axis]), extra=log_dict)
        # positions = pidevice.qPOS(Axis)
        print('position of axis', Axis, '=', positions[Axis])
        # for Axis in pidevice.axes:
        #    print('position of axis {} = {:.2f}'.format(Axis, positions[Axis]))

    return positions[Axis]
