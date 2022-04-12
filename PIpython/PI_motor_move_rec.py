#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This example helps you to get used to PIPython."""

import csv, time, datetime
from pipython import GCSDevice, pitools

CONTROLLERNAME = 'C-663.12'
STAGES = ('M-228.10S')  # connect stages to axes
REFMODE = ('FNL')  # reference the connected stages

def date_now():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today = str(today)
    return today

def time_now():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    now = str(now)
    return now

def move(target):
    """Connect, setup system and move stages and display the positions in a loop."""
    # We recommend to use GCSDevice as context manager with "with".
    # The CONTROLLERNAME decides which PI GCS DLL is loaded.
    with GCSDevice(CONTROLLERNAME) as pidevice:
        # InterfaceSetupDlg() is an interactive dialog.
        pidevice.InterfaceSetupDlg(key='sample')
        # ------------------------------------------------------------------------------#
        # Each PI controller supports the qIDN() command which returns an
        # identification string with a trailing line feed character which
        # we "strip" away.
        # print('connected: {}'.format(pidevice.qIDN().strip()))
        # Show the version info which is helpful for PI support when there
        # are any issues.
        # if pidevice.HasqVER():
        #    print('version info: {}'.format(pidevice.qVER().strip()))
        # ------------------------------------------------------------------------------#
        # The "startup" function will initialize your system. There are controllers that
        # cannot discover the connected stages hence we set them with the
        # "stages" argument. The desired referencing method (see controller
        # user manual) is passed as "refmode" argument
        print('initialize connected stages...')
        pitools.startup(pidevice, stages=STAGES, refmodes=REFMODE, servostates=True)
        # ------------------------------------------------------------------------------#
        # Now we query the allowed motion range of all connected stages.
        # GCS commands often return an (ordered) dictionary with axes/channels
        # as "keys" and the according values as "values".
        # The GCS commands qTMN() and qTMX() used above are query commands.
        # They don't need an argument and will then return all available
        # information, e.g. the limits for _all_ axes. With setter commands
        # however you have to specify the axes/channels. GCSDevice provides
        # a property "axes" which returns the names of all connected axes.
        rangemin = list(pidevice.qTMN().values())
        rangemax = list(pidevice.qTMX().values())
        print('move stages...')
        pidevice.MOV(pidevice.axes, target)

        # To check the on target state of an axis there is the GCS command
        # qONT(). But it is more convenient to just call "waitontarget".
        pitools.waitontarget(pidevice)

        # GCS commands usually can be called with single arguments, with
        # lists as arguments or with a dictionary.
        # If a query command is called with an argument the keys in the
        # returned dictionary resemble the arguments. If it is called
        # without an argument the keys are always strings.

        positions = pidevice.qPOS(pidevice.axes)
        for axis in pidevice.axes:
            print('position of axis {} = {:.2f}'.format(axis, positions[axis]))
    return positions[axis]

if __name__ == '__main__':
    # To see what is going on in the background you can remove the following
    # two hashtags. Then debug messages are shown.
    # import logging
    # logging.basicConfig(level=logging.DEBUG)

    move(1.5)
