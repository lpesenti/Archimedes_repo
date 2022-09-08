__author__ = "Luca Pesenti"
__credits__ = ["Luca Pesenti"]
__version__ = "0.0.1"
__maintainer__ = "no longer supported"
__email__ = "lpesenti@uniss.it"
__status__ = "Deprecated"

r"""
This script is the first attempt to create an automatic procedure to check the OP27 health status. The idea was to built
a Optical Lever Shield Health Monitor (OLSHM) using a Raspberry Pi 3b+ connected to an MCP3008 ADC that reads
the signal of each OP27. However, only a few tests were made on just one MCP3008 ADC connected on one side to a
Raspberry and the other part the connections were free.
"""

import time

import adafruit_mcp3xxx.mcp3008 as MCP
import board
import busio
import digitalio
from adafruit_mcp3xxx.analog_in import AnalogIn
from matplotlib import mlab


def make_psd(data):
    psd_s, psd_f = mlab.psd(data, NFFT=num, Fs=freq, detrend="linear")
    psd_s = psd_s[1:]
    psd_f = psd_f[1:]


# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the cs (chip select)
cs = digitalio.DigitalInOut(board.D5)

# create the mcp object
mcp = MCP.MCP3008(spi, cs)

# create an analog input channel on pin 0
channel = AnalogIn(mcp, MCP.P0)

while True:
    i = 0
    while i < 10:
        print('Raw ADC Value: ', channel.value)
        print('ADC Voltage: ' + str(channel.voltage) + 'V')
        time.sleep(0.0001)
