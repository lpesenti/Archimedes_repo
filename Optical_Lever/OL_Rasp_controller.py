import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import time
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
