from obspy import read, read_inventory
from obspy import UTCDateTime
from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import numpy as np
import math
from scipy import signal
import scipy.io
import scipy.fftpack
import re
import datetime

def myfromtimestampfunction(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def myvectorizer(input_func):
    def output_func(array_of_numbers):
        return [input_func(a) for a in array_of_numbers]
    return output_func

############################
import matplotlib as mpl
#mpl.use('Agg') 
import matplotlib.ticker
import matplotlib.gridspec as gridspec
axis_font = {'fontname':'DejaVu Sans', 'size':'11.5'} # <-- magari questo font non ce lo hai, cmq ti da un warning e lo cambia col default
yaxis_font = {'fontname':'DejaVu Sans', 'size':'11.5'}
mpl.rc('xtick', labelsize=11)
mpl.rc('ytick', labelsize=11)
mpl.rcParams['xtick.major.size']=5
mpl.rcParams['xtick.minor.size']=2
mpl.rcParams['ytick.major.size']=5
mpl.rcParams['ytick.minor.size']=2
mpl.rcParams['axes.linewidth']=1.5
mpl.rcParams['xtick.major.width']=1.5
mpl.rcParams['xtick.minor.width']=1.5
mpl.rcParams['ytick.major.width']=1.5
mpl.rcParams['ytick.minor.width']=1.5 
mpl.rcParams['xtick.top']=True
mpl.rcParams['ytick.right']=True
mpl.rcParams['xtick.direction']="in"
mpl.rcParams['ytick.direction']="in"
#new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f','#bcbd22', '#17becf',]
             # bluetto , arancione,  verde   , rosso scuro, viola  , marrone  ,  rosa    , grigio   , giallo  , azzurro
new_colors = ['cornflowerblue', 'g', 'b', 'k']

locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
############################
params = {'axes.labelsize': 16,     # Fontsize of the x and y labels
          'axes.titlesize': 16,     # Fontsize of the axes title
          'figure.titlesize':16, # Size of the figure title (Figure.suptitle())
          'xtick.labelsize':14,     # Fontsize of the tick labels
          'ytick.labelsize':14,     # Fontsize of the tick labels
          'legend.fontsize':14,     # Fontsize for legends (plt.legend(), fig.legend())
          'legend.title_fontsize':10} # Fontsize for legend titles, None
plt.rcParams.update(params)
############################


datapath = "/Users/drozza/Documents/UNISS/CAoS/Data/"
year="2020/"

filenamexml=datapath+"ET_network.xml"
invxml = read_inventory(filenamexml)
print(invxml)
invxml.plot()
invxml0 = invxml.select(station='SOE0', channel='HHZ')
invxml1 = invxml.select(station='SOE1', channel='HHZ')
invxml2 = invxml.select(station='SOE2', channel='HHZ')
invxml0.plot_response(0.001, label_epoch_dates=True) 
invxml1.plot_response(0.001, label_epoch_dates=True)
invxml2.plot_response(0.001, label_epoch_dates=True)

sensor="SOE4/"

#filenameSOH=datapath+year+sensor+"ET.SOE4.D0.SOH_centaur-3_6653_20201122_"+str(0).zfill(2)+"0000.miniseed"
#st4soh = read(filenameSOH)
#print("st ",st4soh)
#tr4soh = st4soh[0]
#print("tr ",tr4soh)
#print("stat ",tr4soh.stats)
#print("get ",tr4soh._get_response)

filename=datapath+year+sensor+"ET.SOE4..HHZ_centaur-3_6653_20201122_"+str(0).zfill(2)+"0000.miniseed"
ti = UTCDateTime("2020-11-22T00:00:00")
st4 = read(filename)#,starttime=ti, endtime=ti+1000)
print("st ",st4)
tr4 = st4[0]
print("tr ",tr4)
#tr4n = st4[1]
#tr4e = st4[2]
print("stat ",tr4.stats)
fs4=tr4.stats.sampling_rate
print("fs ",fs4)
trsl4=tr4.slice(ti,ti+1800-1/fs4)
st4.plot()
trsl4.plot()
#tr4.plot()
#tr4n.plot()
#tr4e.plot()
Num4=trsl4.stats.npts
_,f4=mlab.psd(np.ones(Num4),NFFT=Num4,Fs=fs4)
f4=f4[1:] # remove first value that is 0
s4,_=mlab.psd(trsl4.data,NFFT=Num4,Fs=fs4,detrend="linear")
s4=s4[1:]

fig0, ax0=plt.subplots()
ax0.plot(f4,s4,label="SOE4")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlabel("f (Hz)")
ax0.set_ylabel("PSD")
legend = ax0.legend(loc='upper right', shadow=True)#, fontsize='x-large')

plt.show()
