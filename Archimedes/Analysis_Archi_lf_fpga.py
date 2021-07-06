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

params = {'axes.labelsize': 16,     # Fontsize of the x and y labels
          'axes.titlesize': 16,     # Fontsize of the axes title
          'figure.titlesize':16, # Size of the figure title (Figure.suptitle())
          'xtick.labelsize':14,     # Fontsize of the tick labels
          'ytick.labelsize':14,     # Fontsize of the tick labels
          'legend.fontsize':14,     # Fontsize for legends (plt.legend(), fig.legend())
          'legend.title_fontsize':10} # Fontsize for legend titles, None

plt.rcParams.update(params)

def myfromtimestampfunction(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def myvectorizer(input_func):
    def output_func(array_of_numbers):
        return [input_func(a) for a in array_of_numbers]
    return output_func

#def tfe(x, y, Num, fs):
#   """estimate transfer function from x to y, see csd for calling convention"""
#   return mlab.csd(y, x, NFFT=Num,Fs=fs,detrend="linear") / mlab.psd(x, NFFT=Num,Fs=fs,detrend="linear")

xtime=[]
c0=[]
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]
c7=[]
patternDate = "\d{2}[/]\d{2}[/]\d{4}"
patternTime = "\d{2}[:]\d{2}[:]\d{2}[.]\d{6}"
fs = 1000. # Hz
dt = 1./fs
iloop=0
timestamp=0.

# array with all file paths
#lineList = [line.rstrip('\n') for line in open('listafile_psd_87.txt')]
#lineList = [line.rstrip('\n') for line in open('listafile_psd_51.txt')]
#lineList = [line.rstrip('\n') for line in open('listafile_psd_55.txt')]

#lineList = [line.rstrip('\n') for line in open('listafile_hdd_20210228.txt')]
#lineList = [line.rstrip('\n') for line in open('listafile_psd_20210228.txt')]
#lineList = [line.rstrip('\n') for line in open('listafile_psd_20210228_5960.txt')]

#with open('Archi_lf_fpga_Night_55.lvm') as fobj0:
#with open('/Volumes/LaCie/ARCHIMEDES_DATA_STORAGE/SosEnattos_Data/SosEnattos_Data_20210221/test_File_15.lvm') as fobj:
#with open('test_File_19.lvm') as fobj:
#with open('test_File_21_half.lvm') as fobj:
#with open('../SosEnattos_Data_20210219/test_File_163.lvm') as fobj:
#with open('test_File_23.lvm') as fobj:
with open('SCI_21-05-17_1302.lvm') as fobj:
# loop over file in list
#for filename in lineList:
#    print(filename)
#    with open(filename) as fobj:
     for line in fobj:
         if iloop == 0:
            dates = re.findall(patternDate, line)
            times = re.findall(patternTime, line)
            print(dates,times)
            datesstr = ''.join(dates)
            timesstr = ''.join(times)
            print(datesstr)
            humandate = datetime.datetime.strptime(datesstr+" "+timesstr, '%m/%d/%Y %H:%M:%S.%f')
            timestamp = datetime.datetime.timestamp(humandate)#+7200.
            print(timestamp)
         row0 = line.split()
         xtime.append(timestamp+dt*iloop)
         c0.append(float(row0[0]))
         c1.append(float(row0[1]))
         c2.append(float(row0[2]))
         c3.append(float(row0[3]))
         c4.append(float(row0[4]))
         c5.append(float(row0[5]))
         c6.append(float(row0[6]))
#         c7.append(float(row0[7]))
         iloop+=1
print('File Loaded')
#print(dates,times)
#print(c0)
#print(c7)

date_convert = myvectorizer(myfromtimestampfunction)
xtimex = date_convert(xtime)

# Time analysis
fig, ax0=plt.subplots()
ax0=plt.gca()
xfmt = mdates.DateFormatter('%b %d %H:%M')#:%S')
ax0.xaxis.set_major_formatter(xfmt)
ax0.plot(xtimex, c0, label="Interferometer");
ax0.plot(xtimex, c1, label="Pick-off");
ax0.plot(xtimex, c3, label="Error signal")
ax0.plot(xtimex, c4, label="Correction");
legend = ax0.legend(loc='upper right', shadow=True)#, fontsize='x-large')
ax0.grid(True)
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Voltage [V]')

# lambda laser
lambdalaser = 532.e-9 # m
# lenght between mirrors
L = 0.1 # m
# dtheta/dV = alpha/(Vmax-Vmin)
# alpha = lambda/(2piL)
alpha = lambdalaser/(2.*np.pi*L)
Vmax = max(c0) #-4.#6.4 #-4. #-4. #-4.3
Vmin = min(c0) #-9.#7.8 #-6. #-9.
alpha=alpha/(Vmax-Vmin)
print('alpha',alpha)

po_ave = -np.mean(c1)
print('pick-off mean',po_ave)

# PSD
Num = int(60.*fs)
# The values for the power spectrum , The frequencies corresponding to the elements
_,f=mlab.psd(np.ones(Num),NFFT=Num,Fs=fs)#,noverlap=NOL)
f=f[1:] # remove first value that is 0
s1,_=mlab.psd(c3,NFFT=Num,Fs=fs,detrend="linear")#,noverlap=NOL)
s1=s1[1:]
s1=np.sqrt(s1)*alpha*po_ave
#s1=np.sqrt(s1)*1.4e-7*5.765
s2,_=mlab.psd(c1,NFFT=Num,Fs=fs,detrend="linear")#,noverlap=NOL)
s2=s2[1:]
s2=np.sqrt(s2)*alpha

### VIRGO ###
x, y = np.loadtxt('/Users/drozza/Documents/UNISS/Archimedes/Archimedes_at_Virgo_201907/VirgoData_Jul2019.txt',unpack=True, usecols=[0,1])
print('File Loaded')

#for iloop in range(0,len(f)):
#   print(f[iloop],s1[iloop])

fig, ax1=plt.subplots()
ax1.plot(x, y, label="@ Virgo", color="red", linewidth=2)
ax1.plot(f, s1, label="@ Sos-Enattos", color="blue", linewidth=2)#Error-Signal")
ax1.plot(f, s2, label="Pick-Off")
legend = ax1.legend(loc='upper right', shadow=True)#, fontsize='x-large')
ax1.set_xscale("log")
ax1.set_yscale("log")
#ax1.set_xlim([2.,20.])
#ax1.set_ylim([1.e-13,1.e-8])
ax1.grid(True)
ax1.set_xlabel('Frequency [Hz]');
ax1.set_ylabel(r'ASD [rad/$\sqrt{Hz}$]')
#ax1.tick_params(labelcolor='black', labelsize='large', width=3)

plt.show()
