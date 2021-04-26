import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import *
import subprocess
import webbrowser
import numpy as np
import sys
sys.dont_write_bytecode = True

#import LoadFileInfo # Laodfile module
#import Time_code # import the Time Analysis module
import Analysis # import the Analysis module

# Create GUI
def create_window():
    window = tk.Tk()
# do Nothing
def doNothing():
    print("nothing")

# file *.xml
def import_xml0():
    global vxml0
    xml_path0 = askopenfilename()
    print(xml_path0)
    vxml0.set(xml_path0)
# choose channel: N,E,Z
def import_ch0():
    global Ch0
    c0 = entry.get()
    Ch0.set(C0)
# choose dir data file
def import_data0():
    global dir0
    dir_path0 = askdirectory()
    print(dir_path0)
    dir0.set(dir_path0)
# choose sensor name (e.g. SOE*)
def import_data0Name():
    global Name0
    N0 = entry.get()
    Name0.set(N0)
# start time
def import_t0start():
    global t0start
    T0Sta = float(entry.get())
    t0start.set(T0Sta)
# stop time
def import_t0stop():
    global t0stop
    T0Sto = float(entry.get())
    t0stop.set(T0Sto)

# file *.xml
def import_xml1():
    global vxml1
    xml_path1 = askopenfilename()
    print(xml_path1)
    vxml1.set(xml_path1)
# choose channel: N,E,Z
def import_ch1():
    global Ch1
    c1 = entry.get()
    Ch1.set(C1)
# choose dir data file
def import_data1():
    global dir1
    dir_path1 = askdirectory()
    print(dir_path1)
    dir1.set(dir_path1)
# choose sensor name (e.g. SOE*)
def import_data1Name():
    global Name1
    N1 = entry.get()
    Name1.set(N1)
# start time
def import_t1start():
    global t1start
    T1Sta = float(entry.get())
    t1start.set(T1Sta)
# stop time
def import_t1stop():
    global t1stop
    T1Sto = float(entry.get())
    t1stop.set(T1Sto)

# file *.xml
def import_xml2():
    global vxml2
    xml_path2 = askopenfilename()
    print(xml_path2)
    vxml2.set(xml_path2)
# choose channel: N,E,Z
def import_ch2():
    global Ch2
    c2 = entry.get()
    Ch2.set(C2)
# choose dir data file
def import_data2():
    global dir2
    dir_path2 = askdirectory()
    print(dir_path2)
    dir2.set(dir_path2)
# choose sensor name (e.g. SOE*)
def import_data2Name():
    global Name2
    N2 = entry.get()
    Name2.set(N2)
# start time
def import_t2start():
    global t2start
    T2Sta = float(entry.get())
    t2start.set(T2Sta)
# stop time
def import_t2stop():
    global t2stop
    T2Sto = float(entry.get())
    t2stop.set(T2Sto)

# file *.xml
def import_xml3():
    global vxml3
    xml_path3 = askopenfilename()
    print(xml_path3)
    vxml3.set(xml_path3)
# choose channel: N,E,Z
def import_ch3():
    global Ch3
    c3 = entry.get()
    Ch3.set(C3)
# choose dir data file
def import_data3():
    global dir3
    dir_path3 = askdirectory()
    print(dir_path3)
    dir3.set(dir_path3)
# choose sensor name (e.g. SOE*)
def import_data3Name():
    global Name3
    N3 = entry.get()
    Name3.set(N3)
# start time
def import_t3start():
    global t3start
    T3Sta = float(entry.get())
    t3start.set(T3Sta)
# stop time
def import_t3stop():
    global t3stop
    T3Sto = float(entry.get())
    t3stop.set(T3Sto)

# file *.xml
def import_xml4():
    global vxml4
    xml_path4 = askopenfilename()
    print(xml_path4)
    vxml4.set(xml_path4)
# choose channel: N,E,Z
def import_ch4():
    global Ch4
    c4 = entry.get()
    Ch4.set(C4)
# choose dir data file
def import_data4():
    global dir4
    dir_path4 = askdirectory()
    print(dir_path4)
    dir4.set(dir_path4)
# choose sensor name (e.g. SOE*)
def import_data4Name():
    global Name4
    N4 = entry.get()
    Name4.set(N4)
# start time
def import_t4start():
    global t4start
    T4Sta = float(entry.get())
    t4start.set(T4Sta)
# stop time
def import_t4stop():
    global t4stop
    T4Sto = float(entry.get())
    t4stop.set(T4Sto)

# file *.xml
def import_xml5():
    global vxml5
    xml_path5 = askopenfilename()
    print(xml_path5)
    vxml5.set(xml_path5)
# choose channel: N,E,Z
def import_ch5():
    global Ch5
    c5 = entry.get()
    Ch5.set(C5)
# choose dir data file
def import_data5():
    global dir5
    dir_path5 = askdirectory()
    print(dir_path5)
    dir5.set(dir_path5)
# choose sensor name (e.g. SOE*)
def import_data5Name():
    global Name5
    N5 = entry.get()
    Name5.set(N5)
# start time
def import_t5start():
    global t5start
    T5Sta = float(entry.get())
    t5start.set(T5Sta)
# stop time
def import_t5stop():
    global t5stop
    T5Sto = float(entry.get())
    t5stop.set(T5Sto)

# Get time window for PSD
def import_Tw():
    global Tw
    T = int(entry.get())
    Tw.set(T)
def import_OL():
    global OL
    overl = float(entry.get())
    OL.set(overl)
def import_Fi():
    global Fi
    fi = int(entry.get())
    Fi.set(fi)
def import_Ff():
    global Ff
    ff = int(entry.get())
    Ff.set(ff)
def import_LogaX():
    global LogaX
    logax = bool(entry.get())
    LogaX.set(logax)
def import_LogaY():
    global LogaY
    logay = bool(entry.get())
    LogaY.set(logay)
def PSDgraph():
    Analysis.Plot_PSD(vxml0.get(),Ch0.get(),dir0.get(),Name0.get(),t0start.get(),t0stop.get(),
                      vxml1.get(),Ch1.get(),dir1.get(),Name1.get(),t1start.get(),t1stop.get(),
                      vxml2.get(),Ch2.get(),dir2.get(),Name2.get(),t2start.get(),t2stop.get(),
                      vxml3.get(),Ch3.get(),dir3.get(),Name3.get(),t3start.get(),t3stop.get(),
                      vxml4.get(),Ch4.get(),dir4.get(),Name4.get(),t4start.get(),t4stop.get(),
                      vxml5.get(),Ch5.get(),dir5.get(),Name5.get(),t5start.get(),t5stop.get(),
                      float(Tw.get()),float(OL.get()),float(Fi.get()),float(Ff.get()),
                      bool(LogaX.get()),bool(LogaY.get()),verbose=True)

root = tk.Tk()
root.title("SEismic Analysis (SEA) PyGUI")

########################################################################################
frame3 = tk.LabelFrame(root, text='Load datasets', relief=tk.GROOVE)
frame3.grid(row=0, column=0, padx=10, pady=2)

# Choose Dataset
tk.Label(frame3, text='xml file').grid(row=0, column=0)
tk.Label(frame3, text='data file').grid(row=0, column=1)
tk.Label(frame3, text='Set Sensor').grid(row=0, column=2)
tk.Label(frame3, text="Set Channel").grid(row=0, column=3)
tk.Label(frame3, text='Start time').grid(row=0, column=4)
tk.Label(frame3, text='Stop time').grid(row=0, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml0).grid(row=1, column=0)
vxml0 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data/metadata/Sos_Enattos.xml')
entryxml0 = tk.Entry(frame3, textvariable=vxml0).grid(row=1, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data0).grid(row=2, column=0)
dir0 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry0 = tk.Entry(frame3, textvariable=dir0).grid(row=2, column=1)
Name0 = tk.StringVar(value='SOE0')
entryn0 = tk.Entry(frame3, textvariable=Name0).grid(row=2, column=2)
Ch0 = tk.StringVar(value='HHZ')
Ch0_ent = tk.Entry(frame3, textvariable=Ch0).grid(row=2, column=3)
t0start = tk.StringVar(value='2020-11-23T00:00:00')
t0stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti0 = tk.Entry(frame3, textvariable=t0start).grid(row=2, column=4)
entrytf0 = tk.Entry(frame3, textvariable=t0stop).grid(row=2, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml1).grid(row=3, column=0)
vxml1 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data/metadata/Sos_Enattos.xml')
entryxml1 = tk.Entry(frame3, textvariable=vxml1).grid(row=3, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data1).grid(row=4, column=0)
dir1 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry1 = tk.Entry(frame3, textvariable=dir1).grid(row=4, column=1)
Name1 = tk.StringVar(value='SOE1')
entryn1 = tk.Entry(frame3, textvariable=Name1).grid(row=4, column=2)
Ch1 = tk.StringVar(value='HHZ')
Ch1_ent = tk.Entry(frame3, textvariable=Ch1).grid(row=4, column=3)
t1start = tk.StringVar(value='2020-11-23T00:00:00')
t1stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti1 = tk.Entry(frame3, textvariable=t1start).grid(row=4, column=4)
entrytf1 = tk.Entry(frame3, textvariable=t1stop).grid(row=4, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml2).grid(row=5, column=0)
vxml2 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data/metadata/Sos_Enattos.xml')
entryxml2 = tk.Entry(frame3, textvariable=vxml2).grid(row=5, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data2).grid(row=6, column=0)
dir2 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry2 = tk.Entry(frame3, textvariable=dir2).grid(row=6, column=1)
Name2 = tk.StringVar(value='SOE2')
entryn2 = tk.Entry(frame3, textvariable=Name2).grid(row=6, column=2)
Ch2 = tk.StringVar(value='HHZ')
Ch2_ent = tk.Entry(frame3, textvariable=Ch2).grid(row=6, column=3)
t2start = tk.StringVar(value='2020-11-23T00:00:00')
t2stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti2 = tk.Entry(frame3, textvariable=t2start).grid(row=6, column=4)
entrytf2 = tk.Entry(frame3, textvariable=t2stop).grid(row=6, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml3).grid(row=7, column=0)
vxml3 = tk.StringVar()
entryxml3 = tk.Entry(frame3, textvariable=vxml3).grid(row=7, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data3).grid(row=8, column=0)
dir3 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry3 = tk.Entry(frame3, textvariable=dir3).grid(row=8, column=1)
Name3 = tk.StringVar()
entryn3 = tk.Entry(frame3, textvariable=Name3).grid(row=8, column=2)
Ch3 = tk.StringVar(value='HHZ')
Ch3_ent = tk.Entry(frame3, textvariable=Ch3).grid(row=8, column=3)
t3start = tk.StringVar(value='2020-11-23T00:00:00')
t3stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti3 = tk.Entry(frame3, textvariable=t3start).grid(row=8, column=4)
entrytf3 = tk.Entry(frame3, textvariable=t3stop).grid(row=8, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml4).grid(row=9, column=0)
vxml4 = tk.StringVar()
entryxml4 = tk.Entry(frame3, textvariable=vxml4).grid(row=9, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data4).grid(row=10, column=0)
dir4 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry4 = tk.Entry(frame3, textvariable=dir4).grid(row=10, column=1)
Name4 = tk.StringVar()
entryn4 = tk.Entry(frame3, textvariable=Name4).grid(row=10, column=2)
Ch4 = tk.StringVar(value='HHZ')
Ch4_ent = tk.Entry(frame3, textvariable=Ch4).grid(row=10, column=3)
t4start = tk.StringVar(value='2020-11-23T00:00:00')
t4stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti4 = tk.Entry(frame3, textvariable=t4start).grid(row=10, column=4)
entrytf4 = tk.Entry(frame3, textvariable=t4stop).grid(row=10, column=5)

tk.Button(frame3, text='Browse .xml file',command=import_xml5).grid(row=11, column=0)
vxml5 = tk.StringVar()
entryxml5 = tk.Entry(frame3, textvariable=vxml5).grid(row=11, column=1)
tk.Button(frame3, text='Browse Data/ folder',command=import_data5).grid(row=12, column=0)
dir5 = tk.StringVar(value='/Users/drozza/Documents/UNISS/CAoS/Data')
entry5 = tk.Entry(frame3, textvariable=dir5).grid(row=12, column=1)
Name5 = tk.StringVar()
entryn5 = tk.Entry(frame3, textvariable=Name5).grid(row=12, column=2)
Ch5 = tk.StringVar(value='HHZ')
Ch5_ent = tk.Entry(frame3, textvariable=Ch5).grid(row=12, column=3)
t5start = tk.StringVar(value='2020-11-23T00:00:00')
t5stop = tk.StringVar(value='2020-11-23T01:00:00')
entryti5 = tk.Entry(frame3, textvariable=t5start).grid(row=12, column=4)
entrytf5 = tk.Entry(frame3, textvariable=t5stop).grid(row=12, column=5)

########################################################################################
frame4 = tk.LabelFrame(root, text='PSD Analysis', relief=tk.GROOVE)
frame4.grid(row=1, column=0, padx=10, pady=2)

# Choose PSD parameters
Tw = tk.StringVar(value=30)
Tw_lbl = tk.Label(frame4, text="Set PSD time window [s]").grid(row=0, column=0)
Tw_ent = tk.Entry(frame4, textvariable=Tw).grid(row=0, column=1)
OL = tk.StringVar(value=0)
OL_lbl = tk.Label(frame4, text="Set Overlap percentage [%]").grid(row=1, column=0)
OL_ent = tk.Entry(frame4, textvariable=OL).grid(row=1, column=1)
Fi = tk.StringVar(value=1)
Fi_lbl = tk.Label(frame4, text="Set min Freq. to plot [Hz]").grid(row=2, column=0)
Fi_ent = tk.Entry(frame4, textvariable=Fi).grid(row=2, column=1)
Ff = tk.StringVar(value=20)
Ff_lbl = tk.Label(frame4, text="Set max Freq. to plot [Hz]").grid(row=3, column=0)
Ff_ent = tk.Entry(frame4, textvariable=Ff).grid(row=3, column=1)
LogaX = IntVar(value=False)
Checkbutton(frame4, text="Log X", variable=LogaX).grid(row=4, column=0)
LogaY = IntVar(value=True)
Checkbutton(frame4, text="Log Y", variable=LogaY).grid(row=4, column=1)
# Call Time and ASD Graph
tk.Button(frame4, text='Graphs', command=PSDgraph).grid(row=5, column=1)

########################################################################################
# Close
frame5 = tk.LabelFrame(root, text='Quit Analysis', relief=tk.GROOVE)
frame5.grid(row=2, column=0, padx=10, pady=2)
tk.Button(frame5, text='Close',command=root.destroy).grid(row=0, column=1)

menu =  Menu(root)
root.config(menu=menu)
subMenu = Menu(menu)
menu.add_cascade(label="File",menu=subMenu)
subMenu.add_command(label="New", command=create_window)
subMenu.add_command(label="Open", command=doNothing)
subMenu.add_command(label="Restart", command=doNothing)
subMenu.add_command(label="Exit", command=doNothing)
editMenu = Menu(menu)
menu.add_cascade(label = "Help", menu=editMenu)
editMenu.add_command(label="Help", command=doNothing)

root.geometry('+300+250')

root.mainloop()
