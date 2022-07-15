#!/usr/bin/python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import ttk
from tkinter import *
import subprocess
import webbrowser
import numpy as np
import sys

sys.dont_write_bytecode = True

import PI_functions  # import the Analysis module


# Create GUI
def create_window():
    window = tk.Tk()


# do Nothing
def doNothing():
    print("nothing")


def import_VEL():
    global Vel
    v = bool(entry.get())
    Vel.set(v)


def import_NAxis():
    global NAxis
    naxis = int(entry.get())
    NAxis.set(naxis)


def ReadAxisC663():
    global axis0
    axis0 = PI_functions.fz_ReadAxisC663(str(CONTROLLERNAME.get()), str(STAGES.get()), str(REFMODE.get()), str(SN.get()), int(Axis.get()))
    Axis0.set(axis0)


def MoveAxisC663():
    global axis1
    axis1 = PI_functions.fz_MoveAxisC663(str(CONTROLLERNAME.get()), str(STAGES.get()), str(REFMODE.get()), str(SN.get()), int(Axis.get()), float(Targ.get()), float(Vel.get()))
    Axis1.set(axis1)


def MoveAxisOPENC663():
    global axis1OPEN
    axis1OPEN = PI_functions.fz_MoveAxisC663(str(CONTROLLERNAME.get()), str(STAGES.get()), str(REFMODE.get()), str(SN.get()), int(Axis.get()), float(TargOPEN.get()), float(Vel.get()))
    Axis1OPEN.set(axis1OPEN)


def MoveAxisCLOSEC663():
    global axis1CLOSE
    axis1CLOSE = PI_functions.fz_MoveAxisC663(str(CONTROLLERNAME.get()), str(STAGES.get()), str(REFMODE.get()), str(SN.get()), int(Axis.get()), float(TargCLOSE.get()), float(Vel.get()))
    Axis1CLOSE.set(axis1CLOSE)


root = tk.Tk()
root.title("Physical Instrument Python GUI for Archimedes")

########################################################################################

tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tabControl.add(tab1, text='Overview')
tabControl.pack(expand=1, fill="both")
tabControl.add(tab2, text='C-663')
tabControl.add(tab3, text='E-872')

########################################################################################

frameStart = tk.LabelFrame(tab1, text='PI motor GUI', relief=tk.GROOVE)
frameStart.grid(row=0, column=0, padx=10, pady=2)
text= Text(frameStart, width= 100, height= 10, background="gray71",foreground="#fff",font= ('Sans Serif', 13, 'italic bold'))
#Insert the text at the begining
text.insert(INSERT, "This GUI was deployed to connect the PI motors for Archimedes experiment\n\n"
                    "test\n"
                    "\n")

text.pack(expand= 1, fill= BOTH)

########################################################################################
frame1 = tk.LabelFrame(tab2, text='Read and Write', relief=tk.GROOVE)
frame1.grid(row=0, column=1, padx=10, pady=2)

xvar = StringVar()
def selection():
    xvar = radiobutton_variable.get()
    print("You selected the option " + xvar)

radiobutton_variable = StringVar()
Radiobutton(frame1, text="C-663 SN=021550465 stage @ UNISS",  variable = radiobutton_variable, value = '021550465', command=selection).grid(row = 0, column = 0)
Radiobutton(frame1, text="C-663 SN=021550449 stage @ LULA ", variable = radiobutton_variable, value = '021550449', command=selection).grid(row = 1, column = 0)

CONTROLLERNAME = tk.StringVar(value='C-663.12')
STAGES = tk.StringVar(value='M-228.10S')  # connect stages to axes
REFMODE = tk.StringVar(value='FNL')  # reference the connected stages
SN = tk.StringVar(value='021550465')
if xvar == '021550465':
    print(radiobutton_variable.get())
    CONTROLLERNAME = 'C-663.12'
    STAGES = 'M-228.10S'  # connect stages to axes
    REFMODE = 'FNL'  # reference the connected stages
    SN = '021550465'  # 021550449 @ LULA ; 021550465 SN stage @ UNISS
elif xvar == '021550449':
    print(radiobutton_variable.get())
    CONTROLLERNAME = 'C-663.12'
    STAGES = 'M-228.10S'  # connect stages to axes
    REFMODE = 'FNL'  # reference the connected stages
    SN = '021550449'  # 021550449 @ LULA ; 021550465 SN stage @ UNISS
else:
    print('none')

tk.Label(frame1, text='Velocity [mm/s]').grid(row=2, column=0)
Vel = tk.StringVar(value=0.25)
tk.Entry(frame1, textvariable=Vel).grid(row=2, column=1)

# tk.Button(frame1, text='Stop Move', command=StopAbort).grid(row=0, column=3)

tk.Label(frame1, text='Axis').grid(row=3, column=0)
Axis = tk.StringVar(value=1)
tk.Entry(frame1, textvariable=Axis).grid(row=3, column=1)
tk.Button(frame1, text='Read axis value', command=ReadAxisC663).grid(row=3, column=2)
Axis0 = tk.StringVar()
tk.Entry(frame1, textvariable=Axis0).grid(row=3, column=3)

tk.Label(frame1, text='Set value [0-10 mm]').grid(row=4, column=0)
Targ = tk.StringVar(value=5.)
tk.Entry(frame1, textvariable=Targ).grid(row=4, column=1)
tk.Button(frame1, text='Move to target value', command=MoveAxisC663).grid(row=4, column=2)
Axis1 = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1).grid(row=4, column=3)

tk.Label(frame1, text='Set OPEN value [7.5 mm]').grid(row=5, column=0)
TargOPEN = tk.StringVar(value=7.5)
tk.Entry(frame1, textvariable=TargOPEN).grid(row=5, column=1)
tk.Button(frame1, text='Move to OPEN target value', command=MoveAxisOPENC663).grid(row=5, column=2)
Axis1OPEN = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1OPEN).grid(row=5, column=3)

tk.Label(frame1, text='Set CLOSE value [8.3 mm]').grid(row=6, column=0)
TargCLOSE = tk.StringVar(value=8.3)
tk.Entry(frame1, textvariable=TargCLOSE).grid(row=6, column=1)
tk.Button(frame1, text='Move to CLOSE target value', command=MoveAxisCLOSEC663).grid(row=6, column=2)
Axis1CLOSE = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1CLOSE).grid(row=6, column=3)

########################################################################################


########################################################################################
# Close
frameEnd = tk.LabelFrame(tab1, text='Quit program', relief=tk.GROOVE)
frameEnd.grid(row=3, column=0, padx=10, pady=2)
tk.Button(frameEnd, text='Close', command=root.destroy).grid(row=0, column=1)

menu = Menu(root)
root.config(menu=menu)
subMenu = Menu(menu)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="New", command=create_window)
subMenu.add_command(label="Open", command=doNothing)
subMenu.add_command(label="Restart", command=doNothing)
subMenu.add_command(label="Exit", command=doNothing)
editMenu = Menu(menu)
menu.add_cascade(label="Help", menu=editMenu)
editMenu.add_command(label="Help", command=doNothing)

root.geometry('+300+250')

root.mainloop()
