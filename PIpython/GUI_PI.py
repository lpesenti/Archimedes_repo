#!/usr/bin/python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
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


def import_M1():
    global M1
    m1 = bool(entry.get())
    print(m1)
    M1.set(m1)


def import_M2():
    global M2
    m2 = bool(entry.get())
    M2.set(m2)


def import_VEL():
    global Vel
    v = bool(entry.get())
    Vel.set(v)


# def import_StepSize():
#     global StepSize
#     ss = bool(entry.get())
#     StepSize.set(ss)

def import_NAxis():
    global NAxis
    naxis = int(entry.get())
    NAxis.set(naxis)


def ReadAxis():
    global axis0
    axis0 = PI_functions.fz_ReadAxis(bool(M1.get()), bool(M2.get()), int(Axis.get()))
    Axis0.set(axis0)


def MoveAxis():
    global axis1
    axis1 = PI_functions.fz_MoveAxis(bool(M1.get()), bool(M2.get()), int(Axis.get()), float(Targ.get()),
                                     float(Vel.get()))
    Axis1.set(axis1)


def MoveAxisOPEN():
    global axis1OPEN
    axis1OPEN = PI_functions.fz_MoveAxis(bool(M1.get()), bool(M2.get()), int(Axis.get()), float(TargOPEN.get()),
                                         float(Vel.get()))
    Axis1OPEN.set(axis1OPEN)


def MoveAxisCLOSE():
    global axis1CLOSE
    axis1CLOSE = PI_functions.fz_MoveAxis(bool(M1.get()), bool(M2.get()), int(Axis.get()), float(TargCLOSE.get()),
                                          float(Vel.get()))
    Axis1CLOSE.set(axis1CLOSE)


root = tk.Tk()
root.title("Physical Instrument Python GUI for Archimedes")

########################################################################################
frameStart = tk.LabelFrame(root, text='Choose Motor', relief=tk.GROOVE)
frameStart.grid(row=0, column=0, padx=10, pady=2)

M1 = IntVar(value=True)
Checkbutton(frameStart, text="C-663 (1 axis)", variable=M1).grid(row=0, column=0)
M2 = IntVar(value=False)
Checkbutton(frameStart, text="E-872 (2 axis)", variable=M2).grid(row=1, column=0)

tk.Label(frameStart, text='Velocity [mm/s]').grid(row=2, column=0)
Vel = tk.StringVar(value=0.5)
tk.Entry(frameStart, textvariable=Vel).grid(row=2, column=1)

########################################################################################
frame1 = tk.LabelFrame(root, text='Read and Write', relief=tk.GROOVE)
frame1.grid(row=0, column=1, padx=10, pady=2)

tk.Label(frame1, text='Axis').grid(row=0, column=0)
Axis = tk.StringVar(value=1)
tk.Entry(frame1, textvariable=Axis).grid(row=0, column=1)
tk.Button(frame1, text='Read axis value', command=ReadAxis).grid(row=0, column=2)
Axis0 = tk.StringVar()
tk.Entry(frame1, textvariable=Axis0).grid(row=0, column=3)

tk.Label(frame1, text='Set value [0-10 mm]').grid(row=1, column=0)
Targ = tk.StringVar(value=5.)
tk.Entry(frame1, textvariable=Targ).grid(row=1, column=1)
tk.Button(frame1, text='Move to target value', command=MoveAxis).grid(row=1, column=2)
Axis1 = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1).grid(row=1, column=3)

tk.Label(frame1, text='Set OPEN value [2 mm]').grid(row=2, column=0)
TargOPEN = tk.StringVar(value=2.)
tk.Entry(frame1, textvariable=TargOPEN).grid(row=2, column=1)
tk.Button(frame1, text='Move to OPEN target value', command=MoveAxisOPEN).grid(row=2, column=2)
Axis1OPEN = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1OPEN).grid(row=2, column=3)

tk.Label(frame1, text='Set CLOSE value [7 mm]').grid(row=3, column=0)
TargCLOSE = tk.StringVar(value=7.)
tk.Entry(frame1, textvariable=TargCLOSE).grid(row=3, column=1)
tk.Button(frame1, text='Move to CLOSE target value', command=MoveAxisCLOSE).grid(row=3, column=2)
Axis1CLOSE = tk.StringVar()
tk.Entry(frame1, textvariable=Axis1CLOSE).grid(row=3, column=3)

########################################################################################
########################################################################################
# Close
frameEnd = tk.LabelFrame(root, text='Quit program', relief=tk.GROOVE)
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
