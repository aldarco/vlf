'''
This module contains the main functions to process a PAVNET raw data file, 
which contains an IQ signal valued

'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas as pd
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
import time
import datetime as dt
#import itertools



def read_iqfile(filepath, sep=" "):
    '''
    Reads a data IQ file based on PAVNET datafile's format: ANTAR title and datetime on .txt
    returns: complex array and datetime (sx, t0)
    '''
    params = {}
    sx = []
    
    with open(filepath, 'r') as data:
        i = 0
        for line in data:
            if line[0] == '#':
                if i == 0:
                    t0 = line[2:].rstrip()
                    t0 = t0.split(' ')
                    usecond = int(t0[-1].split('.')[1])
                    second = int(t0[-1].split('.')[0])
                    day, month, year, hour, minute = list(map(int,t0[:-1]))
                    t0 = dt.datetime(year=year,
                                        month=month,
                                        day=day,
                                        hour=hour,
                                        minute=minute,
                                        second=second,
                                        microsecond=usecond)
                if i == 1:
                    npts = int(line.split(':')[1])

            else:
                xx = line.split(sep)
                sx.append(float(xx[0]) + 1j*float(xx[1])) # complex values
            i += 1

    return np.asarray(sx), t0

def read_iqfile2(filepath):
    '''
    Reads a data IQ file based on PAVNET datafile's format: ANTAR title and datetime on .txt
    returns: complex array and datetime (sx, t0)
    '''
    params = {}
    sx = []
    
    with open(filepath, 'r') as data:
        i = 0
        for line in data:
            if line[0] == '#':
                if i == 0:
                    t0 = line[2:].rstrip()
                    t0 = t0.split(' ')
                    usecond = int(t0[-1].split('.')[1])
                    second = int(t0[-1].split('.')[0])
                    day, month, year, hour, minute = list(map(int,t0[:-1]))
                    t0 = dt.datetime(year=year,
                                        month=month,
                                        day=day,
                                        hour=hour,
                                        minute=minute,
                                        second=second,
                                        microsecond=usecond)
                if i == 1:
                    npts = int(line.split(':')[1])

            else:
                xx = line.split(' ')
                sx.append(float(xx[0]) + 1j*float(xx[1])) # complex values
            i += 1

    return np.asarray(sx), t0

def read_datafile(file,iqformat=False,sep=","):
    '''
    Load ANTAR-type IQ data from PAVNET VLF receiver
    
    Parameters

    - file: str
            file path
    - iqformat: bool
            if True returns complex array values of signal, 
            else returs a dataframe with I and Q columns

    Uses pd.read_csv for ANTAR files
    '''
    if ("ANTAR" not in file) and (not file.endswith(".txt")):  raise Exception(f"ERROR: File {file} not admitted.")
    
    df = pd.read_csv(file, sep=sep, comment="#", header=None)
    if iqformat:
        iqv = np.array(df.iloc[:,0]+ 1j*df.iloc[:,1])
        return iqv

    return df


def load_data_into_matrix(folder, fnames=None):
    if not fnames:
        fnames = os.listdir(folder)
    for name in fnames: 
        td = time_from_fname(name) 
        df = pd.read_csv(folder+"/"+name)


def time_from_fname(fname):
    if ("ANTAR" not in fname) and (not fname.endswith(".txt")):
        raise Exception(f"ERROR: File name {fname} not admitted.")
    dd = list(map(int,fname.split("_")[3:9]))
    t = dt.datetime(*dd[:3][::-1], *dd[3:])
    return t

def fft_iqfile(filepath, sampling_freq, method="single", module =True,**kwargs):
    '''
    Overlap windows and fft along a signal given a certanin percent of overlap length 
    to finally return te fft average 

    input: 
    - filepath: path with filename
    - sampling_freq: Fs or sampling rate (SPS)
    - method: 'single', 'overlap'. By default 'single'
    - **kwargs for fft_window
        wlen: number of pts to perform FFT
        window: window function
        fac: calibration/escalation factor
    output: S, t0 # spectrum and time
    '''
    #print(kwargs)
    #siq, t0 = read_iqfile(filepath)
    siq = read_datafile(filepath, iqformat=True, sep=kwargs["sep"])
    ff = filepath.split("/")[-1]
    t0 = time_from_fname(ff)
    npts = len(siq)
    #print("before fft ",len(siq))
    if method == 'single':
        S = fft_window(siq, sampling_freq=sampling_freq,module=module, **kwargs)
        #print("^",end="")
    elif method == 'overlap':
        S = fft_overlap(siq, sampling_freq=sampling_freq, **kwargs)
        #print("*",end="")
    else:
        raise ValueError("Unrecognized method.")

    return S, t0 


def fft_multi_iqfiles(fnames, sampling_freq, fft_npts=4096, **kwargs):
    '''
    computes FFT of several IQ data files by calling fft_iqfile function
    
    inputs: 
    - filepath: path with filename
    - sampling_freq: Fs or sampling rate (SPS)
    - **kwargs for fft_window
        wlen: number of pts to perform FFT
        window: window function
        fac: calibration/escalation factor
    output: Dataframe containing S, freqs (as index), t0 (as column names)
    '''
    freqs = np.arange(fft_npts)*sampling_freq/fft_npts

    fftv = []
    timev = []
    for fx in fnames:
        S, t0 = fft_iqfile(fx, sampling_freq=sampling_freq, fft_npts=fft_npts, ret_freq=False,module=True,**kwargs)
        fftv.append(S)
        timev.append(t0)
    
    df = pd.DataFrame(np.asarray(fftv).T, index=freqs, columns=timev)
    df.columns = pd.to_datetime(df.columns)
    return df
   

def fft_window(signal, fft_npts=2**12,window=signal.windows.flattop, fac=1,\
               sampling_freq=50e3, module=True, ret_freq =True, **kwargs):
    #wlen = 2**12
    #whanning = np.hanning(len(data_signal[date0]))
    #print(signal.shape,signal[:int(fft_npts)].shape, window(int(fft_npts),**kwargs).shape)
    if module:
        return fac*fftshift(abs(fft(signal[:int(fft_npts)]*window(int(fft_npts)))))
    else:
        return fac*fftshift(fft(signal[:int(fft_npts)]*window(int(fft_npts))))
        

def fft_overlap(sx, fft_npts=2**12, window=signal.windows.flattop, sampling_freq=50e3, \
                povlp=50, fac=1, phase=False , ret_freq=True,shift=True,**kwargs):
    '''
    sx: complex signal array of 2**n length, expected 2**15 points
    fft_npts: size of window (2**x)
    window: winsow function
    povlp: overlaping percent
    coherent = if True, averages complex values of FFT, else averages module values (|FFT|).
    
    Returns:
       
       FFT_amp
    '''
    sx = np.asarray(sx)
    sxlen = len(sx)
    n_iterations = np.floor((sxlen/fft_npts-1)/(1-povlp/100))
    #Sprint("> {} iteration estimated".format(n_iterations))
    
    start = 0 
    end = fft_npts 
    fft_sum = []
    it_idx = 0
    w = window(fft_npts)

    while end <= sxlen:
        sx_w = sx[int(start):int(end)]
        fftc = fft(sx_w * w )
        if shift:
            fftc = fftshift(fftc)
        else:
            #fftc = abs(fftc)
            fft_sum.append(abs(fftc))
        fftc = abs(fftc)*fac
        fft_sum.append(fftc)

        it_idx += 1
        end = fft_npts*(1 + it_idx*(1-povlp/100)) 
        start = end - fft_npts #wlen*(1 + (it_idx-1*(1-povlp/100))

    #print("> {} iteration excuted".format(it_idx))
    return np.sum(fft_sum, axis=0)/it_idx


if __name__ == "__main__":
    # ... nothing to do yet
    pass
