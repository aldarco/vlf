# functions to use when processing PAVNET data
#
# author: aldo Arriola
# email:  aldo.arriolac@gmail.com

import numpy as np
import tarfile
import os
import sys
sys.path.append("/home/aldo//notebooks/code")
import fft_pavnet
import datetime.datetime as dtime


def get_content(ftar):
    # obtener archivo de datos (TXT) contenido en el comprimido tar.gz
    tar= tarfile.open(ftar, "r:gz")
    fileout = tar.extractfile(tar.getmembers()[0])
    content = fileout.read()
    return content

def get_dt_fname_v3(fname, loc="PLO"):
    if not fname.startswith(f"{loc}_"): return None
    fname = fname.split("h")[0].replace("PLO_", "")
    dt_ = dtime.strptime(fname, "%Y_%m_%d_%H_%M_%S")
    return dt_

def fft_method(st, fft_npts=2**13, wf=np.blackman):
    #return fft_pavnet.fft_window(st, wlen=fft_npts,fw=signal.windows.flattop)
    return fft_pavnet.fft_overlap(st, fft_npts=fft_npts,window=wf)

def basebandiq(x, tt, fc, bw=100, fs=50e3):
    '''
    covnerts a signal to baseband, DC
    x : Numpy array, signal
    tt: Time array, same len as x
    fc: central frequency
    bw: bandwidth frequency around fc
    fs: sampling frequency
    --
    Returns: xi,xq
    [I, Q ] compontes

    '''

    xi = np.cos(2*np.pi*fc*tt)  * x
    xq = -np.sin(2*np.pi*fc*tt) * x
    lpf = firwin(101, cutoff=bw, fs=fs)
    xi = signal.filtfilt(lpf, 1, xi)
    xq = signal.filtfilt(lpf, 1, xq)

    return xi, xq

def IQ_clipping_filter(iq, nstd=None):
    # clip_min, clip_max = np, 1.0
    # iq : array, shape
    if not nstd: nstd = 3
    ithr = nstd * np.std(iq[:, 0])
    qthr = nstd * np.std(iq[:, 1])
    I_clipped = np.clip(iq[:,0], -ithr, ithr)
    Q_clipped = np.clip(iq[:,1], -qthr, qthr)
    return I_clipped, Q_clipped

