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
import scipy.signal as signal
from datetime import datetime as dtime
import time

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
    lpf = signal.firwin(101, cutoff=bw, fs=fs)
    xi = signal.filtfilt(lpf, 1, xi)
    xq = signal.filtfilt(lpf, 1, xq)

    return xi, xq



def IQ_frombin(f):
    '''
    Read IQ data from a binary file (pavnet structured)
    '''
    #file_path = file_list[k]
    #print(f"Now reading {file_path}");
    
    #with open(file_path, 'rb') as f:
    header = f.read(70)
    sizeIQ = f.read(30)
    size_str = sizeIQ.decode(errors='ignore')
    size_start = size_str.find("#size ")
    size_end = size_str.find("\n", size_start)
    sizefileIQ = size_str[size_start+6:size_end].strip()
    sizedata = int(sizefileIQ)
    
    I = np.fromfile(f, dtype=np.float32, count=sizedata)
    Q = np.fromfile(f, dtype=np.float32, count=sizedata)
    PPS = f.read(30)      
    
    return np.array([I,Q])
    
def read_binary_IQ(f):

    header = f.read(70)
    sizeIQ = f.read(30)
    size_str = sizeIQ.decode(errors='ignore')
    size_start = size_str.find("#size ")
    size_end = size_str.find("\n", size_start)
    sizefileIQ = size_str[size_start+6:size_end].strip()
    sizedata = int(sizefileIQ)

    # Read I and Q data as binary and convert
    I_data = f.read(sizedata * 4)  # float32 is 4 bytes
    Q_data = f.read(sizedata * 4)

    I = np.frombuffer(I_data, dtype=np.float32)
    Q = np.frombuffer(Q_data, dtype=np.float32)

    PPS = f.read(30)
    return np.asarray([I,Q])

def is_file_stable(filepath, wait_time=0.2, max_iter=6):
    # file size stable 
    size1 = os.path.getsize(filepath)
    time.sleep(wait_time)
    size2 = os.path.getsize(filepath)
    it = 0
    while size1 != size2 and it<max_iter:
        try:
            size1 = size2
            time.sleep(wait_time)
            size2 = os.path.getsize(filepath)
            #return size1 == size2
        except FileNotFoundError:
            return False
        it += 1
    return size1 == size2