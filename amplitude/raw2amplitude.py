import sys
import os
sys.path.append("/home/aldo//notebooks/code")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import dates as mdates
import pandas as pd
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import firwin, lfilter
import time
import tools       # codigo propio
import datetime
from io import StringIO as strio
import tarfile
import scipy.integrate as integrate
import gc
import fft_pavnet
import utils
gc.collect()

vlf_transmitters = {
                    "NPM":21.4e3, 
                    "NAA":24.0e3, 
                    "NLK":24.8e3, 
                    "NLM":25.2e3,
                    "NAU":40.75e3
                    }

# Gral params
LOC = sys.argv[2] # location PIU , PLO
sampling_freq = 50e3 # frecuencia de muestreo de la señal en la RedPitaya
Npts = 2**14  # signal length tamaño dela señal   
frames_per_file = 1  #signals per file
fft_npts = 2**13
fs = sampling_freq
MINSIZE_THRESHOLD = 80000
TOFFSET = 5 # hours
# a year folder as input
# path/to/yeardata/ with subfolder "PLO_XXX-XXX" -> data/
folder1 = sys.argv[1] # "/data/pavnet/2025/PIURA/09042025/" # contains subfolders
# each subfolder contains a day of data
resfolder = '/'+"/".join(folder1.split("/")[:-2]) # path for outputs)
subfolders = [sf for sf in os.listdir(folder1)]#!ls $folder1
print(
    f'''
    Data: {folder1}
    No. subfolders: {len(subfolders)}
    Output path: {resfolder}
    ''')
signal_len = Npts

method ="overlap" # window or overlap
povlp = 50 # %

# frequency range (band). For f0(central frequency)=20 and bw=20, the resulting spectral will span from 10 to 30 
f0 = 25e3
bw = 40e3

# no guardar espectros para salvar memoria
SAVE_SPECTRUMS = True #False
downsampling_factor = 6 # 6 samples per minute

# trim signal length
# 2**14 ~ 16k
# 2**13 ~ 8k
# 2**12 ~ 4k
order_hpf = 101
order_lpf = 101
cutf_amp = 0.01
# define filters
hpf = signal.firwin(order_hpf, cutoff=15e3, window="flattop", fs=fs, pass_zero="highpass")
lpf_zero = signal.firwin(order_lpf, cutoff=100, fs=fs, pass_zero="lowpass")
cutoff = 100  # Hz (ancho de banda útil)
numtaps = 101
lpf = firwin(numtaps, cutoff=cutoff, fs=fs)

# --- processing loop ---
time_arr = []

amplitudes = {tx:[] for tx in vlf_transmitters.keys()}
freq_arr = np.arange(fft_npts)*sampling_freq/fft_npts
freq_arr_all = np.arange(signal_len)*sampling_freq/signal_len

ftx_indexhood = []
bw = 200 

hpf = signal.firwin(51, cutoff=12e3, window="hamming", fs=sampling_freq)
N = 1
ti = time.time()
for call, f in vlf_transmitters.items():
    i = np.argmin(abs(freq_arr-f))
    id_0 = np.argmin(abs(freq_arr-(f-bw/2)))
    id_f = np.argmin(abs(freq_arr-(f+bw/2)))
    ftx_indexhood.append([id_0, id_f])
fc = 24e3
t = np.arange(signal_len)/fs
data = pd.DataFrame()

def fft_method(st):
    #return fft_pavnet.fft_window(st, wlen=fft_npts,fw=signal.windows.flattop)
    return fft_pavnet.fft_overlap(st, fft_npts=fft_npts,window=signal.windows.flattop)

for subfolder in subfolders[:]:
    path_ = folder1+"/"+subfolder+"/" 
    print("Subfolder:", path_)
    if not os.path.isdir(path_): continue
    fnames = np.array(tools.get_fnames(path_, fmt=".tar.gz"))#
    #print(f"# Data Files: {len(fnames)}")
    
    #file sizes
    fsize = np.array(list(map(
                            lambda x:os.path.getsize("/".join([path_,x])), 
                            fnames
                            )))
    fnames = fnames[fsize>MINSIZE_THRESHOLD]
    #print(f"Removing :\n{fnames[fsize<MINSIZE_THRESHOLD]}")
    nfiles = len(fnames)
    #amplitudes = {tx:[] for tx in vlf_transmitters.keys()}
    for k, f in enumerate(fnames[:]):
        print("File {:60s}\t {:7d} of {:7d} processed \t -> {:3.2f} %".format(f,k+1, nfiles, round((k+1)*100/nfiles, 2) ),end="\r")
        try:
            content = utils.get_content(path_+"/"+f)
            st = pd.read_csv(strio(content.decode()), comment="#", header=None, sep=",").values
            [st[:,0] , st[:,1]] = utils.IQ_clipping_filter(st)
        
            iq = signal.filtfilt(hpf, 1, st[:,0]) +1j*signal.filtfilt(hpf, 1, st[:,1])
                
            # almaecnamos el tiempo en UTC : +5h
            time_arr.append(tools.get_date_from_fname(str(f)) + datetime.timedelta(hours=TOFFSET))
            #print("\napplying FFT {}".format(method))
            S = fft_method(iq) #fft_pavnet.fft_window(st, wlen=fft_npts,fw=signal.windows.flattop)

            for k, (tx_call, tx_f) in enumerate(vlf_transmitters.items()):
                id_0, id_f = ftx_indexhood[k]
                wf = freq_arr[id_0:id_f]
                amp_ = integrate.simpson(S[id_0:id_f], wf)/(wf[-1]-wf[0])
                #amp_2 = max(S[id_0:id_f])
                amplitudes[tx_call].append(amp_)
        except Exception as e:
            print(f"\nFile Err: {f} \n[!] {e}")
    print()
'''
    if kit=True:
        df = pd.DataFrame(amplitudes,index=time_arr)
        df.sort_index(inplace=True)
        data = pd.concat((data,df))
        plt.plot(df["NAA"])
        plt.show(block=False)
'''            
    #print("\n---") 

print(f"Total processing time: {time.time()-ti} s")


ampdf0 = pd.DataFrame(amplitudes,index=time_arr)
ampdf0.sort_index(inplace=True)

# filter for the narrowband
lowp = signal.firwin(101, cutf_amp, fs=1/10)
for i, tx in enumerate(ampdf0.keys()):
    ampdf0[f"{tx}-filt"] = signal.filtfilt(lowp, 1, ampdf0[tx].values)

ampdf0.to_csv(f"/{resfolder}/{LOC}_AMPLITUDE_{ampdf0.index[0].date()}-to-{ampdf0.index[-1].date()}.csv",index=True)

