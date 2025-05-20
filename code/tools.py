'''
Modules to use in the analysis of data from PAVNET
 ---------------------- DEPRECATED ---------------------------

'''

import numpy as np
from scipy import signal
import datetime
from scipy.optimize import curve_fit
from scipy.stats import zscore
import pandas as pd
import time
import os


def segment_index(fv, fc, span):
    '''get the index for a segment with central frequency f0 and a span'''
    idx_min = np.argmin(abs(fv-(fc-span/2)))
    idx_max = np.argmin(abs(fv-(fc+span/2)))

    return idx_min, idx_max

def gaussian_smooth(x_data, y_data, sigma):
    '''Gaussian smooth at each point of the y_data. Smoothing kernel size depends on the sigma value'''
    smooth_y = np.zeros(len(y_data))
    i = 0
    for xi in x_data:
        kernel = np.exp(-(x_data - xi)**2/(2*sigma**2))
        kernel /= sum(kernel)
        smooth_y[i] = sum(y_data*kernel)
        i += 1
    return smooth_y

def get_date_from_fname(fnames):
    '''
    When a file contains a single signal it's simpler to get the datetime
    format name: ANTAR_RAW_<#>_[DAY]_[MONTH]_[YEAR]_[h]_[m]_[s]__0.txt
    '''
    def f(fname):
        [d,mo, y, h, m,s] = fname.split("_")[3:9]
        return datetime.datetime(int(y), int(mo),int(d), int(h), int(m), int(s))
    if type(fnames)==list:
        return list(map(f, fnames))
    elif type(fnames)==str:
        return f(fnames)
    else:
        return None

def get_datetimes(fname, nsegments):
    '''gert the datetime of the each dataframe in the datafile
        fname: filename 
        nsegments: number of data segments
    '''
    try:
        f = open(fname, 'r')
        data = f.readlines()
        f.close()
        dts = []
        #print(data)
        pts_per_seg = int(data[1].split(":")[1])
    except:
        print("ERROR AT ",fname)
    comment_rows = 0
    while data[comment_rows].startswith("#"):
        comment_rows += 1
    #nsegments = len(data)/
    #print(comment_rows, pts_per_seg)
    for i in np.arange(nsegments):
        #k = i*2**15 + i*4
        k = i*pts_per_seg + i*comment_rows 
        #print(data[k])
        d,mo,y,h,mi,s = list(map(float,data[k].rstrip().split(" ")[1:]))
        s, us = divmod(s,1)
        us = round(us*10e6,6)
        
        #dts.append(datetime.datetime.strptime(str(int(y))+' '+str(int(mo))+' '+str(int(d))+' ' + str(int(h))+':'+str(int(mi))+':'+str(int(s)), "%Y %m %d %H:%M:%S"))
        dts.append(datetime.datetime(int(y), int(mo),int(d), int(h), int(mi), int(s)))
        #print(dts[i])
        
    return dts
        
#def save_3cols([datetime, peak_freqs, amplitude], filename):
def datetime2hours(dt):
    time = dt.time()
    return time.hour + time.minute/60 + time.second/3600

def hours2hms(hours):
    h,m = divmod(hours,1)
    m,s = divmod(m*60,1)
    s = int(s*60)
    return list(map(int,[h,m,s]))

#def highpass_filter(signal,fs):
def index_of_datetime(dt_list, dt_target, tinterval=5,errfac=1):
    '''Returns the  closest element index in a list of a given datetime
        tinterval: minutes betweeen measurements. Default tinterval=5min
    ''' 
    datet = dt_target.date()
    hours = datetime2hours(dt_target)
    found = False
    
    for k, dtk in enumerate(dt_list):
        if dtk.date() == datet:
            if abs(datetime2hours(dtk)-hours)*60 < tinterval*errfac:
                found = True
                return k
            else:
                continue
    if not found:
        return None


def iqsignal_threshold_filter(data, threshold=0.01):
    '''
    limits peaks along the iq data signal
    data: (N,2) shape array, representing i,q columns data
    threshold: max amplitude signal value
    '''
    data = np.array(data)
    #threshold = 0.01
    n = len(data)
    data2_lim = np.zeros(data.shape)
    for i in np.arange(n):
        if abs(data[i, 0]) > threshold:
            data2_lim[i, 0] = np.sign(data[i,0])*threshold
        else:
            data2_lim[i, 0] = data[i,0]

        if abs(data[i, 1]) > threshold:
            data2_lim[i, 1] = np.sign(data[i,1])*threshold
        else:
            data2_lim[i, 1] = data[i,1]

    return data2_lim

def fft_overlaping50(iqdata, fft_pts):
    '''overlap and average the fft along a signal (overlaping 50%)'''
    xv = np.zeros(fft_pts)
    N = len(iqdata)
    n = int( 2*(N/fft_pts - 1))
    #print(n)
    for k in range(n+1):
        
        fft = abs(np.fft.fftshift(np.fft.fft(iqdata[int(k*fft_pts/2) : int((1+k/2)*fft_pts)]*np.hanning(fft_pts))))
        xv = xv + fft
    
    return xv/(n+1)
    

def simple_snr(xv):
    mean = np.mean(xv)
    std = np.std(xv)
    return 20*np.log10(abs(np.where(std == 0, 0, mean/std)))


def plot_spectrum_by_dt(dt, fnames, folder, Npts, frames_per_file, comment="#", delimiter=" ", start_col=2, end_col=4):
    '''Plot the spectrum at a given time (the closest one)
    dt represents the datetime
    fnames a list of filenames
    '''
    nflag = 0
    for file in fnames:
        dtv = tools.get_datetimes(folder+'/'+file, frames_per_file)
        data = read_csv(folder+'/'+file, comment=comment, delimiter=delimiter, header=None).to_numpy()[:, start_col:end_col]

        for k, dt in enumerate(dtv):
            #timev.append(dt)
            if nflag==1: 
                break
            if dt == tv1:
                print(dt)
                dframe = np.array((data[k*Npts: (k+1)*Npts, 0], data[k*Npts: (k+1)*Npts, 1])).T
                signal_high = dframe
                nflag+=1
            
        if nflag == 1:
            break
            

def fit_gaussian(x, y):
    npts = len(x)
    mean = sum(x*y)/sum(y)
    sigma = np.sqrt(abs(sum((x-mean)**2*y)/sum(y)))
    
    def gaussian(x, amp, x0, sigma, y0):
        return amp*np.exp(-(x-x0)**2/(2*sigma**2))+y0
    
    popy, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma, min(x)+(max(x)-min(y))/2])
    
    return x, gaussian(x,*popy)
    

def highsnr_window(signal, wlen, step):
    """
    gives the signal window with the highest SNR.
    signal: signal array
    wlen: window length
    step: increasing index step
    """
    idx1 = 0
    idx2 = wlen
    n = len(signal)
    snr0 = 0    
    i1 = i2 = 0
    
    while idx2<n:
        mean = np.mean(abs(signal[idx1:idx2]))
        std = np.std(abs(signal[idx1:idx2]))
        snr = np.where(std==0, 0, mean/std)
        #print(mean,std,snr)
        
        if snr0<snr:
            i1 = idx1
            i2 = idx2
            snr0 = snr
        else:
            pass
        idx1 += step
        idx2 += step
    return signal[i1:i2]

def lowstd_window(signal, wlen, step):
    """
    gives the signal window with the lowest standard deviation.
    signal: signal array
    wlen: window length
    step: increasing index step
    """
    idx1 = 0
    idx2 = wlen
    n = len(signal)
    std0 = None    
    i1 = i2 = 0
    
    while idx2<n:
        mean = np.mean(abs(signal[idx1:idx2]))
        std = np.std(abs(signal[idx1:idx2]))
        
        #print(mean,std,snr)
        
        if std0 == None:
            std0 = std
        elif std<std0:
            i1 = idx1
            i2 = idx2
            std0 = std
        else:
            pass
        idx1 += step
        idx2 += step
    if i2 == 0:
        i2 = wlen
    #print(i1, i2)
    return signal[i1: i2] 

def lowzscore_window(signal, wlen, step):
    """
    gives the signal window with the lowest z-score average.
    signal: signal array
    wlen: window length
    step: increasing index step
    """
    idx1 = 0
    idx2 = wlen
    n = len(signal)
    zs0 = None    
    i1 = i2 = 0
    
    while idx2<n:
        #mean = np.mean(abs(signal[idx1:idx2]))
        std = np.std(abs(signal[idx1:idx2]))
        zs = zscore(abs(signal[idx1:idx2])).mean() #np.where(std==0, 0, mean/std)
        #print(mean,std,snr)
        
        if zs0 == None:
            zs0 = zs
        elif zs<zs0:
            i1 = idx1
            i2 = idx2
            zs0 = zs
        else:
            pass
        idx1 += step
        idx2 += step
    if i2 == 0:
        i2 = wlen
    #print( signal[i1: i2] )
    return signal[i1: i2]

def get_fnames(folder, raw=True, fmt='.txt'):
    os.chdir(folder)

    fnames = []
    files_ = sorted(filter(os.path.isfile, os.listdir(".")), key=os.path.getmtime)
    for fn in files_:
        if not fn.endswith(fmt):
            continue
        if raw:
            if "ANTAR" in fn and "RAW" in fn: 
                fnames.append(fn)
        else:
            if "ANTAR" in fn: 
                fnames.append(fn)
    os.chdir("../") 

    #print("%d RAW files found"%(len(fnames)))
    return fnames

def folder2dframe_signal(fnames, folder, cols,  Npts, sampling_freq, frames_per_file, avoid_days,nnn):
    """
    Load data from all files (ANTAR files) to a single pandas dataframe.

    """
    i1, i2 = cols
    #avoid_days = False
    t0 = time.time()

    freqs = np.arange(Npts)*sampling_freq/Npts

    timev = [] #[None]*frames_per_file*len(fnames)
    signalv = []
    fftv = []


    total_iter = len(fnames)*frames_per_file
    current_iter = 1

    for fname in fnames:
        dts = get_datetimes(folder+'/'+fname, nsegments=frames_per_file)
        #data = np.loadtxt(folder+'/'+fname)
        data = pd.read_csv(folder+'/'+fname, comment="#", delimiter=" ", header=None).to_numpy()
        for k,dt in enumerate(dts):
            if avoid_days:
                if dt.date() in nnn:
                    
                    current_iter += 1
                    #print(">> Percent completed = %.1f // current datetime: %s"%(100*current_iter/total_iter, str(dt)), end="\r", flush=True)
                    continue
            
            dataframe = data[k*Npts: (k+1)*Npts]
            
            #signalv.append(np.sqrt(dataframe[:,2]**2 + dataframe[:,3]**2))
            signalv.append(dataframe[:,i1] + 1j*dataframe[:,i2])
            timev.append(dt)
        
            current_iter += 1
            
            #print(">> Percent completed = %.1f // current datetime: %s"%(100*current_iter/total_iter, str(dt)), end="\r", flush=True)

    del dataframe
    data_signal = pd.DataFrame(np.asarray(signalv).T, columns=timev, index=freqs)
    data_signal.columns = pd.to_datetime(data_signal.columns)
    del signalv

    return timev, data_signal

def folder2dframe_signal_2(fnames, folder, cols,  Npts, sampling_freq, frames_per_file, avoid_days,nnn):
    """
    Load data from all files (ANTAR files) to a single pandas dataframe.

    """
    i1, i2 = cols
    #avoid_days = False
    t0 = time.time()

    freqs = np.arange(Npts)*sampling_freq/Npts

    timev = [] #[None]*frames_per_file*len(fnames)
    signalv = []
    fftv = []


    total_iter = len(fnames)*frames_per_file
    current_iter = 1

    for fname in fnames:
        dts = get_datetimes(folder+'/'+fname, nsegments=frames_per_file)
        #data = np.loadtxt(folder+'/'+fname)
        data = pd.read_csv(folder+'/'+fname, comment="#", delimiter=" ", header=None).to_numpy()
        for k,dt in enumerate(dts):
            if avoid_days:
                if dt.date() in nnn:
                    
                    current_iter += 1
                    print(">> Percent completed = %.1f // current datetime: %s"%(100*current_iter/total_iter, str(dt)), end="\r", flush=True)
                    continue
            
            dataframe = data[k*Npts: (k+1)*Npts]
            
            #signalv.append(np.sqrt(dataframe[:,2]**2 + dataframe[:,3]**2))
            signalv.append(dataframe[:,i1] + 1j*dataframe[:,i2])
            timev.append(dt)
        
            current_iter += 1
            
            print(">> Percent completed = %.1f // current datetime: %s"%(100*current_iter/total_iter, str(dt)), end="\r", flush=True)

    del dataframe
    data_signal = pd.DataFrame(np.asarray(signalv).T, columns=timev, index=freqs)
    data_signal.columns = pd.to_datetime(data_signal.columns)
    del signalv

    return timev, data_signal

def nearest(items, pivot):
    x = min(items, key=lambda x: abs(x - pivot))
    return list(items).index(x)

def fft_window(signal, wlen=2**12,fw=np.hanning, fac=1, **kwargs):
    #wlen = 2**12
    #whanning = np.hanning(len(data_signal[date0]))
    return fac*fftshift(abs(fft(signal[:wlen]*fw(wlen,**kwargs))))