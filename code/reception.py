import sys
import os
sys.path.append("./code") # must be in ./code

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or try 'TkAgg', 'QtAgg', etc.
import matplotlib.pyplot as plt
#from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import dates as mdates
import matplotlib.animation as animation
import time
#import datetime
#from io import StringIO as strio
import tarfile
import signal
import gc
import fft_pavnet
import utils
import station
import json
from  collections import deque
import threading

gc.collect()

with open('config_params.json', 'r') as f:
    config = json.load(f)
    #print(json.dumps(config, indent=4))

buffersize = 3600*24 # seconds in a day
#localpath = config["localdb"]
dspconfig = config["DSP"]
ftpconfig = config["RedPitaya"]
dbconfig = config["DataBase"]
Txs = dspconfig["vlf_transmitters"].keys()
ftp = station.RPftp(ftpconfig)
sproc = station.dsp(dspconfig)

data_buffer = {"DateTime": deque(maxlen=buffersize)}
for k in dspconfig["vlf_transmitters"].keys():
    data_buffer[k] = deque(maxlen=buffersize)
spectrum = np.zeros(sproc.fft_npts)
print("VLF TX", Txs)

def reception():
    lps = [] #line protocol strings
    while not stop_event.is_set():
        data = ftp.available_data()
        t_arr = data["time"]
        iq_arr = data["signal"]
        print("received",len(t_arr), "signals", len(iq_arr))
        savg = np.zeros(dspconfig["fft_npts"])
        t0 = time.perf_counter()
        for t, iq in zip(t_arr, iq_arr):
            
            amps_, spectrum  = sproc.get_amplitudes(iq)
            amps_["DateTime"] = t
            savg += spectrum
            #lps.append(
            #    "{meas}, "
                #)
            # send to DB
            # update buffer
            data_buffer["DateTime"].append(t)
            for k,name in enumerate( Txs):
                data_buffer[name].append(amps_[name])
            #print(f"{t} :"+"\t".join(amps_.values()))
            #print(amps_)
        tf = time.perf_counter()
        print("Avg Time spent processing: ", (tf-t0)/len(t_arr), "s / file  - - - ")
        savg /= len(t_arr)
        spectrum = savg
        #print(sproc.freq_arr.shape, savg.shape)
        #data_buffer["spectrum"] = savg # averge spectrum of the last set of data retrieved
    print("Saving buffer...")
    print("\n\n[!] Reception stopped. \n")
    save_buffer()
    print("All done.")

def update(it):
    if len(data_buffer["DateTime"])==0: return lines
    plotbuff = 1000 
    for  ax, line, name in zip(axs[1:], lines[1:], Txs):
        line.set_data(list(data_buffer["DateTime"])[-plotbuff:],
                               list(data_buffer[name])[-plotbuff:]
                               )
        ax.relim()
        ax.autoscale_view()
    lines[0].set_data(sproc.freq_arr, spectrum)
    axs[0].relim()
    axs[0].autoscale_view()
    
    return lines

def init():
    return lines

def handle_exit( sig=None, frame=None):
    print("\n[main] Shutting down...")
    stop_event.set()
    reception_thread.join()
    
    plt.close('all')
    sys.exit(0)

def save_buffer():
    tosave = {}
    for n in Txs:
        tosave[n] = list(data_buffer[n])
    import pandas as pd
    df = pd.DataFrame(tosave)
    df.to_csv(f'./lastbuffer_{max(data_buffer["DateTime"])}.csv')
    df = pd.DataFrame(list(spectrum), index=sproc.freq_arr)
    df.to_csv(f'./lastspectrum_{max(data_buffer["DateTime"])}.csv')


if __name__ == "__main__":
    stop_event = threading.Event()
    reception_thread = threading.Thread(target=reception, daemon=True)
    fig, axs = plt.subplots(len(Txs)+1,1,figsize=(10,10)) # +1row for spectrum
    reception_thread.start()    
    lines = [axs[0].plot([],[], color="blueviolet")[0]]
    for  ax, name in zip(axs[1:], Txs):
        lines.append(ax.plot([], [], marker="o", lw=0,color="firebrick", markerfacecolor="none")[0])
        ax.set_ylabel(name)
        
    ani = animation.FuncAnimation(fig, update, init_func=init, interval=1000,blit=True)
    signal.signal(signal.SIGINT, handle_exit)     # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)    # kill
    plt.show()
    handle_exit(reception_thread)
   
    
##################################################################
