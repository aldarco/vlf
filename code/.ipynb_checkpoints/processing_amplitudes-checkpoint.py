import station
import os
import json
from collections import deque
import numpy as np
import threading
import signal 
import time
import matplotlib; matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils
import sys

curr_dir = os.path.dirname(__file__)
with open(f'{curr_dir}/config_params.json', 'r') as f:
    config = json.load(f)
    #print(json.dumps(config, indent=4))

dspconfig = config["DSP"]
Txs = dspconfig["vlf_transmitters"].keys()
backup_path = None #"/data/pavnet/datafiles/"
ramdisk = "/run/user/1000/temprpdata/"
SLEEPT = 2 #seconds


def move_to_backup(path_files):
    # to do
    pass
buffersize = 1000
sproc = station.dsp(dspconfig)
data_buffer = {"DateTime": deque(maxlen=buffersize)}
for k in dspconfig["vlf_transmitters"].keys():
    data_buffer[k] = deque(maxlen=buffersize)
spectrum = np.zeros(sproc.fft_npts)

print("VLF TX", Txs)

def process_available_data():
    global spectrum 
    datafiles = [f for f in os.listdir(ramdisk) ]
    Savg = np.zeros(dspconfig["fft_npts"])
    iq_arr = [] ; t_arr = []
    txamp = {name:[] for name in Txs}
    ti = time.perf_counter()
    for f in datafiles:
        pathf = ramdisk+f"{f}"
        if not utils.is_file_stable(pathf): continue
        try: 
            with open(pathf, "+rb") as file:
                iq = utils.read_binary_IQ(file).T
                t = utils.get_dt_fname_v3(f)
                #print(iq)
            #iq_arr.append(   iq.T )#  to (N,2) shape
            #t_arr.append(t)
            #print(f"File: {f}", end="\r")
            amps_, S = sproc.get_amplitudes(iq)
            Savg += S
            data_buffer["DateTime"].append(t)
            for name in Txs:
                # appends to queue buffer
                data_buffer[name].append(amps_[name])
                # to do: store and/or send to db
            os.remove(pathf)
        except ValueError as e:
            print(f"ValueError {e}\nfile: {f}... retrying")
            time.sleep(0.25)
            try:
                with open(pathf, "+rb") as file:
                    iq = utils.read_binary_IQ(file).T
                    #print(iq.shape)

                    t = utils.get_dt_fname_v3(f)
             
                amps_, S = sproc.get_amplitudes(iq)
                Savg += S
                data_buffer["DateTime"].append(t)
                for name in Txs:
                    # appends to queue buffer
                    data_buffer[name].append(amps_[name])
                    # to do: store and/or send to db
                os.remove(pathf)
            except Exception as e:
                print("couldn't... lost {f}")
    if datafiles: 
        spectrum = Savg / len(datafiles) 
        tf = time.perf_counter()
        print("Nfiles:", len(datafiles),"Avg time spent : ", (tf-ti)/len(datafiles), "s / file  - - - ", end="\r")



def run_updater():
    global spectrum
    lps = [] #line protocol strings
    while not stop_event.is_set():
        #t0 = time.perf_counter()
        process_available_data()
        #tf = time.perf_counter()
        time.sleep(SLEEPT)
        
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
    processing_thread.join()

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
    #os.mkdir(ramdisk)
    stop_event = threading.Event()
    processing_thread = threading.Thread(target=run_updater, daemon=True)
    fig, axs = plt.subplots(len(Txs)+1,1,figsize=(10,10)) # +1row for spectrum
    processing_thread.start()    
    lines = [axs[0].plot([],[], color="blueviolet")[0]]
    for  ax, name in zip(axs[1:], Txs):
        lines.append(ax.plot([], [], marker="o", lw=0,color="firebrick", markerfacecolor="none")[0])
        ax.set_ylabel(name)
        
    ani = animation.FuncAnimation(fig, update, init_func=init, interval=1000,blit=True)
    signal.signal(signal.SIGINT, handle_exit)     # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)    # kill
    plt.show()
    handle_exit(processing_thread)
        