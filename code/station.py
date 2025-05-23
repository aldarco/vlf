import utils
import fft_pavnet
import collections
import paramiko
import time
import os
import numpy as np 
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import firwin, lfilter
import scipy.integrate as integrate
import sys

class RPftp:
    def __init__(self, params):
        self.remote_path = params["remote_dir"]
        self.local_path = params["local_dir"]
        self.transport = paramiko.Transport((params["hostname"], 
                                        params["port"]
                                        ))
        self.transport.connect(username=params["username"], 
                               password=params["password"]
                               )
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)
        print(" > SFTP is up")
        self.buffer_size = 20
        self.buffer = collections.deque(maxlen=self.buffer_size)
        self.waitfornewer()

    def available_data(self):
        '''
        looks for available data from /mnt/ramdisk in remote RP
        '''
        available_files = [f for f in self.sftp.listdir(self.remote_path) if f not in self.buffer]
        if not available_files:
            time.sleep(1)
            return self.available_data()
        
        return self.download_files(available_files)

    def download_files(self, files):
        #available_files = 
        data = {"time":[], "signal":[]}
        for fname in files:
            #available_data.append(self.get_remotedata(fname))
            t, s = self.get_remotedata(fname)
            if s is None: continue
            data["time"].append(t)
            data["signal"].append(s)
        return data

    def get_remotedata(self, fname):
        iq = None
        t = utils.get_dt_fname_v3(fname)
        try:
            remote_file = self.remote_path+"/"+fname
            local_file = self.local_path+"/"+fname
            #self.sftp.get(self.remote_path, self.local_path) #downloads directri to local
            with self.sftp.open(remote_file,"+rb") as remf:
                #iq = utils.IQ_frombin(remf)
                iq = utils.read_binary_IQ(remf).T
                remf.write(local_file)
            print(f"Opened remote: {remote_file}.")     
        except Exception as e:
            print(f"[!] Exception: {e}. Lost datafile: {remote_file}.\n")
            #sys.exit(0)
        
        return t, iq
    
    def waitfornewer(self):
        current = self.sftp.listdir(self.remote_path)
        print(current)
        while current == self.sftp.listdir(self.remote_path):
            time.sleep(1)
        
                


## processing funtion for a file
class dsp:
    def __init__(self, config):
        self.sampling_freq = config["sampling_frequency"]
        self.fft_npts = config["fft_npts"]
        self.method = config["method"]
        self.povlp = config["overlap_percent"]
        self.vlf_transmitters = config["vlf_transmitters"]
        self.LOC = config["location"]
        self.TOFFSET = config["ut_timeoffset"]
        self.MINSIZE_THRESHOLD = config["min_filesize"]
        self.freq_arr = np.arange(self.fft_npts)*self.sampling_freq/self.fft_npts
        self.ftx_indexrange = []
        self.bw = 150
        
        for call, f in self.vlf_transmitters.items():
            i = np.argmin(abs(self.freq_arr-f))
            id_0 = np.argmin(abs(self.freq_arr-(f-self.bw/2)))
            id_f = np.argmin(abs(self.freq_arr-(f+self.bw/2)))
            self.ftx_indexrange.append([id_0, id_f])
        
        self.hpf = signal.firwin(51, cutoff=12e3, window="hamming", fs=self.sampling_freq)
        print("DSP Set Ok")
    
    def get_amplitudes(self, x):
        '''
        x: signal in time
        hpf: high pass filter

        returns amplitudes of each Tx by using the spectrum of a signal x
        '''
        #print(x)
        #print("signal shape", x.shape)
        
        # file = strio(content.decode())
        #st = pd.read_csv(, comment="#", header=None, sep=",").values
        [x[:,0] , x[:,1]] = self.IQ_clipping_filter(x)

        iq = signal.filtfilt(self.hpf, 1, x[:,0]) +1j*signal.filtfilt(self.hpf, 1, x[:,1])
                
        # local universal time : localt +5h
        #t = utils.get_dt_fname_v3(str(f)) + datetime.timedelta(hours=self.TOFFSET))
        #print("\napplying FFT {}".format(method))
        S = self.fft_method(iq, fft_npts=self.fft_npts) #fft_pavnet.fft_window(st, wlen=fft_npts,fw=signal.windows.flattop)
        amplitudes = {f"{tx}":[] for tx in self.vlf_transmitters.keys()}
        #print(amplitudes)
        for k, (tx_call, tx_f) in enumerate(self.vlf_transmitters.items()):
            id_0, id_f = self.ftx_indexrange[k]
            wf = self.freq_arr[id_0:id_f]
            amp_ = integrate.simpson(S[id_0:id_f], wf)/(wf[-1]-wf[0])
            #amp_2 = max(S[id_0:id_f])
            #print(amp_, type(amp_))
            amplitudes[tx_call].append(amp_)
        return amplitudes , S
    
    def IQ_clipping_filter(self, iq, nstd=None):
        # clip_min, clip_max = np, 1.0
        # iq : array, shape (N, 2)
        if not nstd: nstd = 3
        ithr = nstd * np.std(iq[:, 0])
        qthr = nstd * np.std(iq[:, 1])
        I_clipped = np.clip(iq[:,0], -ithr, ithr)
        Q_clipped = np.clip(iq[:,1], -qthr, qthr)
        return I_clipped, Q_clipped

    def fft_method(self, st, fft_npts=2**13, wf=np.blackman):
        #return fft_pavnet.fft_window(st, wlen=fft_npts,fw=signal.windows.flattop)
        return fft_pavnet.fft_overlap(st, fft_npts=fft_npts,window=wf)

    def get_location(self):
        return self.LOC    
        
class sender:
    def __init__(self):
        pass

if __name__ == "__main__":
    # test
    # Connection details
    hostname = '10.42.0.42'
    port = 22
    username = 'root'
    password = 'escondido'
    remote_dir = '/mnt/ramdisk/'
    local_dir = '/home/aldo/jupyter/test_ssh_download/'

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Connect via SFTP
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # List and download all files
    for filename in sftp.listdir(remote_dir):
        remote_file = remote_dir + filename
        local_file = os.path.join(local_dir, filename)
        #sftp.get(remote_file, local_file)
        sftp.open
        print(f'Downloaded: {filename}')

    sftp.close()
    transport.close()