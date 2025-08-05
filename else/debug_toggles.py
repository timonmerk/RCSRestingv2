import pandas as pd
import mne
from matplotlib import pyplot as plt
import pandas as pd
import os
from pathlib import Path
import xmltodict
from scipy import signal, stats
import scipy
import numpy as np

raw_prt = mne.io.read_raw_brainvision("/Users/Timon/Downloads/PRT1.vhdr")
raw_prt.pick("Ref_Scalp")
raw_prt.resample(250)
#data = raw_prt.get_data()[:, :250 * 100] # 250 * 100
data = raw_prt.get_data()[:, 250 * 1407-70*250: 250 * 1437] # 250 * 100

# start: 42 seconds: end 42 + 1427 = 1469

raw_new = mne.io.RawArray(data, raw_prt.info)
# notch filter 
raw_new.notch_filter(np.arange(60, 125, 60), filter_length="auto", phase="zero-double", notch_widths=5)
raw_new.notch_filter(np.arange(60, 125, 60), filter_length="auto", phase="zero-double", notch_widths=5)
raw_new.notch_filter(np.arange(60, 125, 60), filter_length="auto", phase="zero-double", notch_widths=5)
# low pass filter at 10 Hz
raw_new.filter(l_freq=10, h_freq=None, filter_length="auto", phase="zero-double")

# show a time frequency plot
data_ = raw_new.get_data()
times = raw_new.times

f, t, Zxx = signal.stft(data_[0], fs=250, nperseg=128, noverlap=5, nfft=500)
plt.pcolormesh(t, f, np.log(np.abs(Zxx)), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('STFT Magnitude')
#plt.clim(-5, 0)
#plt.clim(-15, -8)
#plt.clim(-15, -12)   # File 1
plt.clim(-15, -9)   # File 2
plt.colorbar(label='Magnitude')
plt.show(block=True)


raw_new.plot(block=True, scalings={"eeg": 1})
raw_new.plot_psd()


raw_prt.plot(duration=10, scalings={"eeg": 1}, block=True)
# clip
 # clip to 2 hours
raw_prt.plot_psd()
plt.show(block=True)

