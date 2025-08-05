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


subject = 3
if subject == 3:
    data_folder = "data_003"
elif subject == 2:
    data_folder = "data_002"

df = pd.read_excel(f"files_useful.xlsx").query("Subject == @subject")

file_xml = df["Filename1"].iloc[4]
# 0 left is pretty much without artifacts, right can be cleaned (to some extent...)
# 1 has only artifacts

# file_xml = 'OCD001_2019_01_10_17_07_49__MR_0.xml'
# file_xml = "ADBS03_2019_04_15_13_45_52__MR_0.xml"

file = file_xml.replace(".xml", ".txt")
f_path = None
f_path_xml = None
for f in list(Path(data_folder).rglob("*.txt")):
    if file == f.name:
        f_path = f
        f_path_xml = f.with_suffix(".xml")
        break
data = pd.read_csv(f_path, header=None)
meta_data = xmltodict.parse(open(f_path_xml, 'rb').read())
fs_str = meta_data['RecordingItem']['SenseChannelConfig']['TDSampleRate']
fs = int(fs_str.split(" ")[0])

data = data.loc[:, (data != 0).any(axis=0)]
data.columns = ['ch_left', 'ch_right']
time = data.index / fs
data['time'] = time
# get the middle 3 min of recording
start_idx = int(len(data) / 2 - 90 * fs)
end_idx = int(len(data) / 2 + 90 * fs)

data_ = data.iloc[start_idx:end_idx]
raw = mne.io.RawArray(data_[['ch_left', 'ch_right']].values.T, mne.create_info(['ch_left', 'ch_right'], sfreq=fs, ch_types='eeg'))
raw.plot_psd()
plt.show(block=True)

raw.plot(block=True, scalings={"eeg": 10})


## use of PARRM
from pyparrm import get_example_data_paths, PARRM
from pyparrm._utils._power import compute_psd

parrm = PARRM(
    data=np.expand_dims(data_["ch_left"].values, axis=0), 
    sampling_freq=200,
    artefact_freq=150,
    verbose=False,
)

parrm.find_period()

parrm.create_filter(period_half_width=0.02, filter_half_width=5000)

filtered_ecog = parrm.filter_data()

parrm.explore_filter_params()

raw_f_parrm = mne.io.RawArray(filtered_ecog, mne.create_info(['ch_right'], sfreq=200, ch_types='eeg'))
raw_f_parrm.plot(block=True, scalings={"eeg": 10})
raw_f_parrm.plot_psd()


start_seconds = 130
end_seconds = -1
if end_seconds == -1:
    end_seconds = len(data_) / fs
start_idx = int(start_seconds * fs)
end_idx = int(end_seconds * fs)
data = data_.iloc[start_idx:end_idx].reset_index(drop=True)
data["time"] = data.index / fs

data_ = data.iloc[start_idx:end_idx]
raw = mne.io.RawArray(data_[['ch_left', 'ch_right']].values.T, mne.create_info(['ch_left', 'ch_right'], sfreq=fs, ch_types='eeg'))
raw.filter(h_freq=None, l_freq=10)


data_filt = raw.get_data()
ch_right = data_filt[1, :]
#plt.plot(ch_right, label='ch_right')
#plt.show(block=True)

peak_pos, peak_height = signal.find_peaks(ch_right, height=0.02,)
#plt.figure()
#plt.plot(ch_right, label='ch_right')
#plt.plot(peak_pos, peak_height['peak_heights'], 'x', label='peaks')
#plt.legend()
#plt.show(block=True)

dist_mean = int(np.diff(peak_pos).mean())

ch_right_orig = data['ch_right'].values

avg_ = []
for i in range(len(peak_pos) - 1):
    center = peak_pos[i]
    pre = center - dist_mean // 2
    post = center + dist_mean // 2
    if pre < 0 or post >= len(ch_right_orig):
        continue
    avg_.append(ch_right_orig[pre:post])

avg_arr = np.array(avg_).mean(axis=0)

data_clean = ch_right_orig.copy()
for i in range(len(avg_)):
    peak_pos_i = peak_pos[i]
    data_clean[peak_pos_i - dist_mean // 2:peak_pos_i + dist_mean // 2] -= avg_arr

plt.figure()
plt.plot(ch_right_orig, label='ch_right_orig')
plt.plot(data_clean, label='ch_right_clean')
plt.legend()
plt.show(block=True)

f, Psd_clean = scipy.signal.welch(data_clean[avg_arr.shape[0]:-avg_arr.shape[0]], fs, nperseg=fs*2)
f, Psd_orig = scipy.signal.welch(ch_right_orig[avg_arr.shape[0]:-avg_arr.shape[0]], fs, nperseg=fs*2)
plt.figure()
plt.plot(f, np.log10(Psd_clean), label='ch_right_clean')
plt.plot(f, np.log10(Psd_orig), label='ch_right_orig')
plt.legend()
plt.show(block=True)


plt.figure()
for i in range(len(avg_)):
    plt.plot(avg_[i], color='gray', alpha=0.1)
plt.plot(avg_arr, label='average', color='red')
plt.legend()
plt.show(block=True)

ch_left = data['ch_right'].values
plt.plot(data['time'], ch_left, label='ch_right')
plt.show(block=True)

f, Psd_left = scipy.signal.welch(data['ch_left'].values, fs, nperseg=fs*2)
f, Psd_right = scipy.signal.welch(data['ch_right'].values, fs, nperseg=fs*2)
plt.figure()
plt.plot(f, np.log10(Psd_left), label='ch_left')
plt.plot(f, np.log10(Psd_right), label='ch_right')
plt.legend()
plt.show(block=True)


raw.plot(block=True)
plt.plot(data['time'], ch_left, label='ch_right')
plt.show(block=True)



file_path = "/Users/Timon/Documents/Houston/resting_state_OCD/data_002/2018-11-08-selected/SensingProgrammerFiles/Tasks Files/OCD001_2018_11_08_13_26_57__MR_0.txt"


# remove columsn that are only 0
raw = mne.io.RawArray(data[['ch_left', 'ch_right']].values.T, mne.create_info(['ch_left', 'ch_right'], sfreq=fs, ch_types='eeg'))
raw.plot(block=True)
raw.plot_psd()
plt.show(block=True)

#