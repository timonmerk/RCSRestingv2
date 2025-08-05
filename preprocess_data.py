import mne

def preprocess_data(data_, fs):

    data_filtered = mne.filter.filter_data(
        data_,
        sfreq=fs,
        l_freq=65,
        h_freq=55,
        method='iir',
        verbose=False
    )
    data_filtered = mne.filter.filter_data(
        data_filtered,
        sfreq=fs,
        l_freq=0.5,
        h_freq=100,
        method='iir',
        verbose=False
    )

    data_filtered = mne.filter.filter_data(
        data_filtered,
        sfreq=fs,
        l_freq=None,
        h_freq=100,
        method='iir',
        verbose=False
    )

    data_filtered = mne.filter.filter_data(
        data_filtered,
        sfreq=fs,
        l_freq=None,
        h_freq=100,
        method='iir',
        verbose=False
    )

    return data_filtered