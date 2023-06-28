
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

import physio

import seaborn as sns

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n1bis_prep_info import *

debug = False





################################
######## LOAD DATA ########
################################

#sujet, session_i = 'PD01', 1
def open_raw_data_session(sujet, session_i):

    #### open raw and adjust for sujet
    os.chdir(os.path.join(path_data, 'eeg'))

    sujet_eeg_open = sujet[-2:] + sujet[:-2]

    if sujet_eeg_open == 'NT28' and session_i == 0:

        raw = mne.io.read_raw_brainvision(f'{sujet_eeg_open}_ses0{session_i+2}.vhdr', preload=True)
        raw_2 = mne.io.read_raw_brainvision(f'{sujet_eeg_open}_ses0{session_i+2}_2.vhdr', preload=True)

        raw = mne.concatenate_raws([raw, raw_2])

    elif sujet_eeg_open == 'AR30' and session_i == 2:

        raw = mne.io.read_raw_brainvision(f'{sujet_eeg_open}_ses0{session_i+2}.vhdr', preload=True)
        srate = int(raw.info['sfreq'])
        raw.crop(tmin=1076000/srate, tmax=None)

    else:

        raw = mne.io.read_raw_brainvision(f'{sujet_eeg_open}_ses0{session_i+2}.vhdr', preload=True)

    srate = int(raw.info['sfreq'])

    if srate != 500:
        raise ValueError(f'#### WARNING : {sujet_eeg_open} srate != 500 ####')

    #### Data vizualisation
    if debug == True :
        duration = 4.
        n_chan = 20
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    #raw.info['ch_names'] # verify

    #### identify EOG and rename chans
    # raw[raw.info['ch_names'].index('36'), :] = raw.get_data()[raw.info['ch_names'].index('Fp1'), :]
    # mne.rename_channels(raw.info, {'36' : 'VEOG'})

    if debug:
        data_plot = raw[raw.info['ch_names'].index('Fp1'), :][0][0,:int(1e10)]
        plt.plot(data_plot)
        plt.show()
    
    #### select raw_eeg
    raw_eeg = raw.copy()
    drop_chan = ['PRESS','ECG','TRIG']
    raw_eeg.info['ch_names']
    raw_eeg.drop_channels(drop_chan)

    #### select aux chan
    raw_aux = raw.copy()
    select_chan = ['PRESS','ECG','TRIG']
    raw_aux = raw_aux.pick_channels(select_chan)

    if debug:
        plt.plot(zscore(raw_aux.get_data()[0,:]), label='PRESS')
        plt.plot(zscore(raw_aux.get_data()[1,:]), label='ECG')
        plt.plot(zscore(raw_aux.get_data()[2,:]), label='TRIG')
        plt.legend()
        plt.show()

    #### extract trig
    trig_sig = raw_aux.get_data()[-1,:]*-1
    peaks = scipy.signal.find_peaks(trig_sig, height=trig_sig.max()/2, distance=srate*60)[0]

    if debug:
        respi = raw_aux.get_data()[0,:]
        plt.plot(respi)
        plt.vlines(peaks, ymin=respi.min(), ymax=respi.max(), color='r')
        plt.show()

        #### tag block ending
        trig_list = [1,3,5,7]
        peaks[trig_list]

        plt.plot(respi)
        plt.vlines(peaks[trig_list], ymin=respi.min(), ymax=respi.max(), color='r')
        plt.show()

    #### generate trig
    trig = {}
    #cond = conditions[0]
    for cond in conditions:
        
        _stop = dict_trig_sujet[sujet][f'ses0{session_i+2}'][cond]
        _start = _stop - (srate*5*60)
        trig[cond] = np.array([_start, _stop])

        if debug:
            respi = raw_aux.get_data()[0,:]
            plt.plot(respi)
            plt.vlines([_start, _stop], ymin=respi.min(), ymax=respi.max(), color='r')
            plt.show()

    #### remove trig
    raw_aux.drop_channels(['TRIG'])

    #raw_eeg.info # verify
    #raw_aux.info # verify
    
    del raw

    return raw_eeg, raw_aux, trig, srate







################################
######## AUX PREPROC ########
################################

def ecg_detection(raw_aux):

    print('#### ECG DETECTION ####', flush=True)

    #### params
    data_aux = raw_aux.get_data()
    chan_list_aux = raw_aux.info['ch_names']
    srate = int(raw_aux.info['sfreq'])
    
    #### adjust ECG
    if sujet_ecg_adjust.get(sujet) == 'inverse':
        data_aux[1,:] = data_aux[1,:] * -1

    #### filtre
    ecg_clean = physio.preprocess(data_aux[1,:], srate, band=[5., 45.], ftype='bessel', order=5, normalize=True)

    ecg_events_time = physio.detect_peak(ecg_clean, srate, thresh=10, exclude_sweep_ms=4.0) # thresh = n MAD

    if debug:
        plt.plot(data_aux[1,:])
        plt.plot(ecg_clean)
        plt.vlines(ecg_events_time, ymin=ecg_clean.min(), ymax=ecg_clean.max(), color='r')
        plt.show()
    
    #### replace
    data_aux[1,:] = ecg_clean.copy()

    ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
    raw_aux = mne.io.RawArray(data_aux, info_aux)

    # raw_aux.notch_filter(50, picks='misc', verbose='critical')

    # ECG
    # event_id = 999
    # ch_name = 'ECG'
    # qrs_threshold = .5 #between o and 1
    # ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold, verbose='critical')
    # ecg_events_time = list(ecg_events[0][:,0])

    return raw_aux, ecg_events_time



def respi_preproc(raw_aux):

    raw_aux.info['ch_names']
    srate = raw_aux.info['sfreq']
    respi = raw_aux.get_data()[0,:]

    #### inspect Pxx
    if debug:
        plt.plot(np.arange(respi.shape[0])/srate, respi)
        plt.show()

        srate = raw_aux.info['sfreq']
        nwind = int(10*srate)
        nfft = nwind
        noverlap = np.round(nwind/2)
        hannw = scipy.signal.windows.hann(nwind)
        hzPxx, Pxx = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx, label='respi')
        plt.legend()
        plt.xlim(0,60)
        plt.show()

    #### invert if needed to have inspi down
    if sujet_respi_adjust[sujet] == 'inverse':
        respi *= -1

    #### filter respi   
    # fcutoff = 1.5
    # transw  = .2
    # order   = np.round( 7*srate/fcutoff )
    # if order%2==0:
    #     order += 1

    # shape   = [ 1,1,0,0 ]
    # frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]

    # filtkern = scipy.signal.firls(order,frex,shape,fs=srate)

    # respi_filt = scipy.signal.filtfilt(filtkern,1,respi)

    #### filter respi physio
    respi_filt = physio.preprocess(respi, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    respi_filt = physio.smooth_signal(respi_filt, srate, win_shape='gaussian', sigma_ms=40.0)

    if debug:
        plt.plot(respi, label='respi')
        plt.plot(respi_filt, label='respi_filtered')
        plt.legend()
        plt.show()

        hzPxx, Pxx_pre = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        hzPxx, Pxx_post = scipy.signal.welch(respi_filt,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx_pre, label='pre')
        plt.semilogy(hzPxx, Pxx_post, label='post')
        plt.legend()
        plt.xlim(0,60)
        plt.show()


    #### replace respi 
    data = raw_aux.get_data()
    data[0,:] = respi_filt
    raw_aux._data = data

    #### verif
    #plt.plot(raw_aux.get_data()[0,:]),plt.show()

    return raw_aux











################################
######## COMPARISON ########
################################


# to compare during preprocessing
def compare_pre_post(raw, raw_post, nchan):

    # compare before after
    srate = raw.info['sfreq']
    nchan_i = chan_list_eeg.index(nchan)
    x_pre = raw.get_data()[nchan_i,:]
    x_post = raw_post.get_data()[nchan_i,:]
    time = np.arange(x_pre.shape[0]) / srate

    nwind = int(10*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx_pre = scipy.signal.welch(x_pre,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
    hzPxx, Pxx_post = scipy.signal.welch(x_post,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

    plt.plot(time, x_pre, label='x_pre')
    plt.plot(time, x_post, label='x_post')
    plt.title(nchan)
    plt.legend()
    plt.show()

    plt.semilogy(hzPxx, Pxx_pre, label='Pxx_pre')
    plt.semilogy(hzPxx, Pxx_post, label='Pxx_post')
    plt.title(nchan)
    plt.legend()
    # plt.xlim(60,360)
    plt.show()










################################
######## PREPROCESSING ########
################################


def reref_eeg(raw, new_ref):

    raw_eeg_reref = raw.copy()
    raw_eeg_reref, refdata = mne.set_eeg_reference(raw, ref_channels=new_ref)

    if debug == True :
        duration = 3.
        n_chan = 20
        raw_eeg_reref.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    raw = raw_eeg_reref.copy() # if reref ok

    return raw




def mean_centered(raw):
    
    data = raw.get_data()
    
    # mean centered
    data_mc = np.zeros((np.size(data,0),np.size(data,1)))
    for chan in range(np.size(data,0)):
        data_mc[chan,:] = data[chan,:] - np.mean(data[chan,:])
        #### no detrend to keep low derivation
        #data_mc[chan,:] = scipy.signal.detrend(data_mc[chan,:]) 

    # fill raw
    for chan in range(np.size(data,0)):
        raw[chan,:] = data_mc[chan,:]

    del data_mc    

    # verif
    if debug == True :
        # all data
        duration = .5
        n_chan = 10
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

        # compare before after
        compare_pre_post(raw, raw, 4)


    return raw




def line_noise_removing(raw):

    linenoise_freq = [50, 100, 150]

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    raw_post.notch_filter(linenoise_freq, verbose='critical')

    
    if debug == True :

        # compare before after
        compare_pre_post(raw, raw_post, 4)

    return raw_post





def high_pass(raw, h_freq, l_freq):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    #filter_length = int(srate*10) # give sec
    filter_length = 'auto'

    if debug == True :
        h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
        flim = (0.1, srate / 2.)
        mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

    raw_eeg_mc_hp = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2', verbose='critical')

    if debug == True :
        duration = 60.
        n_chan = 20
        raw_eeg_mc_hp.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    return raw_post






def low_pass(raw, h_freq, l_freq):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    filter_length = int(srate*10) # in samples

    if debug == True :
        h = mne.filter.create_filter(raw_post.get_data(), srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
        flim = (0.1, srate / 2.)
        mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

    raw_post = raw_post.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hann', fir_design='firwin2', verbose='critical')

    if debug == True :
        duration = .5
        n_chan = 10
        raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    return raw_post




def ICA_computation(raw):

    if debug == True :
        duration = .5
        n_chan = 10
        raw.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    # n_components = np.size(raw.get_data(),0) # if int, use only the first n_components PCA components to compute the ICA decomposition
    n_components = 15
    random_state = 27
    method = 'fastica'
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, method=method)

    reject = None
    decim = None
    # picks = mne.pick_types(raw.info, eeg=True, eog=True)
    picks = mne.pick_types(raw.info)
    ica.fit(raw)

    # for eeg signal
    ica.plot_sources(raw)
    ica.plot_components()
        
    # apply ICA
    raw_ICA = raw.copy()
    ica.apply(raw_ICA) # exclude component

    # verify
    if debug == True :

        # compare before after
        compare_pre_post(raw, raw_ICA, 0)

        duration = .5
        n_chan = 10
        raw_ICA.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    return raw_ICA





def average_reref(raw):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    raw_post.set_eeg_reference('average')

    if debug == True :
        duration = .5
        n_chan = 10
        raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    return raw_post




def csd_computation(raw):

    raw_csd = surface_laplacian(raw=raw, m=4, leg_order=50, smoothing=1e-5) # MXC way

    # compare before after
    # compare_pre_post(raw, raw_post, 4)

    return raw_csd



#raw, prep_step = raw_eeg, prep_step_wb
def preprocessing_ieeg(raw, prep_step):

    print('#### PREPROCESSING ####', flush=True)

    #### Execute preprocessing

    if debug:
        raw_init = raw.copy() # first data


    if prep_step['reref']['execute']:
        print('reref', flush=True)
        raw_post = reref_eeg(raw, prep_step['reref']['params'])
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['mean_centered']['execute']:
        print('mean_centered', flush=True)
        raw_post = mean_centered(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['line_noise_removing']['execute']:
        print('line_noise_removing', flush=True)
        raw_post = line_noise_removing(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['high_pass']['execute']:
        print('high_pass', flush=True)
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        raw_post = high_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['low_pass']['execute']:
        print('low_pass', flush=True)
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        raw_post = low_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post

    if prep_step['csd_computation']['execute']:
        print('csd_computation', flush=True)
        raw_post = csd_computation(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post

    if prep_step['ICA_computation']['execute']:
        print('ICA_computation', flush=True)
        raw_post = ICA_computation(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['average_reref']['execute']:
        print('average_reref', flush=True)
        raw_post = average_reref(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post

    #compare_pre_post(raw_init, raw, 5)

    return raw






################################
######## VIEWER ########
################################


def view_data(data):

    chan_selection_list = [['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1'],
                           ['Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']]
    
    for select_chanlist_i in range(2):

        chan_i_selected = [chan_list_eeg.index(chan) for chan in chan_selection_list[select_chanlist_i]]

        data_sel = data[chan_i_selected,:]

        trig = pd.read_excel(os.path.join(path_prep, sujet, 'info', f"{sujet}_{session_i}_trig.xlsx")).drop(columns=['Unnamed: 0'])

        #### downsample
        print('resample')
        srate_downsample = 50

        time_vec = np.linspace(0,data_sel.shape[-1],data_sel.shape[-1])/srate
        time_vec_resample = np.linspace(0,data_sel.shape[-1],int(data_sel.shape[-1] * (srate_downsample / srate)))/srate

        data_resampled = np.zeros((data_sel.shape[0], time_vec_resample.shape[0]))

        for chan_i in range(data_sel.shape[0]):
            f = scipy.interpolate.interp1d(time_vec, data_sel[chan_i,:], kind='quadratic', fill_value="extrapolate")
            data_resampled[chan_i,:] = f(time_vec_resample)

        trig_data = {'start' : [], 'stop' : []}

        for cond_i in conditions:
            _start = [i for i in trig.query(f"name == '{cond_i}'")['time'].values[0][1:-1].split(' ') if len(i) != 0]
            _start = int(_start[0])/srate
            _stop = [i for i in trig.query(f"name == '{cond_i}'")['time'].values[0][1:-1].split(' ') if len(i) != 0]
            _stop = int(_stop[1])/srate
            trig_data['start'].append(_start)
            trig_data['stop'].append(_stop)

        trig = pd.DataFrame(trig_data)

        print('plot')

        fig, ax = plt.subplots()

        for chan_i, chan in enumerate(chan_selection_list[select_chanlist_i]):
        
            x = data_resampled[chan_i,:]
            ax.plot(time_vec_resample, zscore(x)+(chan_i), label=chan)

        ax.vlines(trig['start'].values, ymin=zscore(data_resampled[0,:]).min(), ymax=(zscore(x)+(chan_i)).max(), colors='g', label='start')
        ax.vlines(trig['stop'].values, ymin=zscore(data_resampled[0,:]).min(), ymax=(zscore(x)+(chan_i)).max(), colors='r', label='stop')
        
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

        plt.show()


  





################################
######## CHOP & SAVE ########
################################

#band_preproc, export_info = 'wb', True
def chop_save_trc(raw_eeg, raw_aux, trig, ecg_events_time, band_preproc, session_i, export_info):

    print('#### SAVE ####', flush=True)
    
    #### save alldata + stim chan
    chan_list_eeg = raw_eeg.info['ch_names']
    chan_list_aux = raw_aux.info['ch_names']

    data_all = np.vstack(( raw_eeg.get_data(), raw_aux.get_data(), np.zeros(( raw_aux.get_data().shape[1] )) ))
    chan_list_all = chan_list_eeg + chan_list_aux + ['ECG_cR']

    ch_types = ['seeg'] * (len(chan_list_all)-4) + ['misc'] * 4
    srate = raw_eeg.info['sfreq'] 
    info = mne.create_info(chan_list_all, srate, ch_types=ch_types)
    raw_all = mne.io.RawArray(data_all, info)

    del data_all

    #### save chan_list
    os.chdir(os.path.join(path_anatomy, sujet))
    keep_plot_textfile = open(sujet + "_chanlist.txt", "w")
    for element in chan_list_all[:-4]:
        keep_plot_textfile.write(element + "\n")
    keep_plot_textfile.close()

    #### add cR events
    event_cR = np.zeros((len(ecg_events_time),3))
    for cR in range(len(ecg_events_time)):
        event_cR[cR, 0] = ecg_events_time[cR]
        event_cR[cR, 2] = 10

    raw_all.add_events(event_cR, stim_channel='ECG_cR', replace=True)
    raw_all.info['ch_names']

    #### prepare trig
    trig_df = pd.DataFrame({'name' : trig.keys(), 'time' : trig.values()}, columns=['name', 'time'])

    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    #### save all cond
    odor_code = odor_order[sujet][f'ses0{session_i+2}']
    raw_all.save(f'{sujet}_{odor_code}_allcond_{band_preproc}.fif')

    #### save every cond
    #cond = conditions_allsubjects[0]
    for cond in conditions:

        raw_chunk = raw_all.copy()
        raw_chunk.crop( tmin = trig_df.query(f"name == '{cond}'")['time'].values[0][0]/srate , tmax= trig_df.query(f"name == '{cond}'")['time'].values[0][1]/srate )

        raw_chunk.save(f'{sujet}_{odor_code}_{cond}_{band_preproc}.fif')

        del raw_chunk

        #### verif respi
        if debug:
            srate = raw_aux.info['sfreq']
            nwind = int(10*srate)
            nfft = nwind
            noverlap = np.round(nwind/2)
            hannw = scipy.signal.windows.hann(nwind)
            hzPxx, Pxx = scipy.signal.welch(raw_chunk._data[-2,:],fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
            plt.plot(hzPxx, Pxx, label='respi')
            plt.title(cond)
            plt.legend()
            plt.xlim(0,1)
            plt.show()

    if export_info == True :
    
        #### export trig, count_session, cR
        os.chdir(os.path.join(path_prep, sujet, 'info'))
        
        trig_df.to_excel(f'{sujet}_{session_i}_trig.xlsx')

        cR = pd.DataFrame(ecg_events_time, columns=['cR_time'])
        cR.to_excel(sujet +'_cR_time.xlsx')

    del raw_all

    return 





########################################
######## DETECT ARTIFACT ########
########################################



def detect_cross(sig, threshold):

    """
    Detect crossings
    ------
    inputs =
    - sig : numpy 1D array
    - show : plot figure showing rising zerox in red and decaying zerox in green (default = False)
    output =
    - pandas dataframe with index of rises and decays
    """

    rises, = np.where((sig[:-1] <=threshold) & (sig[1:] >threshold)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=threshold) & (sig[1:] <threshold)) # detect where sign inversion from + to -

    if rises.size != 0:

        if rises[0] > decays[0]: # first point detected has to be a rise
            decays = decays[1:] # so remove the first decay if is before first rise
        if rises[-1] > decays[-1]: # last point detected has to be a decay
            rises = rises[:-1] # so remove the last rise if is after last decay

        return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T
    
    else:

        return None
    



def compute_rms(x):

    """Fast root mean square."""
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i] ** 2
    ms /= n

    return np.sqrt(ms)




def sliding_rms(x, sf, window=0.5, step=0.2, interp=True):

    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = last
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > last))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf

    # Now loop over successive epochs
    for i in range(idx.size):
        out[i] = compute_rms(x[beg[i] : end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = scipy.interpolate.interp1d(t, out, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        t = np.arange(n) / sf
        out = f(t)

    return t, out

#sig = data
def iirfilt(sig, srate, lowcut=None, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0):

    if len(sig.shape) == 1:

        axis = 0

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    return filtered_sig



def med_mad(data, constant = 1.4826):

    median = np.median(data)
    mad = np.median(np.abs(data - median)) * constant

    return median , mad



def compute_artifact_features(inds, srate):

    artifacts = pd.DataFrame()
    artifacts['start_ind'] = inds['rises'].astype(int)
    artifacts['stop_ind'] = inds['decays'].astype(int)
    artifacts['start_t'] = artifacts['start_ind'] / srate
    artifacts['stop_t'] = artifacts['stop_ind'] / srate
    artifacts['duration'] = artifacts['stop_t'] - artifacts['start_t']

    return artifacts



#data = data[16,:]
def detect_movement_artifacts(data, srate, n_chan_artifacted=5, n_deviations=5, low_freq=40 , high_freq=150, wsize=1, step=0.2):
    
    eeg_filt = iirfilt(data, srate, low_freq, high_freq, ftype='bessel', order=2, axis=1)

    if len(eeg_filt.shape) != 1:

        masks = np.zeros((eeg_filt.shape), dtype='bool')

        for i in range(eeg_filt.shape[0]):

            print_advancement(i, eeg_filt.shape[0], steps=[25, 50, 75])

            sig_chan_filtered = eeg_filt[i,:]
            t, rms_chan = sliding_rms(sig_chan_filtered, sf=srate, window = wsize, step = step) 
            pos, dev = med_mad(rms_chan)
            detect_threshold = pos + n_deviations * dev
            masks[i,:] = rms_chan > detect_threshold

        compress_chans = masks.sum(axis = 0)
        inds = detect_cross(compress_chans, n_chan_artifacted+0.5)
        artifacts = compute_artifact_features(inds, srate)

    else:

        sig_chan_filtered = eeg_filt
        t, rms_chan = sliding_rms(sig_chan_filtered, sf=srate, window = wsize, step = step) 
        pos, dev = med_mad(rms_chan)
        detect_threshold = pos + n_deviations * dev
        masks = rms_chan > detect_threshold

        compress_chans = masks*1
        inds = detect_cross(masks, 0.5)
        artifacts = compute_artifact_features(inds, srate)

    return artifacts


# chan_artifacts = artifacts
def insert_noise(sig, srate, chan_artifacts, freq_min=30., margin_s=0.2, seed=None):

    sig_corrected = sig.copy()

    margin = int(srate * margin_s)
    up = np.linspace(0, 1, margin)
    down = np.linspace(1, 0, margin)
    
    noise_size = np.sum(chan_artifacts['stop_ind'].values - chan_artifacts['start_ind'].values) + 2 * margin * chan_artifacts.shape[0]
    
    # estimate psd sig
    freqs, spectrum = scipy.signal.welch(sig, nperseg=noise_size, nfft=noise_size, noverlap=0, scaling='spectrum', window='box', return_onesided=False, average='median')
    
    spectrum = np.sqrt(spectrum)
    
    # pregenerate long noise piece
    rng = np.random.RandomState(seed=seed)
    
    long_noise = rng.randn(noise_size)
    noise_F = np.fft.fft(long_noise)
    #long_noise = np.fft.ifft(np.abs(noise_F) * spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = np.fft.ifft(spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = long_noise.astype(sig.dtype)
    sos = scipy.signal.iirfilter(2, freq_min / (srate / 2), analog=False, btype='highpass', ftype='bessel', output='sos')
    long_noise = scipy.signal.sosfiltfilt(sos, long_noise, axis=0)
    
    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=0)
    rms_sig = np.median(filtered_sig**2)
    rms_noise = np.median(long_noise**2)
    factor = np.sqrt(rms_sig) / np.sqrt(rms_noise)
    long_noise *= factor
    
    noise_ind = 0

    for _, artifact in chan_artifacts.iterrows():

        ind0, ind1 = int(artifact['start_ind']), int(artifact['stop_ind'])
        
        n_samples = ind1 - ind0 + 2 * margin
        
        sig_corrected[ind0:ind1] = 0
        sig_corrected[ind0-margin:ind0] *= down
        sig_corrected[ind1:ind1+margin] *= up
        
        noise = long_noise[noise_ind: noise_ind + n_samples]
        noise_ind += n_samples
        
        noise += np.linspace(sig[ind0-1-margin], sig[ind1+1+margin], n_samples)
        noise[:margin] *= up
        noise[-margin:] *= down
        
        sig_corrected[ind0-margin:ind1+margin] += noise
        
    return sig_corrected




#raw = raw_preproc_wb 
def remove_artifacts(raw, srate, trig, odor_code):

    data = raw.get_data()

    #### detect on all chan
    print('#### ARTIFACT DETECTION ALLCHAN ####', flush=True)
    artifacts_raw = detect_movement_artifacts(data, srate, n_chan_artifacted=5, n_deviations=5, low_freq=40 , high_freq=150, wsize=1, step=0.2)
    artifacts_mask = np.zeros((artifacts_raw.shape[0]), dtype='bool')

    #### exclude artifact in intertrial on all chan
    #cond = 'FR_CV_1'
    for cond in conditions:

        start, stop = trig[cond][0], trig[cond][1]
        mask_include = (artifacts_raw['start_ind'].values >= start) & (artifacts_raw['stop_ind'].values <= stop)
        artifacts_mask = np.bitwise_or(artifacts_mask, mask_include)

    artifacts = artifacts_raw[artifacts_mask]

    #### add artifacts manualy if needed
    if len(dict_artifacts_sujet[sujet]['start']) != 0:

        artifact_add_manual_dict = {}
        for col in artifacts.columns:
            artifact_add_manual_dict[col] = []

        for artifact_i, artifact_val in enumerate(dict_artifacts_sujet[sujet]['start']):

            artifact_add_manual_dict['start_ind'].append(int(artifact_val))
            artifact_add_manual_dict['stop_ind'].append(int(dict_artifacts_sujet[sujet]['stop'][artifact_i]))
            artifact_add_manual_dict['start_t'].append(artifact_val/srate)
            artifact_add_manual_dict['stop_t'].append(dict_artifacts_sujet[sujet]['stop'][artifact_i]/srate)
            artifact_add_manual_dict['duration'].append(dict_artifacts_sujet[sujet]['stop'][artifact_i]/srate - artifact_val/srate)

        artifact_all_manual = pd.DataFrame(artifact_add_manual_dict)

        artifacts = pd.concat([artifacts, artifact_all_manual]).sort_values('start_ind')

    if artifacts.shape[0] == 0:

        return raw

    if debug:

        sig = data[1,:]
        plt.plot(sig)
        plt.vlines(artifacts['start_ind'].values, ymin=sig.min(), ymax=sig.max(), color='r', label='start')
        plt.vlines(artifacts['stop_ind'].values, ymin=sig.min(), ymax=sig.max(), color='g', label='stop')
        for cond in conditions:
            plt.vlines(trig[cond][0], ymin=sig.min(), ymax=sig.max(), color='k', label='start', linewidth=2)
            plt.vlines(trig[cond][1], ymin=sig.min(), ymax=sig.max(), color='k', linestyle='dashed', label='stop', linewidth=2)
        plt.legend()
        plt.show()

        sig_artifact = np.array([])
        sig_artifact_starts = np.array([])
        sig_append_i = 0
        min_max_artifacts = np.array([])
        len_artifacts = np.array([])

        for start, stop in zip(artifacts['start_ind'], artifacts['stop_ind']):

            sig_append = sig[start:stop] - sig[start:stop].mean()
            sig_artifact = np.append(sig_artifact, sig_append)
            sig_append_i += sig_append.shape[0]
            sig_artifact_starts = np.append(sig_artifact_starts, sig_append_i)
            min_max_artifacts = np.append(min_max_artifacts, np.abs(sig_append.max() - sig_append.min()))
            len_artifacts = np.append(len_artifacts, sig_artifact.shape[0])

        plt.plot(sig_artifact)
        plt.vlines(sig_artifact_starts, ymin=sig_artifact.min(), ymax=sig_artifact.max(), color='r')
        plt.show()

    #### correct on all chan
    print('#### ARTIFACT CORRECTION ALLCHAN ####', flush=True)
    data_corrected = data.copy()

    for chan_i, chan_name in enumerate(chan_list_eeg):

        data_corrected[chan_i,:] = insert_noise(data[chan_i,:], srate, artifacts, freq_min=30., margin_s=0.2, seed=None)

    if debug:

        chan_i = 0

        plt.plot(data[chan_i,:], label='raw')
        plt.plot(data_corrected[chan_i,:], label='corrected')
        plt.vlines(artifacts['start_ind'].values, ymin=sig.min(), ymax=sig.max(), color='r', label='start')
        plt.vlines(artifacts['stop_ind'].values, ymin=sig.min(), ymax=sig.max(), color='g', label='stop')
        for cond in conditions:
            plt.vlines(trig[cond][0], ymin=sig.min(), ymax=sig.max(), color='k', label='start', linewidth=2)
            plt.vlines(trig[cond][1], ymin=sig.min(), ymax=sig.max(), color='k', linestyle='dashed', label='stop', linewidth=2)
        plt.legend()
        plt.show()

        n_chan_plot = 5

        fig, ax = plt.subplots()

        for chan_i, chan_name in enumerate(chan_list_eeg[:n_chan_plot]):
        
            ax.plot(zscore(data[chan_i,:])+3*chan_i, label=f"raw : {chan_name}")
            ax.plot(zscore(data_corrected[chan_i,:])+3*chan_i, label=f"correct raw : {chan_name}")

        for cond in conditions:

            plt.vlines(trig[cond][0], ymin=0, ymax=3*chan_i, color='k', label='start', linewidth=2)
            plt.vlines(trig[cond][1], ymin=0, ymax=3*chan_i, color='k', linestyle='dashed', label='stop', linewidth=2)

        plt.vlines(artifacts['start_ind'].values, ymin=0, ymax=3*chan_i, color='r', label='start')
        plt.vlines(artifacts['stop_ind'].values, ymin=0, ymax=3*chan_i, color='r', label='stop')
        
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

        plt.show()

    #### clean idividual chan
    if sujet in segment_to_clean.keys() and odor_code in segment_to_clean[sujet].keys():

        print('#### ARTIFACT DETECTION SPECFIC CHAN ####', flush=True)

        #chan_to_correct = segment_to_clean[sujet][odor_code][0]
        for chan_to_correct in segment_to_clean[sujet][odor_code]:

            print(chan_to_correct, flush=True)

            chan_to_correct_i = chan_list_eeg.index(chan_to_correct)

            sig_to_correct = data_corrected[chan_to_correct_i,:]

            artifacts_raw = detect_movement_artifacts(sig_to_correct, srate, n_chan_artifacted=1, n_deviations=10, low_freq=40 , high_freq=150, wsize=1, step=0.2)
            artifacts_mask = np.zeros((artifacts_raw.shape[0]), dtype='bool')

            #### exclude artifact in intertrial in idividual chan
            #cond = 'FR_CV_1'
            for cond in conditions:

                start, stop = trig[cond][0], trig[cond][1]
                mask_include = (artifacts_raw['start_ind'].values >= start) & (artifacts_raw['stop_ind'].values <= stop)
                artifacts_mask = np.bitwise_or(artifacts_mask, mask_include)

            artifacts = artifacts_raw[artifacts_mask]

            if artifacts.shape[0] == 0:

                continue

            if debug:

                sig = sig_to_correct
                plt.plot(sig)
                plt.vlines(artifacts['start_ind'].values, ymin=sig.min(), ymax=sig.max(), color='r', label='start')
                plt.vlines(artifacts['stop_ind'].values, ymin=sig.min(), ymax=sig.max(), color='g', label='stop')
                for cond in conditions:
                    plt.vlines(trig[cond][0], ymin=sig.min(), ymax=sig.max(), color='k', label='start', linewidth=2)
                    plt.vlines(trig[cond][1], ymin=sig.min(), ymax=sig.max(), color='k', linestyle='dashed', label='stop', linewidth=2)
                plt.legend()
                plt.show()

            #### correct in idividual chan
            print('#### ARTIFACT CORRECTION SPECFIC CHAN ####', flush=True)
            sig_corrected = insert_noise(sig_to_correct, srate, artifacts, freq_min=30., margin_s=0.2, seed=None)

            if debug:

                sig = sig_to_correct
                plt.plot(sig)
                plt.plot(sig_corrected)
                plt.vlines(artifacts['start_ind'].values, ymin=sig.min(), ymax=sig.max(), color='r', label='start')
                plt.vlines(artifacts['stop_ind'].values, ymin=sig.min(), ymax=sig.max(), color='g', label='stop')
                for cond in conditions:
                    plt.vlines(trig[cond][0], ymin=sig.min(), ymax=sig.max(), color='k', label='start', linewidth=2)
                    plt.vlines(trig[cond][1], ymin=sig.min(), ymax=sig.max(), color='k', linestyle='dashed', label='stop', linewidth=2)
                plt.legend()
                plt.show()

            data_corrected[chan_to_correct_i,:] = sig_corrected

    #### inject corrected data
    raw_clean = raw.copy()
    for chan_i, chan_name in enumerate(chan_list_eeg):
        raw_clean[chan_i,:] = data_corrected[chan_i,:]

    del data_corrected, data, raw

    return raw_clean








################################
######## EXECUTE ########
################################


if __name__== '__main__':

    #sujet = sujet_list[4]
    for sujet in sujet_list:

        ########################################
        ######## CONSTRUCT ARBORESCENCE ########
        ########################################

        construct_token = generate_folder_structure(sujet)

        if construct_token != 0 :
            
            raise ValueError("""Folder structure has been generated 
            Lauch the script again for preproc""")

        ########################
        ######## PARAMS ########
        ########################

        # sujet = '01PD'
        # sujet = '02MJ'
        # sujet = '03VN'
        # sujet = '04GB'
        # sujet = '05LV'
        # sujet = '06EF'
        # sujet = '07PB'
        # sujet = '08DM'
        # sujet = '09TA'
        # sujet = '10BH'
        # sujet = '11FA'
        # sujet = '12BD'
        # sujet = '13FP'
        # sujet = '14MD'
        # sujet = '15LG'
        # sujet = '16GM'
        # sujet = '17JR'
        # sujet = '18SE'
        # sujet = '19TM'
        # sujet = '20TY'
        # sujet = '21ZV'
        # sujet = '22DI'
        # sujet = '23LF'
        # sujet = '24TJ'
        # sujet = '25DF'
        # sujet = '26MN'
        # sujet = '27BD'
        # sujet = '28NT'
        # sujet = '29SC'
        # sujet = '30AR'
        # sujet = '31HJ'
        # sujet = '32CM'
        # sujet = '33MA'

        # session_i = 0
        # session_i = 1
        # session_i = 2

        for session_i in range(3):

            #### pass if already computed
            odor_code = odor_order[sujet][f'ses0{session_i+2}']

            if os.path.exists(os.path.join(path_prep, sujet, 'sections', f'{sujet}_{odor_code}_allcond_wb.fif')):

                print(f"{sujet} {odor_code} ALREADY COMPTUED", flush=True)
                continue

            else:

                print(f'#### COMPUTE {sujet} {odor_code} ####', flush=True)

            ################################
            ######## EXTRACT DATA ########
            ################################

            raw_eeg, raw_aux, trig, srate = open_raw_data_session(sujet, session_i)
            #raw_eeg.info['ch_names'] # verify
            
            #### verif power
            if debug == True:
                mne.viz.plot_raw_psd(raw_eeg)

                view_data(raw_eeg.get_data())

            ################################
            ######## AUX PROCESSING ########
            ################################

            #### verif ecg and respi orientation
            if debug:
                _ecg = raw_aux.get_data()[-1]
                plt.plot(_ecg)
                plt.show()

                _respi = raw_aux.get_data()[0]
                plt.plot(_respi)
                plt.show()

            raw_aux, ecg_events_time = ecg_detection(raw_aux)

            raw_aux = respi_preproc(raw_aux)

            if debug == True:
                #### verif ECG
                chan_list_aux = raw_aux.info['ch_names']
                ecg_i = chan_list_aux.index('ECG')
                ecg = raw_aux.get_data()[ecg_i,:]
                plt.plot(ecg)
                plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
                trig_values = []
                for trig_i in trig.values():
                    [trig_values.append(i) for i in trig_i]
                plt.vlines(trig_values, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)

                plt.legend()
                plt.show()

                #### add events if necessary
                corrected = []
                cR_init = trig['time'].values
                ecg_events_corrected = np.hstack([cR_init, np.array(corrected)])

                #### find an event to remove
                around_to_find = 1000
                value_to_find = 3265670    
                ecg_cR_array = np.array(ecg_events_time) 
                ecg_cR_array[ ( np.array(ecg_events_time) >= (value_to_find - around_to_find) ) & ( np.array(ecg_events_time) <= (value_to_find + around_to_find) ) ] 

                #### verify add events
                plt.plot(ecg)
                plt.vlines(ecg_events_time, ymin=min(ecg), ymax=max(ecg), colors='k')
                plt.vlines(ecg_events_corrected, ymin=min(ecg), ymax=max(ecg), colors='r', linewidth=3)
                plt.legend()
                plt.show()

            ########################################################
            ######## PREPROCESSING & ARTIFACT CORRECTION ########
            ########################################################

            raw_preproc_wb = preprocessing_ieeg(raw_eeg, prep_step_wb)
            #compare_pre_post(raw_eeg, raw_preproc_wb, 'F4') # to verify

            if sujet in sujet_extra_ICA:

                raw_preproc_wb = ICA_computation(raw_preproc_wb)

            #view_data(raw_preproc_wb.get_data())

            raw_preproc_wb_clean = remove_artifacts(raw_preproc_wb, srate, trig, odor_code)
            #compare_pre_post(raw_preproc_wb, raw_preproc_wb_clean, 'F4') # to verify

            #view_data(raw_preproc_wb.get_data())

            ################################
            ######## CHOP AND SAVE ########
            ################################

            chop_save_trc(raw_preproc_wb_clean, raw_aux, trig, ecg_events_time, band_preproc='wb', session_i=session_i, export_info=True)

            del raw_preproc_wb

            #### verif
            if debug == True:
                compare_pre_post(raw_eeg, raw_preproc_wb, 0)




