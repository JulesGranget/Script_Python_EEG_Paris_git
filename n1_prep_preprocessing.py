
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n1bis_prep_trigger_info import *

debug = False





################################
######## LOAD DATA ########
################################

#sujet, session_i = 'PD01', 1
def open_raw_data_session(sujet, session_i):

    #### open raw
    os.chdir(os.path.join(path_data, sujet, f'ses0{str(session_i+1)}'))

    raw = mne.io.read_raw_brainvision(f'{sujet}_ses0{session_i+1}.vhdr', preload=True)

    srate = int(raw.info['sfreq'])

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

    #### generate triggers
    if debug:
        plt.plot(zscore(raw_aux.get_data()[0,:]), label='PRESS')
        plt.plot(zscore(raw_aux.get_data()[2,:]), label='TRIG')
        plt.legend()
        plt.show()

    trig = dict_trig_sujet[sujet][f'ses0{session_i+1}']

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

    print('#### ECG DETECTION ####')

    #### params
    data_aux = raw_aux.get_data()
    chan_list_aux = raw_aux.info['ch_names']
    srate = int(raw_aux.info['sfreq'])
    
    #### adjust ECG
    if sujet_ecg_adjust.get(sujet) == 'inverse':
        data_aux[0,:] = data_aux[0,:] * -1
    
    #### notch ECG
    ch_types = ['misc'] * (np.size(data_aux,0)) # ‘ecg’, ‘stim’, ‘eog’, ‘misc’, ‘seeg’, ‘eeg’

    info_aux = mne.create_info(chan_list_aux, srate, ch_types=ch_types)
    raw_aux = mne.io.RawArray(data_aux, info_aux)

    raw_aux.notch_filter(50, picks='misc', verbose='critical')

    # ECG
    event_id = 999
    ch_name = 'ECG'
    qrs_threshold = .5 #between o and 1
    ecg_events = mne.preprocessing.find_ecg_events(raw_aux, event_id=event_id, ch_name=ch_name, qrs_threshold=qrs_threshold, verbose='critical')
    ecg_events_time = list(ecg_events[0][:,0])

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

    #### filter respi   
    fcutoff = 1.5
    transw  = .2
    order   = np.round( 7*srate/fcutoff )
    if order%2==0:
        order += 1

    shape   = [ 1,1,0,0 ]
    frex    = [ 0, fcutoff, fcutoff+fcutoff*transw, srate/2 ]

    filtkern = scipy.signal.firls(order,frex,shape,fs=srate)

    respi_filt = scipy.signal.filtfilt(filtkern,1,respi)

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
    data[-1,:] = respi_filt
    raw_aux._data = data

    #plt.plot(raw_aux.get_data()[-1,:]),plt.show()

    return raw_aux











################################
######## COMPARISON ########
################################


# to compare during preprocessing
def compare_pre_post(raw, raw_post, nchan):

    # compare before after
    srate = raw.info['sfreq']
    x_pre = raw.get_data()[nchan,:]
    x_post = raw_post.get_data()[nchan,:]
    time = np.arange(x_pre.shape[0]) / srate

    nwind = int(10*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx_pre = scipy.signal.welch(x_pre,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
    hzPxx, Pxx_post = scipy.signal.welch(x_post,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

    plt.plot(time, x_pre, label='x_pre')
    plt.plot(time, x_post, label='x_post')
    plt.legend()
    plt.show()

    plt.semilogy(hzPxx, Pxx_pre, label='Pxx_pre')
    plt.semilogy(hzPxx, Pxx_post, label='Pxx_post')
    plt.legend()
    plt.xlim(60,360)
    plt.show()










################################
######## PREPROCESSING ########
################################

#raw, prep_step = raw_eeg, prep_step_wb
def preprocessing_ieeg(raw, prep_step):

    print('#### PREPROCESSING ####')

    #### 1. Initiate preprocessing step

    srate = raw.info['sfreq']

    def reref_eeg(raw, new_ref):

        raw.info['ch_names']
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

        # verify
        if debug == True :

            # compare before after
            compare_pre_post(raw, raw_post, 4)

        return raw_csd




    # 2. Execute preprocessing 

    if debug:
        raw_init = raw.copy() # first data


    if prep_step['reref']['execute']:
        print('reref')
        raw_post = reref_eeg(raw, prep_step['reref']['params'])
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['mean_centered']['execute']:
        print('mean_centered')
        raw_post = mean_centered(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['line_noise_removing']['execute']:
        print('line_noise_removing')
        raw_post = line_noise_removing(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['high_pass']['execute']:
        print('high_pass')
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        raw_post = high_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['low_pass']['execute']:
        print('low_pass')
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        raw_post = low_pass(raw, h_freq, l_freq)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['ICA_computation']['execute']:
        print('ICA_computation')
        raw_post = ICA_computation(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['average_reref']['execute']:
        print('average_reref')
        raw_post = average_reref(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post


    if prep_step['csd_computation']['execute']:
        print('csd_computation')
        raw_post = csd_computation(raw)
        #compare_pre_post(raw.get_data(), raw_post.get_data(), 5)
        raw = raw_post

    #compare_pre_post(raw_init, raw, 5)

    return raw









################################
######## CHOP & SAVE ########
################################

#raw_eeg, raw_aux, trig, ecg_events_time, band_preproc, session_i, export_info = raw_preproc_wb, raw_aux, trig, ecg_events_time, 'wb', session_i, True
def chop_save_trc(raw_eeg, raw_aux, trig, ecg_events_time, band_preproc, session_i, export_info):

    print('#### SAVE ####')
    
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
    keep_plot_textfile = open(sujet + "_chanlist_ieeg.txt", "w")
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
    odor_code = odor_order[sujet][f'ses0{session_i}']
    raw_all.save(f'{sujet}_{odor_code}_allcond_{band_preproc}.fif')

    #### save every cond
    #cond = conditions_allsubjects[1]
    for cond in conditions_allsubjects:

        raw_chunk = raw_all.copy()
        raw_chunk.crop( tmin = (trig_df['time'][trig_df['name'] == f'{cond}_start'].values[0])/srate , tmax= (trig_df['time'][trig_df['name'] == f'{cond}_stop'].values[0])/srate )

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
        
        trig_df.to_excel(sujet + '_trig.xlsx')

        cR = pd.DataFrame(ecg_events_time, columns=['cR_time'])
        cR.to_excel(sujet +'_cR_time.xlsx')

    del raw_all

    return 











################################
######## EXECUTE ########
################################


if __name__== '__main__':


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


    # sujet, session_i = 'Pilote', 1
    sujet, session_i = 'PD01', 1




    ################################
    ######## EXTRACT DATA ########
    ################################

    raw_eeg, raw_aux, trig, srate = open_raw_data_session(sujet, session_i)
    #raw_eeg.info['ch_names'] # verify
    
    #### verif power
    if debug == True:
        mne.viz.plot_raw_psd(raw_eeg)

    ################################
    ######## AUX PROCESSING ########
    ################################

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

    ################################################
    ######## PREPROCESSING, CHOP AND SAVE ########
    ################################################

    raw_preproc_wb  = preprocessing_ieeg(raw_eeg, prep_step_wb)
    #compare_pre_post(raw_eeg, raw_preproc_whole_band, 5) # to verify

    chop_save_trc(raw_preproc_wb, raw_aux, trig, ecg_events_time, band_preproc='wb', session_i=session_i, export_info=True)


    del raw_preproc_wb

    #### verif
    if debug == True:
        compare_pre_post(raw_eeg, raw_preproc_wb, 0)




