



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
import mne
import pandas as pd
import joblib
import xarray as xr

from n0_config import *
from n0bis_analysis_functions import *

debug = False





################################
######## LOAD DATA ########
################################


#### open raw
os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Data/Pilote')

raw = mne.io.read_raw_brainvision(f'Pilote_01.vhdr', preload=True)

#### Data vizualisation
if debug == True :
    duration = 4.
    n_chan = 20
    raw.plot(scalings='auto',duration=duration,n_channels=n_chan)# verify

    mne.viz.plot_raw_psd(raw)

#### get params
srate = int(raw.info['sfreq'])
chan_list = raw.info['ch_names']

#### select raw_eeg
raw_eeg = raw.copy()
drop_chan = ['EMG','PRESSION']
raw_eeg.info['ch_names']
raw_eeg.drop_channels(drop_chan)

#### select aux chan
raw_aux = raw.copy()
select_chan = ['EMG','PRESSION']
raw_aux = raw_aux.pick_channels(select_chan)

#### generate triggers
trig = pd.DataFrame(raw.annotations)
trig_time = (trig['onset'].values[1:] * raw.info['sfreq']).astype(int)

trig_names = []
for trig_i in trig['description'].values:
    if trig_i == 'New Segment/':
        continue
    else:
        trig_names.append(trig_i[10:].replace(' ', ''))

trig = {'time' : trig_time, 'name' : trig_names}

#raw_eeg.info # verify
#raw_aux.info # verify

#### implement montage
dict_chan_types = {}
for nchan in chan_list[:-2]:
    if nchan == 'EOG':
        dict_chan_types[nchan] = 'eog'
    else:
        dict_chan_types[nchan] = 'eeg'

raw_eeg.set_channel_types(dict_chan_types)
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
raw_eeg.set_montage(ten_twenty_montage)


#### verif and adjust trig for some patients
chan_name = 'PRESSION'
data_plot = raw_aux.get_data()
chan_list_plot = raw_aux.info['ch_names']
start = 0 * 60 * int(raw_aux.info['sfreq']) # give min
stop =  57 * 60 * int(raw_aux.info['sfreq']) # give min

chan_i = chan_list_plot.index(chan_name)
times = np.arange(np.size(data_plot,1))
trig_keep = (trig['time'] >= start) & (trig['time'] <= stop)
x = data_plot[chan_i,start:stop]*-1
time = times[start:stop]

plt.plot(time, x)
plt.vlines(trig['time'][trig_keep], ymin=np.min(x), ymax=np.max(x), colors='k')
plt.show()

#### adjust trig
adjust_trig = np.array([2640,   6063,   9702,  13149,  17024,  20603,  24404,
    27895,  31782,  35549,  39205,  42542,  46152,  49676,  53177,
    57032,  61178,  64796,  68287,  71653,  75061,  78324,  81694,
    85452,  88795,  92265,  95728,  99346, 103724, 107521, 111285,
    114273, 117833, 121375, 125391, 129424, 133314, 136761,
    139924, 143796, 147160, 151076, 154669, 158080, 160952, 163870,
    167124, 170553, 173490, 176560, 179890, 183015, 186187, 189125,
    192476, 195851, 199576, 203542, 207473, 210823, 213965, 217001,
    220164, 224159, 227951, 231101, 234260, 237681, 241326, 244689,
    247760, 250968, 254331, 257473, 261084, 264512, 267559, 271160,
    274718, 279585, 282933, 287051, 290555, 294204, 297683,
    301115, 304858, 308171, 311647, 315054, 318459, 322059, 325506,
    328994, 332839, 337098, 341062,
    349628, 353050, 356419, 359782, 363495, 366692, 369991, 373068,
    376452, 380313, 383578, 387209, 390278, 393419, 396962, 400199,
    403471, 406521, 409723, 412822, 415980, 419123, 422782, 426397,
    429473, 432916, 436000, 439635, 442843, 446436, 450024, 453406,
    456550, 459848, 463186, 466988, 470779, 474059, 477687, 481597,
    486394, 489936, 493065, 496609, 500027,
    503760, 507312])


trig = {'time' : adjust_trig, 'name' : trig_names}

sniff_peaks = adjust_trig.copy()






######## EOG ########

if debug:

    eog_evoked = mne.preprocessing.create_eog_epochs(raw_eeg).average()
    eog_evoked.apply_baseline(baseline=(None, -0.5))
    eog_evoked.plot_joint()

    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw_eeg)
    ica

    ica.plot_sources(raw, show_scrollbars=False)
    ica.plot_components()
    ica.plot_overlay(raw, exclude=[0], picks='eeg')

    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(raw_eeg)
    ica.exclude = eog_indices

    ica.plot_scores(eog_scores)
    ica.plot_properties(raw_eeg, picks=eog_indices)

    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw, show_scrollbars=False)

    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    ica.plot_sources(eog_evoked)



################################
######## PREPROC ########
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
    plt.xlim(0,150)
    plt.show()








#prep_step = prep_step_lf
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


        n_components = np.size(raw.get_data(),0) # if int, use only the first n_components PCA components to compute the ICA decomposition
        random_state = 27
        method = 'fastica'
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, method=method)

        reject = None
        decim = None
        picks = mne.pick_types(raw.info, eeg=True, eog=True)
        ica.fit(raw, picks=picks,decim=decim,reject=reject)

        # for eeg signal
        ica.plot_sources(raw)
        ica.plot_components()
            
        # apply ICA
        raw_ICA = raw.copy()
        ica.apply(raw_ICA) # exclude component

        # verify
        if debug == True :

            # compare before after
            compare_pre_post(raw, raw_post, 4)

        # remove EOG
        drop_chan = ['EOG']
        raw_ICA.drop_channels(drop_chan)
        #raw_ICA.info['ch_names'] # verify

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
######## EXECUTE ########
################################


#### reref A1 - A2
raw_eeg_reref = raw_eeg.copy()
raw_eeg_reref, _ = mne.set_eeg_reference(raw_eeg_reref, ref_channels=['A1', 'A2'])




#### preproc
prep_step_eeg = {
'reref' : {'execute': False, 'params' : ['chan']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': True, 'params' : {'l_freq' : 0.05, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : None, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
'csd_computation' : {'execute': False},
}

raw_preproc_wb  = preprocessing_ieeg(raw_eeg_reref, prep_step_eeg)

if debug:
    compare_pre_post(raw_eeg, raw_eeg_reref, 0)



########################
######## PLOT ########
########################



#### params siff
t_start_SNIFF = -1.5
t_stop_SNIFF = .5

# baseline = [-3, -2]

#### verif EOG

if debug:

    plt.plot(raw_eeg.get_data()[-1,:])
    plt.plot(raw_eeg.get_data()[0,:])
    plt.plot(raw_eeg.get_data()[5,:])
    plt.show()


# mne.viz.plot_raw_psd(raw_eeg)
mne.viz.plot_raw_psd(raw_preproc_wb)

#### baseline
# data = raw_eeg.get_data()
data = raw_preproc_wb.get_data()
times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
data_epoch = np.zeros((len(chan_list), len(sniff_peaks), len(times)))
for nchan in range(len(chan_list[:-3])):
    for sniff_i, sniff_time in enumerate(sniff_peaks):
        _t_start = sniff_time + int(t_start_SNIFF*srate) 
        _t_stop = sniff_time + int(t_stop_SNIFF*srate)

        sniff_data = data[nchan, _t_start:_t_stop]

        _baseline = np.mean(sniff_data[:srate])

        baseline_sniff_data = sniff_data / _baseline

        data_epoch[nchan, sniff_i, :] = baseline_sniff_data



#### no baseline
data = raw_preproc_wb.get_data()
times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
data_epoch = np.zeros((len(chan_list), len(sniff_peaks), len(times)))
for nchan in range(len(chan_list[:-3])):
    for sniff_i, sniff_time in enumerate(sniff_peaks):
        _t_start = sniff_time + int(t_start_SNIFF*srate) 
        _t_stop = sniff_time + int(t_stop_SNIFF*srate)

        sniff_data = data[nchan, _t_start:_t_stop]

        data_epoch[nchan, sniff_i, :] = sniff_data


#### zscore
nchan = 'Cz'
chan_i = chan_list.index(nchan)

epoch_mean = np.mean(data_epoch[chan_i, :, :])
epoch_std = np.std(data_epoch[chan_i, :, :])

data_epoch_zscored = np.zeros(( data_epoch.shape[1], data_epoch.shape[2] ))

for sniff_i, sniff_time in enumerate(sniff_peaks):
    x = data_epoch[chan_i, sniff_i, :]
    data_epoch_zscored[sniff_i, :] = (x - np.mean(x)) / np.std(x) 

if debug:
    for sniff_i, sniff_time in enumerate(sniff_peaks):
        plt.plot(data_epoch_zscored[sniff_i, :])
    plt.hlines([std_thresh, -std_thresh], xmin=0, xmax=data_epoch_zscored.shape[1], color='r')
    plt.show()

#### artifact rejection
std_thresh = 3
epoch_to_reject = np.array(())
epoch_keep = np.array(())

for sniff_i, sniff_time in enumerate(sniff_peaks):
    if data_epoch_zscored[sniff_i, :].max() > std_thresh or data_epoch_zscored[sniff_i, :].min() < -std_thresh:
        epoch_to_reject = np.append(epoch_to_reject, sniff_i)
    else:
        epoch_keep = np.append(epoch_keep, sniff_i)


if debug:
    for sniff_i in epoch_keep.astype(np.int32):
        plt.plot(data_epoch_zscored[sniff_i, :])
    plt.hlines([std_thresh, -std_thresh], xmin=0, xmax=data_epoch_zscored.shape[1], color='r')
    plt.show()



ppi = np.mean(data_epoch_zscored[epoch_keep.astype(np.int32), :], axis=0)
ppi = np.mean(data_epoch_zscored[:, :], axis=0)


times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
plt.plot(times, ppi)
plt.vlines(0, ymin=ppi.min(), ymax=ppi.max(), color='r')
plt.gca().invert_yaxis()
plt.show()





dims = ['chan_list', 'sniffs', 'times']
coords = [chan_list, range(len(sniff_peaks)), times]
xr_epoch_SNIFF = xr.DataArray(data_epoch, coords=coords, dims=dims)

#### select
chan = 'Cz'
chan_i = chan_list.index(chan)

xr_epoch_SNIFF[chan_i, :, :].values

for sniff_i in range(xr_epoch_SNIFF.shape[1])[:20]:

    if sniff_i == 8:
        continue

    plt.plot(xr_epoch_SNIFF['times'], xr_epoch_SNIFF[chan_i, sniff_i, :])
    plt.show()

plt.show()


#### plot
respi = xr_epoch_SNIFF[-1, :, :].mean('sniffs')*-1
plt.plot(times, respi)
plt.vlines([0], ymin=respi.min(), ymax=respi.max(), color='r')
plt.show()

os.chdir('/home/jules/smb4k/CRNLDATA/crnldata/cmo/multisite/DATA_MANIP/EEG_Paris_J/Results')


times = xr_epoch_SNIFF['times'].data
#nchan_i = chan_list.index('Cz')
for nchan in chan_list[:-3]:

    nchan_i = chan_list.index(nchan)

    epochs = xr_epoch_SNIFF[nchan_i, :, :]
    mean = np.zeros(epochs.shape[1])
    for i in range(epochs.shape[0]):
        if i == 8:
            continue
        else:
            mean += epochs[i, :]

    mean /= epochs.shape[0] -1

    x = xr_epoch_SNIFF[nchan_i, :, :].mean('sniffs')
    x = mean

    plt.plot(times, x)
    plt.title(nchan)
    plt.vlines([0], ymin=x.min(), ymax=x.max(), color='r')
    #plt.savefig(f'{nchan}_PPI.png')
    plt.show()

    plt.close()



#### mne
events = []
for sniff in sniff_peaks:
    events.append([sniff, 0, 1])

event_dict = {'sniff': 1}
epochs = mne.Epochs(raw_preproc_wb, events, tmin=-1.5, tmax=.5, event_id=event_dict, preload=True)
evoked = epochs['sniff'].average()

times = np.arange(-5, .5, 0.1)
times = np.array([-1.5, .5, .4, .3, .2, .1, 0, .1, .2, .3])
evoked.apply_baseline((-2, -1))
evoked.plot_topomap(times, ch_type='eeg', time_unit='s')





######## BIPOLARIZE ########

# raw_bip_ref = mne.set_bipolar_reference(raw, anode=['A1'], cathode=['A2'])

bip_A = 'Cz'
bip_B = 'A1'

#### reref
reref = ( (raw_eeg.get_data()[chan_list.index('A1'), :] + raw_eeg.get_data()[chan_list.index('A2'), :]) / 2 )

# plt.plot(reref)
# plt.show()

data_bipo_a = raw_eeg.get_data()[chan_list.index(bip_A), :] - reref
data_bipo_b = raw_eeg.get_data()[chan_list.index(bip_B), :] - reref

data_bipo = data_bipo_a - data_bipo_b 

plt.plot(data_bipo)
plt.show()

#### epoch
times = np.arange(t_start_SNIFF, t_stop_SNIFF, 1/srate)
data_epoch_bipo = np.zeros((len(sniff_peaks), len(times)))
for sniff_i, sniff_time in enumerate(sniff_peaks):
    _t_start = sniff_time + int(t_start_SNIFF*srate) 
    _t_stop = sniff_time + int(t_stop_SNIFF*srate)

    sniff_data = data_bipo[_t_start:_t_stop]

    _baseline = np.mean(sniff_data[:srate])

    baseline_sniff_data = sniff_data / _baseline

    data_epoch_bipo[sniff_i, :] = baseline_sniff_data


ERP_bipo = np.mean(data_epoch_bipo, axis=0)

ERP_bipo_filt = scipy.signal.savgol_filter(ERP_bipo, int(srate*.5+1), 3) # window size 51, polynomial order 3

#### plot
plt.plot(times, ERP_bipo)
plt.title(f'{bip_A} - {bip_B}')
plt.vlines(0, ymin=ERP_bipo.min(), ymax=ERP_bipo.max(), color='r')
plt.show()

plt.plot(times, ERP_bipo_filt)
plt.title(f'{bip_A} - {bip_B}')
plt.vlines(0, ymin=ERP_bipo_filt.min(), ymax=ERP_bipo_filt.max(), color='r')
plt.show()

plt.show(time)







reref A1 A2

filtre : 0.05 40
notch 50


artifact rejection : enlever les epochs qui posent problème
zscore sur l'ensemble des epochs
vire ceux qui sont > à 3SD
potentiellement faire une deuxième vague de zscore après une première rejection

attention à pas enlever endessous de 100

#### PPI
-1.5 -> 0.5
pente > -0.5 = PPI
fit_lm de matlab

plot avec l'axe des y inveré pas * -1






