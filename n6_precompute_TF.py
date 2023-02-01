
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False



################################
######## STRETCH TF ########
################################



def compute_stretch_tf(tf, session_eeg, cond, session_i, respfeatures_allcond, stretch_point_TF):

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond[f's{session_eeg+1}'][cond][session_i], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        tf_mean_allchan[n_chan,:,:] = tf_mean

    return tf_mean_allchan

#condition, resp_features, freq_band, stretch_point_TF = conditions[0], list(resp_features_allcond.values())[0], freq_band, stretch_point_TF
def compute_stretch_tf_dB(session_eeg, tf, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'Baselines'))
    
    baselines = np.load(f'{sujet}_s{session_eeg+1}_{band}_baselines.npy')

    #### apply baseline
    for n_chan in range(np.size(tf,0)):
        
        for fi in range(np.size(tf,1)):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        if n_chan/np.size(tf,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(tf,0)))

        tf_mean = np.zeros((np.size(tf,1),int(stretch_point_TF)))
        for fi in range(np.size(tf,1)):

            x = tf[n_chan,fi,:]
            x_stretch, ratio = stretch_data(respfeatures_allcond[f's{session_eeg+1}'][cond][session_i], stretch_point_TF, x, srate)
            tf_mean[fi,:] = np.mean(x_stretch, axis=0)

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(np.size(tf,0)))

    tf_mean_allchan = np.zeros((np.size(tf,0), np.size(tf,1), stretch_point_TF))

    for n_chan in range(np.size(tf,0)):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan


def compute_stretch_tf_itpc(tf, session_eeg, cond, session_i, respfeatures_allcond, stretch_point_TF):
    
    #### identify number stretch
    x = tf[0,:]
    x_stretch, ratio = stretch_data(respfeatures_allcond[f's{session_eeg+1}'][cond][session_i], stretch_point_TF, x, srate)
    nb_cycle = np.size(x_stretch, 0)
    
    #### compute tf
    tf_stretch = np.zeros((nb_cycle, np.size(tf,0), int(stretch_point_TF)), dtype='complex')

    for fi in range(np.size(tf,0)):

        x = tf[fi,:]
        x_stretch, ratio = stretch_data(respfeatures_allcond[f's{session_eeg+1}'][cond][session_i], stretch_point_TF, x, srate)
        tf_stretch[:,fi,:] = x_stretch

    return tf_stretch






################################
######## PRECOMPUTE TF ########
################################


def precompute_tf(session_eeg, cond, session_i, band_prep_list):

    print('TF PRECOMPUTE')

    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, conditions_allsubjects)

    #### select prep to load
    #band_prep_i, band_prep = 1, 'hf'
    for band_prep in band_prep_list:

        #### select data without aux chan
        data = load_data_sujet(sujet, band_prep, cond, session_i)[:len(chan_list_ieeg),:]

        freq_band = freq_band_dict[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            if os.path.exists(f'{sujet}_s{session_eeg+1}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)
            print('COMPUTE')

            #### select wavelet parameters
            if band_prep == 'lf':
                wavetime = np.arange(-2,2,1/srate)
                nfrex = nfrex_lf
                ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

            elif band_prep == 'hf':
                wavetime = np.arange(-.5,.5,1/srate)
                nfrex = nfrex_hf
                ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

            elif band_prep == 'wb':
                wavetime = np.arange(-2,2,1/srate)
                nfrex = nfrex_wb
                ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex)


            #### compute wavelets
            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle_list[fi] / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()

            os.chdir(path_memmap)
            tf_allchan = np.memmap(f'{sujet}_s{session_eeg+1}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(np.size(data,0), nfrex, np.size(data,1)))

            def compute_tf_convolution_nchan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)))

                for fi in range(nfrex):
                    
                    tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_allchan[n_chan,:,:] = tf

                return

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(np.size(data,0)))

            #### stretch
            print('STRETCH')
            tf_allband_stretched = compute_stretch_tf_dB(session_eeg, tf_allchan, cond, session_i, respfeatures_allcond, stretch_point_TF, band, band_prep, nfrex)
            
            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            np.save(f'{sujet}_s{session_eeg+1}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy', tf_allband_stretched)
            
            os.chdir(path_memmap)
            os.remove(f'{sujet}_s{session_eeg+1}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}_precompute_convolutions.dat')





################################
######## PRECOMPUTE ITPC ########
################################



def precompute_tf_itpc(session_eeg, cond, session_i, band_prep_list):

    print('ITPC PRECOMPUTE')

    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, conditions_allsubjects)
    
    #### select prep to load
    for band_prep in band_prep_list:

        #### select data without aux chan
        data = load_data_sujet(sujet, band_prep, cond, session_i)[:len(chan_list_ieeg),:]

        freq_band = freq_band_dict[band_prep]

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

            if os.path.exists(f'{sujet}_s{session_eeg+1}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy') == True :
                print('ALREADY COMPUTED')
                continue
            
            print(band, ' : ', freq)

            #### select wavelet parameters
            if band_prep == 'lf':
                wavetime = np.arange(-2,2,1/srate)
                nfrex = nfrex_lf
                ncycle_list = np.linspace(ncycle_list_lf[0], ncycle_list_lf[1], nfrex) 

            elif band_prep == 'hf':
                wavetime = np.arange(-.5,.5,1/srate)
                nfrex = nfrex_hf
                ncycle_list = np.linspace(ncycle_list_hf[0], ncycle_list_hf[1], nfrex)

            elif band_prep == 'wb':
                wavetime = np.arange(-2,2,1/srate)
                nfrex = nfrex_wb
                ncycle_list = np.linspace(ncycle_list_wb[0], ncycle_list_wb[1], nfrex)

            #### compute wavelets
            frex  = np.linspace(freq[0],freq[1],nfrex)
            wavelets = np.zeros((nfrex,len(wavetime)) ,dtype=complex)

            # create Morlet wavelet family
            for fi in range(0,nfrex):
                
                s = ncycle_list[fi] / (2*np.pi*frex[fi])
                gw = np.exp(-wavetime**2/ (2*s**2)) 
                sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
                mw =  gw * sw

                wavelets[fi,:] = mw
                
            # plot all the wavelets
            if debug == True:
                plt.pcolormesh(wavetime,frex,np.real(wavelets))
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Real part of wavelets')
                plt.show()

            #### compute itpc
            print('COMPUTE, STRETCH & ITPC')
            def compute_itpc_n_chan(n_chan):

                if n_chan/np.size(data,0) % .2 <= .01:
                    print("{:.2f}".format(n_chan/np.size(data,0)))
                x = data[n_chan,:]

                tf = np.zeros((nfrex,np.size(x)), dtype='complex')

                for fi in range(nfrex):
                    
                    tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                #### stretch
                tf_stretch = compute_stretch_tf_itpc(tf, session_eeg, cond, session_i, respfeatures_allcond, stretch_point_TF)

                #### ITPC
                tf_angle = np.angle(tf_stretch)
                tf_cangle = np.exp(1j*tf_angle) 
                itpc = np.abs(np.mean(tf_cangle,0))

                if debug == True:
                    time = range(stretch_point_TF)
                    frex = range(nfrex)
                    plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
                    plt.show()

                return itpc 

            compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(np.size(data,0)))
            
            itpc_allchan = np.zeros((np.size(data,0),nfrex,stretch_point_TF))

            for n_chan in range(np.size(data,0)):

                itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

            #### save
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            np.save(f'{sujet}_s{session_eeg+1}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy', itpc_allchan)

            del itpc_allchan






################################
######## CHECK STATUS ########
################################

def check_precompute_status():

    print('#### For TF to compute : ####')

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    to_compute = []
    #session_eeg, band_prep, cond, session_i = 0, band_prep_list[0], 'RD_FV', 0
    for band_prep in band_prep_list:
        for session_eeg in range(3):
            for band, freq in freq_band_dict[band_prep].items():
                for cond in conditions:
                    for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                        to_compute.append(f'{sujet}_s{session_eeg+1}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy')
    
    print([file_i for file_i in to_compute if file_i not in os.listdir()])

    print('#### For ITPC to compute : ####')

    os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

    to_compute = []
    #session_eeg, band_prep, cond, session_i = 0, band_prep_list[0], 'RD_FV', 0
    for band_prep in band_prep_list:
        for session_eeg in range(3):
            for band, freq in freq_band_dict[band_prep].items():
                for cond in conditions:
                    for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                        to_compute.append(f'{sujet}_s{session_eeg+1}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}_{str(session_i+1)}.npy')
        
    print([file_i for file_i in to_compute if file_i not in os.listdir()])






################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #### load data
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, conditions_allsubjects)
    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)

    #### check status
    check_precompute_status()

    #### compute all
    print('######## PRECOMPUTE TF & ITPC ########')

    #### compute and save tf
    #cond = 'RD_SV'
    #session_i = 0
    #session_eeg = 1

    for session_eeg in range(3):

        for cond in conditions:

            print(cond)

            if len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1:
        
                #precompute_tf(session_eeg, cond, 0, band_prep_list)
                execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf', [session_eeg, cond, 0, band_prep_list])
                #precompute_tf_itpc(session_eeg, cond, 0, band_prep_list)
                execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf_itpc', [session_eeg, cond, 0, band_prep_list])
            
            elif len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1:

                for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                    #precompute_tf(session_eeg, cond, session_i, band_prep_list)
                    execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf', [session_eeg, cond, session_i, band_prep_list])
                    #precompute_tf_itpc(session_eeg, cond, session_i, band_prep_list)
                    execute_function_in_slurm_bash('n6_precompute_TF', 'precompute_tf_itpc', [session_eeg, cond, session_i, band_prep_list])









