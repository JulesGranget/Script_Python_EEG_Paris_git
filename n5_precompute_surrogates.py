
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config_params import *
from n0bis_config_analysis_functions import *

import joblib

debug = False



################################################
######## PRECOMPUTE AND SAVE SURROGATES ########
################################################


def shuffle_CycleFreq(x):

    cut = int(np.random.randint(low=0, high=len(x), size=1))
    x_cut1 = x[:cut]
    x_cut2 = x[cut:]*-1
    x_shift = np.concatenate((x_cut2, x_cut1), axis=0)

    return x_shift
    

def shuffle_Cxy(x):
   half_size = x.shape[0]//2
   ind = np.random.randint(low=0, high=half_size)
   x_shift = x.copy()
   
   x_shift[ind:ind+half_size] *= -1
   if np.random.rand() >=0.5:
       x_shift *= -1

   return x_shift


def precompute_surrogates_coh(band_prep, session_eeg, cond, session_i):
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
    
    print(cond)

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, conditions_allsubjects)
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(sujet + f'_s{session_eeg+1}_' + cond + '_' + str(session_i+1) + '_Coh.npy'):
        print('ALREADY COMPUTED')
        return

    respi_i = chan_list.index('Respi')

    respi = data_tmp[respi_i,:]

    hzCxy = np.linspace(0,srate/2,int(nfft/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    surrogates_n_chan = np.zeros((np.size(data_tmp,0),len(hzCxy)))

    def compute_surrogates_coh_n_chan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]
        y = respi

        surrogates_val_tmp = np.zeros((n_surrogates_coh,len(hzCxy)))
        for surr_i in range(n_surrogates_coh):
            
            #if surr_i%100 == 0:
            #    print(surr_i) 

            x_shift = shuffle_Cxy(x)
            #y_shift = shuffle_Cxy(y)
            hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, y, fs=srate, window=hannw, nperseg=None, noverlap=noverlap, nfft=nfft)

            surrogates_val_tmp[surr_i,:] = Cxy[mask_hzCxy]

        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i = int(np.floor(n_surrogates_coh*percentile_coh))
        compute_surrogates_coh_tmp = surrogates_val_tmp_sorted[percentile_i,:]

        return compute_surrogates_coh_tmp
    
    compute_surrogates_coh_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_coh_n_chan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[n_chan,:] = compute_surrogates_coh_results[n_chan]

    np.save(sujet + f'_s{session_eeg+1}_' + cond + '_' + str(session_i+1) + '_Coh.npy', surrogates_n_chan)



#band_prep, session_eeg, cond, session_i = 'wb', 0, 'FR_CV', 0
def precompute_surrogates_cyclefreq(band_prep, session_eeg, cond, session_i):
    
    print(cond)

    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    data_tmp = load_data_sujet(sujet, band_prep, cond, session_i)

    if os.path.exists(sujet + f'_s{session_eeg+1}_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy'):
        print('ALREADY COMPUTED')
        return

    surrogates_n_chan = np.zeros((3,np.size(data_tmp,0), stretch_point_surrogates))

    respfeatures_i = respfeatures_allcond[f's{session_eeg+1}'][cond][session_i]

    def compute_surrogates_cyclefreq_nchan(n_chan):

        if n_chan/np.size(data_tmp,0) % .2 <= .01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]

        surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq,stretch_point_surrogates))
        for surr_i in range(n_surrogates_cyclefreq):
            
            #if surr_i%100 == 0:
            #    print(surr_i)

            x_shift = shuffle_CycleFreq(x)
            #y_shift = shuffle_CycleFreq(y)

            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

            x_stretch_mean = np.mean(x_stretch, axis=0)

            surrogates_val_tmp[surr_i,:] = x_stretch_mean

        mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
        surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
        percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
        percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

        up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
        dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

        return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

    compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

    for n_chan in range(np.size(data_tmp,0)):

        surrogates_n_chan[0,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][0]
        surrogates_n_chan[1,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][1]
        surrogates_n_chan[2,n_chan,:] = compute_surrogates_cyclefreq_results[n_chan][2]
    
    np.save(sujet + f'_s{session_eeg+1}_' + cond + '_' + str(session_i+1) + '_cyclefreq_' +  band_prep + '.npy', surrogates_n_chan)





################################
######## CHECK STATUS ########
################################

def check_precompute_status():

    print('#### For Cxy to compute : ####')

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    to_compute = []
    #session_eeg, band_prep, cond, session_i = 0, band_prep_list[0], 'RD_FV', 0
    for session_eeg in range(3):
        for band_prep in band_prep_list:
            for cond in conditions:
                for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                    to_compute.append(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_Coh.npy')
    
    print([file_i for file_i in to_compute if file_i not in os.listdir() and file_i.find('cyclefreq') == -1])

    print('#### For Cyclefreq to compute : ####')

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    to_compute = []
    #session_eeg, band_prep, cond, session_i = 0, band_prep_list[0], 'RD_FV', 0
    for session_eeg in range(3):
        for band_prep in band_prep_list:
            for cond in conditions:
                for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):
                    to_compute.append(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy')
    
    print([file_i for file_i in to_compute if file_i not in os.listdir() and file_i.find('Coh') == -1])



################################
######## EXECUTE ########
################################

if __name__ == '__main__':


    #### load data
    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet, conditions_allsubjects)
    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)

    #### params surrogates
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    #### indicate which file is missing
    check_precompute_status()

    #### compute and save
    print('######## COMPUTE SURROGATES ########')

    #session_eeg = 1
    for session_eeg in range(3):

        #band_prep = band_prep_list[0]
        for band_prep in band_prep_list:

            print('COMPUTE FOR ' + band_prep)

            #cond = 'RD_FV'
            for cond in conditions:

                if len(respfeatures_allcond[f's{session_eeg+1}'][cond]) == 1:

                    #precompute_surrogates_cyclefreq(band_prep, session_eeg, cond, 0)
                    execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_cyclefreq', [band_prep, session_eeg, cond, 0])

                    if band_prep == 'wb':
                        #precompute_surrogates_coh(band_prep, session_eeg, cond, 0)
                        execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_coh', [band_prep, session_eeg, cond, 0])

                elif len(respfeatures_allcond[f's{session_eeg+1}'][cond]) > 1:

                    for session_i in range(len(respfeatures_allcond[f's{session_eeg+1}'][cond])):

                        #precompute_surrogates_cyclefreq(band_prep, session_eeg, cond, session_i)
                        execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_cyclefreq', [band_prep, session_eeg, cond, session_i])

                        if band_prep == 'wb':
                            #precompute_surrogates_coh(band_prep, session_eeg, cond, session_i)
                            execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_coh', [band_prep, session_eeg, cond, session_i])







