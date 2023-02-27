

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib
import pickle

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





################################################
######## PSD & COH WHOLE COMPUTATION ########
################################################


def load_surrogates_session(session_eeg, prms):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    surrogates_allcond = {'Cxy' : {}}

    for band_prep in band_prep_list:
        surrogates_allcond[f'cyclefreq_{band_prep}'] = {}

        #cond = 'FR_CV'
        for cond in conditions_allsubjects:

            if prms['count_session'][f's{session_eeg+1}'][cond] == 1:

                surrogates_allcond['Cxy'][cond] = [np.load(f'{sujet}_s{session_eeg+1}_{cond}_1_Coh.npy')]
                surrogates_allcond[f'cyclefreq_{band_prep}'][cond] = [np.load(f'{sujet}_s{session_eeg+1}_{cond}_1_cyclefreq_{band_prep}.npy')]

            elif prms['count_session'][f's{session_eeg+1}'][cond] > 1:

                data_load_Cxy = []
                data_load_cyclefreq = []

                for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):

                    data_load_Cxy.append(np.load(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_Coh.npy'))
                    data_load_cyclefreq.append(np.load(f'{sujet}_s{session_eeg+1}_{cond}_{str(session_i+1)}_cyclefreq_{band_prep}.npy'))                 

                surrogates_allcond['Cxy'][cond] = data_load_Cxy
                surrogates_allcond[f'cyclefreq_{band_prep}'][cond] = data_load_cyclefreq

    #### verif 
    if debug:
        for cond in list(surrogates_allcond['Cxy'].keys()):
            for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):
                print(f'#### for {cond}, session {session_i+1} :')
                print('Cxy : ', surrogates_allcond['Cxy'][cond][session_i].shape)
                print('cyclefreq : ', surrogates_allcond['cyclefreq_wb'][cond][session_i].shape)

    return surrogates_allcond





#### compute Pxx & Cxy & Cyclefreq
def compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, nb_point_by_cycle, respfeatures_allcond, prms):
    
    print(cond)

    #### extract data
    chan_i = prms['chan_list'].index('Respi')
    respi = load_data_sujet(sujet, band_prep, cond, odor_i)[chan_i,:]
    data_tmp = load_data_sujet(sujet, band_prep, cond, odor_i)
    if stretch_point_surrogates == stretch_point_TF:
        nb_point_by_cycle = stretch_point_surrogates
    else:
        raise ValueError('Not the same stretch point')

    #### prepare analysis
    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    #### compute
    Cxy_for_cond = np.zeros(( np.size(data_tmp,0), len(hzCxy)))
    Pxx_for_cond = np.zeros(( np.size(data_tmp,0), len(hzPxx)))
    cyclefreq_for_cond = np.zeros(( np.size(data_tmp,0), nb_point_by_cycle))

    for n_chan in range(np.size(data_tmp,0)):

        #### script avancement
        if n_chan/np.size(data_tmp,0) % .2 <= 0.01:
            print('{:.2f}'.format(n_chan/np.size(data_tmp,0)))

        x = data_tmp[n_chan,:]
        hzPxx, Pxx = scipy.signal.welch(x,fs=srate,window=prms['hannw'],nperseg=prms['nwind'],noverlap=prms['noverlap'],nfft=prms['nfft'])

        y = respi
        hzPxx, Cxy = scipy.signal.coherence(x, y, fs=srate, window=prms['hannw'], nperseg=None, noverlap=prms['noverlap'], nfft=prms['nfft'])

        x_stretch, trash = stretch_data(respfeatures_allcond[f's{session_eeg+1}'][cond][session_i], nb_point_by_cycle, x, srate)
        x_stretch_mean = np.mean(x_stretch, 0)

        Cxy_for_cond[n_chan,:] = Cxy[mask_hzCxy]
        Pxx_for_cond[n_chan,:] = Pxx
        cyclefreq_for_cond[n_chan,:] = x_stretch_mean

    return Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond

        




def compute_all_PxxCxyCyclefreq(session_eeg, respfeatures_allcond, prms):

    #### initiate dict
    Cxy_allcond = {}
    Pxx_allcond = {}
    cyclefreq_allcond = {}
    for band_prep in band_prep_list:
        Pxx_allcond[band_prep] = {}
        cyclefreq_allcond[band_prep] = {}

    #band_prep = band_prep_list[0]
    for band_prep in band_prep_list:

        print(band_prep)

        for cond in conditions_allsubjects:

            if ( prms['count_session'][f's{session_eeg+1}'][cond] == 1 ) & (band_prep == 'lf' or band_prep == 'wb'):

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond[band_prep][cond] = [Pxx_for_cond]
                Cxy_allcond[cond] = [Cxy_for_cond]
                cyclefreq_allcond[band_prep][cond] = [cyclefreq_for_cond]

            elif ( prms['count_session'][f's{session_eeg+1}'][cond] == 1 ) & (band_prep == 'hf') :

                Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, 0, stretch_point_surrogates, respfeatures_allcond, prms)

                Pxx_allcond[band_prep][cond] = [Pxx_for_cond]
                cyclefreq_allcond[band_prep][cond] = [cyclefreq_for_cond]

            elif ( prms['count_session'][f's{session_eeg+1}'][cond] > 1) & (band_prep == 'lf' or band_prep == 'wb'):

                Pxx_load = []
                Cxy_load = []
                cyclefreq_load = []

                for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    Cxy_load.append(Cxy_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond[band_prep][cond] = Pxx_load
                Cxy_allcond[cond] = Cxy_load
                cyclefreq_allcond[band_prep][cond] = cyclefreq_load

            elif (prms['count_session'][f's{session_eeg+1}'][cond] > 1) & (band_prep == 'hf'):

                Pxx_load = []
                cyclefreq_load = []

                for session_i in range(prms['count_session'][f's{session_eeg+1}'][cond]):

                    Pxx_for_cond, Cxy_for_cond, cyclefreq_for_cond = compute_PxxCxyCyclefreq_for_cond(band_prep, session_eeg, cond, session_i, stretch_point_surrogates, respfeatures_allcond, prms)

                    Pxx_load.append(Pxx_for_cond)
                    cyclefreq_load.append(cyclefreq_for_cond)

                Pxx_allcond[band_prep][cond] = Pxx_load
                cyclefreq_allcond[band_prep][cond] = cyclefreq_load

    return Pxx_allcond, Cxy_allcond, cyclefreq_allcond






#dict2reduce = surrogates_allcond
def reduce_data(dict2reduce, session_eeg, prms):

    #### for Pxx and Cyclefreq
    if np.sum([True for i in list(dict2reduce.keys()) if i in band_prep_list]) > 0:
    
        #### generate dict
        dict_reduced = {}
        for band_prep in band_prep_list:
            dict_reduced[band_prep] = {}
            for cond in conditions_allsubjects:
                dict_reduced[band_prep][cond] = []

        for band_prep in band_prep_list:

            for cond in conditions_allsubjects:

                n_session = prms['count_session'][f's{session_eeg+1}'][cond]

                #### reduce
                if n_session == 1:
                    dict_reduced[band_prep][cond].append(dict2reduce[band_prep][cond])

                elif n_session > 1:
                    
                    for session_i in range(n_session):

                        if session_i == 0:
                            dict_reduced[band_prep][cond] = dict2reduce[band_prep][cond][session_i]
                        else:
                            dict_reduced[band_prep][cond] = (dict_reduced[band_prep][cond] + dict2reduce[band_prep][cond][session_i])/2

        #### verify
        for band_prep in band_prep_list:
            for cond in conditions_allsubjects:
                if len(dict_reduced[band_prep][cond].shape) != 2:
                    raise ValueError(f'reducing false for Pxx or Cyclefreq : {band_prep}, {cond}')

    #### for Cxy
    elif np.sum([True for i in list(dict2reduce.keys()) if i in conditions_allsubjects]) > 0:

        #### generate dict
        dict_reduced = {}
        for cond in conditions_allsubjects:
            dict_reduced[cond] = []

        for cond in conditions_allsubjects:

            n_session = prms['count_session'][f's{session_eeg+1}'][cond]

            #### reduce
            if n_session == 1:
                dict_reduced[cond].append(dict2reduce[cond])

            elif n_session > 1:
                
                for session_i in range(n_session):

                    if session_i == 0:
                        dict_reduced[cond] = dict2reduce[cond][session_i]
                    else:
                        dict_reduced[cond] = (dict_reduced[cond] + dict2reduce[cond][session_i])/2

        #### verify
        for cond in conditions_allsubjects:
            if len(dict_reduced[cond].shape) != 2:
                raise ValueError(f'reducing false for Cxy :, {cond}')

    #### for surrogates
    else:
        
        #### generate dict
        dict_reduced = {}
        for key in list(dict2reduce.keys()):
            dict_reduced[key] = {}
            for cond in conditions_allsubjects:
                dict_reduced[key][cond] = []

        #key = 'Cxy'
        for key in list(dict2reduce.keys()):

            for cond in conditions_allsubjects:

                n_session = prms['count_session'][f's{session_eeg+1}'][cond]

                #### reduce
                if n_session == 1:
                    dict_reduced[key][cond].append(dict2reduce[cond])

                elif n_session > 1:
                    
                    for session_i in range(n_session):

                        if session_i == 0:
                            dict_reduced[key][cond] = dict2reduce[key][cond][session_i]
                        else:
                            dict_reduced[key][cond] = (dict_reduced[key][cond] + dict2reduce[key][cond][session_i])/2

        #### verify
        for key in list(dict2reduce.keys()):
            if key == 'Cxy':
                for cond in conditions_allsubjects:
                    if len(dict_reduced[key][cond].shape) != 2:
                        raise ValueError(f'reducing false for Surrogates : {key}, {cond}')
            else:
                for cond in conditions_allsubjects:
                    if len(dict_reduced[key][cond].shape) != 3:
                        raise ValueError(f'reducing false for Surrogates : {key}, {cond}')

    return dict_reduced

                    




def reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond, session_eeg, prms):

    Pxx_allcond_red = reduce_data(Pxx_allcond, session_eeg, prms)
    cyclefreq_allcond_red = reduce_data(cyclefreq_allcond, session_eeg, prms)

    Cxy_allcond_red = reduce_data(Cxy_allcond, session_eeg, prms)
    surrogates_allcond_red = reduce_data(surrogates_allcond, session_eeg, prms)
    
    return Pxx_allcond_red, cyclefreq_allcond_red, Cxy_allcond_red, surrogates_allcond_red






def compute_reduced_PxxCxyCyclefreqSurrogates(session_eeg, respfeatures_allcond, surrogates_allcond, prms):


    if os.path.exists(os.path.join(path_precompute, sujet, 'PSD_Coh', f'{sujet}_s{session_eeg+1}_Pxx_allcond.pkl')) == False:
    
        Pxx_allcond, Cxy_allcond, cyclefreq_allcond = compute_all_PxxCxyCyclefreq(session_eeg, respfeatures_allcond, prms)

        Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond = reduce_PxxCxy_cyclefreq(Pxx_allcond, Cxy_allcond, cyclefreq_allcond, surrogates_allcond, session_eeg, prms)

        save_Pxx_Cxy_Cyclefreq_Surrogates_allcond(session_eeg, Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond)

        print('COMPUTE Pxx CF Cxy Surr')

    else:

        print('ALREADY COMPUTED')

    print('done') 






def save_Pxx_Cxy_Cyclefreq_Surrogates_allcond(session_eeg, Pxx_allcond, cyclefreq_allcond, Cxy_allcond, surrogates_allcond):

    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    with open(f'{sujet}_s{session_eeg+1}_Pxx_allcond.pkl', 'wb') as f:
        pickle.dump(Pxx_allcond, f)

    with open(f'{sujet}_s{session_eeg+1}_Cxy_allcond.pkl', 'wb') as f:
        pickle.dump(Cxy_allcond, f)

    with open(f'{sujet}_s{session_eeg+1}_surrogates_allcond.pkl', 'wb') as f:
        pickle.dump(surrogates_allcond, f)

    with open(f'{sujet}_s{session_eeg+1}_cyclefreq_allcond.pkl', 'wb') as f:
        pickle.dump(cyclefreq_allcond, f)




def get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, session_eeg):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
        
    with open(f'{sujet}_s{session_eeg+1}_Pxx_allcond.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'{sujet}_s{session_eeg+1}_Cxy_allcond.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'{sujet}_s{session_eeg+1}_surrogates_allcond.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'{sujet}_s{session_eeg+1}_cyclefreq_allcond.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond




################################################
######## PLOT & SAVE PSD AND COH ########
################################################

#n_chan = 0
def plot_save_PSD_Coh(n_chan, session_eeg):

    #### load data
    Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond = get_Pxx_Cxy_Cyclefreq_Surrogates_allcond(sujet, session_eeg)
    prms = get_params()
    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)
    
    #### compute
    chan_name = prms['chan_list_ieeg'][n_chan]

    odor = odor_order[sujet][f's{session_eeg+1}']

    if n_chan/len(prms['chan_list_ieeg']) % .2 <= 0.01:
        print('{:.2f}'.format(n_chan/len(prms['chan_list_ieeg'])))

    hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
    mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
    hzCxy = hzCxy[mask_hzCxy]

    for band_prep in band_prep_list:

        fig, axs = plt.subplots(nrows=4, ncols=len(conditions_allsubjects))
        plt.suptitle(f'{sujet}_{odor}_{chan_name}')
        
        for c, cond in enumerate(conditions_allsubjects):

            respi_mean = respi_mean_allcond[f's{session_eeg+1}'][cond]
            
            #### plot
            ax = axs[0,c]
            ax.set_title(cond, fontweight='bold', rotation=0)
            ax.semilogy(hzPxx,Pxx_allcond[band_prep][cond][n_chan,:], color='k')
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond[band_prep][cond][n_chan,:]), color='r')
            ax.set_xlim(0,60)

            ax = axs[1,c]
            ax.plot(hzPxx[remove_zero_pad:],Pxx_allcond[band_prep][cond][n_chan,:][remove_zero_pad:], color='k')
            ax.set_xlim(0, 2)
            ax.vlines(respi_mean, ymin=0, ymax=max(Pxx_allcond[band_prep][cond][n_chan,:]), color='r')

            ax = axs[2,c]
            ax.plot(hzCxy,Cxy_allcond[cond][n_chan,:], color='k')
            ax.plot(hzCxy,surrogates_allcond['Cxy'][cond][n_chan,:], color='c')
            ax.vlines(respi_mean, ymin=0, ymax=1, color='r')

            ax = axs[3,c]
            ax.plot(cyclefreq_allcond[band_prep][cond][n_chan,:], color='k')
            ax.plot(surrogates_allcond[f'cyclefreq_{band_prep}'][cond][0, n_chan,:], color='b')
            ax.plot(surrogates_allcond[f'cyclefreq_{band_prep}'][cond][1, n_chan,:], color='c', linestyle='dotted')
            ax.plot(surrogates_allcond[f'cyclefreq_{band_prep}'][cond][2, n_chan,:], color='c', linestyle='dotted')
            if stretch_TF_auto:
                ax.vlines(prms['respi_ratio_allcond'][cond]*stretch_point_surrogates, ymin=np.min( surrogates_allcond[f'cyclefreq_{band_prep}'][cond][2, n_chan,:] ), ymax=np.max( surrogates_allcond[f'cyclefreq_{band_prep}'][cond][1, n_chan,:] ), colors='r')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=np.min( surrogates_allcond[f'cyclefreq_{band_prep}'][cond][2, n_chan,:] ), ymax=np.max( surrogates_allcond[f'cyclefreq_{band_prep}'][cond][1, n_chan,:] ), colors='r') 
        #plt.show()
        
        #### save
        fig.savefig(f'{sujet}_{chan_name}_odor_{odor}_{band_prep}.jpeg', dpi=600)
        plt.close()

    











################################
######## LOAD TF & ITPC ########
################################

#tf_mode = 'ITPC'
def compute_TF_ITPC(session_eeg, prms):

    for tf_mode in ['TF', 'ITPC']:
    
        if tf_mode == 'TF':
            print('######## LOAD TF ########')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'TF', f'{sujet}_s{session_eeg+1}_tf_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue
            
        elif tf_mode == 'ITPC':
            print('######## LOAD ITPC ########')
            os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
            if os.path.exists(os.path.join(path_precompute, sujet, 'ITPC', f'{sujet}_s{session_eeg+1}_itpc_stretch_allcond.pkl')):
                print('ALREADY COMPUTED')
                continue

        #### generate str to search file
        freq_band_str = {}

        for band_prep in band_prep_list:

            freq_band = freq_band_dict[band_prep]

            for band, freq in freq_band.items():
                freq_band_str[band] = str(freq[0]) + '_' + str(freq[1])


        #### load file with reducing to one TF

        tf_stretch_allcond = {}

        for band_prep in band_prep_list:

            tf_stretch_allcond[band_prep] = {}

            for cond in conditions_allsubjects:

                tf_stretch_onecond = {}

                if prms['count_session'][f's{session_eeg+1}'][cond] == 1:

                    #### generate file to load
                    load_file = []
                    for file in os.listdir(): 
                        if file.find(cond) != -1:
                            load_file.append(file)
                        else:
                            continue

                    #### impose good order in dict
                    for band, freq in freq_band_dict[band_prep].items():
                        tf_stretch_onecond[band] = 0

                    #### file load
                    for file in load_file:

                        for i, (band, freq) in enumerate(freq_band_dict[band_prep].items()):

                            if (file.find(freq_band_str[band]) != -1) and (file.find(f's{session_eeg+1}') != -1):
                                tf_stretch_onecond[band] = np.load(file)
                            else:
                                continue
                                
                    tf_stretch_allcond[band_prep][cond] = tf_stretch_onecond

                elif prms['count_session'][f's{session_eeg+1}'][cond] > 1:

                    #### generate file to load
                    load_file = []
                    for file in os.listdir(): 
                        if file.find(cond) != -1:
                            load_file.append(file)
                        else:
                            continue

                    #### implement count
                    for band, freq in freq_band_dict[band_prep].items():
                        tf_stretch_onecond[band] = 0

                    #### load file
                    for file in load_file:

                        for i, (band, freq) in enumerate(freq_band_dict[band_prep].items()):

                            if (file.find(freq_band_str[band]) != -1) and (file.find(f's{session_eeg+1}') != -1):

                                if np.sum(tf_stretch_onecond[band]) != 0:

                                    session_load_tmp = ( np.load(file) + tf_stretch_onecond[band] ) /2
                                    tf_stretch_onecond[band] = session_load_tmp

                                else:
                                    
                                    tf_stretch_onecond[band] = np.load(file)

                            else:

                                continue

                    tf_stretch_allcond[band_prep][cond] = tf_stretch_onecond

        #### verif
        for cond in conditions_allsubjects:
            for band, freq in freq_band_dict[band_prep].items():
                if len(tf_stretch_allcond[band_prep][cond][band]) != len(prms['chan_list_ieeg']) :
                    print('ERROR FREQ BAND : ' + band)
                    
        #### save
        if tf_mode == 'TF':
            with open(f'{sujet}_s{session_eeg+1}_tf_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)
        elif tf_mode == 'ITPC':
            with open(f'{sujet}_s{session_eeg+1}_itpc_stretch_allcond.pkl', 'wb') as f:
                pickle.dump(tf_stretch_allcond, f)

    print('done')
    





def get_tf_itpc_stretch_allcond(session_eeg, tf_mode):

    source_path = os.getcwd()

    if tf_mode == 'TF':

        os.chdir(os.path.join(path_precompute, sujet, 'TF'))

        with open(f'{sujet}_s{session_eeg+1}_tf_stretch_allcond.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)


    elif tf_mode == 'ITPC':
        
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        with open(f'{sujet}_s{session_eeg+1}_itpc_stretch_allcond.pkl', 'rb') as f:
            tf_stretch_allcond = pickle.load(f)

    os.chdir(source_path)

    return tf_stretch_allcond



########################################
######## PLOT & SAVE TF & ITPC ########
########################################



#n_chan, session_eeg, tf_mode, band_prep = 0, 0, 'TF', 'wb'
def save_TF_ITPC_n_chan(n_chan, session_eeg, tf_mode, band_prep):

    #### load data
    prms = get_params()
    tf_stretch_allcond = get_tf_itpc_stretch_allcond(session_eeg, tf_mode)


    if tf_mode == 'TF':
        os.chdir(os.path.join(path_results, sujet, 'TF', 'summary'))
    elif tf_mode == 'ITPC':
        os.chdir(os.path.join(path_results, sujet, 'ITPC', 'summary'))
    
    odor = odor_order[sujet][f's{session_eeg+1}']

    chan_name = prms['chan_list_ieeg'][n_chan]

    if n_chan/len(prms['chan_list_ieeg']) % .2 <= .01:
        print('{:.2f}'.format(n_chan/len(prms['chan_list_ieeg'])))

    time = range(stretch_point_TF)
    freq_band = freq_band_dict[band_prep]

    #### determine plot scale
    scales = {'vmin_val' : np.array(()), 'vmax_val' : np.array(()), 'median_val' : np.array(())}

    for c, cond in enumerate(conditions_allsubjects):

        for i, (band, freq) in enumerate(freq_band.items()) :

            if band == 'whole' or band == 'l_gamma':
                continue

            data = tf_stretch_allcond[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))

            scales['vmin_val'] = np.append(scales['vmin_val'], np.min(data))
            scales['vmax_val'] = np.append(scales['vmax_val'], np.max(data))
            scales['median_val'] = np.append(scales['median_val'], np.median(data))

    median_diff = np.max([np.abs(np.min(scales['vmin_val']) - np.median(scales['median_val'])), np.abs(np.max(scales['vmax_val']) - np.median(scales['median_val']))])

    vmin = np.median(scales['median_val']) - median_diff
    vmax = np.median(scales['median_val']) + median_diff

    #### plot

    nrows = len(freq_band)

    if band_prep == 'hf':
        fig, axs = plt.subplots(nrows=nrows, ncols=len(conditions_allsubjects))
    else:
        fig, axs = plt.subplots(nrows=nrows, ncols=len(conditions_allsubjects))
    
    plt.suptitle(f'{sujet}_{odor}_{chan_name}')

    #### for plotting l_gamma down
    if band_prep == 'hf':
        keys_list_reversed = list(freq_band.keys())
        keys_list_reversed.reverse()
        freq_band_reversed = {}
        for key_i in keys_list_reversed:
            freq_band_reversed[key_i] = freq_band[key_i]
        freq_band = freq_band_reversed

    for c, cond in enumerate(conditions_allsubjects):

        #### plot
        for i, (band, freq) in enumerate(freq_band.items()) :

            data = tf_stretch_allcond[band_prep][cond][band][n_chan, :, :]
            frex = np.linspace(freq[0], freq[1], np.size(data,0))
        
            if len(conditions_allsubjects) == 1:
                ax = axs[i]
            else:
                ax = axs[i,c]

            if i == 0 :
                ax.set_title(cond, fontweight='bold', rotation=0)

            ax.pcolormesh(time, frex, data, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))

            if c == 0:
                ax.set_ylabel(band)

            if stretch_TF_auto:
                ax.vlines(prms['respi_ratio_allcond'][cond][0]*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
            else:
                ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=freq[0], ymax=freq[1], colors='g')
    #plt.show()

    #### save
    fig.savefig(f'{sujet}_{chan_name}_odor_{odor}_{band_prep}.jpeg', dpi=600)
    plt.close()





########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq(session_eeg):
    
    prms = get_params()
    respfeatures_allcond, respi_mean_allcond = load_respfeatures(sujet)
        
    surrogates_allcond = load_surrogates_session(session_eeg, prms)

    compute_reduced_PxxCxyCyclefreqSurrogates(session_eeg, respfeatures_allcond, surrogates_allcond, prms)
    
    #### compute joblib

    os.chdir(os.path.join(path_results, sujet, 'PSD_Coh', 'summary'))

    print('######## PLOT & SAVE PSD AND COH ########')

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(plot_save_PSD_Coh)(n_chan, session_eeg) for n_chan, session_eeg in zip(range(len(prms['chan_list_ieeg'])), [session_eeg]*len(prms['chan_list_ieeg'])))

    

def compilation_compute_TF_ITPC(session_eeg):

    prms = get_params()

    compute_TF_ITPC(session_eeg, prms)
    
    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'TF':
            print('######## PLOT & SAVE TF ########')
        if tf_mode == 'ITPC':
            print('######## PLOT & SAVE ITPC ########')
        
        for band_prep in band_prep_list: 

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(save_TF_ITPC_n_chan)(n_chan, session_eeg, tf_mode, band_prep) for n_chan, session_eeg, tf_mode, band_prep in zip(range(len(prms['chan_list_ieeg'])), [session_eeg]*len(prms['chan_list_ieeg']), [tf_mode]*len(prms['chan_list_ieeg']), [band_prep]*len(prms['chan_list_ieeg'])))








################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #session_eeg = 0
    for session_eeg in range(3):
    
        #### Pxx Cxy CycleFreq
        #compilation_compute_Pxx_Cxy_Cyclefreq(session_eeg_i)
        execute_function_in_slurm_bash('n7_power_analysis', 'compilation_compute_Pxx_Cxy_Cyclefreq', [session_eeg])


        #### TF & ITPC
        #compilation_compute_TF_ITPC(session_eeg_i)
        execute_function_in_slurm_bash('n7_power_analysis', 'compilation_compute_TF_ITPC', [session_eeg])

    



