

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from n0_config import *

debug = False



########################################
######## COMPUTE RESPI FEATURES ########
########################################

#resp_sig, sr, t_start, condition = raw_allcond[cond][session_i].get_data()[respi_i, :], srate, 0, cond
def analyse_resp(resp_sig, sr, t_start, condition):
    
    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt iEEG inspi = '-'
        # for abdominal belt EEG inspi = '-'
        # for nasal thermistance inspi = '+'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '+',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = True,
                                                    manual_baseline = 0.,

                                                    high_pass_filter = None,
                                                    constrain_frequency = None,
                                                    median_windows_filter = None,

                                                    # clean
                                                    eliminate_time_shortest_ratio = 8,
                                                    eliminate_amplitude_shortest_ratio = 4,
                                                    eliminate_mode = 'OR', ) # 'AND')
    


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g')
    #ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    







def analyse_resp_debug(resp_sig, sr, t_start, condition, params):

    if params['mean_smooth'] :
        if (2*sr)%2 != 1:
            win = 2*sr + 1
        else:
            win = 2*sr
        resp_sig = scipy.signal.savgol_filter(resp_sig, win, 5)
    
    # compute signal features
        # indicate if inspiration is '+' or '-'
        # for abdominal belt inspi = '-'
        # for nasal thermistance inspi = '+'
    cycle_indexes = respirationtools.detect_respiration_cycles(resp_sig, sr, t_start=t_start, output = 'index',
                                                    inspiration_sign = '+',
                                                    # baseline
                                                    #baseline_with_average = False,
                                                    baseline_with_average = params.get('baseline_with_average'),
                                                    manual_baseline = params.get('manual_baseline'),

                                                    high_pass_filter = params.get('high_pass_filter'),
                                                    constrain_frequency = params.get('constrain_frequency'),
                                                    median_windows_filter = params.get('median_windows_filter'),

                                                    # clean
                                                    eliminate_time_shortest_ratio = params.get('eliminate_time_shortest_ratio'),
                                                    eliminate_amplitude_shortest_ratio = params.get('eliminate_amplitude_shortest_ratio'),
                                                    eliminate_mode = params.get('eliminate_mode') )
    


    resp_sig_mc = resp_sig.copy()
    resp_sig_mc -= np.mean(resp_sig_mc)
    resp_features = respirationtools.get_all_respiration_features(resp_sig_mc, sr, cycle_indexes, t_start = 0.)
    #print(resp_features.columns)
    
    cycle_amplitudes = resp_features['total_amplitude'].values
    cycle_durations = resp_features['cycle_duration'].values # idem as : cycle_durations = np.diff(cycle_indexes[:, 0])/sr
    cycle_freq = resp_features['cycle_freq'].values
    
    # figure 
    
    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(condition)
    times = np.arange(resp_sig.size)/sr
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, resp_sig)
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g')
    #ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:-1, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(condition)

        # histogram cycle freq
    ax = axs[0]
    count, bins = np.histogram(cycle_freq, bins=np.arange(0,1.5,0.01))
    ax.plot(bins[:-1], count)
    ax.set_xlim(0,.6)
    ax.set_ylabel('n')
    ax.set_xlabel('freq')
    ax.axvline(np.median(cycle_freq), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(cycle_freq)))
    W, pval = scipy.stats.shapiro(cycle_freq)
    ax.plot(0, 0, label='Shapiro W = {:.3f}, pval = {:.3f}'.format(W, pval)) # for plotting shapiro stats
    ax.legend()
    
        # histogram inspi/expi ratio
    ax = axs[1]
    ratio = (cycle_indexes[:-1, 1] - cycle_indexes[:-1, 0]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()

    plt.close()
   
    return resp_features, fig0, fig1
    





########################################
######## VERIF RESPIFEATURES ########
########################################



if __name__ == '__main__':

    ############################
    ######## PARAMETERS ########
    ############################

    from n0_config import *
    sujet = 'Pilote'


    ############################
    ######## LOAD DATA ########
    ############################


    #### adjust conditions
    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    dirlist_subject = os.listdir()

    cond_keep = []
    for cond in conditions_allsubjects:

        for file in dirlist_subject:

            if file.find(cond) != -1 : 
                cond_keep.append(cond)
                break

    conditions = cond_keep

    #### load data
    raw_allcond = {}

    for session_EEG in range(3):

        raw_allcond_session = {}
        
        for cond in conditions:

            load_i = []
            for session_i, session_name in enumerate(os.listdir()):
                if np.sum([(session_name.find(cond) > 0), (session_name.find(f's{session_EEG+1}') > 0), (session_name.find('lf') != -1 or session_name.find('wb') != -1)]) == 3:                    
                    load_i.append(session_i)
                else:
                    continue

            load_list = [os.listdir()[i] for i in load_i]

            data = []
            for load_name in load_list:
                data.append(mne.io.read_raw_fif(load_name, preload=True))

            raw_allcond_session[cond] = data

        raw_allcond[f's{session_EEG+1}'] = raw_allcond_session


    srate = int(raw_allcond['s1'][os.listdir()[0][10:15]][0].info['sfreq'])
    chan_list = raw_allcond['s1'][os.listdir()[0][10:15]][0].info['ch_names']


    respi_allcond = {}

    for session_EEG in range(3):

        respi_allcond_session = {}

        for cond in conditions:
            
            data = []
            for session_i in range(len(raw_allcond[f's{session_EEG+1}'][cond])):
                if cond == 'FR_MV' :
                    respi_i = chan_list.index('ventral')
                else :
                    respi_i = chan_list.index('Respi')

                data.append(analyse_resp(raw_allcond[f's{session_EEG+1}'][cond][session_i].get_data()[respi_i, :], srate, 0, cond))

            respi_allcond_session[cond] = data

        respi_allcond[f's{session_EEG+1}'] = respi_allcond_session

        
    del raw_allcond


    ########################################
    ######## VERIF RESPIFEATURES ########
    ########################################
    
    if debug == True :

        # info to debug
        cond_len = {}
        for session_eeg in range(3):
            cond_len[f's{session_eeg+1}'] = {}
            for cond in conditions:
                cond_len[f's{session_eeg+1}'][cond] = len(respi_allcond[f's{session_eeg+1}'][cond])
        
        cond_len
        #cond = 'FR_CV'
        cond = 'RD_CV' 
        #cond = 'RD_FV' 
        #cond = 'RD_SV'
        
        session_eeg, session_i = 's1', 0

        respi_allcond[session_eeg][cond][session_i][1].show(), respi_allcond[session_eeg][cond][session_i][2].show()

        #### recompute if necessary
        params = {

        'mean_smooth' : True,

        'baseline_with_average' : True,
        'manual_baseline' : 0.,

        'high_pass_filter' : True,
        'constrain_frequency' : None,
        'median_windows_filter' : False,

        'eliminate_time_shortest_ratio' : 8,
        'eliminate_amplitude_shortest_ratio' : 10,
        'eliminate_mode' : 'OR'

        }

        resp_features, fig0, fig1 = analyse_resp_debug(raw_allcond[session_eeg][cond][session_i].get_data()[respi_i, :], srate, 0, cond, params)
        fig0.show()
        fig1.show()

        #### replace
        respi_allcond[session_eeg][cond][session_i] = [resp_features, fig0, fig1]



    ################################
    ######## SAVE FIG ########
    ################################


    #### when everything ok
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))

    for session_eeg in range(3):

        for cond_i in conditions:

            for i in range(len(respi_allcond[f's{session_eeg+1}'][cond_i])):

                respi_allcond[f's{session_eeg+1}'][cond_i][i][0].to_excel(sujet + f'_s{session_eeg+1}' + '_' + cond_i + '_' + str(i+1) + '_respfeatures.xlsx')
                respi_allcond[f's{session_eeg+1}'][cond_i][i][1].savefig(sujet + f'_s{session_eeg+1}' + '_' + cond_i + '_' + str(i+1) + '_fig0.jpeg')
                respi_allcond[f's{session_eeg+1}'][cond_i][i][2].savefig(sujet + f'_s{session_eeg+1}' + '_' + cond_i + '_' + str(i+1) + '_fig1.jpeg')

