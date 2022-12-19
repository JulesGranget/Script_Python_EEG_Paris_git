

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools

from bycycle.cyclepoints import find_extrema, find_zerox
from bycycle.plts import plot_cyclepoints_array

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





############################
######## LOAD DATA ########
############################

def load_respi_allcond_data(sujet):


    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    raw_allcond = {}

    for cond in conditions_allsubjects:

        load_i = []
        for session_i, session_name in enumerate(os.listdir()):
            if session_name.find(cond) > 0 and (session_name.find('lf') != -1 or session_name.find('wb') != -1):
                load_i.append(session_i)
            else:
                continue

        load_list = [os.listdir()[i] for i in load_i]

        data = []

        for load_name in load_list:

            load_data = mne.io.read_raw_fif(load_name, preload=True)
            load_data = load_data.pick_channels(['PRESS']).get_data() 
            
            data.append(load_data)

        raw_allcond[cond] = data


    #### compute
    respi_allcond = {}
    for cond in conditions:
        
        respi_to_analyze = raw_allcond[cond]
        data.append(analyse_resp(raw_allcond[cond], srate, 0, cond))

        respi_allcond[cond] = data



    respi_allcond_bybycle = {}
    for cond in conditions:
        
        data = []
        for session_i in range(len(raw_allcond[cond])):
            if cond == 'FR_MV' :
                respi_i = chan_list.index('ventral')
            else :
                respi_i = chan_list.index('nasal')
            
            respi_sig = raw_allcond[cond][session_i].get_data()[respi_i, :]
            resp_features_i = correct_resp_features(respi_sig, detection_bycycle(respi_sig, srate), cond, srate)
            data.append(resp_features_i)

        respi_allcond_bybycle[cond] = data

    return raw_allcond, respi_allcond, respi_allcond_bybycle, conditions, chan_list, srate



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
                                                    inspiration_sign = '-',
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
    ax.plot(times[cycle_indexes[:, 0]], resp_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='r', label='inspi')
    ax.plot(times[cycle_indexes[:, 1]], resp_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='g', label='expi')
    #ax.set_xlim(0,120)
    ax.set_ylabel('resp')
    ax.legend()
    

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
                                                    inspiration_sign = '-',
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
    





#sig = raw_allcond[cond][0].get_data()[-4, :]
def detection_bycycle(sig, srate):

    if debug:
        plt.plot(sig)
        plt.show()

        plt.plot(sig_low)
        plt.show()


    #### filter
    sig_low = mne.filter.filter_data(sig, srate, l_freq, h_freq, filter_length='auto', verbose='CRITICAL')

    if sujet in sujet_for_more_filter:
        sig_low = mne.filter.filter_data(sig, srate, 0, .2, filter_length='auto', verbose='CRITICAL')

        cycle_indexes = respirationtools.detect_respiration_cycles(sig_low, srate, t_start=0, output = 'index',
                                                    inspiration_sign = '-',
                                                    # baseline
                                                    baseline_with_average = False,
                                                    # baseline_with_average = True,
                                                    manual_baseline = 0. - sig_low.std()/4,

                                                    high_pass_filter = None,
                                                    constrain_frequency = None,
                                                    median_windows_filter = None,

                                                    # clean
                                                    eliminate_time_shortest_ratio = 8,
                                                    eliminate_amplitude_shortest_ratio = 4,
                                                    eliminate_mode = 'OR', ) # 'AND')

        

        resp_features = respirationtools.get_all_respiration_features(sig_low, srate, cycle_indexes, t_start = 0.)

    if sujet == 'pat_03105_1551':

        cycle_indexes_inspi =   np.array([9.4e2, 3.13e3, 5.05e3, 7.26e3, 8.91e3, 1.019e4, 1.211e4, 1.328e4, 1.497e4, 1.650e4, 1.831e4, 1.998e4, 2.160e4, 2.293e4, 2.459e4, 2.677e4, 2.840e4, 3.045e4, 3.184e4, 3.301e4, 3.445e4, 3.633e4, 3.805e4, 3.949e4, 4.152e4, 4.332e4, 4.496e4, 4.649e4, 
                                4.851e4, 4.998e4, 5.160e4, 5.309e4, 5.473e4, 5.647e4, 5.813e4, 5.970e4, 6.146e4, 6.285e4, 6.496e4, 6.656e4, 6.830e4, 7.007e4, 7.176e4, 7.338e4, 7.474e4, 7.644e4, 7.797e4, 7.960e4, 8.087e4, 8.271e4, 8.391e4, 8.561e4, 8.721e4, 8.872e4, 9.037e4, 9.157e4, 9.317e4, 
                                9.467e4, 9.657e4, 9.844e4, 1.0157e5, 1.0318e5, 1.0487e5, 1.0634e5, 1.0819e5, 1.0974e5, 1.1110e5, 1.1300e5, 1.1562e5, 1.1779e5, 1.1923e5, 1.2085e5, 1.2234e5, 1.2393e5, 1.2598e5, 1.2742e5, 1.2889e5, 1.3017e5, 1.3154e5, 1.3300e5, 1.3454e5, 1.3608e5, 
                                1.3746e5, 1.3866e5, 1.4042e5, 1.4188e5, 1.4379e5, 1.4534e5, 1.4679e5, 1.4852e5, 1.5009e5])
        cycle_indexes_expi =    np.array([1.46e3, 3.68e3, 5.55e3, 7.83e3, 9.39e3, 1.094e4, 1.254e4, 1.376e4, 1.548e4, 1.699e4, 1.879e4, 2.048e4, 2.208e4, 2.362e4, 2.525e4, 2.729e4, 2.893e4, 3.099e4, 3.249e4, 3.351e4, 3.501e4, 3.702e4, 3.860e4, 4.001e4, 4.211e4, 4.375e4, 4.551e4, 4.713e4, 
                                4.900e4, 5.054e4, 5.208e4, 5.366e4, 5.526e4, 5.687e4, 5.868e4, 6.021e4, 6.202e4, 6.348e4, 6.558e4, 6.709e4, 6.882e4, 7.062e4, 7.229e4, 7.392e4, 7.518e4, 7.701e4, 7.851e4, 8.000e4, 8.144e4, 8.326e4, 8.468e4, 8.612e4, 8.765e4, 8.925e4, 9.077e4, 
                                9.199e4, 9.371e4, 9.520e4, 9.756e4, 1.0031e5, 1.0205e5, 1.0371e5, 1.0544e5, 1.0707e5, 1.0862e5, 1.1020e5, 1.1164e5, 1.1367e5, 1.1629e5, 1.1829e5, 1.1992e5, 1.2132e5, 1.2287e5, 1.2447e5, 1.2650e5, 1.2797e5, 1.2940e5, 1.3075e5, 1.3213e5, 1.3355e5, 
                                1.3503e5, 1.3647e5, 1.3788e5, 1.3930e5, 1.4088e5, 1.4251e5, 1.4431e5, 1.4578e5, 1.4728e5, 1.4905e5, -1])

    if sujet == 'pat_03128_1591':

        cycle_indexes_inspi =   np.array([2.54e3, 5.02e3, 9.08e3, 1.231e4, 1.520e4, 1.753e4, 2.040e4, 2.282e4, 2.612e4, 2.872e4, 3.178e4, 3.453e4, 3.688e4, 3.923e4, 4.176e4, 4.428e4, 4.707e4, 4.920e4, 5.199e4, 5.456e4, 5.728e4, 6.030e4, 6.294e4, 6.579e4, 6.900e4, 7.138e4, 7.417e4, 7.661e4, 
                                7.929e4, 8.176e4, 8.428e4, 8.670e4, 8.900e4, 9.163e4, 9.464e4, 9.730e4, 9.983e4, 1.0215e5, 1.0460e5, 1.0717e5, 1.0961e5, 1.1214e5, 1.1436e5, 1.1706e5, 1.1973e5, 1.2241e5, 1.2496e5, 1.2730e5, 1.2950e5, 1.3216e5, 1.3466e5, 1.3719e5, 1.3957e5, 1.4197e5, 
                                1.4460e5, 1.4711e5, 1.4923e5])
        cycle_indexes_expi =    np.array([3.10e3, 5.54e3, 9.78e3, 1.290e4, 1.577e4, 1.843e4, 2.090e4, 2.351e4, 2.674e4, 2.953e4, 3.242e4, 3.509e4, 3.745e4, 3.976e4, 4.238e4, 4.493e4, 4.766e4, 4.982e4, 5.259e4, 5.510e4, 5.790e4, 6.094e4, 6.352e4, 6.643e4, 6.952e4, 7.191e4, 7.474e4, 7.705e4, 
                                7.986e4, 8.234e4, 8.486e4, 8.724e4, 8.964e4, 9.226e4, 9.520e4, 9.785e4, 1.0041e5, 1.0258e5, 1.0514e5, 1.0776e5, 1.1024e5, 1.1272e5, 1.1496e5, 1.1755e5, 1.2048e5, 1.2296e5, 1.2546e5, 1.2777e5, 1.3014e5, 1.3272e5, 1.3520e5, 1.3763e5, 1.4018e5, 1.4264e5, 1.4526e5, 1.4753e5, 
                                1.4988e5])
        
        cycle_indexes = np.zeros((cycle_indexes_inspi.shape[0], 2))
        for i, (x, y) in enumerate(zip(cycle_indexes_inspi, cycle_indexes_expi)):
            cycle_indexes[i,:] = [x, y]
        cycle_indexes = np.asarray(cycle_indexes, np.int64)

        resp_features = respirationtools.get_all_respiration_features(sig_low, srate, cycle_indexes, t_start = 0.)

    if debug:

        times = np.arange(0, sig.shape[0])/srate
        plt.plot(times, sig, label='original')
        plt.plot(times, sig_low, label='filtered')
        plt.legend()
        plt.show()

        plt.plot(times, sig_low, label='filtered')
        plt.plot(times[cycle_indexes.T[0,:]], sig_low[cycle_indexes.T[0,:]], ls='None', marker='o', color='r', label='decays')
        plt.plot(times[cycle_indexes.T[1,:]], sig_low[cycle_indexes.T[1,:]], ls='None', marker='o', color='b', label='rises')
        plt.legend()
        plt.show()

    #### detect
    if sujet in sujet_for_more_filter or sujet in sujet_manual_detection:
        rises, decays = cycle_indexes.T[1,:-1], cycle_indexes.T[0,:]

        # peaks, _ = scipy.signal.find_peaks(sig_low, distance=3*srate, prominence=sig_low.mean()+sig_low.std()/15)
        # troughs, _ = scipy.signal.find_peaks(sig_low*-1, distance=3*srate, prominence=sig_low.mean()+sig_low.std()/15)
    else:
        peaks, troughs = find_extrema(sig_low, srate, f_theta)
        rises, decays = find_zerox(sig_low, peaks, troughs)

    if debug:
        times = np.arange(0, sig_low.shape[0])/srate
        plt.plot(times, sig_low)
        plt.plot(times[rises], sig_low[rises], ls='None', marker='o', color='r', label='rises')
        plt.plot(times[decays], sig_low[decays], ls='None', marker='o', color='b', label='decays')
        plt.plot(times[peaks], sig_low[peaks], ls='None', marker='o', color='g', label='peaks')
        plt.plot(times[troughs], sig_low[troughs], ls='None', marker='o', color='k', label='troughs')
        plt.legend()
        plt.show()

    #### adjust detection
    
    if decays[0] > rises[0]:
        decays = decays[1:]
        troughs = troughs[1:]

    if decays[-1] < rises[-1]:
        rises = rises[:-1]
        troughs = troughs[:-1]

    if sujet not in sujet_for_more_filter and sujet not in sujet_manual_detection: 

        if peaks[0] < decays[0]:
            peaks = peaks[1:]

        if troughs[-1] > decays[-1]:
            troughs = troughs[:-1]
        
    #### generate df
    #### INSPI SIGN = -
    data_detection = {'cycle_num' : range(rises.shape[0]), 'inspi_index' : decays[:-1], 'expi_index' : rises, 
    'inspi_time' : decays[:-1]/srate, 'expi_time' : rises/srate, 'select' : [1]*rises.shape[0]}
    
    df_detection = pd.DataFrame(data_detection, columns=['cycle_num', 'inspi_index', 'expi_index', 'inspi_time', 'expi_time', 'select'])

    df_detection['cycle_duration'] = np.diff(decays/srate)
    df_detection['insp_duration'] = df_detection['expi_time'] - df_detection['inspi_time']
    df_detection['exp_duration'] = df_detection['cycle_duration'] - df_detection['insp_duration']
    df_detection['cycle_freq'] = 1/df_detection['cycle_duration']

    if sujet in sujet_for_more_filter or sujet in sujet_manual_detection:
        df_detection['insp_amplitude'] = resp_features['insp_amplitude']
        df_detection['exp_amplitude'] = resp_features['exp_amplitude']
        df_detection['total_amplitude'] = resp_features['total_amplitude']
    else:  
        df_detection['insp_amplitude'] = sig[peaks] - sig[df_detection['inspi_index'].values]
        df_detection['exp_amplitude'] = np.abs(sig[troughs] - sig[df_detection['expi_index'].values])
        df_detection['total_amplitude'] = df_detection['insp_amplitude'] + df_detection['exp_amplitude']

    #### verif
    if debug:
        plot_cyclepoints_array(sig_low, srate, peaks=peaks, troughs=troughs, 
        rises=df_detection['inspi_index'], decays=df_detection['expi_index'])
        plt.show()

    #### supress cycle freq based
    mean_freq = np.mean(df_detection['cycle_freq'].values)
    std_freq = np.std(df_detection['cycle_freq'].values)

    delete_i_freq = np.where( (df_detection['cycle_freq'].values > mean_freq + SD_delete_cycles_freq*std_freq) | (df_detection['cycle_freq'].values < mean_freq - SD_delete_cycles_freq*std_freq) )[0]

    if delete_i_freq.shape[0] != 0:
        df_detection['select'][delete_i_freq] = np.array([0]*delete_i_freq.shape[0])
    df_detection_deleted = df_detection.loc[delete_i_freq]

    #### supress cycle amp based
    mean_amp = np.mean(df_detection['total_amplitude'].values)
    std_amp = np.std(df_detection['total_amplitude'].values)

    delete_i_amp = np.where( (df_detection['total_amplitude'].values > mean_amp + SD_delete_cycles_amp*std_amp) | (df_detection['total_amplitude'].values < mean_amp - SD_delete_cycles_amp*std_amp) )[0]

    if delete_i_amp.shape[0] != 0:
        df_detection['select'][delete_i_amp] = np.array([0]*delete_i_amp.shape[0])
    df_detection_deleted = pd.concat([df_detection_deleted, df_detection.loc[delete_i_amp]])

    #### verif
    if debug:
        plot_cyclepoints_array(sig_low, srate, peaks=peaks, troughs=troughs, 
        rises=df_detection['inspi_index'], decays=df_detection['expi_index'])
        plt.show()
        
        plot_cyclepoints_array(sig_low, srate, peaks=peaks, troughs=troughs, 
        rises=df_detection_deleted['inspi_index'], decays=df_detection_deleted['expi_index'])
        plt.show()

    return df_detection





#df_detection, respi_sig = detection_bycycle(respi_sig, srate), respi_sig
def correct_resp_features(respi_sig, df_detection, cond, srate):

    cycle_indexes = np.concatenate((df_detection['expi_index'][df_detection['select'] == 1].values.reshape(-1,1), 
                                    df_detection['inspi_index'][df_detection['select'] == 1].values.reshape(-1,1)), axis=1)
    cycle_freq = df_detection['cycle_freq'][df_detection['select'] == 1].values
    cycle_amplitudes = df_detection['total_amplitude'][df_detection['select'] == 1].values

    fig0, axs = plt.subplots(nrows=3, sharex=True)
    plt.suptitle(cond)
    times = np.arange(respi_sig.size)/srate
    
        # respi signal with inspi expi markers
    ax = axs[0]
    ax.plot(times, respi_sig)
    ax.plot(times[cycle_indexes[:, 1]], respi_sig[cycle_indexes[:, 1]], ls='None', marker='o', color='r', label='inspi')
    ax.plot(times[cycle_indexes[:, 0]], respi_sig[cycle_indexes[:, 0]], ls='None', marker='o', color='g', label='expi')
    ax.set_ylabel('resp')
    ax.legend()
    

        # instantaneous frequency
    ax = axs[1]
    ax.plot(times[cycle_indexes[:, 0]], cycle_freq)
    ax.set_ylim(0, max(cycle_freq)*1.1)
    ax.axhline(np.median(cycle_freq), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_freq)))
    ax.legend()
    ax.set_ylabel('freq')

        # instantaneous amplitude
    ax = axs[2]
    ax.plot(times[cycle_indexes[:, 0]], cycle_amplitudes)
    ax.axhline(np.median(cycle_amplitudes), color='m', linestyle='--', label='median={:.3f}'.format(np.median(cycle_amplitudes)))
    ax.set_ylabel('amplitude')
    ax.legend()
    # plt.show()

    plt.close()
    
    
    # respi cycle features

    fig1, axs = plt.subplots(nrows=2)
    plt.suptitle(cond)

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
    ratio = (cycle_indexes[:-1, 0] - cycle_indexes[:-1, 1]).astype('float64') / (cycle_indexes[1:, 0] - cycle_indexes[:-1, 0])
    count, bins = np.histogram(ratio, bins=np.arange(0, 1., 0.01))
    ax.plot(bins[:-1], count)
    ax.axvline(np.median(ratio), color='m', linestyle='--', label='median = {:.3f}'.format(np.median(ratio)))
    ax.set_ylabel('n')
    ax.set_xlabel('ratio')
    ax.legend()
    # plt.show()

    plt.close()

    return [df_detection, fig0, fig1]





########################################
######## EDIT CYCLES SELECTED ########
########################################


#respi_allcond = respi_allcond_bybycle
def edit_df_for_sretch_cycles_deleted(sujet, respi_allcond, raw_allcond):

    conditions, chan_list, chan_list_ieeg, srate = extract_chanlist_srate_conditions(sujet)

    for cond in conditions:
        
        for session_i in range(len(raw_allcond[cond])):

            #### params
            respi_i = chan_list.index('nasal')
            respi = raw_allcond[cond][session_i].get_data()[respi_i, :]
            cycle_times = respi_allcond[cond][session_i][0][['inspi_time', 'expi_time']].values
            mean_cycle_duration = np.mean(respi_allcond[cond][session_i][0][['insp_duration', 'exp_duration']].values, axis=0)
            mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
            times = np.arange(0,respi.shape[0])/srate

            #### stretch
            clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                    respi, times, cycle_times, nb_point_by_cycle=stretch_point_TF, inspi_ratio=ratio_stretch_TF)

            i_to_update = respi_allcond[cond][session_i][0].index.values[~np.isin(respi_allcond[cond][session_i][0].index.values, cycles)]
            respi_allcond[cond][session_i][0]['select'][i_to_update] = np.array([0]*i_to_update.shape[0])



def export_sniff_count(sujet, respi_allcond):

    #### generate df
    df_count_cycle = pd.DataFrame(columns={'sujet' : [], 'cond' : [], 'trial' : [], 'count' : []})

    for cond in conditions:
        
        for session_i in range(len(raw_allcond[cond])):

            data_i = {'sujet' : [sujet], 'cond' : [cond], 'trial' : [session_i+1], 'count' : [np.sum(respi_allcond[cond][session_i][0]['select'].values)]}
            df_i = pd.DataFrame(data_i, columns=data_i.keys())
            df_count_cycle = pd.concat([df_count_cycle, df_i])

    #### export
    os.chdir(os.path.join(path_results, sujet, 'RESPI'))
    df_count_cycle.to_excel(f'{sujet}_count_cycles.xlsx')




############################
######## SAVE DATA ########
############################


def save_all_respfeatures(respi_allcond, respi_allcond_bybycle, conditions, export):

    #### when everything ok classic
    if export == 'sam':
        os.chdir(os.path.join(path_results, sujet, 'RESPI'))

        for cond_i in conditions:

            for i in range(len(respi_allcond[cond_i])):

                respi_allcond[cond_i][i][0].to_excel(sujet + '_' + cond_i + '_' + str(i+1) + '_respfeatures.xlsx')
                respi_allcond[cond_i][i][1].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig0.jpeg')
                respi_allcond[cond_i][i][2].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig1.jpeg')

    #### when everything ok bycycle
    if export == 'bycycle':
        os.chdir(os.path.join(path_results, sujet, 'RESPI'))

        for cond_i in conditions:

            for i in range(len(respi_allcond_bybycle[cond_i])):

                respi_allcond_bybycle[cond_i][i][0].to_excel(sujet + '_' + cond_i + '_' + str(i+1) + '_respfeatures.xlsx')
                respi_allcond_bybycle[cond_i][i][1].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig0.jpeg')
                respi_allcond_bybycle[cond_i][i][2].savefig(sujet + '_' + cond_i + '_' + str(i+1) + '_fig1.jpeg')




if __name__ == '__main__':

    ############################
    ######## LOAD DATA ########
    ############################

    
    #### whole protocole
    sujet = 'PD01'

    #### load data
    raw_allcond, respi_allcond, respi_allcond_bybycle, conditions, chan_list, srate = load_respi_allcond_data(sujet)



    ########################################
    ######## VERIF RESPIFEATURES ########
    ########################################
    
    if debug == True :

        # info to debug
        cond_len = {}
        for cond in conditions:
            cond_len[cond] = len(respi_allcond[cond])
        
        cond_len
        cond = 'RD_CV' 
        cond = 'RD_FV' 
        cond = 'RD_SV'

        cond = 'FR_CV'

        cond = 'RD_AV'
        cond = 'FR_MV'
        
        session_i = 0

        respi_allcond[cond][session_i][1].show()
        respi_allcond[cond][session_i][2].show()

        respi_allcond_bybycle[cond][session_i][1].show()
        respi_allcond_bybycle[cond][session_i][2].show()

        #### recompute
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

        #### adjust for MOUTH VENTILATION
        if cond == 'FR_MV':
            respi_i = chan_list.index('ventral')
        else:
            respi_i = chan_list.index('nasal')

        resp_features, fig0, fig1 = analyse_resp_debug(raw_allcond[cond][session_i].get_data()[respi_i, :], srate, 0, cond, params)
        fig0.show()
        fig1.show()

        #### changes
        # CHEe : 'eliminate_time_shortest_ratio' : 2, 'eliminate_amplitude_shortest_ratio' : 10, 'baseline_with_average' : False
        # TREt : 'median_windows_filter' : True, 'eliminate_time_shortest_ratio' : 8, 'eliminate_amplitude_shortest_ratio' : 10, 'mean_smooth' : True

        #### replace
        respi_allcond[cond][session_i] = [resp_features, fig0, fig1]




    ########################################
    ######## EDIT CYCLES SELECTED ########
    ########################################

    edit_df_for_sretch_cycles_deleted(sujet, respi_allcond_bybycle, raw_allcond)

    export_sniff_count(sujet, respi_allcond_bybycle)

    ################################
    ######## SAVE FIG ########
    ################################

    #### select export
    # export = 'sam'
    export = 'bycycle'

    save_all_respfeatures(respi_allcond, respi_allcond_bybycle, conditions, export)


    
