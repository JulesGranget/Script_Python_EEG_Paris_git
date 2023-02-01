

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import physio

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



def exclude_bad_cycles_locally(respi, cycles, srate, metric='integral', exclusion_metrics='med', metric_coeff_exclusion=3, window=30, 
                            count_exclusion_coeff=0.5, respi_scale=[0.05, 0.55], plot_detection=False, plot_n_cycles=-1):

    """

    Local detection for cycle exclusion based on drift from the mean, med or mode.

    To be exclude definitively a cycle need to be exclude in a certain number of window define by the count_exclusion_coeff.
    If None, every cycle excluded at least one time is taking in account.

    If a cycle frequency is outside the resp_scale, the cycle is automaticaly excluded.

    """

    #### get inspi
    inspi_starts = cycles[:-1,0]

    #### compute cycle metric
    amplitudes = np.zeros(inspi_starts.shape[0])
    sums = np.zeros(inspi_starts.shape[0])

    for cycle_i in range(inspi_starts.shape[0]):
        if cycle_i == inspi_starts.shape[0]-1:
            start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]
        else:
            start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

        amplitudes[cycle_i] = np.abs(np.min(respi[start_i:stop_i])) + np.abs(np.max(respi[start_i:stop_i])) 
        sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i]))

    if metric == 'amplitude':
        cycle_metrics = amplitudes
    if metric == 'integral':
        cycle_metrics = sums

    #### compute cycles durations
    durations = np.diff(inspi_starts/srate)

    #### initiate containers
    cycle_excluded_indx = np.array([])
    cycle_excluded_duration_indx = np.array([])
    cycle_metrics_allwindow_val = np.array([])

    if plot_n_cycles != -1:
        inspi_starts = inspi_starts[:plot_n_cycles]

    #### identify cycles to exclude
    #cycle_i, inspi_i = 10, inspi_starts[10]
    for cycle_i, inspi_i in enumerate(inspi_starts):

        #### chunk the signal
        if (inspi_i + window/2*srate) > respi.shape[0]:
            stop_i = respi.shape[0]
            start_i = stop_i - window*srate

        if (inspi_i - window/2*srate) < 0:
            start_i = 0
            stop_i = start_i + window*srate

        else:
            start_i = inspi_i - window/2*srate
            stop_i = inspi_i + window/2*srate

        cycle_sel_mask = (inspi_starts >= start_i) & (inspi_starts <= stop_i)
        chunk_metrics_val = cycle_metrics[cycle_sel_mask]
        
        #### exclude with cycle freq
        chunk_cycle_duration_val = durations[cycle_sel_mask[:-1]]
        chunk_cycle_duration_excluded_i = [i for i, val in enumerate(chunk_cycle_duration_val) if (val > 1/respi_scale[0] or val < 1/respi_scale[1])]
        cycle_excluded_duration_i = np.arange(durations.shape[0])[cycle_sel_mask[:-1]][chunk_cycle_duration_excluded_i]

        #### exclude
        if exclusion_metrics == 'med':
            med, mad = physio.compute_median_mad(chunk_metrics_val)
            chunk_metrics_excluded = chunk_metrics_val[(chunk_metrics_val < (med - mad*metric_coeff_exclusion)) | (chunk_metrics_val > (med + mad*metric_coeff_exclusion))]
        if exclusion_metrics == 'mean':
            _mean, _std = chunk_metrics_val.mean(), chunk_metrics_val.std()
            chunk_metrics_excluded = chunk_metrics_val[(chunk_metrics_val < (_mean - _std*metric_coeff_exclusion)) | (chunk_metrics_val > (_mean + _std*metric_coeff_exclusion))]
        if exclusion_metrics == 'mod':
            med, mad = physio.compute_median_mad(chunk_metrics_val)
            mod = physio.get_empirical_mode(respi)
            chunk_metrics_excluded = chunk_metrics_val[(chunk_metrics_val < (mod - mad*metric_coeff_exclusion)) | (chunk_metrics_val > (mod + mad*metric_coeff_exclusion))]

        chunk_cycles_excluded_i = [i for i, val in enumerate(cycle_metrics) if val in chunk_metrics_excluded]

        #### plot cycles
        if plot_detection:

            chunk_cycle_time = inspi_starts[cycle_sel_mask]/srate

            cycle_excluded_i = [i for i, val in enumerate(chunk_metrics_val) if val in chunk_metrics_excluded]
            cycle_metrics_excluded_time = chunk_cycle_time[cycle_excluded_i]

            cycle_excluded_duration_time = chunk_cycle_time[chunk_cycle_duration_excluded_i]
            cycle_excluded_duration_val = chunk_metrics_val[chunk_cycle_duration_excluded_i]

            plt.plot(chunk_cycle_time, chunk_metrics_val)
            plt.scatter(chunk_cycle_time, chunk_metrics_val, color='g', label='selected')
            plt.scatter(cycle_metrics_excluded_time, chunk_metrics_excluded, color='r', label='excluded metric')
            plt.scatter(cycle_excluded_duration_time, cycle_excluded_duration_val, color='k', label='excluded duration', marker='x')

            if exclusion_metrics == 'med':
                plt.axhline(med, color='r')
                plt.axhline(med + mad*metric_coeff_exclusion, color='g')
                plt.axhline(med - mad*metric_coeff_exclusion, color='g')
            if exclusion_metrics == 'mean':
                plt.axhline(_mean, color='r')
                plt.axhline(_mean + _std*metric_coeff_exclusion, color='g')
                plt.axhline(_mean - _std*metric_coeff_exclusion, color='g')
            if exclusion_metrics == 'mod':
                plt.axhline(mod, color='r')
                plt.axhline(mod + mad*metric_coeff_exclusion, color='g')
                plt.axhline(mod - mad*metric_coeff_exclusion, color='g')

            plt.axvline(inspi_i/srate, color='b')
            plt.title(f'{cycle_i+1}/{inspi_starts[:-1].shape[0]}')
            plt.legend()

            plt.show()

        cycle_excluded_indx = np.concatenate((cycle_excluded_indx, chunk_cycles_excluded_i), axis=0)
        cycle_excluded_duration_indx = np.concatenate((cycle_excluded_duration_indx, cycle_excluded_duration_i), axis=0)
        cycle_metrics_allwindow_val = np.concatenate((cycle_metrics_allwindow_val, chunk_metrics_val), axis=0)

    if plot_detection:
        return
        
    #### metric exclusion
    cycle_excluded_indx_unique, cycle_exclude_count = np.unique(cycle_excluded_indx, return_counts=True)
    cycle_excluded_indx_unique = cycle_excluded_indx_unique.astype('int')

    cycle_metrics_allwindow_val_unique, cycle_metrics_allwindow_val_count = np.unique(cycle_metrics_allwindow_val, return_counts=True)

    if cycle_metrics_allwindow_val_unique.shape[0] != cycle_metrics.shape[0]:
        raise ValueError('Not all cycle metric have been detected')    
    
    if count_exclusion_coeff == None:
        cycle_excluded_metric_indx = cycle_excluded_indx_unique
    else:
        mask_exclusion = (cycle_exclude_count / cycle_metrics_allwindow_val_count[cycle_excluded_indx_unique]) >= count_exclusion_coeff
        cycle_excluded_metric_indx = cycle_excluded_indx_unique[mask_exclusion]

    #### duration exclusion
    cycle_excluded_duration_indx_unique = np.unique(cycle_excluded_duration_indx).astype('int')
    cycle_exclude_final = np.unique(np.concatenate((cycle_excluded_metric_indx, cycle_excluded_duration_indx_unique))).astype('int')

    #### fig for all detection
    time_vec = np.arange(respi.shape[0])/srate
    
    fig_respi_exclusion, ax = plt.subplots()
    ax.plot(time_vec, respi)
    ax.scatter(inspi_starts/srate, respi[inspi_starts], color='g', label='selected')
    ax.scatter(inspi_starts[cycle_excluded_metric_indx]/srate, respi[inspi_starts[cycle_excluded_metric_indx]], color='r', label='excluded_metric')
    ax.scatter(inspi_starts[cycle_excluded_duration_indx_unique]/srate, respi[inspi_starts[cycle_excluded_duration_indx_unique]], color='k', label='excluded_duration', marker='x')
    plt.legend()
    # plt.show()

    return cycle_exclude_final, fig_respi_exclusion





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
    sujet = 'MJ02'
    sujet = 'VN03'
    sujet = 'GB04'
    sujet = 'LV05'
    sujet = 'EF06'
    sujet = 'PB07'
    sujet = 'DM08'
    sujet = 'TA09'
    sujet = 'BH10'
    sujet = 'FA11'
    sujet = 'BD12'
    sujet = 'FP13'
    sujet = 'MD14'
    sujet = 'LG15'
    sujet = 'GM16'
    sujet = 'JR17'
    sujet = 'SE18'
    sujet = 'TM19'
    sujet = 'TY20'
    sujet = 'ZV21'
    sujet = 'DI22'
    sujet = 'LF23'
    sujet = 'TJ24'
    sujet = 'DF25'
    sujet = 'MN26'
    sujet = 'BD27'
    sujet = 'NT28'
    sujet = 'SC29'
    sujet = 'AR30'
    sujet = 'HJ31'
    sujet = 'CM32'
    sujet = 'MA33'

    #### load data
    os.chdir(os.path.join(path_data, 'respi_detection'))

    srate = get_params(sujet)['srate']

    cond = 'FR_CV_1'

    session_i = 0
    file_df = [file_i for file_i in os.listdir() if file_i.find(f'{sujet}_ses0{session_i+2}') != -1][0]

    if cond == 'FR_CV_1':
        raw_df = pd.read_excel(file_df, sheet_name='Variables cycle BL')
    if cond == 'MECA':
        raw_df = pd.read_excel(file_df, sheet_name='Variables cycle T1')
    if cond == 'CO2':
        raw_df = pd.read_excel(file_df, sheet_name='Variables cycle T2')
    if cond == 'FR_CV_2':
        raw_df = pd.read_excel(file_df, sheet_name='VS POST')
    
    raw_df_val = raw_df['Temps de début de cycle'].values[1:].astype('float')
    cycle_inspi_val = raw_df_val[~np.isnan(raw_df_val)].astype('int')*srate

    data = load_data_sujet(sujet, 'wb', cond, 'o')
    respi = data[-3,:]

    plt.plot(respi)
    plt.vlines(cycle_inspi_val, ymin=respi.min(), ymax=respi.max(), color='r')
    plt.show()

    
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


    
