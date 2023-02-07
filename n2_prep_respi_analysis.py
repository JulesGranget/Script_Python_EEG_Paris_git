

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






########################################
######## COMPUTE RESPI FEATURES ########
########################################


#respi = respi_allcond[cond][odor_i]
def exclude_bad_cycles(respi, cycles, srate, exclusion_metrics='med', metric_coeff_exclusion=3, inspi_coeff_exclusion=2, respi_scale=[0.1, 0.35]):

    cycles_init = cycles.copy()

    #### exclude regarding inspi/expi diff
    _diff = np.log(np.diff(cycles_init[:-1,:], axis=1).reshape(-1))

    if debug:
        plt.plot(zscore(_diff))
        plt.plot(zscore(np.log(_diff)))
        plt.show()

    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(_diff)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = _diff.mean(), _diff.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(_diff)
        mod = physio.get_empirical_mode(_diff)
        metric_center, metric_dispersion = mod, med

    # inspi_time_excluded = _diff[(_diff < (metric_center - metric_dispersion*inspi_coeff_exclusion)) | (_diff > (metric_center + metric_dispersion*inspi_coeff_exclusion))]
    inspi_time_excluded = _diff[(_diff < (metric_center - metric_dispersion*inspi_coeff_exclusion))]
    inspi_time_included_i = [i for i, val in enumerate(_diff) if val not in inspi_time_excluded]
    inspi_time_excluded_i = [i for i, val in enumerate(_diff) if val in inspi_time_excluded]

    cycle_inspi_excluded_i = [i for i, val in enumerate(cycles_init[:,0]) if val in cycles_init[inspi_time_excluded_i][:,0]]

    cycles = cycles_init[inspi_time_included_i,:]
    inspi_starts = cycles[:,0]

    if debug:

        inspi_starts_init = cycles_init[:-1,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        ax.scatter(inspi_starts_init[cycle_inspi_excluded_i], respi[inspi_starts_init[cycle_inspi_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts_init, _diff, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*inspi_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*inspi_coeff_exclusion, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### compute cycle metric
    sums = np.zeros(inspi_starts.shape[0])

    for cycle_i in range(inspi_starts.shape[0]):
        if cycle_i == inspi_starts.shape[0]-1:
            start_i, stop_i = inspi_starts[cycle_i], respi.shape[0]
        else:
            start_i, stop_i = inspi_starts[cycle_i], inspi_starts[cycle_i+1] 

        sums[cycle_i] = np.sum(np.abs(respi[start_i:stop_i] - respi[start_i:stop_i].mean()))

    cycle_metrics = np.log(sums)

    if debug:
        plt.plot(zscore(sums))
        plt.plot(zscore(np.log(sums)), label='log')
        plt.legend()
        plt.show()

    #### exclude regarding duration
    durations = np.diff(inspi_starts/srate)

    # cycle_duration_sel_i = [i for i, val in enumerate(durations) if (val > 1/respi_scale[0] or val < 1/respi_scale[1]) == False]
    # cycle_duration_excluded_i = [i for i, val in enumerate(durations) if (val > 1/respi_scale[0] or val < 1/respi_scale[1])]
    cycle_duration_sel_i = [i for i, val in enumerate(durations) if (val < 1/respi_scale[1]) == False]
    cycle_duration_excluded_i = [i for i, val in enumerate(durations) if (val < 1/respi_scale[1])]
    cycle_metrics_cleaned = cycle_metrics[cycle_duration_sel_i]

    cycles = cycles[cycle_duration_sel_i, :]

    #### exclude regarding metric
    if exclusion_metrics == 'med':
        med, mad = physio.compute_median_mad(cycle_metrics_cleaned)
        metric_center, metric_dispersion = med, mad

    if exclusion_metrics == 'mean':
        metric_center, metric_dispersion = cycle_metrics_cleaned.mean(), cycle_metrics_cleaned.std()

    if exclusion_metrics == 'mod':
        med, mad = physio.compute_median_mad(cycle_metrics_cleaned)
        mod = physio.get_empirical_mode(cycle_metrics_cleaned)
        metric_center, metric_dispersion = mod, med

    # chunk_metrics_excluded = cycle_metrics_cleaned[(cycle_metrics_cleaned < (metric_center - metric_dispersion*metric_coeff_exclusion)) | (cycle_metrics_cleaned > (metric_center + metric_dispersion*metric_coeff_exclusion))]
    cycle_metrics_excluded = cycle_metrics_cleaned[(cycle_metrics_cleaned < (metric_center - metric_dispersion*metric_coeff_exclusion))]
    cycle_metrics_excluded_i = [i for i, val in enumerate(cycle_metrics_cleaned) if val in cycle_metrics_excluded]

    #### verif cycle metrics
    if debug:

        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[cycle_duration_excluded_i], respi[inspi_starts[cycle_duration_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts, cycle_metrics, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*metric_coeff_exclusion, color='r', linestyle='--')
        plt.legend()
        plt.show()

    #### final cleaning
    cycles_final = np.concatenate((cycles, cycles_init[-1, :].reshape(1, -1)), axis=0)
    cycles_mask_keep = np.ones((cycles_final.shape[0]), dtype='int')
    cycles_mask_keep[cycle_metrics_excluded_i] = 0
    cycles_mask_keep[-1] = 0

    #### fig for all detection
    time_vec = np.arange(respi.shape[0])/srate
    
    inspi_starts_init = cycles_init[:-1,0]
    fig_respi_exclusion, ax = plt.subplots(figsize=(18, 10))
    ax.plot(time_vec, respi)
    ax.scatter(inspi_starts_init/srate, respi[inspi_starts_init], color='g', label='inspi_selected')
    ax.scatter(cycles_init[:-1,1]/srate, respi[cycles_init[:-1,1]], color='c', label='expi_selected', marker='s')
    ax.scatter(inspi_starts_init[cycle_inspi_excluded_i]/srate, respi[inspi_starts_init[cycle_inspi_excluded_i]], color='m', label='excluded_inspi', marker='+', s=200)
    ax.scatter(inspi_starts[cycle_duration_excluded_i]/srate, respi[inspi_starts[cycle_duration_excluded_i]], color='k', label='excluded_duration', marker='x', s=200)
    ax.scatter(cycles[:,0][cycle_metrics_excluded_i]/srate, respi[cycles[:,0][cycle_metrics_excluded_i]], color='r', label='excluded_metric')
    plt.legend()
    # plt.show()
    plt.close()

    #### fig final
    inspi_starts_init = cycles_init[:-1,0]
    fig_final, ax = plt.subplots(figsize=(18, 10))
    ax.plot(time_vec, respi)
    ax.scatter(cycles[:,0]/srate, respi[cycles[:,0]], color='g', label='inspi_selected')
    ax.scatter(cycles[:-1,1]/srate, respi[cycles[:-1,1]], color='c', label='expi_selected', marker='s')
    ax.scatter(cycles[:,0][cycle_metrics_excluded_i]/srate, respi[cycles[:,0][cycle_metrics_excluded_i]], color='r', label='excluded_metric')
    plt.legend()
    # plt.show()
    plt.close()

    return cycles_final, cycles_mask_keep, fig_respi_exclusion, fig_final






def get_all_respiration_features(respi, srate, cycles, cycles_mask_keep, t_start = 0.):
    
    times = np.arange(respi.size)/srate + t_start
    
    assert cycles.dtype.kind == 'i'
    
    n = cycles.shape[0]-1
    
    index = np.arange(n, dtype = 'int64')
    df = pd.DataFrame(index = index)
    
    insp = times[cycles[:-1,0]]
    expi = times[cycles[:-1,1]]
    insp_next = times[cycles[1:,0]]

    ix1 = cycles[:-1,0]
    ix2 = cycles[:-1,1]
    ix3 = cycles[1:,0]
    
    df['cycle_num'] = pd.Series(range(n) , index = index, dtype = int)
    df['inspi_index'] = pd.Series(ix1 , index = index, dtype = 'int64')
    df['expi_index'] = pd.Series(ix2 , index = index, dtype = 'int64')
    df['inspi_time'] = pd.Series(insp , index = index)
    df['expi_time'] = pd.Series(expi , index = index)
    df['cycle_duration'] = pd.Series(insp_next - insp  , index = index)
    df['insp_duration'] = pd.Series(expi - insp  , index = index)
    df['exp_duration'] = pd.Series(insp_next - expi  , index = index)
    df['cycle_freq'] = 1./df['cycle_duration']
    for k in ('insp_volume', 'exp_volume', 'total_amplitude', 'insp_amplitude', 'exp_amplitude'):
        df[k] = pd.Series(index = index)
    
    #missing cycle
    mask = ix2==-1
    df.loc[mask, ['expi_time', 'cycle_duration', 'insp_duration', 'exp_duration', 'cycle_freq']] = np.nan
    
    for c in range(n):
        i1, i2, i3 = ix1[c], ix2[c], ix3[c]
        if i2 == -1:
            #this is a missing cycle in the middle
            continue
        #~ print(i1, i2, i3)
        df.loc[c, 'insp_volume'] = np.sum(respi[i1:i2])/srate
        df.loc[c, 'exp_volume'] = np.sum(respi[i2:i3])/srate
        df.loc[c, 'insp_amplitude'] = np.max(np.abs(respi[i1:i2]))
        df.loc[c, 'exp_amplitude'] = np.max(np.abs(respi[i2:i3]))
    
    df['total_amplitude'] = df['insp_amplitude']+df['exp_amplitude']

    #### ad select columns
    select_vec = np.ones((df.index.shape[0]), dtype='int')
    select_vec[cycles_mask_keep] = 0
    df.insert(df.columns.shape[0], 'select', select_vec)

    #### insert last cycle
    last_cycle = {}
    for columns_i in df.columns:
        if columns_i == 'cycle_num':
            last_cycle[columns_i] = df['cycle_num'].shape[0]
        elif columns_i == 'inspi_index':
            last_cycle[columns_i] = cycles[-1,0]
        elif columns_i == 'expi_index':
            last_cycle[columns_i] = cycles[-1,1]
        elif columns_i == 'inspi_time':
            last_cycle[columns_i] = cycles[-1,0]/srate
        else:
            last_cycle[columns_i] = 0

    df = df.append(last_cycle, ignore_index=True)     
        
    return df






############################
######## LOAD DATA ########
############################



def load_respi_allcond_data(sujet, cycle_detection_params):

    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    srate = get_params(sujet)['srate']

    raw_allcond = {}

    for cond in conditions_allsubjects:

        raw_allcond[cond] = {}

        for odor_i in odor_list:

            load_i = []
            for session_i, session_name in enumerate(os.listdir()):
                if session_name.find(cond) != -1 and session_name.find(odor_i) != -1 and (session_name.find('lf') != -1 or session_name.find('wb') != -1):
                    load_i.append(session_i)
                else:
                    continue

            load_name = [os.listdir()[i] for i in load_i][0]

            load_data = mne.io.read_raw_fif(load_name, preload=True)
            load_data = load_data.pick_channels(['PRESS']).get_data().reshape(-1)

            raw_allcond[cond][odor_i] = load_data

    #### preproc respi
    respi_allcond = {}

    for cond in conditions:

        respi_allcond[cond] = {}

        for odor_i in odor_list:
        
            resp_clean = physio.preprocess(raw_allcond[cond][odor_i], srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            respi_allcond[cond][odor_i] = resp_clean_smooth

    #### detect
    respfeatures_allcond = {}

    #cond = 'FR_CV_1'
    for cond in conditions:

        respfeatures_allcond[cond] = {}

        #odor_i = '-'
        for odor_i in odor_list:

            cycles = physio.detect_respiration_cycles(respi_allcond[cond][odor_i], srate, baseline_mode='median',inspration_ajust_on_derivative=True)

            cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][odor_i], cycles, srate, 
                                        exclusion_metrics=cycle_detection_params['exclusion_metrics'], metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], 
                                        inspi_coeff_exclusion=cycle_detection_params['inspi_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'])

            resp_features_i = get_all_respiration_features(respi_allcond[cond][odor_i], srate, cycles, cycles_mask_keep, t_start=0.)

            respfeatures_allcond[cond][odor_i] = [resp_features_i, fig_respi_exclusion, fig_final]


    return raw_allcond, respi_allcond, respfeatures_allcond







def load_respi_allcond_data_recompute(sujet, cond, odor_i, cycle_detection_params):


    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    srate = get_params(sujet)['srate']

    load_i = []
    for session_i, session_name in enumerate(os.listdir()):
        if session_name.find(cond) != -1 and session_name.find(odor_i) != -1 and (session_name.find('lf') != -1 or session_name.find('wb') != -1):
            load_i.append(session_i)
        else:
            continue

    load_name = [os.listdir()[i] for i in load_i][0]

    load_data = mne.io.read_raw_fif(load_name, preload=True)
    load_data = load_data.pick_channels(['PRESS']).get_data().reshape(-1)

    raw = load_data

    #### preproc respi
    resp_clean = physio.preprocess(raw, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

    respi = resp_clean_smooth

    #### detect

    cycles = physio.detect_respiration_cycles(respi, srate, baseline_mode='median',inspration_ajust_on_derivative=True)

    cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][odor_i], cycles, srate, 
                                exclusion_metrics=cycle_detection_params['exclusion_metrics'], metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], 
                                inspi_coeff_exclusion=cycle_detection_params['inspi_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'])

    resp_features_i = get_all_respiration_features(respi_allcond[cond][odor_i], srate, cycles, cycles_mask_keep, t_start=0.)

    respfeatures = [resp_features_i, fig_respi_exclusion, fig_final]

    return raw, respi, respfeatures








########################################
######## EDIT CYCLES SELECTED ########
########################################


#respi_allcond = respi_allcond_bybycle
def edit_df_for_sretch_cycles_deleted(sujet, respi_allcond, respfeatures_allcond):

    prms = get_params(sujet)

    for cond in conditions:
        
        for odor_i in odor_list:

            #### stretch
            cycles = respfeatures_allcond[cond][odor_i][0][['inspi_index', 'expi_index']].values/srate
            times = np.arange(respi_allcond[cond][odor_i].shape[0])/srate
            clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
                    respi_allcond[cond][odor_i], times, cycles, nb_point_by_cycle=stretch_point_TF, inspi_ratio=ratio_stretch_TF)

            if debug:
                plt.plot(data_stretch_linear)
                plt.show()

            i_to_update = respfeatures_allcond[cond][odor_i][0].index.values[~np.isin(respfeatures_allcond[cond][odor_i][0].index.values, cycles)]
            respfeatures_allcond[cond][odor_i][0]['select'][i_to_update] = np.array([0]*i_to_update.shape[0])

    



def export_sniff_count(sujet, respfeatures_allcond):

    #### generate df
    df_count_cycle = pd.DataFrame(columns={'sujet' : [], 'cond' : [], 'odor' : [], 'count' : []})

    for cond in conditions:
        
        for odor_i in odor_list:

            data_i = {'sujet' : [sujet], 'cond' : [cond], 'odor' : [odor_i], 'count' : [int(np.sum(respfeatures_allcond[cond][odor_i][0]['select'].values))]}
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










############################
######## EXECUTE ########
############################



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
    
    raw_allcond, respi_allcond, respfeatures_allcond = load_respi_allcond_data(sujet, cycle_detection_params)





    ########################################
    ######## VERIF RESPIFEATURES ########
    ########################################
    
    if debug == True :

        cond = 'FR_CV_1'
        cond = 'FR_CV_2' 
        cond = 'MECA' 
        cond = 'CO2'
        
        odor_i = 'o'
        odor_i = '-'
        odor_i = '+'

        respfeatures_allcond[cond][odor_i][1].show()
        respfeatures_allcond[cond][odor_i][2].show()




    ########################################
    ######## EDIT CYCLES SELECTED ########
    ########################################

    respfeatures_allcond= edit_df_for_sretch_cycles_deleted(sujet, respi_allcond, respfeatures_allcond)

    export_sniff_count(sujet, respfeatures_allcond)





    ################################
    ######## SAVE FIG ########
    ################################

    #### select export
    # export = 'sam'
    export = 'bycycle'

    save_all_respfeatures(respi_allcond, respfeatures_allcond, conditions, export)


    
