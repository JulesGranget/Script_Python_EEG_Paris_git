

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

# from bycycle.cyclepoints import find_extrema, find_zerox
# from bycycle.plts import plot_cyclepoints_array

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False






########################################
######## PHYSIO TOOL DEBUGGED ########
########################################


#resp = respi_allcond[cond][odor_i]
def detect_respiration_cycles(resp, srate, baseline_mode='manual', baseline=None, 
                              epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False):
    """
    Detect respiration cycles based on:
      * crossing zeros (or crossing baseline)
      * some cleanning with euristicts

    Parameters
    ----------
    resp: np.array
        Preprocess traces of respiratory signal.
    srate: float
        Sampling rate
    baseline_mode: 'manual' / 'zero' / 'median' / 'mode'
        How to compute the baseline for zero crossings.
    baseline: float or None
        External baseline when baseline_mode='manual'
    inspration_ajust_on_derivative: bool (default False)
        For the inspiration detection, the zero crossing can be refined to auto detect the inflection point.
        This can be usefull when expiration ends with a long plateau.
    Returns
    -------
    cycles: np.array
        Indices of inspiration and expiration. shape=(num_cycle, 3)
        with [index_inspi, index_expi, index_next_inspi]
    """

    # baseline = get_respiration_baseline(resp, srate, baseline_mode=baseline_mode, baseline=baseline)
    baseline = resp.mean()

    #~ q90 = np.quantile(resp, 0.90)
    q10 = np.quantile(resp, 0.10)
    epsilon = (baseline - q10) / 100.

    baseline_dw = baseline - epsilon * epsilon_factor1
    baseline_insp = baseline - epsilon * epsilon_factor2

    resp0 = resp[:-1]
    resp1 = resp[1:]

    ind_dw, = np.nonzero((resp0 >= baseline_dw) & (resp1 < baseline_dw))
    
    ind_insp, = np.nonzero((resp0 >= baseline_insp) & (resp1 < baseline_insp))
    ind_insp_no_clean = ind_insp.copy()
    keep_inds = np.searchsorted(ind_insp, ind_dw, side='left')
    keep_inds = keep_inds[keep_inds > 0]
    ind_insp = ind_insp[keep_inds - 1]
    ind_insp = np.unique(ind_insp)

    ind_exp, = np.nonzero((resp0 < baseline) & (resp1 >= baseline))
    keep_inds = np.searchsorted(ind_exp, ind_insp, side='right')
    keep_inds = keep_inds[keep_inds<ind_exp.size]
    ind_exp = ind_exp[keep_inds]
    
    # this is tricky to read but quite simple in concept
    # this remove ind_exp assigned to the same ind_insp
    bad, = np.nonzero(np.diff(ind_exp) == 0)
    keep = np.ones(ind_insp.size, dtype='bool')
    keep[bad + 1] = False
    ind_insp = ind_insp[keep]
    keep = np.ones(ind_exp.size, dtype='bool')
    keep[bad + 1] = False
    ind_exp = ind_exp[keep]

    #~ import matplotlib.pyplot as plt
    #~ fig, ax = plt.subplots()
    #~ ax.plot(resp)
    #~ ax.scatter(ind_insp_no_clean, resp[ind_insp_no_clean], color='m', marker='*', s=100)
    #~ ax.scatter(ind_dw, resp[ind_dw], color='orange', marker='o', s=30)
    #~ ax.scatter(ind_insp, resp[ind_insp], color='g', marker='o')
    #~ ax.scatter(ind_exp, resp[ind_exp], color='r', marker='o')
    #~ ax.axhline(baseline, color='r')
    #~ ax.axhline(baseline_insp, color='g')
    #~ ax.axhline(baseline_dw, color='orange')
    #~ ax.axhline(q10, color='k')
    #~ plt.show()


    if ind_insp.size == 0:
        print('no cycle dettected')
        return


    mask = (ind_exp > ind_insp[0]) & (ind_exp < ind_insp[-1])
    ind_exp = ind_exp[mask]

    if inspiration_adjust_on_derivative:
        # lets find local minima on second derivative
        # this can be slow
        delta_ms = 10.
        delta = int(delta_ms * srate / 1000.)
        derivate1 = np.gradient(resp)
        derivate2 = np.gradient(derivate1)
        for i in range(ind_exp.size):
            i0, i1 = ind_insp[i], ind_exp[i]
            i0 = max(0, i0 - delta)
            i1 = i0 + np.argmin(resp[i0:i1])
            d1 = derivate1[i0:i1]
            i1 = i0 + np.argmin(d1)
            if (i1 - i0) >2:
                # find the last crossing zeros in this this short segment
                d2 = derivate2[i0:i1]
                i1 = i0 + np.argmin(d2)
                if (i1 - i0) >2:
                    d2 = derivate2[i0:i1]
                    mask = (d2[:-1] >=0) & (d2[1:] < 0)
                    if np.any(mask):
                        ind_insp[i] = i0 + np.nonzero(mask)[0][-1]

    if ind_exp.shape[0] != ind_insp[:-1].shape[0]:

        ind_insp = ind_insp[:-1]
    
    cycles = np.zeros((ind_insp.size - 1, 3), dtype='int64')
    cycles[:, 0] = ind_insp[:-1]
    cycles[:, 1] = ind_exp
    cycles[:, 2] = ind_insp[1:]

    if debug:

        plt.plot(resp)
        plt.scatter(ind_insp[:-1], resp[ind_insp[:-1]], label='inspi')
        plt.scatter(ind_exp, resp[ind_exp], label='expi')
        plt.legend()
        plt.show()

    return cycles







########################################
######## COMPUTE RESPI FEATURES ########
########################################


#respi, cycles_init = respi_allcond[cond][odor_i], cycles
def exclude_bad_cycles(respi, cycles_init, srate, exclusion_metrics='med', metric_coeff_exclusion=3, inspi_coeff_exclusion=2, respi_scale=[0.1, 0.35]):

    next_inspi = cycles_init[:,-1]

    if debug:

        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        plt.show()

    #### exclude regarding inspi/expi diff
    _diff = np.log(np.diff(cycles_init[:,:2], axis=1).reshape(-1))

    if debug:
        plt.plot(np.arange(respi.shape[0])/srate, respi)
        plt.show()

        plt.plot(zscore(_diff))
        plt.plot(zscore(np.log(_diff)))
        plt.title('inspi/expi diff')
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

    cycles = cycles_init[inspi_time_included_i,:2]
    next_inspi = next_inspi[inspi_time_included_i]
    inspi_starts = cycles[:,0]

    if debug:

        inspi_starts_init = cycles_init[:,0]
        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts_init, respi[inspi_starts_init], color='g')
        ax.scatter(inspi_starts_init[cycle_inspi_excluded_i], respi[inspi_starts_init[cycle_inspi_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts_init, _diff, color='r', label=exclusion_metrics)
        ax2.axhline(metric_center, color='r')
        ax2.axhline(metric_center - metric_dispersion*inspi_coeff_exclusion, color='r', linestyle='--')
        ax2.axhline(metric_center + metric_dispersion*inspi_coeff_exclusion, color='r', linestyle='--')
        plt.title('inspi/expi diff')
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

    #### exclude regarding duration
    durations = np.diff(inspi_starts/srate)

    # cycle_duration_sel_i = [i for i, val in enumerate(durations) if (val > 1/respi_scale[0] or val < 1/respi_scale[1]) == False]
    # cycle_duration_excluded_i = [i for i, val in enumerate(durations) if (val > 1/respi_scale[0] or val < 1/respi_scale[1])]
    cycle_duration_sel_i = [i for i, val in enumerate(durations) if (val < 1/respi_scale[1]) == False]
    cycle_duration_excluded_i = [i for i, val in enumerate(durations) if (val < 1/respi_scale[1])]
    cycle_metrics_cleaned = cycle_metrics[cycle_duration_sel_i]

    if debug:

        fig, ax = plt.subplots()
        ax.plot(respi)
        ax.scatter(inspi_starts, respi[inspi_starts], color='g')
        ax.scatter(inspi_starts[cycle_duration_excluded_i], respi[inspi_starts[cycle_duration_excluded_i]], color='k', marker='x', s=100)

        ax2 = ax.twinx()
        ax2.scatter(inspi_starts[1:], 1/durations, color='r', label=exclusion_metrics)
        ax2.axhline(respi_scale[0], color='r')
        ax2.axhline(respi_scale[1], color='r')
        plt.title('durations')
        plt.legend()
        plt.show()

    cycles = cycles[cycle_duration_sel_i, :]
    next_inspi = next_inspi[cycle_duration_sel_i]

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

        plt.plot(respi)
        plt.show()

    #### final cleaning
    next_inspi_final = np.append(cycles[1:,0], next_inspi[-1])
    cycles_final = np.concatenate((cycles, next_inspi_final.reshape(-1,1)), axis=1)
    cycles_mask_keep = np.ones((cycles_final.shape[0]), dtype='int')
    cycles_mask_keep[cycle_metrics_excluded_i] = 0
    cycles_mask_keep[-1] = 0

    #### fig for all detection
    time_vec = np.arange(respi.shape[0])/srate
    
    inspi_starts_init = cycles_init[:,0]
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
    inspi_starts_init = cycles_init[:,0]
    fig_final, ax = plt.subplots(figsize=(18, 10))
    ax.plot(time_vec, respi)
    ax.scatter(cycles[:,0]/srate, respi[cycles[:,0]], color='g', label='inspi_selected')
    ax.scatter(cycles[:-1,1]/srate, respi[cycles[:-1,1]], color='c', label='expi_selected', marker='s')
    ax.scatter(cycles[:,0][cycle_metrics_excluded_i]/srate, respi[cycles[:,0][cycle_metrics_excluded_i]], color='r', label='excluded_metric')
    plt.legend()
    # plt.show()
    plt.close()

    return cycles_final, cycles_mask_keep, fig_respi_exclusion, fig_final






############################
######## LOAD DATA ########
############################



def load_respi_allcond_data(sujet, cycle_detection_params):

    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    srate = get_params()['srate']

    raw_allcond = {}

    for cond in conditions:

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

    #### extract zscore params
    zscore_prms = {'mean' : {}, 'std' : {}}

    #odor = odor_list[1]
    for odor in odor_list:

        respi_concat = []

        #cond = 'MECA'
        for cond in conditions:

            resp_clean = physio.preprocess(raw_allcond[cond][odor_i], srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            respi_concat = np.concatenate((respi_concat, resp_clean_smooth), axis=0)

        zscore_prms['mean'][odor] = np.array(respi_concat).mean()
        zscore_prms['std'][odor] = np.array(respi_concat).std()

    #### preproc respi
    respi_allcond = {}

    for cond in conditions:

        respi_allcond[cond] = {}

        for odor_i in odor_list:
        
            resp_clean = physio.preprocess(raw_allcond[cond][odor_i], srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
            resp_clean_smooth = physio.smooth_signal(resp_clean, srate, win_shape='gaussian', sigma_ms=40.0)

            resp_clean_smooth = (resp_clean_smooth - zscore_prms['mean'][odor]) / zscore_prms['std'][odor]

            respi_allcond[cond][odor_i] = resp_clean_smooth

    #### detect
    respfeatures_allcond = {}

    #cond = 'CO2'
    for cond in conditions:

        respfeatures_allcond[cond] = {}

        #odor_i = '-'
        for odor_i in odor_list:

            # cycles = physio.detect_respiration_cycles(respi_allcond[cond][odor_i], srate, baseline_mode='median',
            #                                           baseline=None, epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False)
            cycles = detect_respiration_cycles(respi_allcond[cond][odor_i], srate, baseline_mode='median',
                                                      baseline=None, epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False)
            
            if debug:

                fig, ax = plt.subplots()
                ax.plot(respi_allcond[cond][odor_i])
                ax.scatter(cycles[:,0], respi_allcond[cond][odor_i][cycles[:,0]], color='g')
                plt.show()

            if sujet in ['07PB', '11FA', '16GM', '18SE', '20TY', '24TJ', '25DF', '26MN', '28NT', '30AR']:

                cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][odor_i], cycles, srate, 
                        exclusion_metrics=cycle_detection_params['exclusion_metrics'], metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], 
                        inspi_coeff_exclusion=cycle_detection_params['inspi_coeff_exclusion'], respi_scale=[0.1, 0.5])
            
            elif sujet in ['32CM']:

                cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][odor_i], cycles, srate, 
                        exclusion_metrics=cycle_detection_params['exclusion_metrics'], metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], 
                        inspi_coeff_exclusion=cycle_detection_params['inspi_coeff_exclusion'], respi_scale=[0.1, 0.6])

            else:

                cycles, cycles_mask_keep, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi_allcond[cond][odor_i], cycles, srate, 
                        exclusion_metrics=cycle_detection_params['exclusion_metrics'], metric_coeff_exclusion=cycle_detection_params['metric_coeff_exclusion'], 
                        inspi_coeff_exclusion=cycle_detection_params['inspi_coeff_exclusion'], respi_scale=cycle_detection_params['respi_scale'])
                
            if debug:

                fig, ax = plt.subplots()
                ax.plot(respi_allcond[cond][odor_i])
                ax.scatter(cycles[:,0], respi_allcond[cond][odor_i][cycles[:,0]], color='r')
                ax.scatter(cycles[:,0][cycles_mask_keep.astype('bool')], respi_allcond[cond][odor_i][cycles[:,0][cycles_mask_keep.astype('bool')]], color='g')
                plt.show()

            #### get resp_features
            resp_features_i = physio.compute_respiration_cycle_features(respi_allcond[cond][odor_i], srate, cycles, baseline=None)
    
            select_vec = np.ones((resp_features_i.index.shape[0]), dtype='int')
            select_vec[cycles_mask_keep] = 0
            resp_features_i.insert(resp_features_i.columns.shape[0], 'select', select_vec)
            
            respfeatures_allcond[cond][odor_i] = [resp_features_i, fig_respi_exclusion, fig_final]


    return raw_allcond, respi_allcond, respfeatures_allcond







def load_respi_allcond_data_recompute(sujet, cond, odor_i, cycle_detection_params):


    #### load data
    os.chdir(os.path.join(path_prep, sujet, 'sections'))

    srate = get_params()['srate']

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

    #### get resp_features
    resp_features_i = physio.compute_respiration_cycle_features(respi_allcond[cond][odor_i], srate, cycles, baseline=None)

    select_vec = np.ones((resp_features_i.index.shape[0]), dtype='int')
    select_vec[cycles_mask_keep] = 0
    resp_features_i.insert(resp_features_i.columns.shape[0], 'select', select_vec)

    respfeatures = [resp_features_i, fig_respi_exclusion, fig_final]

    return raw, respi, respfeatures








########################################
######## EDIT CYCLES SELECTED ########
########################################


#respi_allcond = respi_allcond_bybycle
# def edit_df_for_sretch_cycles_deleted(respi_allcond, respfeatures_allcond):

#     for cond in conditions:
        
#         for odor_i in odor_list:

#             #### stretch
#             cycles = respfeatures_allcond[cond][odor_i][0][['inspi_index', 'expi_index']].values/srate
#             times = np.arange(respi_allcond[cond][odor_i].shape[0])/srate
#             clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = respirationtools.deform_to_cycle_template(
#                     respi_allcond[cond][odor_i], times, cycles, nb_point_by_cycle=stretch_point_TF, inspi_ratio=ratio_stretch_TF)

#             if debug:
#                 plt.plot(data_stretch_linear)
#                 plt.show()

#             i_to_update = respfeatures_allcond[cond][odor_i][0].index.values[~np.isin(respfeatures_allcond[cond][odor_i][0].index.values, cycles)]
#             for i_to_update_i in i_to_update:
                
#                 if i_to_update_i == respfeatures_allcond[cond][odor_i][0].shape[0] - 1:
#                     continue

#                 else:
#                     respfeatures_allcond[cond][odor_i][0]['select'][i_to_update_i] = 0

#     return respfeatures_allcond



def export_cycle_count(sujet, respfeatures_allcond):

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



####################################
######## PLOT MEAN RESPI ########
####################################

def plot_mean_respi(sujet):

    print(sujet)

    time_vec = np.arange(stretch_point_TF)
    colors_respi = {'FR_CV_1' : 'b', 'MECA' : 'r', 'CO2' : 'g', 'FR_CV_2' : 'tab:purple'}
    colors_respi_sem = {'FR_CV_1' : 'c', 'MECA' : 'm', 'CO2' : 'y', 'FR_CV_2' : 'tab:pink'}

    respfeatures = load_respfeatures(sujet)
    respi_allcond = {}
    sem_allcond = {}
    lim = {'min' : np.array([]), 'max' : np.array([])} 

    odor_zscore_params = {}

    for odor in odor_list:

        _respi = np.array([])

        for cond in conditions:

            x = load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]
            _respi = np.concat([_respi, x], axis=0)

        odor_zscore_params[odor] = {'mean' : _respi.mean(), 'std' : _respi.std()}

    #### load
    #cond = 'VS'
    for cond in conditions:

        respi_allcond[cond] = {}
        sem_allcond[cond] = {}

        for odor in odor_list:

            x = load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]
            x = scipy.signal.detrend(x, type='linear')
            x = iirfilt(x, srate, lowcut=0.05, highcut=None)
            x = iirfilt(x, srate, lowcut=None, highcut=45)
            data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond][odor], stretch_point_TF, x, srate)
            data_stretch = (data_stretch - odor_zscore_params[odor]['mean'])/odor_zscore_params[odor]['std']
            respi_allcond[cond][odor] = data_stretch.mean(axis=0)
            sem_allcond[cond][odor] = data_stretch.std(axis=0)/np.sqrt(data_stretch.shape[0])
            lim['min'], lim['max'] = np.append(lim['min'], respi_allcond[cond][odor].min()-sem_allcond[cond][odor]), np.append(lim['max'], respi_allcond[cond][odor].max()+sem_allcond[cond][odor])

    #### plot
    fig, axs = plt.subplots(ncols=3)

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        #cond = 'VS'
        for cond in conditions:

            ax.plot(time_vec, respi_allcond[cond][odor], color=colors_respi[cond], label=cond)
            ax.fill_between(time_vec, respi_allcond[cond][odor]+sem_allcond[cond][odor], respi_allcond[cond][odor]-sem_allcond[cond][odor], alpha=0.25, color=colors_respi_sem[cond])

        ax.vlines(stretch_point_TF/2, ymin=lim['min'].min(), ymax=lim['max'].max(), color='r')
        ax.set_title(odor)
        plt.ylim(lim['min'].min(), lim['max'].max())

    plt.suptitle(sujet)
    plt.legend()
    # plt.show()

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI', 'mean_respi_allsujet'))
    plt.savefig(f"{sujet}_respi_mean.png")

    plt.close('all')




def generate_df_cycle_count():

    df_data = {'sujet' : [], 'cond' : [], 'odor' : [], 'ncycle' : []}

    for sujet_i, sujet in enumerate(sujet_list):

        respfeature_sujet = load_respfeatures(sujet)
        
        for cond_i, cond in enumerate(conditions):

            for odor_i, odor in enumerate(odor_list):

                df_data['sujet'].append(sujet)
                df_data['cond'].append(cond)
                df_data['odor'].append(odor)
                df_data['ncycle'].append(respfeature_sujet[cond][odor].query(f"select == 1").shape[0])

    df_ncycle = pd.DataFrame(df_data)

    ncycle_thresh = 40

    for ncycle_thresh in np.arange(30, 60, 5):
        df_thresh = df_ncycle.query(f"cond not in ['MECA', 'FR_CV_2'] and ncycle >= {ncycle_thresh}")
        _, counts = np.unique(df_thresh['sujet'], return_counts=True)   
        allcond_count = len(['FR_CV_1', 'CO2']) * len(odor_list)
        sujet_list_thresh = sujet_list[counts == 6]
        print(f"ncycle:{ncycle_thresh}, nsujet:{sujet_list_thresh.size}")



############################
######## EXECUTE ########
############################



if __name__ == '__main__':

    ############################
    ######## LOAD DATA ########
    ############################

    
    #### whole protocole
    sujet = '01PD'
    sujet = '02MJ'
    sujet = '03VN'
    sujet = '04GB'
    sujet = '05LV'
    sujet = '06EF'
    sujet = '07PB'
    sujet = '08DM'
    sujet = '09TA'
    sujet = '10BH'
    sujet = '11FA'
    sujet = '12BD'
    sujet = '13FP'
    sujet = '14MD'
    sujet = '15LG'
    sujet = '16GM'
    sujet = '17JR'
    sujet = '18SE'
    sujet = '19TM'
    sujet = '20TY'
    sujet = '21ZV'
    sujet = '22DI'
    sujet = '23LF'
    sujet = '24TJ'
    sujet = '25DF'
    sujet = '26MN'
    sujet = '27BD'
    sujet = '28NT'
    sujet = '29SC'
    sujet = '30AR'
    sujet = '31HJ'
    sujet = '32CM'
    sujet = '33MA'

    for sujet in sujet_list:

        print(f"#### #### ####")
        print(f"#### {sujet} ####")
        print(f"#### #### ####")

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

        # respfeatures_allcond = edit_df_for_sretch_cycles_deleted(respi_allcond, respfeatures_allcond)

        export_cycle_count(sujet, respfeatures_allcond)

        if debug :

            for cond in conditions:

                for odor in odor_list:

                    _respi = respi_allcond[cond][odor_i]
                    plt.plot(_respi)
                    plt.vlines(respfeatures_allcond[cond][odor_i][0]['inspi_index'].values, ymin=_respi.min(), ymax=_respi.max(), color='r')
                    plt.title(f"{odor} {cond}")
                    plt.show()

        ################################
        ######## SAVE FIG ########
        ################################

        os.chdir(os.path.join(path_results, sujet, 'RESPI'))

        for cond in conditions:

            for odor_i in odor_list:

                respfeatures_allcond[cond][odor_i][0].to_excel(f"{sujet}_{cond}_{odor_i}_respfeatures.xlsx")
                respfeatures_allcond[cond][odor_i][1].savefig(f"{sujet}_{cond}_{odor_i}_fig0.jpeg")
                respfeatures_allcond[cond][odor_i][2].savefig(f"{sujet}_{cond}_{odor_i}_fig1.jpeg")

        
    ################################
    ######## PLOT ########
    ################################
    
    #sujet = sujet_list[0]
    for sujet in sujet_list:

        plot_mean_respi(sujet)

    ################################
    ######## CYCLE COUNT ########
    ################################

    generate_df_cycle_count()

    ################################
    ######## AGGREGATES ########
    ################################

    df_respi_allsujet = pd.DataFrame({'sujet' : [], 'cond' : [], 'odor' : [], 
                                      'inspi_duration' : [], 'expi_duration' : [], 'cycle_freq' : [], 'inspi_volume' : [], 
                                      'expi_volume' : [], 'total_amplitude' : [], 'inspi_amplitude' : [], 'expi_amplitude' : [], })
    
    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)

        respfeatures = load_respfeatures(sujet)

        for cond in conditions:

            for odor in odor_list:

                _df = respfeatures[cond][odor]

                _df_respi_allsujet = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'odor' : [odor], 
                                      'inspi_duration' : [np.median(_df['inspi_duration'])], 'expi_duration' : [np.median(_df['expi_duration'])], 
                                      'cycle_freq' : [np.median(_df['cycle_freq'])], 'inspi_volume' : [np.median(_df['inspi_volume'])], 
                                      'expi_volume' : [np.median(_df['expi_volume'])], 'total_amplitude' : [np.median(_df['total_amplitude'])], 
                                      'inspi_amplitude' : [np.median(_df['inspi_amplitude'])], 'expi_amplitude' : [np.median(_df['expi_amplitude'])], })
                
                df_respi_allsujet = pd.concat([df_respi_allsujet, _df_respi_allsujet])

    os.chdir(os.path.join(path_results, 'allplot', 'RESPI'))
    df_respi_allsujet.to_excel('allsujet_df_respi.xlsx')

        









