

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal 
import os
import pandas as pd
import xarray as xr
import neurokit2 as nk
import mne
import physio


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import confusion_matrix


import joblib 
import seaborn as sns
import pandas as pd

import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n1bis_prep_trigger_info import *
from n4_precompute_hrv import *

debug = False





################################
######## LOAD DATA ########
################################

    
def load_ecg_sig(sujet, odor_i, band_prep):

    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    
    raw = mne.io.read_raw_fif(f'{sujet}_{odor_i}_allcond_{band_prep}.fif', preload=True, verbose='critical')

    data = raw.get_data()
    ecg = data[chan_list.index('ECG'), :]

    del raw

    return ecg






################################
######## COMPUTE ########
################################

def get_label_vec(sujet, odor_i, ecg):

    #### generate trig
    ses_i = list(odor_order[sujet].keys())[list(odor_order[sujet].values()).index(odor_i)]

    trig = {}

    #cond = conditions[0]
    for cond in conditions:
        
        _stop = dict_trig_sujet[sujet][ses_i][cond]
        _start = _stop - (srate*5*60)
        trig[cond] = np.array([_start, _stop])

    #### generate label vec
    label_vec = np.zeros((ecg.shape[0]))

    #cond = conditions[0]
    for cond in conditions:

        _start, _stop = trig[cond][0], trig[cond][-1]
        label_vec[_start:_stop] = cond_label_tracker[cond]

    return label_vec, trig



def generate_xr_data_compact(xr_data):

    xr_ecg = xr_data.loc[:, 'ses02', ['free', 'confort', 'coherence'], :, 'ecg', :]
    order = []

    #sujet_i = xr_ecg['participant'].values[1]
    for sujet_i in xr_ecg['participant'].values:
        
        #trial_i = xr_ecg['trial'].values[0]
        for trial_i in xr_ecg['trial'].values:

            #bloc_i = xr_ecg['bloc'].values[0]
            for bloc_i in xr_ecg['bloc'].values: 

                if trial_i == xr_ecg['trial'].values[0] and bloc_i == xr_ecg['bloc'].values[0]:
                    
                    ecg_sig_i = xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values

                else:

                    ecg_sig_i = np.concatenate((ecg_sig_i, xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values), axis=0)
                
                if sujet_i == xr_ecg['participant'].values[0]:
                
                    order.append(f'{trial_i}_{bloc_i}_{int(xr_ecg.loc[sujet_i, bloc_i, trial_i, :].values.shape[0]/srate/60)}min')

        if sujet_i == xr_ecg['participant'].values[0]:

            data_ecg = ecg_sig_i.reshape(1,len(ecg_sig_i))

        else:

            ecg_sig_i = ecg_sig_i.reshape(1,len(ecg_sig_i))
            data_ecg = np.concatenate((data_ecg, ecg_sig_i), axis=0)

    return data_ecg, order






########################################
######## TRACKING FUNCTION ########
########################################


def zscore(data):
    zscore_data = (data - np.mean(data))/np.std(data)
    return zscore_data



def hrv_tracker(ecg_cR, win_size, srate):

    #### load cR
    ecg_cR_val = np.where(ecg_cR != 0)[0]/srate
    RRI = np.diff(ecg_cR_val)

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR_val[0], ecg_cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)
    times = [ecg_cR_val[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR_val)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)
        
        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(ecg_cR_val[cR_i])

    times = np.array(times)

    return df_res, times




#sujet_i = 1
#ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm(ecg_cR, srate, classifier, prms_tracker):

    win_size_sec, metric_used, odor_trig_n_bpm = prms_tracker['win_size_sec'], prms_tracker['metric_list'], prms_tracker['odor_trig_n_bpm']
    win_size = int(win_size_sec*srate)

    #### load cR
    ecg_cR_val = np.where(ecg_cR != 0)[0]/srate
    RRI = np.diff(ecg_cR_val)

    #### verif
    if debug:
        #### RRI
        plt.plot(RRI)
        plt.show()

    #### load sliding win
    ecg_cR_sliding_win = np.array((ecg_cR_val[0], ecg_cR_val[1]))
    cR_initial = 2

    while ecg_cR_sliding_win[-1] <= win_size/srate:
        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win, ecg_cR_val[cR_initial])
        cR_initial += 1

    #### first point sliding win
    RRI_win = np.diff(ecg_cR_sliding_win)
    df_res = get_hrv_metrics_win(RRI_win)
    predictions = classifier.predict(df_res.values)
    trig_odor = [0]
    times = [ecg_cR_val[cR_initial]]

    #### progress bar
    # bar = IncrementalBar('Countdown', max = len(ecg_cR_val)-cR_initial)

    #### sliding on other cR
    for cR_i in range(len(ecg_cR_val)-cR_initial):

        # bar.next()

        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)
        predictions = np.append(predictions, classifier.predict(df_slide.values))
        df_res = pd.concat([df_res, df_slide], axis=0)

        if predictions.shape[0] >= odor_trig_n_bpm:
            trig_odor_win = predictions.copy()[-odor_trig_n_bpm:]
            trig_odor_win[trig_odor_win < 2] = 0
            trig_odor_win[(trig_odor_win == cond_label_tracker['MECA']) | (trig_odor_win == cond_label_tracker['CO2'])] = 1
            trig_odor_pred_i = np.round(np.mean(trig_odor_win))

            if trig_odor_pred_i != 0:
                trig_odor.append(1)
            else:
                trig_odor.append(0)
        
        else:
            trig_odor.append(0)

        times.append(ecg_cR_val[cR_i])

    # bar.finish()

    times = np.array(times)
    trig_odor = np.array(trig_odor)

    return df_res, times, predictions, trig_odor




def get_data_tracking(ecg_cR, prms_tracker):

    #### params
    win_size_sec, jitter = prms_tracker['win_size_sec'], prms_tracker['jitter']
    win_size = int(win_size_sec*srate)

    #### compute tracking data
    df_hrv, times = hrv_tracker(ecg_cR, win_size, srate)

    return df_hrv, times















################################
######## COMPILATION ########
################################


def hrv_compute_compilation(sujet):

    ########################
    ######## VERIF ########
    ########################

    if debug:

        ecg_cR_n = {}

        for sujet in sujet_list:

            print(sujet)

            for odor_i in odor_list:

                #### load
                ecg, ecg_cR = load_ecg_sig(sujet, odor_i, band_prep)

                ecg_cR_n[f'{sujet}_{odor_i}'] = (ecg_cR == 10).sum()

        plt.hist(ecg_cR_n.values(), bins=100)
        plt.show()

        mask = np.array(list(ecg_cR_n.values())) < 3000
        np.array(list(ecg_cR_n.keys()))[mask]



    ########################
    ######## PARAMS ########
    ########################

    prms_tracker = {
    'metric_list' : ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'AUC_LF', 'AUC_HF', 'LF_HF_ratio', 'SD1', 'SD2'],
    'win_size_sec' : 30,
    'odor_trig_n_bpm' : 75,
    'jitter' : 0,
    }

    band_prep = band_prep_list[0]

    odor_ref = 'o'
    odor_list_test = [odor_i for odor_i in odor_list if odor_i != odor_ref]

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    if os.path.exists(os.path.join(path_precompute, sujet, 'HRV', f'{sujet}_hrv_tracker.nc')):
        print('ALREADY COMPUTED')
        return

    xr_dict = {'sujet' : [sujet], 'odor' : np.array(odor_list_test), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((1, len(odor_list_test), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : [sujet], 'odor' : np.array(odor_list_test)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((1, len(odor_list_test))), dims=xr_dict.keys(), coords=xr_dict.values())


    print('################')
    print(f'#### {sujet} {odor_ref} ####')
    print('################')

    ########################
    ######## MODEL ########
    ########################

    #### load
    ecg = load_ecg_sig(sujet, odor_ref, band_prep)
    label_vec, trig = get_label_vec(sujet, odor_ref, ecg)

    ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks] = 10

    df_hrv, times = get_data_tracking(ecg_cR, prms_tracker)
    _times = times*srate
    label_vec = label_vec[_times.astype('int')].astype('int')

    if debug:

        plt.plot(ecg)
        plt.show()
        
        plt.plot(ecg_cR)
        plt.show()

        plt.plot(label_vec)
        plt.show()

    #### split values
    X, y = df_hrv.values, label_vec.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {
    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [0.001, 0.1, 1, 10, 100, 10e5], 
    'SVM__gamma' : [0.1, 0.01]
    }

    print('train')
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    print('done')



    ########################
    ######## TEST ########
    ########################

    #odor_i = odor_list[2]
    for odor_i in odor_list_test:
        
        #### load
        ecg = load_ecg_sig(sujet, odor_i, band_prep)
        label_vec, trig = get_label_vec(sujet, odor_i, ecg)

        ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

        ecg_cR = np.zeros((ecg.shape[0]))
        ecg_cR[ecg_peaks] = 10

        df_hrv, times = get_data_tracking(ecg_cR, prms_tracker)
        _times = times*srate
        label_vec = label_vec[_times.astype('int')].astype('int')

        if debug:

            plt.plot(ecg)
            plt.show()
            
            plt.plot(ecg_cR)
            plt.show()

            plt.plot(label_vec)
            plt.show()


        #### get values
        df_res, times, predictions, trig_odor = hrv_tracker_svm(ecg_cR, srate, classifier, prms_tracker)

        #### trim vectors
        #cond_i, cond = 1, conditions[1]
        for cond_i, cond in enumerate(conditions):

            if cond_i == 0:

                start = trig[cond][0]/srate - trim_edge
                stop = trig[cond][1]/srate + trim_between

                mask_start = (start <= times) & (times <= stop)

                predictions_trim = predictions[mask_start] 
                label_vec_trim = label_vec[mask_start]
                trig_odor_trim = trig_odor[mask_start] 

            elif cond_i == len(conditions)-1:

                start = trig[cond][0]/srate - trim_between
                stop = trig[cond][1]/srate + trim_edge

                mask_start = (start <= times) & (times <= stop) 

                predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

            else:

                start = trig[cond][0]/srate - trim_between
                stop = trig[cond][1]/srate + trim_between

                mask_start = (start <= times) & (times <= stop)

                predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

            if debug:

                plt.plot(predictions_trim, label='prediction', linestyle='--')
                plt.plot(label_vec_trim, label='real')
                plt.plot(trig_odor_trim, label='odor_trig')
                plt.legend()
                plt.show()

        #### resample
        f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions_trim.shape[-1]), predictions_trim, kind='linear')
        predictions_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

        f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec_trim.shape[-1]), label_vec_trim, kind='linear')
        label_vec_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

        f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor_trim.shape[-1]), trig_odor_trim, kind='linear')
        trig_odor_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

        #### plot predictions
        fig_whole, ax = plt.subplots(figsize=(15,10))
        ax.plot(predictions, color='y', label='prediction', linestyle='--')
        ax.plot(label_vec, color='k', label='real')
        ax.plot(trig_odor, color='r', label='odor_trig')
        ax.set_title(f'{sujet}{odor_i}, ref : {odor_ref}, perf : {np.round(classifier_score, 3)}')
        plt.legend()
        # fig_whole.show()
        plt.close()

        fig_trim, ax = plt.subplots(figsize=(15,10))
        ax.plot(predictions_trim_resampled, color='y', label='prediction', linestyle='--')
        ax.plot(label_vec_trim_resampled, color='k', label='real')
        ax.plot(trig_odor_trim_resampled, color='r', label='odor_trig')
        ax.set_title(f'{sujet}{odor_i}, ref : {odor_ref}, perf : {np.round(classifier_score, 3)}')
        ax.legend()
        # fig_trim.show()
        plt.close()

        #### save
        os.chdir(os.path.join(path_results, sujet, 'HRV'))
        fig_whole.savefig(f'{sujet}_{odor_i}_hrv_tracker_whole.png')
        fig_trim.savefig(f'{sujet}_{odor_i}_hrv_tracker_trim.png')

        #### load results
        xr_hrv_tracker.loc[sujet, odor_i, 'prediction', :] = predictions_trim_resampled
        xr_hrv_tracker.loc[sujet, odor_i, 'label', :] = label_vec_trim_resampled
        xr_hrv_tracker.loc[sujet, odor_i, 'trig_odor', :] = trig_odor_trim_resampled

        xr_hrv_tracker_score.loc[sujet, odor_i] = np.round(classifier_score, 5)

    #### save hrv tracker values
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
    xr_hrv_tracker.to_netcdf(f'{sujet}_hrv_tracker.nc')
    xr_hrv_tracker_score.to_netcdf(f'{sujet}_hrv_tracker_score.nc')







def hrv_compute_compilation_restrain_train():

    
    ########################
    ######## PARAMS ########
    ########################

    n_sujet_train = 5

    prms_tracker = {
    'metric_list' : ['MeanNN', 'SDNN', 'RMSSD', 'pNN50', 'AUC_LF', 'AUC_HF', 'LF_HF_ratio', 'SD1', 'SD2'],
    'win_size_sec' : 30,
    'odor_trig_n_bpm' : 75,
    'jitter' : 0,
    }

    band_prep = band_prep_list[0]

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'HRV', f'hrv_tracker_allsujet_train{n_sujet_train}.nc')):
        print('ALREADY COMPUTED')
        return
    
    sujet_list_clean = np.array([sujet for sujet in sujet_list if sujet != '31HJ'])
    sujet_list_shuffle = sujet_list_clean.copy()
    np.random.shuffle(sujet_list_shuffle)
    sujet_list_train, sujet_list_test = sujet_list_shuffle[:n_sujet_train], sujet_list_shuffle[n_sujet_train:]

    xr_dict = {'sujet' : sujet_list_test, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list_test.shape[0], len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list_test, 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list_test.shape[0], len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())
    
    #odor_i = odor_list[0]
    for odor_i in odor_list:

        ########################
        ######## TRAIN ########
        ########################

        #sujet_i, sujet = 1, sujet_list_train[1]
        for sujet_i, sujet in enumerate(sujet_list_train):

            ecg = load_ecg_sig(sujet, odor_i, band_prep)
            ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

            ecg_cR = np.zeros((ecg.shape[0]))
            ecg_cR[ecg_peaks] = 10

            if sujet_i == 0:

                df_hrv, times = get_data_tracking(ecg_cR, prms_tracker)
                _times = times*srate

                label_vec, trig = get_label_vec(sujet, odor_i, ecg)
                label_vec = label_vec[_times.astype('int')].astype('int')

            else:

                df_hrv_i, times = get_data_tracking(ecg_cR, prms_tracker)
                _times = times*srate

                label_vec_i, trig = get_label_vec(sujet, odor_i, ecg)
                label_vec_i = label_vec_i[_times.astype('int')].astype('int')

                df_hrv = pd.concat([df_hrv, df_hrv_i], axis=0)
                label_vec = np.concatenate((label_vec, label_vec_i), axis=0)

        X, y = df_hrv.values, label_vec.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        
        #### make pipeline
        #SVC().get_params()
        steps = [('scaler', StandardScaler()), ('SVM', SVC())]
        pipeline = Pipeline(steps)

        #### find best model
        params = {
        # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
        # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
        'SVM__C' : [0.001, 0.1, 1, 10, 100, 10e5], 
        'SVM__gamma' : [0.1, 0.01]
        }

        print('train')
        grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
        grid.fit(X_train, y_train)
        classifier_score = grid.best_score_
        classifier = grid.best_estimator_
        print('done')

        ########################
        ######## TEST ########
        ########################

        #sujet = sujet_list_test[0]
        for sujet in sujet_list_test:

            print(f'#### {sujet} {odor_i} ####')

            #### load
            ecg = load_ecg_sig(sujet, odor_i, band_prep)
            label_vec, trig = get_label_vec(sujet, odor_i, ecg)

            ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

            ecg_cR = np.zeros((ecg.shape[0]))
            ecg_cR[ecg_peaks] = 10

            df_hrv, times = get_data_tracking(ecg_cR, prms_tracker)
            _times = times*srate
            label_vec = label_vec[_times.astype('int')].astype('int')


            ################################
            ######## SVM PREDICTION ########
            ################################

            #### get values
            df_res, times, predictions, trig_odor = hrv_tracker_svm(ecg_cR, srate, classifier, prms_tracker)

            #### trim vectors
            #cond_i, cond = 1, conditions[1]
            for cond_i, cond in enumerate(conditions):

                if cond_i == 0:

                    start = trig[cond][0]/srate - trim_edge
                    stop = trig[cond][1]/srate + trim_between

                    mask_start = (start <= times) & (times <= stop)

                    predictions_trim = predictions[mask_start] 
                    label_vec_trim = label_vec[mask_start]
                    trig_odor_trim = trig_odor[mask_start] 

                elif cond_i == len(conditions)-1:

                    start = trig[cond][0]/srate - trim_between
                    stop = trig[cond][1]/srate + trim_edge

                    mask_start = (start <= times) & (times <= stop) 

                    predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

                else:

                    start = trig[cond][0]/srate - trim_between
                    stop = trig[cond][1]/srate + trim_between

                    mask_start = (start <= times) & (times <= stop)

                    predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

                if debug:

                    plt.plot(predictions_trim, label='prediction', linestyle='--')
                    plt.plot(label_vec_trim, label='real')
                    plt.plot(trig_odor_trim, label='odor_trig')
                    plt.legend()
                    plt.show()

            #### resample
            f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions_trim.shape[-1]), predictions_trim, kind='linear')
            predictions_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

            f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec_trim.shape[-1]), label_vec_trim, kind='linear')
            label_vec_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

            f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor_trim.shape[-1]), trig_odor_trim, kind='linear')
            trig_odor_trim_resampled = f(np.linspace(0, 1, n_pnts_trim_resample))

            if debug:

                #### plot predictions
                fig_whole, ax = plt.subplots(figsize=(15,10))
                ax.plot(predictions, color='y', label='prediction', linestyle='--')
                ax.plot(label_vec, color='k', label='real')
                ax.plot(trig_odor, color='r', label='odor_trig')
                ax.set_title(f'perf : {np.round(classifier_score, 3)}')
                plt.legend()
                # fig_whole.show()
                plt.close()

                fig_trim, ax = plt.subplots(figsize=(15,10))
                ax.plot(predictions_trim_resampled, color='y', label='prediction', linestyle='--')
                ax.plot(label_vec_trim_resampled, color='k', label='real')
                ax.plot(trig_odor_trim_resampled, color='r', label='odor_trig')
                ax.legend()
                # fig_trim.show()
                plt.close()

                #### save
                os.chdir(os.path.join(path_results, sujet, 'HRV'))
                fig_whole.savefig(f'{sujet}_{odor_i}_hrv_tracker_whole.png')
                fig_trim.savefig(f'{sujet}_{odor_i}_hrv_tracker_trim.png')

            #### load results
            xr_hrv_tracker.loc[sujet, odor_i, 'prediction', :] = predictions_trim_resampled
            xr_hrv_tracker.loc[sujet, odor_i, 'label', :] = label_vec_trim_resampled
            xr_hrv_tracker.loc[sujet, odor_i, 'trig_odor', :] = trig_odor_trim_resampled

            xr_hrv_tracker_score.loc[sujet, odor_i] = np.round(classifier_score, 5)

    #### save hrv tracker values
    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))
    xr_hrv_tracker.to_netcdf(f'hrv_tracker_allsujet_train{n_sujet_train}.nc')
    xr_hrv_tracker_score.to_netcdf(f'hrv_tracker_score_allsujet_train{n_sujet_train}.nc')











################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':



    ########################################
    ######## EXECUTE CLUSTER ########
    ########################################

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #hrv_compute_compilation(sujet)
        execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_compute_compilation', [sujet])

    #hrv_compute_compilation_restrain_train(sujet)
    execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_compute_compilation_restrain_train', [])



