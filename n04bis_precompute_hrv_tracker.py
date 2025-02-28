

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
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif



import joblib 
import seaborn as sns
import pandas as pd

import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n01bis_prep_info import *
from n04_precompute_hrv import *

debug = False





################################
######## FUNCTIONS ########
################################

    
def load_ecg_sig(sujet, odor_i, band_prep):

    os.chdir(os.path.join(path_prep, sujet, 'sections'))
    
    raw = mne.io.read_raw_fif(f'{sujet}_{odor_i}_allcond_{band_prep}.fif', preload=True, verbose='critical')

    data = raw.get_data()
    ecg = data[chan_list.index('ECG'), :]

    del raw

    return ecg




def get_label_vec(sujet, odor_i, ecg, hrv_tracker_mode):

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

    if hrv_tracker_mode == '4classes':
        return label_vec, trig

    else:

        for time_i in range(label_vec.shape[0]):
            if label_vec[time_i] in [0, 1]:
                label_vec[time_i] = 0
            else:
                label_vec[time_i] = 1
                
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




def split_data(X, y, train_size, balance=True):

    #### extract classes and trig for X and y
    y_trig = np.diff(y)
    schift_list = np.where(y_trig != 0)[0] + 1
    schift_list = np.insert(schift_list, 0, np.array([0]))
    schift_list_y = np.array([y[i] for i in schift_list])
    
    classes = np.unique(schift_list_y)

    if debug:

        plt.plot(y)
        plt.show()

    classe_sig = {'X' : {}, 'y' : {}}

    #classe = 0
    for classe in classes:

        classe_start = np.where(schift_list_y == classe)[0]
        classe_start_i_list = schift_list[classe_start]
        classe_stop_i_list = []

        for _classe_start in classe_start:
            try:
                _stop = schift_list[_classe_start+1] 
            except:
                _stop = y.shape[0]

            classe_stop_i_list.append(_stop)

        if debug:
            plt.plot(y)
            plt.vlines(classe_start_i_list, ymin=y.min(), ymax=y.max(), label='start', colors='g')
            plt.vlines(classe_stop_i_list, ymin=y.min(), ymax=y.max(), label='stop', colors='r')
            plt.legend()
            plt.show()

        #### extract all classes signal for sorting
        for classe_start_i, classe_start_val in enumerate(classe_start_i_list):

            _X = X[classe_start_val:classe_stop_i_list[classe_start_i],:]
            _y = y[classe_start_val:classe_stop_i_list[classe_start_i]]

            if classe_start_i == 0:
                classe_sig_X = _X
                classe_sig_y = _y
            else:
                classe_sig_X = np.append(classe_sig_X, _X, axis=0)
                classe_sig_y = np.append(classe_sig_y, _y, axis=0)

        classe_sig['X'][classe] = classe_sig_X
        classe_sig['y'][classe] = classe_sig_y

        if debug:

            plt.plot(classe_sig['y'][classe])
            plt.show()

    #### identify tiniest classe to balance with the other
    if balance:
        classes_shape = np.array([_X.shape[0] for _X in classe_sig['X'].values()])
        min_shape = classes_shape.min()

        #### random sel and balancing of classes
        classe_random_sel = np.array([np.random.choice(classes_shape[int(classe_i)], min_shape, replace=False) for classe_i in range(classes.shape[0])])

        classe_sig_balanced = {'X' : {}, 'y' : {}}

        for classe_i, classe in enumerate(classes):
        
            classe_sig_balanced['X'][classe] = classe_sig['X'][classe][classe_random_sel[classe_i,:],:]
            classe_sig_balanced['y'][classe] = classe_sig['y'][classe][classe_random_sel[classe_i,:]]

        if debug:
            
            for classe_i, classe in enumerate(classes):

                classe_sig_balanced['X'][classe].shape
                classe_sig_balanced['y'][classe].shape

                np.unique(classe_random_sel[classe_i,:]).max()
    else:

        classe_sig_balanced = {'X' : {}, 'y' : {}}

        for classe_i, classe in enumerate(classes):
        
            classe_sig_balanced['X'][classe] = classe_sig['X'][classe]
            classe_sig_balanced['y'][classe] = classe_sig['y'][classe]

    #### random extraction of every classes
    X_train, X_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])

    #classe_i, classe = 0, 0
    for classe_i, classe in enumerate(classes):
            
        _X_train, _X_test, _y_train, _y_test = train_test_split(classe_sig_balanced['X'][classe], classe_sig_balanced['y'][classe], train_size=train_size, random_state=5)

        if classe_i == 0:
            X_train = _X_train
            X_test = _X_test
            y_train = _y_train
            y_test = _y_test
        else:
            X_train = np.append(X_train, _X_train, axis=0)
            X_test = np.append(X_test, _X_test, axis=0)
            y_train = np.append(y_train, _y_train, axis=0)
            y_test = np.append(y_test, _y_test, axis=0)

    return X_train, X_test, y_train, y_test


########################################
######## TRACKING FUNCTION ########
########################################


def zscore(data):
    zscore_data = (data - np.mean(data))/np.std(data)
    return zscore_data



def get_data_hrv_tracker(ecg, prms_tracker):

    #### extract params
    srate = prms_tracker['srate']
    win_size_sec, jitter = prms_tracker['win_size_sec'], prms_tracker['jitter']
    win_size = int(win_size_sec*srate)

    #### precompute ecg
    ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks] = 10

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
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    times = [ecg_cR_val[cR_initial]]

    #### sliding on other cR
    for cR_i in range(len(ecg_cR_val)-cR_initial):
        cR_i += cR_initial

        ecg_cR_sliding_win = np.append(ecg_cR_sliding_win[1:], ecg_cR_val[cR_i])
        RRI_win = np.diff(ecg_cR_sliding_win)
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        
        df_res = pd.concat([df_res, df_slide], axis=0)

        times.append(ecg_cR_val[cR_i])

    times = np.array(times)

    return df_res, times




#sujet_i = 1
#ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm(ecg, classifier, prms_tracker, hrv_tracker_mode):

    srate = prms_tracker['srate']

    win_size_sec, metric_used, odor_trig_n_bpm = prms_tracker['win_size_sec'], prms_tracker['metric_list'], prms_tracker['odor_trig_n_bpm']
    win_size = int(win_size_sec*srate)

    #### load cR
    ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks] = 10

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
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
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
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        predictions = np.append(predictions, classifier.predict(df_slide.values))
        df_res = pd.concat([df_res, df_slide], axis=0)

        if predictions.shape[0] >= odor_trig_n_bpm:
            trig_odor_win = predictions.copy()[-odor_trig_n_bpm:]
            
            if hrv_tracker_mode == '4classes':
                trig_odor_win[trig_odor_win < 2] = 0
                trig_odor_win[(trig_odor_win == cond_label_tracker['MECA']) | (trig_odor_win == cond_label_tracker['CO2'])] = 1

                trig_odor_pred_i = np.round(np.mean(trig_odor_win))

            if hrv_tracker_mode == '2classes':

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







################################################
######## TEST NUMBER OF FEATURES ########
################################################



#sujet_i = 1
#ecg, win_size, srate, srate_resample_hrv, classifier, metric_used, odor_trig_n_bpm, labels_dict = ecg_test, win_size, srate, srate_resample_hrv, model, labels_used, odor_trig_n_bpm, labels_dict
def hrv_tracker_svm_test_features(ecg, classifier, prms_tracker, features_selected):

    srate = prms_tracker['srate']

    win_size_sec, metric_used, odor_trig_n_bpm = prms_tracker['win_size_sec'], prms_tracker['metric_list'], prms_tracker['odor_trig_n_bpm']
    win_size = int(win_size_sec*srate)

    #### load cR
    ecg, ecg_peaks = physio.compute_ecg(ecg, srate)

    ecg_cR = np.zeros((ecg.shape[0]))
    ecg_cR[ecg_peaks] = 10

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
    df_res = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
    df_res = df_res[features_selected]
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
        df_slide = get_hrv_metrics_win(RRI_win)[prms_tracker['metric_list']]
        df_slide = df_slide[features_selected]
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



def hrv_tracker_features_selection(sujet, hrv_tracker_mode):

    print('################', flush=True)
    print(f'#### {sujet} ####', flush=True)
    print('################', flush=True)

    if os.path.exists(os.path.join(path_precompute, sujet, 'HRV', f'{sujet}_hrv_tracker_test_features.nc')):
        print('ALREADY COMPUTED', flush=True)
        return

    #### params
    features_test_list = ['MeanNN_SDNN_RMSSD', 'pNN50', 'SD1_SD2', 'COV', 'pNN50_SD1_SD2_COV', 'MeanNN_SDNN_RMSSD_pNN50_SD1_SD2_COV']

    band_prep = 'wb'
    odor_ref = 'o'

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    xr_dict = {'sujet' : [sujet], 'features' : np.array(features_test_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((1, len(features_test_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : [sujet], 'features' : np.array(features_test_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((1, len(features_test_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    ######### LOAD #########
    ecg = load_ecg_sig(sujet, odor_ref, band_prep)
    df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
    label_vec, trig = get_label_vec(sujet, odor_ref, ecg, hrv_tracker_mode)
    label_vec = label_vec[(times*srate).astype('int')]

    test_value = 0.2

    #features_i = features_test_list[0]
    for features_i in features_test_list:

        print(f'compute tracker {sujet} {features_i}', flush=True)

        features_selected = [f"HRV_{feature_i}" for feature_i in features_i.split('_')]
        df_hrv_selected = df_hrv[features_selected]

        if debug:

            plt.plot(ecg)
            plt.show()

            plt.plot(label_vec)
            plt.show()

        ######### COMPUTE MODEL #########
        #### split values
        X, y = df_hrv_selected.values, label_vec.copy()
        X_train, X_test, y_train, y_test = split_data(X, y, test_value, balance=True)
        
        #### make pipeline
        #SVC().get_params()
        steps = [('scaler', StandardScaler()), ('SVM', SVC())]
        pipeline = Pipeline(steps)

        #### find best model
        params = {
        # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
        # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
        'SVM__C' : [1e4], 
        'SVM__gamma' : [0.1]
        }

        print('train', flush=True)
        grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
        grid.fit(X_train, y_train)
        classifier_score = grid.best_score_
        classifier = grid.best_estimator_
        print('train done', flush=True)
        
        ######### TEST MODEL #########
        
        #### get values
        df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm_test_features(ecg, classifier, prms_tracker, features_selected)

        #### resample label_vec_time
        # f = scipy.interpolate.interp1d(label_vec_time, label_vec, kind='linear')
        # label_vec_time_resampled = f(predictions_time)

        #### trim vectors
        predictions_trim = np.array([])
        label_vec_trim = np.array([])
        trig_odor_trim = np.array([])

        #cond_i, cond = 1, conditions[1]
        for cond_i, cond in enumerate(conditions):

            if cond_i != len(conditions)-1:

                #### cond
                start = trig[cond][0]/srate
                stop = trig[cond][1]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= stop)

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_cond_resampled = np.round(label_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

                #### intercond
                start = trig[cond][1]/srate
                stop = trig[conditions[cond_i+1]][0]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= start + trim_between) | (stop - trim_between <= predictions_time) & (predictions_time <= stop)

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_inter_cond_resampled = np.round(pred_inter_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_inter_cond_resampled = np.round(label_inter_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_inter_cond_resampled = np.round(trig_inter_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_inter_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_inter_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_inter_cond_resampled), axis=0)

            elif cond_i == len(conditions)-1:

                start = trig[cond][0]/srate
                stop = trig[cond][1]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= stop) 

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_cond_resampled = np.round(label_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

            if debug:

                plt.plot(predictions_trim, label='prediction', linestyle='--')
                plt.plot(label_vec_trim, label='real')
                plt.plot(trig_odor_trim, label='odor_trig')
                plt.legend()
                plt.show()

        #### load results
        xr_hrv_tracker.loc[sujet, features_i, 'prediction', :] = predictions_trim
        xr_hrv_tracker.loc[sujet, features_i, 'label', :] = label_vec_trim
        xr_hrv_tracker.loc[sujet, features_i, 'trig_odor', :] = trig_odor_trim

        xr_hrv_tracker_score.loc[sujet, features_i] = np.round(classifier_score, 5)

    ######## SAVE ########
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
    xr_hrv_tracker.to_netcdf(f'{sujet}_hrv_tracker_test_features.nc')
    xr_hrv_tracker_score.to_netcdf(f'{sujet}_hrv_tracker_test_features_score.nc')

    # mutual_info_classif(X, y)
    # prms_tracker['metric_list']






########################################
######## HRV TRACKER TEST ########
########################################






def hrv_tracker_no_ref_modify_train(sujet, hrv_tracker_mode):

    print('################', flush=True)
    print(f'#### {sujet} ####', flush=True)
    print('################', flush=True)

    if os.path.exists(os.path.join(path_precompute, sujet, 'HRV', f'{hrv_tracker_mode}_no_ref_{sujet}_hrv_tracker_alltestsize.nc')):
        print('ALREADY COMPUTED', flush=True)
        return

    ########################
    ######## PARAMS ########
    ########################

    band_prep = 'wb'

    train_percentage_values = [0.5, 0.6, 0.7, 0.8]

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    xr_dict = {'sujet' : [sujet], 'train_percentage' : train_percentage_values, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((1, len(train_percentage_values), len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : [sujet], 'train_percentage' : train_percentage_values, 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((1, len(train_percentage_values), len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    predictions_dict = {}
    for train_value in train_percentage_values:

        predictions_dict[train_value] = {}
        for odor_i in odor_list:

            predictions_dict[train_value][odor_i] = {}
            for trim_type in ['trim', 'no_trim']:

                predictions_dict[train_value][odor_i][trim_type] = {}
                for data_type in ['real', 'predict', 'odor_trig', 'score']:

                    predictions_dict[train_value][odor_i][trim_type][data_type] = []

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        #train_value = train_percentage_values[0]
        for train_value in train_percentage_values:

            print(f'compute tracker {sujet} {odor_i} {train_value}', flush=True)

            ######### LOAD #########
            ecg = load_ecg_sig(sujet, odor_i, band_prep)

            df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
            label_vec, trig = get_label_vec(sujet, odor_i, ecg, hrv_tracker_mode)
            label_vec = label_vec[(times*srate).astype('int')]

            if debug:

                plt.plot(ecg)
                plt.show()

                plt.plot(label_vec)
                plt.show()

            ######### COMPUTE MODEL #########
            #### split values
            X, y = df_hrv.values, label_vec.copy()
            X_train, X_test, y_train, y_test = split_data(X, y, train_value, balance=True)
            
            #### make pipeline
            #SVC().get_params()
            steps = [('scaler', StandardScaler()), ('SVM', SVC())]
            pipeline = Pipeline(steps)

            #### find best model
            params = {
            # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
            # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
            'SVM__C' : [1e6, 1e5], 
            'SVM__gamma' : [1, 0.1]
            }

            print('train', flush=True)
            grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
            grid.fit(X_train, y_train)
            classifier_score = grid.best_score_
            classifier = grid.best_estimator_
            SVM_params = [classifier.get_params()['SVM'].get_params()['kernel'], classifier.get_params()['SVM'].get_params()['C'], classifier.get_params()['SVM'].get_params()['gamma']]
            print('train done', flush=True)

            ######### TEST MODEL #########
            #### get values
            df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker, hrv_tracker_mode)

            #### resample label_vec_time
            # f = scipy.interpolate.interp1d(label_vec_time, label_vec, kind='linear')
            # label_vec_time_resampled = f(predictions_time)

            #### trim vectors
            predictions_trim = np.array([])
            label_vec_trim = np.array([])
            trig_odor_trim = np.array([])

            #cond_i, cond = 1, conditions[1]
            for cond_i, cond in enumerate(conditions):

                if cond_i != len(conditions)-1:

                    #### cond
                    start = trig[cond][0]/srate
                    stop = trig[cond][1]/srate

                    mask_start = (start <= predictions_time) & (predictions_time <= stop)

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                    pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                    label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    label_cond_resampled = np.round(label_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                    trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                    predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

                    #### intercond
                    start = trig[cond][1]/srate
                    stop = trig[conditions[cond_i+1]][0]/srate

                    mask_start = (start <= predictions_time) & (predictions_time <= start + trim_between) | (stop - trim_between <= predictions_time) & (predictions_time <= stop)

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                    pred_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    pred_inter_cond_resampled = np.round(pred_inter_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                    label_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    label_inter_cond_resampled = np.round(label_inter_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                    trig_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    trig_inter_cond_resampled = np.round(trig_inter_cond_resampled).astype('int')

                    predictions_trim = np.concatenate((predictions_trim, pred_inter_cond_resampled), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_inter_cond_resampled), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_inter_cond_resampled), axis=0)

                elif cond_i == len(conditions)-1:

                    start = trig[cond][0]/srate
                    stop = trig[cond][1]/srate

                    mask_start = (start <= predictions_time) & (predictions_time <= stop) 

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                    pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                    label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    label_cond_resampled = np.round(label_cond_resampled).astype('int')

                    f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                    trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                    trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                    predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

                if debug:

                    plt.plot(predictions_trim, label='prediction', linestyle='--')
                    plt.plot(label_vec_trim, label='real')
                    plt.plot(trig_odor_trim, label='odor_trig')
                    plt.legend()
                    plt.show()

            #### load res
            for trim_type in ['trim', 'no_trim']:

                if trim_type == 'trim':
                    data_load = [label_vec_trim, predictions_trim, trig_odor_trim, classifier_score]
                if trim_type == 'no_trim':
                    data_load = [label_vec, predictions, trig_odor, classifier_score]

                for data_type_i, data_type in enumerate(['real', 'predict', 'odor_trig', 'score']):

                    predictions_dict[train_value][odor_i][trim_type][data_type] = data_load[data_type_i]

    ######## PLOT ########
    for train_value in train_percentage_values:
        fig_whole, axs = plt.subplots(ncols=len(odor_list), figsize=(18,9))
        for odor_i, odor in enumerate(odor_list):
            ax = axs[odor_i]
            ax.plot(predictions_dict[train_value][odor]['no_trim']['real'], color='k', label='real', linestyle=':', linewidth=3)
            ax.plot(predictions_dict[train_value][odor]['no_trim']['predict'], color='y', label='prediction')
            ax.plot(predictions_dict[train_value][odor]['no_trim']['odor_trig'], color='r', label='odor_trig', linestyle='--')
            ax.set_title(f"ref : {odor}, predict : {odor}, perf : {np.round(predictions_dict[train_value][odor]['no_trim']['score'], 3)}")
        plt.suptitle(f'{sujet} RAW, train value : {train_value}')
        plt.legend()
        # fig_whole.show()
        plt.close()

        fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
        for odor_i, odor in enumerate(odor_list):
            ax = axs[odor_i]
            ax.plot(predictions_dict[train_value][odor]['trim']['real'], color='k', label='real', linestyle=':', linewidth=3)
            ax.plot(predictions_dict[train_value][odor]['trim']['predict'], color='y', label='prediction')
            ax.plot(predictions_dict[train_value][odor]['trim']['odor_trig'], color='r', label='odor_trig', linestyle='--')
            ax.set_title(f"ref : {odor}, predict : {odor}, perf : {np.round(predictions_dict[train_value][odor]['trim']['score'], 3)}")
        plt.suptitle(f'{sujet} TRIMMED, train value : {train_value}')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, sujet, 'HRV'))
        fig_whole.savefig(f'no_ref_{sujet}_{hrv_tracker_mode}_train_{str(train_value*100)}_hrv_tracker_whole.png')
        fig_trim.savefig(f'no_ref_{sujet}_{hrv_tracker_mode}_train_{str(train_value*100)}_hrv_tracker_trim.png')

    #### load results
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
    
    for train_value in train_percentage_values:
        
        for odor_i, odor in enumerate(odor_list):
            xr_hrv_tracker.loc[sujet, train_value, odor, 'prediction', :] = predictions_dict[train_value][odor]['trim']['predict']
            xr_hrv_tracker.loc[sujet, train_value, odor, 'label', :] = predictions_dict[train_value][odor]['trim']['real']
            xr_hrv_tracker.loc[sujet, train_value, odor, 'trig_odor', :] = predictions_dict[train_value][odor]['trim']['odor_trig']

            xr_hrv_tracker_score.loc[sujet, train_value, odor] = np.round(predictions_dict[train_value][odor]['trim']['score'], 5)

    xr_hrv_tracker.to_netcdf(f'{hrv_tracker_mode}_no_ref_{sujet}_hrv_tracker_alltestsize.nc')
    xr_hrv_tracker_score.to_netcdf(f'{hrv_tracker_mode}_no_ref_{sujet}_hrv_tracker_score_alltestsize.nc')






def hrv_tracker_with_ref(sujet, hrv_tracker_mode):

    print('################', flush=True)
    print(f'#### {sujet} ####', flush=True)
    print('################', flush=True)

    if os.path.exists(os.path.join(path_precompute, sujet, 'HRV', f'{hrv_tracker_mode}_o_ref_{sujet}_hrv_tracker.nc')):
        print('ALREADY COMPUTED', flush=True)
        return

    #### params
    band_prep = 'wb'

    odor_ref = 'o'
    # odor_list_test = [odor_i for odor_i in odor_list if odor_i != odor_ref]

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    xr_dict = {'sujet' : [sujet], 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((1, len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : [sujet], 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((1, len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    predictions_dict = {}
    for odor_i in odor_list:

        predictions_dict[odor_i] = {}
        for trim_type in ['trim', 'no_trim']:

            predictions_dict[odor_i][trim_type] = {}
            for data_type in ['real', 'predict', 'odor_trig', 'score']:

                predictions_dict[odor_i][trim_type][data_type] = []

    print(f'compute tracker {sujet} {odor_ref}', flush=True)

    ######### LOAD #########
    ecg = load_ecg_sig(sujet, odor_ref, band_prep)

    df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
    label_vec, trig = get_label_vec(sujet, odor_ref, ecg, hrv_tracker_mode)
    label_vec = label_vec[(times*srate).astype('int')]

    if debug:

        plt.plot(ecg)
        plt.show()

        plt.plot(label_vec)
        plt.show()

    ######### COMPUTE MODEL #########
    #### split values
    X, y = df_hrv.values, label_vec.copy()
    train_size=0.8
    X_train, X_test, y_train, y_test = split_data(X, y, train_size, balance=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {
    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [1e6, 1e5], 
    'SVM__gamma' : [1, 0.1]
    }

    print('train', flush=True)
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    print('train done', flush=True)
    
    ######### TEST MODEL #########
    #odor_i, odor = 0, odor_list[0]
    for odor_i, odor in enumerate(odor_list):

        #### load data
        ecg = load_ecg_sig(sujet, odor, band_prep)

        df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
        label_vec, trig = get_label_vec(sujet, odor, ecg, hrv_tracker_mode)
        label_vec = label_vec[(times*srate).astype('int')]
        
        #### get values
        df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker, hrv_tracker_mode)

        #### trim vectors
        predictions_trim = np.array([])
        label_vec_trim = np.array([])
        trig_odor_trim = np.array([])

        #cond_i, cond = 1, conditions[1]
        for cond_i, cond in enumerate(conditions):

            if cond_i != len(conditions)-1:

                #### cond
                start = trig[cond][0]/srate
                stop = trig[cond][1]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= stop)

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_cond_resampled = np.round(label_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

                #### intercond
                start = trig[cond][1]/srate
                stop = trig[conditions[cond_i+1]][0]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= start + trim_between) | (stop - trim_between <= predictions_time) & (predictions_time <= stop)

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_inter_cond_resampled = np.round(pred_inter_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_inter_cond_resampled = np.round(label_inter_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_inter_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_inter_cond_resampled = np.round(trig_inter_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_inter_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_inter_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_inter_cond_resampled), axis=0)

            elif cond_i == len(conditions)-1:

                start = trig[cond][0]/srate
                stop = trig[cond][1]/srate

                mask_start = (start <= predictions_time) & (predictions_time <= stop) 

                f = scipy.interpolate.interp1d(np.linspace(0, 1, predictions[mask_start].shape[0]), predictions[mask_start], kind='linear')
                pred_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                pred_cond_resampled = np.round(pred_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, label_vec[mask_start].shape[0]), label_vec[mask_start], kind='linear')
                label_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                label_cond_resampled = np.round(label_cond_resampled).astype('int')

                f = scipy.interpolate.interp1d(np.linspace(0, 1, trig_odor[mask_start] .shape[0]), trig_odor[mask_start] , kind='linear')
                trig_cond_resampled = f(np.linspace(0, 1, points_per_cond))
                trig_cond_resampled = np.round(trig_cond_resampled).astype('int')

                predictions_trim = np.concatenate((predictions_trim, pred_cond_resampled), axis=0)
                label_vec_trim = np.concatenate((label_vec_trim, label_cond_resampled), axis=0)
                trig_odor_trim = np.concatenate((trig_odor_trim, trig_cond_resampled), axis=0)

            if debug:

                plt.plot(predictions_trim, label='prediction', linestyle='--')
                plt.plot(label_vec_trim, label='real')
                plt.plot(trig_odor_trim, label='odor_trig')
                plt.legend()
                plt.show()

        #### load res
        for trim_type in ['trim', 'no_trim']:

            if trim_type == 'trim':
                data_load = [label_vec_trim, predictions_trim, trig_odor_trim, classifier_score]
            if trim_type == 'no_trim':
                data_load = [label_vec, predictions, trig_odor, classifier_score]

            for data_type_i, data_type in enumerate(['real', 'predict', 'odor_trig', 'score']):

                predictions_dict[odor][trim_type][data_type] = data_load[data_type_i]

    ######## PLOT ########
    fig_whole, axs = plt.subplots(ncols=len(odor_list), figsize=(18,9))
    for odor_i, odor in enumerate(odor_list):
        ax = axs[odor_i]
        ax.plot(predictions_dict[odor]['no_trim']['real'], color='k', label='real', linestyle=':', linewidth=3)
        ax.plot(predictions_dict[odor]['no_trim']['predict'], color='y', label='prediction')
        ax.plot(predictions_dict[odor]['no_trim']['odor_trig'], color='r', label='odor_trig', linestyle='--')
        ax.set_title(f"{odor_i}, ref : {odor_ref}, predict : {odor}, perf o : {np.round(predictions_dict[odor]['no_trim']['score'], 3)}")
    plt.suptitle(f'{sujet} RAW')
    plt.legend()
    # fig_whole.show()
    plt.close()

    fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
    for odor_i, odor in enumerate(odor_list):
        ax = axs[odor_i]
        ax.plot(predictions_dict[odor]['trim']['real'], color='k', label='real', linestyle=':', linewidth=3)
        ax.plot(predictions_dict[odor]['trim']['predict'], color='y', label='prediction')
        ax.plot(predictions_dict[odor]['trim']['odor_trig'], color='r', label='odor_trig', linestyle='--')
        ax.set_title(f"ref : {odor_ref}, predict : {odor}, perf : {np.round(predictions_dict[odor]['trim']['score'], 3)}")
    plt.suptitle(f'{sujet} TRIMMED')
    plt.legend()
    # fig_trim.show()
    plt.close()

    ######## SAVE ########
    os.chdir(os.path.join(path_results, sujet, 'HRV'))
    fig_whole.savefig(f'o_ref_{sujet}_{hrv_tracker_mode}_hrv_tracker_whole.png')
    fig_trim.savefig(f'o_ref_{sujet}_{hrv_tracker_mode}_hrv_tracker_trim.png')

    #### load results
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
    for odor_i, odor in enumerate(odor_list):
        xr_hrv_tracker.loc[sujet, odor, 'prediction', :] = predictions_dict[odor]['trim']['predict']
        xr_hrv_tracker.loc[sujet, odor, 'label', :] = predictions_dict[odor]['trim']['real']
        xr_hrv_tracker.loc[sujet, odor, 'trig_odor', :] = predictions_dict[odor]['trim']['odor_trig']

    xr_hrv_tracker_score.loc[sujet, odor] = np.round(classifier_score, 5)

    xr_hrv_tracker.to_netcdf(f'{hrv_tracker_mode}_o_ref_{sujet}_hrv_tracker.nc')
    xr_hrv_tracker_score.to_netcdf(f'{hrv_tracker_mode}_o_ref_{sujet}_hrv_tracker_score.nc')










def hrv_tracker_test_SVM(sujet):

    print('################', flush=True)
    print(f'#### {sujet} ####', flush=True)
    print('################', flush=True)

    if os.path.exists(os.path.join(path_precompute, sujet, 'HRV', f'{sujet}_hrv_tracker_SVM_test.xlsx')):
        print('ALREADY COMPUTED', flush=True)
        return

    ########################
    ######## PARAMS ########
    ########################

    band_prep = 'wb'

    train_percentage_values = [0.5, 0.6, 0.7, 0.8]

    data_df = {'sujet' : [], 'hrv_tracker_mode' : [], 'ref' : [], 'balanced' : [], 'train_percentage' : [], 'odor' : [], 'kernel' : [], 'C' : [], 'gamma' : [], 'score' : []}

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    for hrv_tracker_mode in ['4classes', '2classes']:

        for balanced in [True, False]:

            #train_value = train_percentage_values[0]
            for train_value in train_percentage_values:
                    
                #odor = odor_list[0]
                for odor in odor_list:

                    if odor == 'o':

                        print(f'compute tracker {sujet} {odor} {train_value}, ref : o', flush=True)

                        ######### LOAD #########
                        ecg = load_ecg_sig(sujet, odor, band_prep)

                        df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
                        label_vec, trig = get_label_vec(sujet, odor, ecg, hrv_tracker_mode)
                        label_vec = label_vec[(times*srate).astype('int')]

                        if debug:

                            plt.plot(ecg)
                            plt.show()

                            plt.plot(label_vec)
                            plt.show()

                        ######### COMPUTE MODEL #########
                        #### split values
                        X, y = df_hrv.values, label_vec.copy()
                        X_train, X_test, y_train, y_test = split_data(X, y, train_value, balance=True)
                        
                        #### make pipeline
                        #SVC().get_params()
                        steps = [('scaler', StandardScaler()), ('SVM', SVC())]
                        pipeline = Pipeline(steps)

                        #### find best model
                        params = {
                        # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
                        # 'SVM__kernel' : ['linear', 'rbf'],    
                        'SVM__C' : [1e4, 1e5, 1e6], 
                        'SVM__gamma' : [1, 0.1, 0.01]
                        }

                        print('train', flush=True)
                        grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
                        grid.fit(X_train, y_train)
                        classifier = grid.best_estimator_
                        print('train done', flush=True)

                        #odor_to_test_ref = odor_list[0]
                        for odor_to_test_ref in odor_list:

                            ######### LOAD #########
                            ecg = load_ecg_sig(sujet, odor_to_test_ref, band_prep)

                            df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
                            label_vec, trig = get_label_vec(sujet, odor_to_test_ref, ecg, hrv_tracker_mode)
                            label_vec = label_vec[(times*srate).astype('int')]

                            ######### COMPUTE MODEL #########
                            #### split values
                            X, y = df_hrv.values, label_vec.copy()
                            X_train, X_test, y_train, y_test = split_data(X, y, train_value, balance=True)

                            #### get values
                            df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker, hrv_tracker_mode)

                            #### get accuracy
                            _accuracy = ((predictions-y == 0)*1).sum() / y.shape[0]

                            if debug:

                                plt.plot(predictions)
                                plt.plot(y)
                                plt.show()

                                plt.plot(predictions-y)
                                plt.show()

                            ######### LOAD #########

                            if odor_to_test_ref == 'o':

                                data_df['sujet'].append(sujet)
                                data_df['hrv_tracker_mode'].append(hrv_tracker_mode)
                                data_df['ref'].append('no_ref')
                                data_df['balanced'].append(balanced)
                                data_df['train_percentage'].append(train_value)
                                data_df['odor'].append(odor_to_test_ref)
                                data_df['kernel'].append(classifier.get_params()['SVM'].get_params()['kernel'])
                                data_df['C'].append(classifier.get_params()['SVM'].get_params()['C'])
                                data_df['gamma'].append(classifier.get_params()['SVM'].get_params()['gamma'])
                                data_df['score'].append(_accuracy)

                            else:

                                data_df['sujet'].append(sujet)
                                data_df['hrv_tracker_mode'].append(hrv_tracker_mode)
                                data_df['ref'].append('o')
                                data_df['balanced'].append(balanced)
                                data_df['train_percentage'].append(train_value)
                                data_df['odor'].append(odor_to_test_ref)
                                data_df['kernel'].append(classifier.get_params()['SVM'].get_params()['kernel'])
                                data_df['C'].append(classifier.get_params()['SVM'].get_params()['C'])
                                data_df['gamma'].append(classifier.get_params()['SVM'].get_params()['gamma'])
                                data_df['score'].append(_accuracy)

                    else:

                        print(f'compute tracker {sujet} {odor} {train_value}', flush=True)

                        ######### LOAD #########
                        ecg = load_ecg_sig(sujet, odor, band_prep)

                        df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
                        label_vec, trig = get_label_vec(sujet, odor, ecg, hrv_tracker_mode)
                        label_vec = label_vec[(times*srate).astype('int')]

                        if debug:

                            plt.plot(ecg)
                            plt.show()

                            plt.plot(label_vec)
                            plt.show()

                        ######### COMPUTE MODEL #########
                        #### split values
                        X, y = df_hrv.values, label_vec.copy()
                        X_train, X_test, y_train, y_test = split_data(X, y, train_value, balance=True)
                        
                        #### make pipeline
                        #SVC().get_params()
                        steps = [('scaler', StandardScaler()), ('SVM', SVC())]
                        pipeline = Pipeline(steps)

                        #### find best model
                        params = {
                        # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
                        # 'SVM__kernel' : ['linear', 'rbf'],    
                        'SVM__C' : [1e4, 1e5, 1e6], 
                        'SVM__gamma' : [1, 0.1, 0.01]
                        }

                        print('train', flush=True)
                        grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
                        grid.fit(X_train, y_train)
                        classifier = grid.best_estimator_
                        print('train done', flush=True)

                        ######### LOAD #########

                        data_df['sujet'].append(sujet)
                        data_df['hrv_tracker_mode'].append(hrv_tracker_mode)
                        data_df['ref'].append('no_ref')
                        data_df['balanced'].append(balanced)
                        data_df['train_percentage'].append(train_value)
                        data_df['odor'].append(odor)
                        data_df['kernel'].append(classifier.get_params()['SVM'].get_params()['kernel'])
                        data_df['C'].append(classifier.get_params()['SVM'].get_params()['C'])
                        data_df['gamma'].append(classifier.get_params()['SVM'].get_params()['gamma'])
                        data_df['score'].append(grid.best_score_)

    df_sujet = pd.DataFrame(data_df)
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
    df_sujet.to_excel(f'{sujet}_hrv_tracker_SVM_test.xlsx')







################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':



    ########################################
    ######## EXECUTE CLUSTER ########
    ########################################

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #hrv_tracker_mode = '2classes'
        for hrv_tracker_mode in ['4classes', '2classes']:

            # hrv_tracker_no_ref_modify_train(sujet, hrv_tracker_mode)
            execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_tracker_no_ref_modify_train', [sujet, hrv_tracker_mode])

            # hrv_tracker_with_ref(sujet, hrv_tracker_mode)
            execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_tracker_with_ref', [sujet, hrv_tracker_mode])

            # hrv_tracker_features_selection(sujet, hrv_tracker_mode)
            # execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_tracker_features_selection', [sujet, hrv_tracker_mode])
            
        # hrv_tracker_test_SVM(sujet)
        execute_function_in_slurm_bash('n4bis_precompute_hrv_tracker', 'hrv_tracker_test_SVM', [sujet])



