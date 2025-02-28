

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

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n01bis_prep_info import *
from n04_precompute_hrv import *
from n04bis_precompute_hrv_tracker import *

debug = False






################################
######## COMPILATION ########
################################


def get_classifier_allsujet_hrv_tracker_o_ref(hrv_tracker_mode):

    print('################', flush=True)
    print(f'#### O REF ####', flush=True)
    print('################', flush=True)

    ########################
    ######## PARAMS ########
    ########################

    band_prep = 'wb'

    train_size = 0.8

    odor = 'o'

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    print(f'compute tracker {odor}', flush=True)

    ######### LOAD #########
    ecg_allsujet = np.array([])
    label_vec_allsujet = np.array([])

    for sujet_i, sujet in enumerate(sujet_list):

        print_advancement(sujet_i, len(sujet_list), [25, 50, 75])

        ecg = load_ecg_sig(sujet, odor, band_prep)
        ecg_allsujet = np.append(ecg_allsujet, ecg)

        label_vec, trig = get_label_vec(sujet, odor, ecg, hrv_tracker_mode)
        label_vec_allsujet = np.append(label_vec_allsujet, label_vec)

    df_hrv_allsujet, times = get_data_hrv_tracker(ecg_allsujet, prms_tracker)
        
    label_vec_allsujet = label_vec_allsujet[(times*srate).astype('int')]

    if debug:

        plt.plot(ecg_allsujet)
        plt.show()

        plt.plot(label_vec_allsujet)
        plt.show()

    ######### COMPUTE MODEL #########
    #### split values
    X, y = df_hrv_allsujet.values, label_vec_allsujet.copy()
    X_train, X_test, y_train, y_test = split_data(X, y, train_size, balance=True)
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {
    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [1e4, 1e5], 
    'SVM__gamma' : [0.1, 0.01]
    }

    print('train', flush=True)
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    print('train done', flush=True)

    classifier_params = {'kernel' : classifier.get_params()['SVM'].get_params()['kernel'],
                        'C' : classifier.get_params()['SVM'].get_params()['C'],
                        'gamma' : classifier.get_params()['SVM'].get_params()['gamma']}
    
    lines = [f"{key} {value}" for key, value in classifier_params.items()]

    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

    with open('classifier_params.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    return classifier




def allsujet_hrv_tracker_ref_o(classifier, hrv_tracker_mode):

    ########################
    ######## PARAMS ########
    ########################

    n_pnts_trim_resample = (len(conditions) + len(conditions) -1) * points_per_cond

    band_prep = 'wb'

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    ######### TEST MODEL #########

    for sujet in sujet_list:

        print(sujet)

        predictions_dict = {}
        for odor_i in odor_list:

            predictions_dict[odor_i] = {}
            for trim_type in ['trim', 'no_trim']:

                predictions_dict[odor_i][trim_type] = {}
                for data_type in ['real', 'predict', 'odor_trig', 'score']:

                    predictions_dict[odor_i][trim_type][data_type] = []
            
        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            #### load data
            ecg = load_ecg_sig(sujet, odor, band_prep)

            df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
            label_vec, trig = get_label_vec(sujet, odor, ecg, hrv_tracker_mode)
            label_vec = label_vec[(times*srate).astype('int')]
            
            #### get values
            df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker, hrv_tracker_mode)

            #### get accuracy
            _accuracy = ((predictions-label_vec == 0)*1).sum() / label_vec.shape[0]

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
                    data_load = [label_vec_trim, predictions_trim, trig_odor_trim, _accuracy]
                if trim_type == 'no_trim':
                    data_load = [label_vec, predictions, trig_odor, _accuracy]

                for data_type_i, data_type in enumerate(['real', 'predict', 'odor_trig', 'score']):

                    predictions_dict[odor][trim_type][data_type] = data_load[data_type_i]

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
        for odor_i, odor in enumerate(odor_list):
            xr_hrv_tracker.loc[sujet, odor, 'prediction', :] = predictions_dict[odor]['trim']['predict']
            xr_hrv_tracker.loc[sujet, odor, 'label', :] = predictions_dict[odor]['trim']['real']
            xr_hrv_tracker.loc[sujet, odor, 'trig_odor', :] = predictions_dict[odor]['trim']['odor_trig']

            xr_hrv_tracker_score.loc[sujet, odor] = predictions_dict[odor]['trim']['score']

    #### save results
    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    xr_hrv_tracker.to_netcdf(f'{hrv_tracker_mode}_o_ref_allsujettrain_hrv_tracker.nc')
    xr_hrv_tracker_score.to_netcdf(f'{hrv_tracker_mode}_o_ref_allsujettrain_hrv_tracker_score.nc')







################################
######## COMPILATION ######## 
################################

def hrv_tracker_compilation_allsujet(hrv_tracker_mode):

    classifier = get_classifier_allsujet_hrv_tracker_o_ref(hrv_tracker_mode)
    allsujet_hrv_tracker_ref_o(classifier, hrv_tracker_mode)






################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':

    hrv_tracker_mode = '2classes'
    #hrv_tracker_mode = '4classes'

    #hrv_tracker_compilation_allsujet(hrv_tracker_mode)
    execute_function_in_slurm_bash('n17_precompute_allsujet_hrv_tracker', 'hrv_tracker_compilation_allsujet', [hrv_tracker_mode])




