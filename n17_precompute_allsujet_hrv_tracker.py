

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
from n1bis_prep_info import *
from n4_precompute_hrv import *
from n4bis_precompute_hrv_tracker import *

debug = False






################################
######## COMPILATION ########
################################


def get_classifier_allsujet_hrv_tracker_o_ref():

    print('################', flush=True)
    print(f'#### O REF ####', flush=True)
    print('################', flush=True)

    ########################
    ######## PARAMS ########
    ########################

    prms_tracker = {
    'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_COV'],
    'win_size_sec' : 30,
    'odor_trig_n_bpm' : 75,
    'jitter' : 0,
    'srate' : srate
    }

    band_prep = 'wb'

    odor = 'o'

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    print(f'compute tracker {odor}', flush=True)

    ######### LOAD #########
    ecg_allsujet = np.array([])
    label_vec_allsujet = np.array([])

    for sujet_i, sujet in enumerate(sujet_list):

        if sujet in ['DF25', 'HJ31']:
            continue

        print_advancement(sujet_i, len(sujet_list), [25, 50, 75])
        ecg = load_ecg_sig(sujet, odor, band_prep)
        ecg_allsujet = np.append(ecg_allsujet, ecg)
        label_vec, trig = get_label_vec(sujet, odor, ecg)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    #### make pipeline
    #SVC().get_params()
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps)

    #### find best model
    params = {
    # 'SVM__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'], 
    # 'SVM__kernel' : ['linear', 'poly', 'rbf'],    
    'SVM__C' : [10e5], 
    'SVM__gamma' : [0.1, 0.01]
    }

    print('train', flush=True)
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=n_core)
    grid.fit(X_train, y_train)
    classifier_score = grid.best_score_
    classifier = grid.best_estimator_
    print('train done', flush=True)

    return classifier




def allsujet_hrv_tracker_ref_o(classifier):

    ########################
    ######## PARAMS ########
    ########################

    prms_tracker = {
    'metric_list' : ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_SD1', 'HRV_SD2', 'HRV_COV'],
    'win_size_sec' : 30,
    'odor_trig_n_bpm' : 75,
    'jitter' : 0,
    'srate' : srate
    }

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    band_prep = 'wb'

    xr_dict = {'sujet' : sujet_list_erp, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list_erp.shape[0], len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    for sujet in sujet_list_erp:

        predictions_dict = {}

        for odor in odor_list:

            predictions_dict[odor] = {}
            for trim_type in ['trim', 'no_trim']:

                predictions_dict[odor][trim_type] = {}
                for data_type in ['real', 'predict', 'odor_trig', 'score']:

                    predictions_dict[odor][trim_type][data_type] = []

        for odor in odor_list:

            print(f'compute tracker {sujet} {odor}', flush=True)

            ######### LOAD #########
            ecg = load_ecg_sig(sujet, odor, band_prep)

            df_hrv, times = get_data_hrv_tracker(ecg, prms_tracker)
            label_vec, trig = get_label_vec(sujet, odor, ecg)
            label_vec = label_vec[(times*srate).astype('int')]

            if debug:

                plt.plot(ecg)
                plt.show()

                plt.plot(label_vec)
                plt.show()

            ######### TEST MODEL #########
            #### get values
            df_res, predictions_time, predictions, trig_odor = hrv_tracker_svm(ecg, classifier, prms_tracker)

            #### resample label_vec_time
            # f = scipy.interpolate.interp1d(label_vec_time, label_vec, kind='linear')
            # label_vec_time_resampled = f(predictions_time)

            #### trim vectors
            #cond_i, cond = 1, conditions[1]
            for cond_i, cond in enumerate(conditions):

                if cond_i == 0:

                    start = trig[cond][0]/srate - trim_edge
                    stop = trig[cond][1]/srate + trim_between

                    mask_start = (start <= predictions_time) & (predictions_time <= stop)

                    predictions_trim = predictions[mask_start] 
                    label_vec_trim = label_vec[mask_start]
                    trig_odor_trim = trig_odor[mask_start] 

                elif cond_i == len(conditions)-1:

                    start = trig[cond][0]/srate - trim_between
                    stop = trig[cond][1]/srate + trim_edge

                    mask_start = (start <= predictions_time) & (predictions_time <= stop) 

                    predictions_trim = np.concatenate((predictions_trim, predictions[mask_start]), axis=0)
                    label_vec_trim = np.concatenate((label_vec_trim, label_vec[mask_start]), axis=0)
                    trig_odor_trim = np.concatenate((trig_odor_trim, trig_odor[mask_start]), axis=0)

                else:

                    start = trig[cond][0]/srate - trim_between
                    stop = trig[cond][1]/srate + trim_between

                    mask_start = (start <= predictions_time) & (predictions_time <= stop)

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

            #### load res
            for trim_type in ['trim', 'no_trim']:

                if trim_type == 'trim':
                    data_load = [label_vec_trim_resampled, predictions_trim_resampled, trig_odor_trim_resampled]
                if trim_type == 'no_trim':
                    data_load = [label_vec, predictions, trig_odor]

                for data_type_i, data_type in enumerate(['real', 'predict', 'odor_trig']):

                    predictions_dict[odor][trim_type][data_type] = data_load[data_type_i]    

        #### load results
        for odor_i, odor in enumerate(odor_list):
            xr_hrv_tracker.loc[sujet, odor, 'prediction', :] = predictions_dict[odor]['trim']['predict']
            xr_hrv_tracker.loc[sujet, odor, 'label', :] = predictions_dict[odor]['trim']['real']
            xr_hrv_tracker.loc[sujet, odor, 'trig_odor', :] = predictions_dict[odor]['trim']['odor_trig']


    #### save results
    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    xr_hrv_tracker.to_netcdf(f'o_ref_allsujettrain_hrv_tracker.nc')







################################
######## COMPILATION ######## 
################################

def hrv_tracker_compilation_allsujet():

    classifier = get_classifier_allsujet_hrv_tracker_o_ref()
    allsujet_hrv_tracker_ref_o(classifier)






################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':

    #hrv_tracker_compilation_allsujet()
    execute_function_in_slurm_bash('n17_precompute_allsujet_hrv_tracker', 'hrv_tracker_compilation_allsujet', [])




