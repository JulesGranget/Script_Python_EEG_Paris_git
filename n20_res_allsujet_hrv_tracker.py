

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


def allsujet_hrv_tracker_mean():

    print('################')
    print(f'#### NO REF ####')
    print('################')

    ################################################
    ######## COMPUTE MODEL ONE SESSION ########
    ################################################

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    #### load results
    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
        xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'no_ref_{sujet}_hrv_tracker.nc').values, 0)

    xr_hrv_tracker_mean = xr_hrv_tracker.mean(axis=0)
    xr_hrv_tracker_std = xr_hrv_tracker.std(axis=0)

    #### plot
    fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
    for odor_i, odor in enumerate(odor_list):
        # std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
        # std_to_plot_down[std_to_plot_down < 0] = 0
        ax = axs[odor_i]
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
        # ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        # ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.set_title(f"odor : {odor}")
    plt.suptitle(f'TRIMMED')
    plt.legend()
    # fig_trim.show()
    plt.close()

    ######## SAVE ########
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    fig_trim.savefig(f'allsujet_no_ref_hrv_tracker_whole.png')







    ################################################
    ######## COMPUTE MODEL WITH REF ########
    ################################################


    odor_ref = 'o'
    odor_list_test = [odor_i for odor_i in odor_list if odor_i != odor_ref]

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    band_prep = 'wb'

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list_test), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list_test), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list, 'odor' : np.array(odor_list_test)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(odor_list_test))), dims=xr_dict.keys(), coords=xr_dict.values())

    #### load results
    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
        xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'o_ref_{sujet}_hrv_tracker.nc').values, 0)

    xr_hrv_tracker_mean = xr_hrv_tracker.mean(axis=0)

    fig_trim, axs = plt.subplots(ncols=len(odor_list_test), figsize=(18, 9))
    for odor_i, odor in enumerate(odor_list_test):
        ax = axs[odor_i]
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
        ax.set_title(f"odor : {odor}")
    plt.suptitle(f'TRIMMED')
    plt.legend()
    # fig_trim.show()
    plt.close()

    ######## SAVE ########
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    fig_trim.savefig(f'allsujet_o_ref_hrv_tracker_whole.png')

        







########################################################
######## COMPUTE ALLSUJET ONE CLASSIFIER ########
########################################################


def allsujet_hrv_tracker_one_classifier():

    print('################')
    print(f'#### NO REF ####')
    print('################')

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    xr_allsujet = xr.load_dataarray(f'no_ref_allsujettrain_hrv_tracker.nc')
    data_allsujet = xr_allsujet.values

    xr_hrv_tracker_mean = xr_allsujet.mean('sujet')
    xr_hrv_tracker_std = xr_allsujet.std('sujet')

    #### plot
    fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
    for odor_i, odor in enumerate(odor_list):
        std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
        std_to_plot_down[std_to_plot_down < 0] = 0
        ax = axs[odor_i]
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.set_title(f"odor : {odor}")
    plt.suptitle(f'TRIMMED')
    plt.legend()
    # fig_trim.show()
    plt.close()

    ######## SAVE ########
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    fig_trim.savefig(f'one_classifier_allsujet_no_ref_hrv_tracker.png')







########################################################
######## ANALYSIS PERFORMANCE CLASSIFIER ########
########################################################


def analysis_pref():

    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    train_percentage_values = [0.5, 0.6, 0.7, 0.8]

    xr_dict = {'sujet' : sujet_list, 'train_percentage' : train_percentage_values, 'odor' : np.array(odor_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(train_percentage_values), len(odor_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list, 'train_percentage' : train_percentage_values, 'odor' : np.array(odor_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((sujet_list.shape[0], len(train_percentage_values), len(odor_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    #### load results
    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
        xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'no_ref_{sujet}_hrv_tracker_alltestsize.nc').values, 0)

        xr_hrv_tracker_score.loc[sujet,:] = np.squeeze(xr.load_dataarray(f'no_ref_{sujet}_hrv_tracker_score_alltestsize.nc').values, 0)

    #### plot & save
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

    xr_hrv_tracker_score.rename('value')
    df_score = xr_hrv_tracker_score.to_dataframe(name='value').reset_index()
    df_score = df_score.query(f"value != 0")

    sns.pointplot(data=df_score, x='train_percentage', y='value', hue='odor')
    plt.ylim(0.8, 1)
    # plt.show()

    plt.savefig('perf_no_ref_detection_allsize.png')

    sns.pointplot(data=df_score.query(f"odor == 'o'"), x='train_percentage', y='value')
    plt.ylim(0.8, 1)

    plt.savefig('perf_no_ref_detection_allsize_only_o.png')
    

########################################
######## RES FOR LABEL NUMBER ########
########################################

def res_test_labels_number():

    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    features_test_list = ['MeanNN_SDNN_RMSSD', 'pNN50', 'SD1_SD2', 'COV', 'pNN50_SD1_SD2_COV', 'MeanNN_SDNN_RMSSD_pNN50_SD1_SD2_COV']

    trim_edge = 30 #sec  
    trim_between = 180 #sec
    n_pnts_trim_resample = 10000

    sujet_list = [sujet for sujet in sujet_list if sujet not in ['25DF', '31HJ', '21ZV', '22DI']]

    xr_dict = {'sujet' : sujet_list, 'features' : np.array(features_test_list), 'type' : ['prediction', 'label', 'trig_odor'], 'times' : np.arange(n_pnts_trim_resample)}
    xr_hrv_tracker = xr.DataArray(data=np.zeros((len(sujet_list), len(features_test_list), 3, n_pnts_trim_resample)), dims=xr_dict.keys(), coords=xr_dict.values())

    xr_dict = {'sujet' : sujet_list, 'features' : np.array(features_test_list)}
    xr_hrv_tracker_score = xr.DataArray(data=np.zeros((len(sujet_list), len(features_test_list))), dims=xr_dict.keys(), coords=xr_dict.values())

    #### load results
    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
        xr_hrv_tracker.loc[sujet,:,:,:] = np.squeeze(xr.load_dataarray(f'{sujet}_hrv_tracker_test_features.nc').values, 0)

        xr_hrv_tracker_score.loc[sujet,:] = np.squeeze(xr.load_dataarray(f'{sujet}_hrv_tracker_test_features_score.nc').values, 0)

    xr_hrv_tracker_mean = xr_hrv_tracker.mean('sujet')
    xr_hrv_tracker_std = xr_hrv_tracker.std('sujet')
    xr_hrv_tracker_mean_score = xr_hrv_tracker_score.mean('sujet')

    #### plot & save
    for features_i, features in enumerate(features_test_list):
        fig_trim, ax = plt.subplots(figsize=(18, 9))
        std_to_plot_down = xr_hrv_tracker_mean.loc[features, 'trig_odor', :].values - xr_hrv_tracker_std.loc[features, 'prediction', :].values
        std_to_plot_down[std_to_plot_down < 0] = 0
        ax.plot(xr_hrv_tracker_mean.loc[features, 'label', :].values, color='k', label='real', linewidth=3)
        ax.plot(xr_hrv_tracker_mean.loc[features, 'prediction', :].values, color='y', label='prediction')
        ax.plot(xr_hrv_tracker_mean.loc[features, 'trig_odor', :].values, color='r', label='odor_trig')
        ax.plot(xr_hrv_tracker_mean.loc[features, 'trig_odor', :].values + xr_hrv_tracker_std.loc[features, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.set_title(f"{features}")
        plt.suptitle(f'TRIMMED, score : {np.round(xr_hrv_tracker_mean_score.loc[features].values, 3)}')
        plt.legend()
        # fig_trim.show()
        plt.close()

        ######## SAVE ########
        os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
        fig_trim.savefig(f'test_features_{features}.png')




########################################################
######## RES FOR CLASSIFIER ALLSUJET ########
########################################################


def res_one_classifier_allsujet():

    os.chdir(os.path.join(path_precompute, 'allsujet', 'HRV'))

    xr_data = xr.open_dataarray('o_ref_allsujettrain_hrv_tracker.nc')
    xr_score = xr.open_dataarray('o_ref_allsujettrain_hrv_tracker.nc')
    xr_hrv_tracker_mean = xr_data.mean('sujet')
    xr_hrv_tracker_std = xr_data.std('sujet')

    #### plot & save
    fig_trim, axs = plt.subplots(ncols=len(odor_list), figsize=(18, 9))
    for odor_i, odor in enumerate(odor_list):
        std_to_plot_down = xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values - xr_hrv_tracker_std.loc[odor, 'prediction', :].values
        std_to_plot_down[std_to_plot_down < 0] = 0
        ax = axs[odor_i]
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'label', :].values, color='k', label='real', linewidth=3)
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'prediction', :].values, color='y', label='prediction')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values, color='r', label='odor_trig')
        ax.plot(xr_hrv_tracker_mean.loc[odor, 'trig_odor', :].values + xr_hrv_tracker_std.loc[odor, 'prediction', :].values, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.plot(std_to_plot_down, color='r', label='odor_trig', linestyle=':', linewidth=0.5)
        ax.set_title(f"odor : {odor}")
    plt.suptitle(f'TRIMMED')
    plt.legend()
    # fig_trim.show()
    plt.close()

    ######## SAVE ########
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    fig_trim.savefig(f'one_classifier_allsujet_o_ref_hrv_tracker.png')
    


################################
######## SVM PARAMS ########
################################


def df_allsujet_SVM_params():

    for sujet in sujet_list:

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

        if sujet == sujet_list[0]:
            df_allsujet_params = pd.read_excel(f'{sujet}_hrv_tracker_SVM_test.xlsx')
        else:
            df_sujet = pd.read_excel(f'{sujet}_hrv_tracker_SVM_test.xlsx')
            df_allsujet_params = pd.concat((df_allsujet_params, df_sujet))

    df_allsujet_params = df_allsujet_params.drop(columns=['Unnamed: 0'])
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    df_allsujet_params.to_excel('allsujet_hrv_tracker_SVM_params.xlsx')





################################
######## EXECUTE ######## 
################################

if __name__ == '__main__':



    ########################################
    ######## EXECUTE CLUSTER ########
    ########################################

    allsujet_hrv_tracker_mean()
    allsujet_hrv_tracker_one_classifier()
    df_allsujet_SVM_params()
    analysis_pref()
    res_one_classifier_allsujet()
    res_test_labels_number()


