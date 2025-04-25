

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import xarray as xr
import joblib

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False










################################
######## ALL CONV ########
################################




def get_tf_norm_params(sujet):

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'norm_params'))
    if os.path.exists(f'{sujet}_TF_normalization_params.nc'):
        print('TF NORM PARAMS ALREADY COMPUTED')
        return

    params_data = np.zeros((len(conditions), len(odor_list), len(chan_list_eeg), 4, nfrex))

    for odor_i, odor in enumerate(odor_list):
        for cond_i, cond in enumerate(conditions):

            print(f'{cond} {odor}', flush=True)

            #### Load data
            data = load_data_sujet(sujet, cond, odor)
            data = data[:len(chan_list_eeg), :]  # Keep only EEG channels

            wavelets = get_wavelets()            

            for chan_i in np.arange(len(chan_list_eeg)):

                print_advancement(chan_i, len(chan_list_eeg), steps=[25, 50, 75])

                x = data[chan_i, :]
                tf_conv = np.zeros((nfrex, data.shape[1]), dtype=np.float64)

                for fi in range(nfrex):
                    tf_conv[fi, :] = abs(scipy.signal.fftconvolve(x, wavelets[fi, :], 'same'))**2 

                params_data[cond_i, odor_i, chan_i, 0, :] = np.mean(tf_conv, axis=1)
                params_data[cond_i, odor_i, chan_i, 1, :] = np.std(tf_conv, axis=1)
                params_data[cond_i, odor_i, chan_i, 2, :] = np.median(tf_conv, axis=1)
                params_data[cond_i, odor_i, chan_i, 3, :] = scipy.stats.median_abs_deviation(tf_conv, axis=1)

    #### Save results
    xr_data = np.zeros((4, len(chan_list_eeg), nfrex))
    xr_dict = {'params' : ['mean', 'std', 'median', 'mad'], 'chan_list' : chan_list_eeg, 'nfrex' : np.arange(nfrex)}
    
    xr_data[0] = np.mean(np.mean(params_data[:,:,:,0], axis=0), axis=0)
    xr_data[1] = np.std(np.std(params_data[:,:,:,1], axis=0), axis=0)
    xr_data[2] = np.median(np.median(params_data[:,:,:,2], axis=0), axis=0)
    xr_data[3] = scipy.stats.median_abs_deviation(scipy.stats.median_abs_deviation(params_data[:,:,:,3], axis=0), axis=0)
    
    xr_normal_params = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'norm_params'))
    xr_normal_params.to_netcdf(f'{sujet}_TF_normalization_params.nc')

    print('done', flush=True)








#sujet = sujet_list[0]
def precompute_tf_all_conv(sujet):

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
    if os.path.exists(f'{sujet}_tf_conv_allcond.npy'):
        print(f'{sujet} TF CONV ALREADY COMPUTED')
        return

    #### Load response features
    respfeatures_allcond = load_respfeatures(sujet)

    # normalization = 'rscore'

    tf_conv_allcond = np.zeros((len(conditions), len(odor_list), len(chan_list_eeg), nfrex, stretch_point_TF))

    for odor_i, odor in enumerate(odor_list):
        for cond_i, cond in enumerate(conditions):

            print(f'CONV {cond} {odor}', flush=True)

            #### Load data
            data = load_data_sujet(sujet, cond, odor)
            data = data[:len(chan_list_eeg), :]  # Keep only EEG channels

            if debug:
                time = np.arange(data.shape[-1]) / srate
                plt.plot(time, data[10, :])
                plt.show()

            #### Create wavelets
            wavelets = get_wavelets()            

            # chan_i = 0
            for chan_i in np.arange(len(chan_list_eeg)):

                print_advancement(chan_i, len(chan_list_eeg), steps=[25, 50, 75])

                #### conv
                x = data[chan_i, :]
                tf_conv = np.zeros((nfrex, data.shape[1]), dtype=np.float64)

                for fi in range(nfrex):
                    tf_conv[fi, :] = abs(scipy.signal.fftconvolve(x, wavelets[fi, :], 'same'))**2 

                if debug:
                    time_plot = 30*srate
                    plt.pcolormesh(np.arange(time_plot), frex, tf_conv[:,:time_plot])
                    plt.yscale('log')
                    plt.yticks([2, 8, 10, 30, 50, 100, 150], labels=[2, 8, 10, 30, 50, 100, 150])
                    plt.title(chan_list[chan_i])
                    plt.show()

                #### norm
                print('NORM', flush=True)

                # os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'norm_params'))
                # xr_normal_params = xr.open_dataarray(f'{sujet}_TF_normalization_params.nc')
                    
                # if normalization == 'zscore':
                #     tf_conv_norm = (tf_conv - xr_normal_params.loc['mean', chan_list[chan_i]].values.reshape(-1,1)) / xr_normal_params.loc['std', chan_list[chan_i]].values.reshape(-1,1)
                # if normalization == 'rscore':
                #     tf_conv_norm = (tf_conv - xr_normal_params.loc['median', chan_list[chan_i]].values.reshape(-1,1)) * 0.6745 / xr_normal_params.loc['mad', chan_list[chan_i]].values.reshape(-1,1)

                tf_conv_norm = (tf_conv - np.median(tf_conv, axis=1).reshape(-1,1)) * 0.6745 / scipy.stats.median_abs_deviation(tf_conv, axis=1).reshape(-1,1)

                if debug:

                    time_plot = 30*srate
                    plt.pcolormesh(np.arange(time_plot), frex, tf_conv_norm[:,:time_plot])
                    plt.yscale('log')
                    plt.yticks([2, 8, 10, 30, 50, 100, 150], labels=[2, 8, 10, 30, 50, 100, 150])
                    plt.title(chan_list[chan_i])
                    plt.show()

                #### Stretch
                print('STRETCH', flush=True)
                tf_stretch, _ = stretch_data_tf(respfeatures_allcond[cond][odor], stretch_point_TF, tf_conv_norm, srate)
                cycle_sel = np.random.choice(np.arange(tf_stretch.shape[0]), ncycle_for_TF, replace=False)
                tf_stretch = tf_stretch[cycle_sel] 

                if debug:

                    plt.pcolormesh(np.arange(tf_stretch.shape[-1]), frex, np.median(tf_stretch, axis=0))
                    plt.yscale('log')
                    plt.yticks([2, 8, 10, 30, 50, 100, 150], labels=[2, 8, 10, 30, 50, 100, 150])
                    plt.title(chan_list[chan_i])
                    plt.show()

                tf_conv_allcond[cond_i, odor_i, chan_i] = np.median(tf_stretch, axis=0)

    #### Save results
    print('SAVE', flush=True)
    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
    np.save(f'{sujet}_tf_conv_allcond.npy', tf_conv_allcond)

    print('done', flush=True)


















################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    execute_function_in_slurm_bash('n06_precompute_TF', 'get_tf_norm_params', [[sujet] for sujet in sujet_list], n_core=n_core, mem='15G')
    # sync_folders__push_to_crnldata()

    execute_function_in_slurm_bash('n06_precompute_TF', 'precompute_tf_all_conv', [[sujet] for sujet in sujet_list], n_core=n_core, mem='30G')
    # sync_folders__push_to_crnldata()



        

