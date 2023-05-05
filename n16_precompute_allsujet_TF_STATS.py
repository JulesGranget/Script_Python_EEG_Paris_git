
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False











################################
######## SHUFFLE ########
################################


def get_pixel_extrema_shuffle(tf_stretch_baselines, tf_stretch_cond, tf_percentile_sel_stats):

    #### define ncycle
    n_cycle_baselines = tf_stretch_baselines.shape[0]
    n_cycle_cond = tf_stretch_cond.shape[0]

    #### random selection
    draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
    sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
    sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

    #### extract max min
    tf_shuffle = np.median(np.concatenate((tf_stretch_baselines[sel_baseline, :, :], tf_stretch_cond[sel_cond, :, :])), axis=0)
    if isinstance(tf_percentile_sel_stats, str):
        min, max = tf_shuffle.min(axis=1), tf_shuffle.max(axis=1)
    else:
        min, max = np.percentile(tf_shuffle, int(tf_percentile_sel_stats/2), axis=1), np.percentile(tf_shuffle, int(100 - tf_percentile_sel_stats/2), axis=1)

    del tf_shuffle

    if debug:

        plt.pcolormesh(tf_shuffle)
        plt.colorbar()
        plt.show()

        for cycle_i in range(10):
            plt.pcolormesh(tf_stretch_baselines[cycle_i,:,:])
            plt.show()

        for cycle_i in range(10):
            plt.pcolormesh(tf_stretch_cond[cycle_i,:,:])
            plt.show()

        plt.plot(np.median(np.median(tf_stretch_baselines, axis=0), axis=1))
        plt.plot(np.median(np.median(tf_stretch_cond, axis=0), axis=1))
        plt.show()

    return min, max

    

def get_pixel_extrema_shuffle_wavelet(nchan, n_surrogates_tf, tf_stretch_baselines, tf_stretch_cond):

    wavelet_shuffle = np.zeros((tf_stretch_baselines.shape[2], n_surrogates_tf, tf_stretch_baselines.shape[-1]))

    for surrogate_i in range(n_surrogates_tf):

        #### define ncycle
        n_cycle_baselines = tf_stretch_baselines.shape[1]
        n_cycle_cond = tf_stretch_cond.shape[1]
        n_cycle_tot = n_cycle_baselines + n_cycle_cond

        #### random selection
        sel = np.random.randint(low=0, high=n_cycle_tot, size=n_cycle_cond)
        sel_baseline = np.array([i for i in sel if i <= n_cycle_baselines-1])
        sel_cond = np.array([i for i in sel - n_cycle_baselines if i >= 0])

        #### extract max min
        tf_shuffle = np.median(np.concatenate((tf_stretch_baselines[nchan, sel_baseline, :, :], tf_stretch_cond[nchan, sel_cond, :, :])), axis=0)
    
        wavelet_shuffle[:, surrogate_i,:] = tf_shuffle

    if debug:

        for wavelet_i in range(tf_stretch_baselines.shape[2]):

            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 0, axis=0), linestyle='-', color='b', alpha=0.5)
            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 100, axis=0), linestyle='-', color='b', alpha=0.5)
            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 2.5, axis=0), linestyle='--', color='b', alpha=0.5)
            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 97.5, axis=0), linestyle='--', color='b', alpha=0.5)
            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 5, axis=0), linestyle=':', color='b', alpha=0.5)
            plt.plot(np.percentile(wavelet_shuffle[wavelet_i,:,:], 95, axis=0), linestyle=':', color='b', alpha=0.5)

            plt.plot(np.median(tf_stretch_cond[nchan,:,:,:], axis=0)[wavelet_i,:], color='r')

            plt.title(f'{np.round(wavelet_i/tf_stretch_baselines.shape[2],2)}')

            plt.show()

    return min, max










################################
######## COMPUTE STATS ########
################################



#tf, nchan = data_allcond[cond][odor_i], n_chan
def get_tf_stats(tf, min, max):

    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] > max[wavelet_i], tf_thresh[wavelet_i, :] < min[wavelet_i])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    return tf_thresh


#nchan = 0
def precompute_tf_STATS(nchan, sujet_list_to_compute):

    print(f'#### COMPUTE TF STATS INTRA NCHAN:{nchan} ####', flush=True)

    cond_to_compute = [cond for cond in conditions if cond != 'FR_CV_1']

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        ######## FOR FR_CV BASELINES ########
        cond = 'FR_CV_1'

        print('#### LOAD BASELINE ####', flush=True)

        #sujet_i, sujet = 0, sujet_list_to_compute[0]
        for sujet_i, sujet in enumerate(sujet_list_to_compute):

            print_advancement(sujet_i, len(sujet_list_to_compute), steps=[25, 50, 75])

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            _tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy')[nchan,:,:,:]

            if sujet_i == 0:
                tf_stretch_baselines = _tf_stretch_baselines
            else:     
                tf_stretch_baselines = np.concatenate((tf_stretch_baselines, _tf_stretch_baselines), axis=0)

            del _tf_stretch_baselines

        ######## FOR OTHER COND ########
        
        #cond = 'MECA'
        for cond in cond_to_compute:

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))

            if os.path.exists(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_intra.npy'):
                print(f'{cond}_{odor_i}_intra ALREADY COMPUTED', flush=True)
                continue

            print('#### LOAD COND ####', flush=True)

            #sujet_i, sujet = 0, sujet_list_to_compute[0]
            for sujet_i, sujet in enumerate(sujet_list_to_compute):

                print_advancement(sujet_i, len(sujet_list_to_compute), steps=[25, 50, 75])

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                _tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy')[nchan,:,:,:]

                if sujet_i == 0:
                    tf_stretch_cond = _tf_stretch_cond
                else:     
                    tf_stretch_cond = np.concatenate((tf_stretch_cond, _tf_stretch_cond), axis=0)

                del _tf_stretch_cond

            #### verif tf
            if debug:

                plt.pcolormesh(np.median(tf_stretch_baselines, axis=0))
                plt.show()

                plt.pcolormesh(np.median(tf_stretch_cond, axis=0))
                plt.show()

            print(f'COMPUTE {cond} {odor_i}', flush=True)

            #### define ncycle
            n_cycle_baselines = tf_stretch_baselines.shape[0]
            n_cycle_cond = tf_stretch_cond.shape[0]

            #### space allocation
            _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
            pixel_based_distrib = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
            tf_shuffle = np.zeros((n_cycle_cond, nfrex, stretch_point_TF))

            #surrogates_i = 0
            for surrogates_i in range(n_surrogates_tf):

                # _min, _max =  get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond, tf_percentile_sel_stats)

                print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

                #### random selection
                draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
                sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
                sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

                #### extract max min
                tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[sel_baseline, :, :]
                tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[sel_cond, :, :]

                _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                
                pixel_based_distrib[:, surrogates_i, 0] = _min
                pixel_based_distrib[:, surrogates_i, 1] = _max

            min, max = np.median(pixel_based_distrib[:,:,0], axis=1), np.median(pixel_based_distrib[:,:,1], axis=1) 
            # min, max = np.percentile(pixel_based_distrib[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib[:,:,1], tf_percentile_sel_stats_up, axis=1) 

            #### plot 
            if debug:

                median_max_diff = np.abs(np.median(tf_stretch_cond, axis=0).reshape(-1) - np.median(np.median(tf_stretch_cond, axis=0))).max()
                vmin = -median_max_diff
                vmax = median_max_diff

                tf_plot = np.median(tf_stretch_cond, axis=0)

                time = np.arange(tf_plot.shape[-1])

                plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                plt.yscale('log')
                plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                plt.title(chan_list_eeg[nchan])
                plt.show()


            ######## SAVE ########

            print(f'SAVE', flush=True)

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))

            np.save(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_intra.npy', pixel_based_distrib)

            del tf_stretch_cond

        #### remove baseline
        del tf_stretch_baselines
        








    print(f'#### COMPUTE TF STATS INTER:{nchan} ####', flush=True)

    odor_to_compute = [odor_i for odor_i in odor_list if odor_i != 'o']

    #cond = conditions[0]
    for cond in conditions:

        ######## FOR FR_CV BASELINES ########
        odor_i = 'o'

        #sujet_i, sujet = 0, sujet_list_to_compute[0]
        for sujet_i, sujet in enumerate(sujet_list_to_compute):

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            _tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy')[nchan,:,:,:]

            if sujet_i == 0:
                tf_stretch_baselines = _tf_stretch_baselines
            else:     
                tf_stretch_baselines = np.concatenate((tf_stretch_baselines, _tf_stretch_baselines), axis=0)

            del _tf_stretch_baselines

        ######## FOR OTHER COND ########
        
        #odor_i = '+'
        for odor_i in odor_to_compute:

            if os.path.exists(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_inter.npy'):
                print(f'{cond} {odor_i} ALREADY COMPUTED', flush=True)
                continue

            #sujet_i, sujet = 0, sujet_list_to_compute[0]
            for sujet_i, sujet in enumerate(sujet_list_to_compute):

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                _tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy')[nchan,:,:,:]

                if sujet_i == 0:
                    tf_stretch_cond = _tf_stretch_cond
                else:     
                    tf_stretch_cond = np.concatenate((tf_stretch_cond, _tf_stretch_cond), axis=0)

                del _tf_stretch_cond

            #### verif tf
            if debug:

                plt.pcolormesh(np.median(tf_stretch_baselines, axis=0))
                plt.show()

                plt.pcolormesh(np.median(tf_stretch_cond, axis=0))
                plt.show()

            print(f'COMPUTE {cond} {odor_i}', flush=True)

            pixel_based_distrib = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)

            #### define ncycle
            n_cycle_baselines = tf_stretch_baselines.shape[0]
            n_cycle_cond = tf_stretch_cond.shape[0]

            #### space allocation
            _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
            pixel_based_distrib = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
            tf_shuffle = np.zeros((n_cycle_cond, nfrex, stretch_point_TF))
            
            #surrogates_i = 0
            for surrogates_i in range(n_surrogates_tf):

                # _min, _max =  get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond, tf_percentile_sel_stats)

                print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

                #### random selection
                draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
                sel_baseline = np.random.randint(low=0, high=n_cycle_baselines, size=(draw_indicator == 1).sum())
                sel_cond = np.random.randint(low=0, high=n_cycle_cond, size=(draw_indicator == 0).sum())

                #### extract max min
                tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[sel_baseline, :, :]
                tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[sel_cond, :, :]

                _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                
                pixel_based_distrib[:, surrogates_i, 0] = _min
                pixel_based_distrib[:, surrogates_i, 1] = _max

            min, max = np.median(pixel_based_distrib[:,:,0], axis=1), np.median(pixel_based_distrib[:,:,1], axis=1) 
            # min, max = np.percentile(pixel_based_distrib[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib[:,:,1], tf_percentile_sel_stats_up, axis=1) 
            
            #### plot 
            if debug:

                median_max_diff = np.abs(np.median(tf_stretch_cond, axis=0).reshape(-1) - np.median(np.median(tf_stretch_cond, axis=0))).max()
                vmin = -median_max_diff
                vmax = median_max_diff

                tf_plot = np.median(tf_stretch_cond, axis=0)

                time = np.arange(tf_plot.shape[-1])

                plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                plt.yscale('log')
                plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                plt.title(chan_list_eeg[nchan])
                plt.show()


            ######## SAVE ########

            print(f'SAVE', flush=True)

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))

            np.save(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_inter.npy', pixel_based_distrib)

            del tf_stretch_cond

        #### remove baseline
        del tf_stretch_baselines







########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':
    
    sujet_list_to_compute = sujet_list.tolist()

    #nchan, nchan_name = 0, chan_list_eeg[0]
    for nchan, nchan_name in enumerate(chan_list_eeg):
        
        # precompute_tf_STATS(nchan, sujet_list_to_compute)
        execute_function_in_slurm_bash_mem_choice('n16_precompute_allsujet_TF_STATS', 'precompute_tf_STATS', [nchan, sujet_list_to_compute], '20G')


        







