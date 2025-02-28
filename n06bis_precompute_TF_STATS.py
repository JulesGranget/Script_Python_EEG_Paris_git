
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import gc
import cv2

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False












################################
######## COMPUTE STATS ########
################################



#tf, nchan = data_allcond[cond][odor_i], n_chan
def get_tf_stats_no_cluster_correction(tf, min, max):

    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] > max[wavelet_i], tf_thresh[wavelet_i, :] < min[wavelet_i])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    return tf_thresh



#tf, nchan = tf_plot, n_chan
def get_tf_stats(tf, pixel_based_distrib):

    #### thresh data
    tf_thresh = tf.copy()
    #wavelet_i = 0
    for wavelet_i in range(tf.shape[0]):
        mask = np.logical_or(tf_thresh[wavelet_i, :] < pixel_based_distrib[wavelet_i, 0], tf_thresh[wavelet_i, :] > pixel_based_distrib[wavelet_i, 1])
        tf_thresh[wavelet_i, mask] = 1
        tf_thresh[wavelet_i, np.logical_not(mask)] = 0

    if debug:

        plt.pcolormesh(tf_thresh)
        plt.show()

    #### if empty return
    if tf_thresh.sum() == 0:

        return tf_thresh

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes, tf_stats_percentile_cluster)  

    if debug:

        plt.hist(sizes, bins=100)
        plt.vlines(np.percentile(sizes,95), ymin=0, ymax=20, colors='r')
        plt.show()

    tf_thresh = np.zeros_like(im_with_separated_blobs)
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            tf_thresh[im_with_separated_blobs == blob + 1] = 1

    if debug:
    
        time = np.arange(tf.shape[-1])

        plt.pcolormesh(time, frex, tf, shading='gouraud', cmap='seismic')
        plt.contour(time, frex, tf_thresh, levels=0, colors='g')
        plt.yscale('log')
        plt.show()

    return tf_thresh










def precompute_tf_STATS(sujet):

    print(f'#### COMPUTE TF STATS INTRA {sujet} ####', flush=True)

    cond_to_compute = [cond for cond in conditions if cond != 'FR_CV_1']

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        
    #odor_i = odor_list[0]
    for odor_i in odor_list:

        ######## FOR FR_CV BASELINES ########
        cond = 'FR_CV_1'
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy', mmap_mode='r')

        ######## FOR OTHER COND ########
        
        #cond = 'CO2'
        for cond in cond_to_compute:

            #### identify if already computed
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_{odor_i}_intra.npy'):
                print(f'{cond} {odor_i} ALREADY COMPUTED', flush=True)
                continue

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy', mmap_mode='r')

            #### verif tf
            if debug:

                nchan = 0
                plt.pcolormesh(np.median(tf_stretch_baselines[nchan,:,:,:], axis=0))
                plt.show()

                plt.pcolormesh(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))
                plt.show()

            print(f'COMPUTE {cond} {odor_i}', flush=True)

            os.chdir(path_memmap)
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_{odor_i}_pixel_distrib.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_eeg), nfrex, 2))
            
            #nchan = 0
            def get_min_max_pixel_based_distrib(nchan):

                print_advancement(nchan, len(chan_list_eeg), steps=[25, 50, 75])

                #### define ncycle
                n_cycle_baselines = tf_stretch_baselines.shape[1]
                n_cycle_cond = tf_stretch_cond.shape[1]

                #### space allocation
                _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
                pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
                tf_shuffle = np.zeros((n_cycle_cond, nfrex, stretch_point_TF))

                #surrogates_i = 0
                for surrogates_i in range(n_surrogates_tf):

                    # print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

                    #### random selection
                    draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
                    sel_baseline = np.random.choice(n_cycle_baselines, size=(draw_indicator == 1).sum(), replace=False)
                    sel_cond = np.random.choice(n_cycle_cond, size=(draw_indicator == 0).sum(), replace=False)

                    #### extract max min
                    tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
                    tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, sel_cond, :, :]

                    _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                    # _min, _max = np.percentile(np.median(tf_shuffle, axis=0), 1, axis=1), np.percentile(np.median(tf_shuffle, axis=0), 99, axis=1)
                    
                    pixel_based_distrib_i[:, surrogates_i, 0] = _min
                    pixel_based_distrib_i[:, surrogates_i, 1] = _max

                min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1) 
                # min, max = np.percentile(pixel_based_distrib_i[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib_i[:,:,1], tf_percentile_sel_stats_up, axis=1)  

                if debug:

                    tf_nchan = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                    time = np.arange(tf_nchan.shape[-1])

                    plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats_no_cluster_correction(tf_nchan, min, max), levels=0, colors='g')
                    plt.yscale('log')
                    plt.show()

                    plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats(tf_nchan, np.concatenate((min.reshape(-1,1), max.reshape(-1,1)), axis=1)), levels=0, colors='g')
                    plt.yscale('log')
                    plt.show()

                    #wavelet_i = 0
                    for wavelet_i in range(20):
                        count, _, _ = plt.hist(tf_nchan[wavelet_i, :], bins=500)
                        plt.vlines([min[wavelet_i], max[wavelet_i]], ymin=0, ymax=count.max(), color='r')
                        plt.show()

                pixel_based_distrib[nchan,:,0] = min
                pixel_based_distrib[nchan,:,1] = max

                del min, max, tf_shuffle
            
            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_min_max_pixel_based_distrib)(nchan) for nchan, _ in enumerate(chan_list_eeg))

            #### plot 
            if debug:

                median_max_diff = np.abs(np.median(tf_stretch_cond, axis=1).reshape(-1) - np.median(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))).max()
                vmin = -median_max_diff
                vmax = median_max_diff

                for nchan, nchan_name in enumerate(chan_list_eeg):

                    tf_plot = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                    time = np.arange(tf_plot.shape[-1])

                    plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                    plt.yscale('log')
                    plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                    plt.title(nchan_name)
                    plt.show()


            ######## SAVE ########

            print(f'SAVE', flush=True)

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            np.save(f'{sujet}_tf_STATS_{cond}_{odor_i}_intra.npy', pixel_based_distrib)

            del tf_stretch_cond
            
            os.chdir(path_memmap)
            try:
                os.remove(f'{sujet}_{cond}_{odor_i}_pixel_distrib.dat')
                del pixel_based_distrib
            except:
                pass

        #### remove baseline
        del tf_stretch_baselines

            





    print(f'#### COMPUTE TF STATS INTER {sujet} ####', flush=True)

    odor_to_compute = [odor_i for odor_i in odor_list if odor_i != 'o']

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        
    #cond = conditions[0]
    for cond in conditions:

        ######## FOR FR_CV BASELINES ########
        odor_i = 'o'
        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
        tf_stretch_baselines = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy', mmap_mode='r')

        ######## FOR OTHER COND ########
        
        #odor_i = '-'
        for odor_i in odor_to_compute:

            #### identify if already computed
            if os.path.exists(f'{sujet}_tf_STATS_{cond}_{odor_i}_inter.npy'):
                print(f'{cond} {odor_i} inter ALREADY COMPUTED', flush=True)
                continue

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            tf_stretch_cond = np.load(f'{sujet}_tf_conv_{cond}_{odor_i}.npy', mmap_mode='r')

            #### verif tf
            if debug:

                nchan = 1
                plt.pcolormesh(np.median(tf_stretch_baselines[nchan,:,:,:], axis=0))
                plt.show()

                plt.pcolormesh(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))
                plt.show()

            print(f'COMPUTE {cond} {odor_i}', flush=True)

            os.chdir(path_memmap)
            pixel_based_distrib = np.memmap(f'{sujet}_{cond}_{odor_i}_pixel_distrib.dat', dtype=np.float32, mode='w+', shape=(len(chan_list_eeg), nfrex, 2))
            
            #nchan = 11
            def get_min_max_pixel_based_distrib(nchan):

                print_advancement(nchan, len(chan_list_eeg), steps=[25, 50, 75])

                #### define ncycle
                n_cycle_baselines = tf_stretch_baselines.shape[1]
                n_cycle_cond = tf_stretch_cond.shape[1]

                #### space allocation
                _min, _max = np.zeros((nfrex)), np.zeros((nfrex))
                pixel_based_distrib_i = np.zeros((nfrex, n_surrogates_tf, 2), dtype=np.float32)
                tf_shuffle = np.zeros((n_cycle_cond, nfrex, stretch_point_TF))

                #surrogates_i = 0
                for surrogates_i in range(n_surrogates_tf):

                    # print_advancement(surrogates_i, n_surrogates_tf, steps=[25, 50, 75])

                    #### random selection
                    draw_indicator = np.random.randint(low=0, high=2, size=n_cycle_cond)
                    sel_baseline = np.random.choice(n_cycle_baselines, size=(draw_indicator == 1).sum(), replace=False)
                    sel_cond = np.random.choice(n_cycle_cond, size=(draw_indicator == 0).sum(), replace=False)

                    #### extract max min
                    tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
                    tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, sel_cond, :, :]

                    _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
                    
                    pixel_based_distrib_i[:, surrogates_i, 0] = _min
                    pixel_based_distrib_i[:, surrogates_i, 1] = _max

                    # gc.collect()

                min, max = np.median(pixel_based_distrib_i[:,:,0], axis=1), np.median(pixel_based_distrib_i[:,:,1], axis=1) 
                # min, max = np.percentile(pixel_based_distrib_i[:,:,0], tf_percentile_sel_stats_dw, axis=1), np.percentile(pixel_based_distrib_i[:,:,1], tf_percentile_sel_stats_up, axis=1)  

                if debug:

                    tf_nchan = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                    time = np.arange(tf_nchan.shape[-1])

                    plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats_no_cluster_correction(tf_nchan, min, max), levels=0, colors='g')
                    plt.yscale('log')
                    plt.show()

                    plt.pcolormesh(time, frex, tf_nchan, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats(tf_nchan, np.concatenate((min.reshape(-1,1), max.reshape(-1,1)), axis=1)), levels=0, colors='g')
                    plt.yscale('log')
                    plt.show()

                    #wavelet_i = 0
                    for wavelet_i in range(20):
                        count, _, _ = plt.hist(tf_nchan[wavelet_i, :], bins=500)
                        plt.vlines([min[wavelet_i], max[wavelet_i]], ymin=0, ymax=count.max(), color='r')
                        plt.show()

                pixel_based_distrib[nchan,:,0] = min
                pixel_based_distrib[nchan,:,1] = max
            
            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_min_max_pixel_based_distrib)(nchan) for nchan, _ in enumerate(chan_list_eeg))

            #### plot 
            if debug:

                median_max_diff = np.abs(np.median(tf_stretch_cond, axis=1).reshape(-1) - np.median(np.median(tf_stretch_cond[nchan,:,:,:], axis=0))).max()
                vmin = -median_max_diff
                vmax = median_max_diff

                for nchan, nchan_name in enumerate(chan_list_eeg):

                    tf_plot = np.median(tf_stretch_cond[nchan,:,:,:], axis=0)

                    time = np.arange(tf_plot.shape[-1])

                    plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap='seismic')
                    plt.contour(time, frex, get_tf_stats(tf_plot, min, max), levels=0, colors='g', vmin=vmin, vmax=vmax)
                    plt.yscale('log')
                    plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                    plt.title(nchan_name)
                    plt.show()


            ######## SAVE ########

            print(f'SAVE', flush=True)

            os.chdir(os.path.join(path_precompute, sujet, 'TF'))

            np.save(f'{sujet}_tf_STATS_{cond}_{odor_i}_inter.npy', pixel_based_distrib)

            del tf_stretch_cond

            os.chdir(path_memmap)
            try:
                os.remove(f'{sujet}_{cond}_{odor_i}_pixel_distrib.dat')
                del pixel_based_distrib
            except:
                pass
        
        #### remove baseline
        del tf_stretch_baselines
        








########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    
    
        # precompute_tf_STATS(sujet)
        execute_function_in_slurm_bash('n6bis_precompute_TF_STATS', 'precompute_tf_STATS', [sujet], '30G')

        







