
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
######## STRETCH TF ########
################################

#tf = tf_conv.copy()
def compute_stretch_tf_dB(sujet, tf, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate):

    #### load baseline
    band = band[:-2]
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    baselines = np.load(f'{sujet}_{odor_i}_{band}_baselines.npy')

    #### apply baseline
    os.chdir(path_memmap)
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            if debug:
                plt.plot(activity)
                plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
                plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        tf_stretch_i = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_stretch_i

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))    

    #### verify cycle number
    n_cycles_stretch_sam = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[0,:,:], srate)[0].shape[0]

    #### extract
    tf_stretch_allchan = np.zeros((tf.shape[0], n_cycles_stretch_sam, tf.shape[1], stretch_point_TF))

    #n_chan = 0
    for n_chan in range(tf.shape[0]):
        tf_stretch_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_stretch_allchan







################################
######## SHUFFLE ########
################################


def get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond):

    #### define ncycle
    n_cycle_baselines = tf_stretch_baselines.shape[1]
    n_cycle_cond = tf_stretch_cond.shape[1]
    n_cycle_tot = n_cycle_baselines + n_cycle_cond

    #### random selection
    sel = np.random.randint(low=0, high=n_cycle_tot, size=n_cycle_cond)
    sel_baseline = np.array([i for i in sel if i <= n_cycle_baselines-1])
    sel_cond = np.array([i for i in sel - n_cycle_baselines if i >= 0])

    #### extract max min
    tf_shuffle = np.concatenate((tf_stretch_baselines[nchan, sel_baseline, :, :], tf_stretch_cond[nchan, sel_cond, :, :]))
    tf_shuffle = np.mean(tf_shuffle, axis=0)
    tf_shuffle = rscore_mat(tf_shuffle)
    max, min = tf_shuffle.max(axis=1), tf_shuffle.min(axis=1)

    if debug:

        plt.pcolormesh(tf_shuffle)
        plt.show()

    return max, min

    






################################
######## COMPUTE STATS ########
################################


def precompute_tf_STATS_intra_session(sujet):

    print('#### COMPUTE TF STATS ####')

    #### identify if already computed for all

    compute_token = 0

    cond_to_compute = [cond for cond in conditions if cond != 'FR_CV_1']

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            if os.path.exists(f'{sujet}_tf_STATS_{conditions[0]}_{odor_list[0]}_intra_{str(freq[0])}_{str(freq[1])}.npy') == False:
                compute_token += 1

    if compute_token == 0:
        print('ALL COND ALREADY COMPUTED')
        return

    #### open params
    respfeatures_allcond = load_respfeatures(sujet)

    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                ######## COMPUTE FOR FR_CV BASELINES ########

                cond = 'FR_CV_1'
                data = load_data_sujet(sujet, band_prep, cond, odor_i)

                #### remove aux chan
                data = data[:-3,:]

                #### convolution
                wavelets, nfrex = get_wavelets(band_prep, freq)

                os.chdir(path_memmap)
                tf_conv = np.memmap(f'{sujet}_{cond}_{odor_i}_intra_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                print(f'CONV baselines {band}')
            
                def compute_tf_convolution_nchan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf_i = np.zeros((nfrex, x.shape[0]))

                    for fi in range(nfrex):
                        
                        tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                    tf_conv[n_chan,:,:] = tf_i

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                #### stretch
                print('STRETCH')
                tf_stretch = compute_stretch_tf_dB(sujet, tf_conv, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate)

                os.chdir(path_memmap)
                try:
                    os.remove(f'{sujet}_{cond}_{odor_i}_intra_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                    del tf_conv
                except:
                    pass

                n_cycles = respfeatures_allcond[cond][odor_i]['select'].sum()

                tf_stretch_baselines = np.memmap(f'{sujet}_baselines_intra_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycles, nfrex, stretch_point_TF))

                tf_stretch_baselines = tf_stretch.copy()

                del tf_stretch

                ######## COMPUTE FOR OTHER COND ########
                
                #cond = 'MECA'
                for cond in cond_to_compute:

                    data = load_data_sujet(sujet, band_prep, cond, odor_i)[:len(chan_list_eeg),:]
                    n_cycle = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, data, srate)[0].shape[0]

                    #### compute stretch for cond
                    data = load_data_sujet(sujet, band_prep, cond, odor_i)[:len(chan_list_eeg),:]

                    #### convolution
                    wavelets, nfrex = get_wavelets(band_prep, freq)

                    os.chdir(path_memmap)
                    tf_conv = np.memmap(f'{sujet}_{cond}_{odor_i}_intra_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
                    
                    print(f'CONV {cond} {odor_i} {band}')

                    def compute_tf_convolution_nchan(n_chan):

                        print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                        x = data[n_chan,:]

                        tf_i = np.zeros((nfrex, x.shape[0]))

                        for fi in range(nfrex):
                            
                            tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                        tf_conv[n_chan,:,:] = tf_i

                    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                    #### stretch
                    tf_stretch_cond = compute_stretch_tf_dB(sujet, tf_conv, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate)

                    os.chdir(path_memmap)
                    try:
                        os.remove(f'{sujet}_{cond}_{odor_i}_intra_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                        del tf_conv
                    except:
                        pass

                    ######## COMPUTE SURROGATES & STATS ########

                    print('SURROGATES')

                    pixel_based_distrib = np.zeros((tf_stretch_baselines.shape[0], 50, 2))

                    #nchan = 0
                    for nchan in range(tf_stretch_baselines.shape[0]):

                        # print_advancement(nchan, tf_stretch_baselines.shape[0], steps=[25, 50, 75])

                        pixel_based_distrib_i = np.zeros((tf_stretch_baselines.shape[2], 2, n_surrogates_tf))

                        #surrogates_i = 0
                        for surrogates_i in range(n_surrogates_tf):

                            pixel_based_distrib_i[:,0,surrogates_i], pixel_based_distrib_i[:,1,surrogates_i] =  get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond)

                        min, max = np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 2.5, axis=-1), np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 97.5, axis=-1) 
                        
                        if debug:
                            plt.hist(pixel_based_distrib_i[0, :, :].reshape(-1), bins=500)
                            plt.show()

                        pixel_based_distrib[nchan, :, 0], pixel_based_distrib[nchan, :, 1] = max, min

                    #### plot 
                    if debug:
                        for nchan in range(20):
                            tf = rscore_mat(tf_stretch_cond[nchan, :, :].mean(axis=0))
                            tf_thresh = tf.copy()
                            #wavelet_i = 0
                            for wavelet_i in range(nfrex):
                                mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
                                tf_thresh[wavelet_i, mask] = 1
                                tf_thresh[wavelet_i, np.logical_not(mask)] = 0
                        
                            plt.pcolormesh(tf)
                            plt.contour(tf_thresh, levels=0)
                            plt.show()

                    ######## SAVE ########
    
                    print(f'SAVE {cond} {band} {odor_i}')

                    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                    np.save(f'{sujet}_tf_STATS_{cond}_{odor_i}_intra_{str(freq[0])}_{str(freq[1])}.npy', pixel_based_distrib)
                    

                #### remove baselines after cond computing
                try:
                    os.remove(f'{sujet}_baselines_intra_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat')
                    del tf_stretch_cond
                except:
                    pass




             



def precompute_tf_STATS_inter_session(sujet):

    print('#### COMPUTE TF STATS ####')

    #### identify if already computed for all

    compute_token = 0

    cond_to_compute = [cond for cond in conditions if cond != 'FR_CV_1']

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            if os.path.exists(f'{sujet}_tf_STATS_{conditions[0]}_{odor_list[0]}_inter_{str(freq[0])}_{str(freq[1])}.npy') == False:
                compute_token += 1

    if compute_token == 0:
        print('ALL COND ALREADY COMPUTED')
        return

    #### open params
    respfeatures_allcond = load_respfeatures(sujet)

    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        freq_band = freq_band_list_precompute[band_prep] 

        #band, freq = list(freq_band.items())[0]
        for band, freq in freq_band.items():

            #cond = conditions[0]
            for cond in conditions:

                ######## COMPUTE FOR o BASELINES ########

                odor_i = 'o'
                data = load_data_sujet(sujet, band_prep, cond, odor_i)

                #### remove aux chan
                data = data[:-3,:]

                #### convolution
                wavelets, nfrex = get_wavelets(band_prep, freq)

                os.chdir(path_memmap)
                tf_conv = np.memmap(f'{sujet}_{cond}_{odor_i}_inter_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                print(f'CONV baselines {band}')
            
                def compute_tf_convolution_nchan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf_i = np.zeros((nfrex, x.shape[0]))

                    for fi in range(nfrex):
                        
                        tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                    tf_conv[n_chan,:,:] = tf_i

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                #### stretch
                print('STRETCH')
                tf_stretch = compute_stretch_tf_dB(sujet, tf_conv, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate)

                os.chdir(path_memmap)
                try:
                    os.remove(f'{sujet}_{cond}_{odor_i}_inter_{band}_{str(freq[0])}_{str(freq[1])}_precompute_convolutions.dat')
                    del tf_conv
                except:
                    pass

                n_cycles = respfeatures_allcond[cond][odor_i]['select'].sum()

                tf_stretch_baselines = np.memmap(f'{sujet}_baselines_inter_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], n_cycles, nfrex, stretch_point_TF))

                tf_stretch_baselines = tf_stretch.copy()

                del tf_stretch

                ######## COMPUTE FOR OTHER ODOR ########
                
                #odor_i = '+'
                for odor_i in ['+', '-']:

                    data = load_data_sujet(sujet, band_prep, cond, odor_i)[:len(chan_list_eeg),:]
                    n_cycle = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, data, srate)[0].shape[0]

                    #### compute stretch for cond
                    data = load_data_sujet(sujet, band_prep, cond, odor_i)[:len(chan_list_eeg),:]

                    #### convolution
                    wavelets, nfrex = get_wavelets(band_prep, freq)

                    os.chdir(path_memmap)
                    tf_conv = np.memmap(f'{sujet}_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_inter_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
                    
                    print(f'CONV {cond} {odor_i} {band}')

                    def compute_tf_convolution_nchan(n_chan):

                        print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                        x = data[n_chan,:]

                        tf_i = np.zeros((nfrex, x.shape[0]))

                        for fi in range(nfrex):
                            
                            tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                        tf_conv[n_chan,:,:] = tf_i

                    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                    #### stretch
                    tf_stretch_cond = compute_stretch_tf_dB(sujet, tf_conv, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate)

                    os.chdir(path_memmap)
                    try:
                        os.remove(f'{sujet}_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_inter_precompute_convolutions.dat')
                        del tf_conv
                    except:
                        pass

                    ######## COMPUTE SURROGATES & STATS ########

                    print('SURROGATES')

                    pixel_based_distrib = np.zeros((tf_stretch_baselines.shape[0], 50, 2))

                    #nchan = 0
                    for nchan in range(tf_stretch_baselines.shape[0]):

                        # print_advancement(nchan, tf_stretch_baselines.shape[0], steps=[25, 50, 75])

                        pixel_based_distrib_i = np.zeros((tf_stretch_baselines.shape[2], 2, n_surrogates_tf))

                        #surrogates_i = 0
                        for surrogates_i in range(n_surrogates_tf):

                            pixel_based_distrib_i[:,0,surrogates_i], pixel_based_distrib_i[:,1,surrogates_i] =  get_pixel_extrema_shuffle(nchan, tf_stretch_baselines, tf_stretch_cond)

                        min, max = np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 2.5, axis=-1), np.percentile(pixel_based_distrib_i.reshape(tf_stretch_baselines.shape[2], -1), 97.5, axis=-1) 
                        
                        if debug:
                            plt.hist(pixel_based_distrib_i[0, :, :].reshape(-1), bins=500)
                            plt.show()

                        pixel_based_distrib[nchan, :, 0], pixel_based_distrib[nchan, :, 1] = max, min

                    #### plot 
                    if debug:
                        for nchan in range(20):
                            tf = rscore_mat(tf_stretch_cond[nchan, :, :].mean(axis=0))
                            tf_thresh = tf.copy()
                            #wavelet_i = 0
                            for wavelet_i in range(nfrex):
                                mask = np.logical_or(tf_thresh[wavelet_i, :] >= pixel_based_distrib[nchan, wavelet_i, 0], tf_thresh[wavelet_i, :] <= pixel_based_distrib[nchan, wavelet_i, 1])
                                tf_thresh[wavelet_i, mask] = 1
                                tf_thresh[wavelet_i, np.logical_not(mask)] = 0
                        
                            plt.pcolormesh(tf)
                            plt.contour(tf_thresh, levels=0)
                            plt.show()

                    ######## SAVE ########
    
                    print(f'SAVE {cond} {band} {odor_i}')

                    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                    np.save(f'{sujet}_tf_STATS_{cond}_{odor_i}_inter_{str(freq[0])}_{str(freq[1])}.npy', pixel_based_distrib)
                    

                #### remove baselines after cond computing
                try:
                    os.remove(f'{sujet}_baselines_inter_{cond}_{odor_i}_{band}_{str(freq[0])}_{str(freq[1])}_stretch.dat')
                    del tf_stretch_cond
                except:
                    pass









########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:    
    
        # precompute_tf_STATS_intra_session(sujet)
        execute_function_in_slurm_bash_mem_choice('n6bis_precompute_TF_STATS', 'precompute_tf_STATS_intra_session', [sujet], '30G')

        # precompute_tf_STATS_inter_session(sujet)
        execute_function_in_slurm_bash_mem_choice('n6bis_precompute_TF_STATS', 'precompute_tf_STATS_inter_session', [sujet], '30G')

        







