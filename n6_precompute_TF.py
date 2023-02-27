

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import respirationtools
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False









################################
######## STRETCH TF ########
################################


#tf = tf_allchan.copy()
def compute_stretch_tf_dB(sujet, tf, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate):

    #### load baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    baselines = np.load(f'{sujet}_{odor_i}_{band[:-2]}_baselines.npy')

    #### apply baseline
    for n_chan in range(tf.shape[0]):
        
        for fi in range(tf.shape[1]):

            activity = tf[n_chan,fi,:]
            baseline_fi = baselines[n_chan, fi]

            #### verify baseline
            #plt.plot(activity)
            #plt.hlines(baseline_fi, xmin=0 , xmax=activity.shape[0], color='r')
            #plt.show()

            tf[n_chan,fi,:] = 10*np.log10(activity/baseline_fi)

    def stretch_tf_db_n_chan(n_chan):

        tf_mean = np.mean(stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate)[0], axis=0)

        if debug:

            plt.pcolormesh(tf_mean)
            plt.show()

        return tf_mean

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))

    #### extract
    tf_mean_allchan = np.zeros((tf.shape[0], tf.shape[1], stretch_point_TF))

    for n_chan in range(tf.shape[0]):
        tf_mean_allchan[n_chan,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_mean_allchan












################################
######## PRECOMPUTE TF ########
################################


def precompute_tf(sujet, cond):

    print('TF PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)

    #### select prep to load
    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #odor_i = odor_list[0]
        for odor_i in odor_list:

            #### select data without aux chan
            data = load_data_sujet(sujet, band_prep, cond, odor_i)

            #### remove aux chan
            data = data[:-3,:]

            freq_band = freq_band_list_precompute[band_prep] 

            #band, freq = list(freq_band.items())[0]
            for band, freq in freq_band.items():

                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                if os.path.exists(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy'):
                    print('ALREADY COMPUTED')
                    continue
                
                print(f"{band} : {freq}")
                print('COMPUTE')

                #### select wavelet parameters
                wavelets, nfrex = get_wavelets(band_prep, freq)

                #### compute
                os.chdir(path_memmap)
                tf_allchan = np.memmap(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))

                def compute_tf_convolution_nchan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf = np.zeros((nfrex, x.shape[0]))

                    for fi in range(nfrex):
                        
                        tf[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                    tf_allchan[n_chan,:,:] = tf

                joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

                #### stretch or chunk
                print('STRETCH')
                tf_allband_stretched = compute_stretch_tf_dB(sujet, tf_allchan, cond, odor_i, respfeatures_allcond, stretch_point_TF, band, srate)
                
                #### save
                print('SAVE')
                os.chdir(os.path.join(path_precompute, sujet, 'TF'))

                np.save(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}.npy', tf_allband_stretched)
                
                os.chdir(path_memmap)
                os.remove(f'{sujet}_tf_{str(freq[0])}_{str(freq[1])}_{cond}_precompute_convolutions.dat')











################################
######## PRECOMPUTE ITPC ########
################################

def precompute_itpc(sujet, cond):

    print('ITPC PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)

    #### select prep to load
    #band_prep_i, band_prep = 0, 'wb'
    for band_prep_i, band_prep in enumerate(band_prep_list):

        #odor_i = odor_list[0]
        for odor_i in odor_list:

            #### select data without aux chan
            data = load_data_sujet(sujet, band_prep, cond, odor_i)

            #### remove aux chan
            data = data[:-3,:]

            freq_band = freq_band_list_precompute[band_prep] 

            #band, freq = list(freq_band.items())[0]
            for band, freq in freq_band.items():

                os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

                if os.path.exists(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy') :
                    print('ALREADY COMPUTED')
                    continue
                
                print(f"{band} : {freq}")
                print('COMPUTE')

                #### select wavelet parameters
                wavelets, nfrex = get_wavelets(band_prep, freq)

                #### compute
                print('COMPUTE, STRETCH & ITPC')
                #n_chan = 0
                def compute_itpc_n_chan(n_chan):

                    print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                    x = data[n_chan,:]

                    tf = np.zeros((nfrex, x.shape[0]), dtype='complex')

                    for fi in range(nfrex):
                        
                        tf[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

                    #### stretch
                    tf_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf, srate)[0]

                    #### ITPC
                    tf_angle = np.angle(tf_stretch)
                    tf_cangle = np.exp(1j*tf_angle) 
                    itpc = np.abs(np.mean(tf_cangle,0))

                    if debug == True:
                        time = range(stretch_point_TF)
                        frex = range(nfrex)
                        plt.pcolormesh(time,frex,itpc,vmin=np.min(itpc),vmax=np.max(itpc))
                        plt.show()

                    return itpc 

                compute_itpc_n_chan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_itpc_n_chan)(n_chan) for n_chan in range(data.shape[0]))
                
                itpc_allchan = np.zeros((data.shape[0],nfrex,stretch_point_TF))

                for n_chan in range(data.shape[0]):

                    itpc_allchan[n_chan,:,:] = compute_itpc_n_chan_res[n_chan]

                #### save
                print('SAVE')
                os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
                np.save(f'{sujet}_itpc_{str(freq[0])}_{str(freq[1])}_{cond}.npy', itpc_allchan)

                del itpc_allchan














################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### compute and save tf
        #cond = 'MECA'
        for cond in conditions:

            print(cond)
        
            #precompute_tf(sujet, cond)
            execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_tf', [sujet, cond], '30G')
            #precompute_itpc(sujet, cond)
            execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_itpc', [sujet, cond], '30G')



