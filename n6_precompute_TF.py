

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import xarray as xr
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False









################################
######## STRETCH ########
################################



#tf = tf_conv
def compute_stretch_tf(tf, cond, odor_i, respfeatures_allcond, stretch_point_TF, srate):

    #n_chan = 0
    def stretch_tf_db_n_chan(n_chan):

        tf_stretch_i = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[n_chan,:,:], srate)[0]

        return tf_stretch_i

    stretch_tf_db_nchan_res = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(stretch_tf_db_n_chan)(n_chan) for n_chan in range(tf.shape[0]))    

    #### verify cycle number
    n_cycles_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, tf[0,:,:], srate)[0].shape[0]

    #### extract
    tf_stretch_allchan = np.zeros((tf.shape[0], n_cycles_stretch, tf.shape[1], stretch_point_TF))

    #n_chan = 0
    for n_chan in range(tf.shape[0]):
        tf_stretch_allchan[n_chan,:,:,:] = stretch_tf_db_nchan_res[n_chan]

    return tf_stretch_allchan








################################
######## ALL CONV ########
################################

def precompute_tf_all_conv(sujet):

    #### identify if already computed for all
    compute_token = 0

    cond_to_compute = [cond for cond in conditions if cond != 'FR_CV_1']

    os.chdir(os.path.join(path_precompute, sujet, 'TF'))

    for odor_i in odor_list:

        #cond = cond_to_compute[0]
        for cond in cond_to_compute:

            if os.path.exists(f'{sujet}_tf_conv_{cond}_{odor_i}.npy') == False:
                compute_token += 1

    if compute_token == 0:
        print('ALL COND ALREADY COMPUTED')
        return

    #### open params
    respfeatures_allcond = load_respfeatures(sujet)

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        #cond = conditions[2]
        for cond in conditions:

            print(f'#### CONV {cond} {odor_i} ####')

            #### load
            data = load_data_sujet(sujet, cond, odor_i)
            data = data[:len(chan_list_eeg),:]

            if debug:
                
                time = np.arange(data.shape[-1])/srate
                plt.plot(time, data[10,:])
                plt.show()

            #### convolution
            wavelets = get_wavelets()

            os.chdir(path_memmap)
            tf_conv = np.memmap(f'{sujet}_{cond}_{odor_i}_precompute_convolutions.dat', dtype=np.float64, mode='w+', shape=(data.shape[0], nfrex, data.shape[1]))
        
            def compute_tf_convolution_nchan(n_chan):

                print_advancement(n_chan, data.shape[0], steps=[25, 50, 75])

                x = data[n_chan,:]

                tf_i = np.zeros((nfrex, x.shape[0]))

                for fi in range(nfrex):
                    
                    tf_i[fi,:] = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2 

                tf_conv[n_chan,:,:] = tf_i

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_tf_convolution_nchan)(n_chan) for n_chan in range(data.shape[0]))

            if debug:
                plt.pcolormesh(tf_conv[0,:,:int(tf_conv.shape[-1]/4)])
                plt.show()

            #### normalize
            print('NORMALIZE')
            tf_conv = norm_tf(sujet, tf_conv, odor_i, norm_method)

            if debug:
                tf_plot = tf_conv[0,:,:int(tf_conv.shape[-1]/5)]
                vmin = np.percentile(tf_plot.reshape(-1), 2.5)
                vmax = np.percentile(tf_plot.reshape(-1), 97.5)
                plt.pcolormesh(tf_conv[0,:,:int(tf_conv.shape[-1]/5)], vmin=vmin, vmax=vmax)
                plt.show()

            #### stretch
            print('STRETCH')
            tf_stretch = compute_stretch_tf(tf_conv, cond, odor_i, respfeatures_allcond, stretch_point_TF, srate)

            if debug:

                for nchan, nchan_name in enumerate(chan_list_eeg):
                    plt.pcolormesh(np.arange(tf_stretch.shape[-1]), frex, np.median(tf_stretch[nchan,:,:,:], axis=0))
                    plt.yscale('log')
                    plt.yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
                    plt.title(nchan_name)
                    plt.show()

            os.chdir(path_memmap)
            try:
                os.remove(f'{sujet}_{cond}_{odor_i}_precompute_convolutions.dat')
                del tf_conv
            except:
                pass

            #### save & transert
            print('SAVE')
            os.chdir(os.path.join(path_precompute, sujet, 'TF'))
            np.save(f'{sujet}_tf_conv_{cond}_{odor_i}.npy', tf_stretch)













################################
######## PRECOMPUTE ITPC ########
################################

def precompute_itpc(sujet, cond):

    print('ITPC PRECOMPUTE')

    respfeatures_allcond = load_respfeatures(sujet)

    #### select prep to load
    #odor_i = odor_list[0]
    for odor_i in odor_list:

        #### select data without aux chan
        data = load_data_sujet(sujet, cond, odor_i)

        #### remove aux chan
        data = data[:len(chan_list_eeg),:]

        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))

        if os.path.exists(f'{sujet}_itpc_{cond}_{odor_i}.npy') :
            print('ALREADY COMPUTED')
            continue

        #### select wavelet parameters
        wavelets = get_wavelets()

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

        if debug:
            plt.pcolormesh(itpc_allchan[0,:,:])
            plt.show()

        #### save
        print('SAVE')
        os.chdir(os.path.join(path_precompute, sujet, 'ITPC'))
        np.save(f'{sujet}_itpc_{cond}_{odor_i}.npy', itpc_allchan)

        del itpc_allchan














################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #sujet = sujet_list[19]
    for sujet in sujet_list:
    
        #precompute_tf_all_conv(sujet)
        execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_tf_all_conv', [sujet], '30G')
        #precompute_itpc(sujet, cond)
        # execute_function_in_slurm_bash_mem_choice('n6_precompute_TF', 'precompute_itpc', [sujet], '30G')



