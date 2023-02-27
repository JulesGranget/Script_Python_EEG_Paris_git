
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n1bis_prep_trigger_info import *

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet, odor_i, band_prep = 'PD01', 'o', 'wb'
def compute_and_save_baseline(sujet, odor_i, band_prep):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    verif_band_compute = []
    for band in list(freq_band_dict[band_prep].keys()):
        if os.path.exists(os.path.join(path_precompute, sujet, 'Baselines', f'{sujet}_{odor_i}_{band}_baselines.npy')):
            verif_band_compute.append(True)

    if np.sum(verif_band_compute) > 0:
        print(f'{sujet}_{odor_i} : BASELINES ALREADY COMPUTED')
        return
            
    #### open raw
    data_allcond = load_data_sujet(sujet, band_prep, 'allcond', odor_i)

    #### Data vizualisation
    if debug == True :
        plt.plot(data_allcond[-1,:])
        plt.show()

    #### remove unused chan
    data_allcond = data_allcond[:-3,:]
    
    #### generate all wavelets to conv
    wavelets_to_conv = {}
        
    #band, freq = 'theta', [2, 10]
    for band, freq in freq_band_dict[band_prep].items():

        #### compute the wavelets
        wavelets_to_conv[band], nfrex = get_wavelets(band_prep, freq)  

    # plot all the wavelets
    if debug == True:
        for band in list(wavelets_to_conv.keys()):
            wavelets2plot = wavelets_to_conv[band]
            plt.pcolormesh(np.arange(wavelets2plot.shape[1]),np.arange(wavelets2plot.shape[0]),np.real(wavelets2plot))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(band)
            plt.show()

    #### get trig values
    session_i = list(odor_order[sujet].keys())[list(odor_order[sujet].values()).index(odor_i)]
    srate = get_params()['srate']

    trig = {}
    for cond in conditions:
        trig_stop = dict_trig_sujet[sujet][session_i][cond]
        trig_start = trig_stop - 300*srate
        trig[cond] = [trig_start, trig_stop]

    #### compute convolutions
    n_band_to_compute = len(list(freq_band_dict[band_prep].keys()))

    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_{odor_i}_baseline_convolutions.dat', dtype=np.float64, mode='w+', shape=(n_band_to_compute, data_allcond.shape[0], nfrex))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, data_allcond.shape[0], steps=[25, 50, 75])

        x = data_allcond[n_chan,:]
        #band_i, band = 0, 'theta'
        for band_i, band in enumerate(list(wavelets_to_conv.keys())):

            baseline_coeff_band = np.array(())
            #fi = 0
            for fi in range(nfrex):
                
                fi_conv = abs(scipy.signal.fftconvolve(x, wavelets_to_conv[band][fi,:], 'same'))**2

                #### chunk data
                fi_conv_chunked = np.concatenate((  fi_conv[trig['FR_CV_1'][0] : trig['FR_CV_1'][1]],
                                                    fi_conv[trig['MECA'][0] : trig['MECA'][1]],
                                                    fi_conv[trig['CO2'][0] : trig['CO2'][1]],
                                                    fi_conv[trig['FR_CV_2'][0] : trig['FR_CV_2'][1]],
                                                    ), axis=0)

                baseline_coeff_band = np.append(baseline_coeff_band, np.median(fi_conv_chunked))
        
            baseline_allchan[band_i, n_chan,:] = baseline_coeff_band

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data_allcond,0)))

    #### save baseline
    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))

    for band_i, band in enumerate(list(freq_band_dict[band_prep].keys())):
    
        np.save(f'{sujet}_{odor_i}_{band}_baselines.npy', baseline_allchan[band_i, :, :])

    #### remove memmap
    os.chdir(path_memmap)
    os.remove(f'{sujet}_{odor_i}_baseline_convolutions.dat')









################################
######## EXECUTE ########
################################


if __name__== '__main__':


    #### params
    sujet = '01PD'
    band_prep = 'wb'
    odor_i = 'o'

    #### compute
    #compute_and_save_baseline(sujet, odor_i, band_prep)
    
    #### slurm execution
    #sujet = sujet_list[0]
    for sujet in sujet_list:
        for odor_i in odor_list:
            for band_prep in band_prep_list:
                execute_function_in_slurm_bash('n3_precompute_baselines', 'compute_and_save_baseline', [sujet, odor_i, band_prep])





