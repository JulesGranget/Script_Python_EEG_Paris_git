
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd
import joblib
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n1bis_prep_info import *

debug = False




########################################
######## COMPUTE BASELINE ######## 
########################################

#sujet, odor_i, band_prep = 'PD01', 'o', 'wb'
def compute_and_save_baseline(sujet, odor_i):

    print('#### COMPUTE BASELINES ####')

    #### verify if already computed
    if os.path.exists(os.path.join(path_precompute, sujet, 'baselines', f'{sujet}_{odor_i}_baselines.nc')):
        print('ALREADY COMPUTED')
        return
            
    #### open raw
    data_allcond = load_data_sujet(sujet, 'allcond', odor_i)
    data_allcond = data_allcond[:len(chan_list_eeg),:]

    #### Data vizualisation
    if debug:
        plt.plot(data_allcond[0,:])
        plt.show()
    
    #### generate all wavelets to conv
    wavelets = get_wavelets()  

    if debug:
        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets[0,:]))
        plt.plot(np.real(wavelets[-1,:]))
        plt.show()

    #### get trig values
    session_i = list(odor_order[sujet].keys())[list(odor_order[sujet].values()).index(odor_i)]

    trig = {}
    for cond in conditions:
        trig_stop = dict_trig_sujet[sujet][session_i][cond]
        trig_start = trig_stop - 300*srate
        trig[cond] = [trig_start, trig_stop]

    #### compute convolutions
    os.chdir(path_memmap)
    baseline_allchan = np.memmap(f'{sujet}_{odor_i}_baseline_convolutions.dat', dtype=np.float64, mode='w+', shape=(data_allcond.shape[0], nfrex, 4))

        #### compute
    #n_chan = 0
    def baseline_convolutions(n_chan):

        print_advancement(n_chan, data_allcond.shape[0], steps=[25, 50, 75])

        x = data_allcond[n_chan,:]

        baseline_coeff = np.zeros((frex.shape[0], 4))

        #fi = 0
        for fi in range(nfrex):
            
            fi_conv = abs(scipy.signal.fftconvolve(x, wavelets[fi,:], 'same'))**2

            #### chunk data
            fi_conv_chunked = np.concatenate((  fi_conv[trig['FR_CV_1'][0] : trig['FR_CV_1'][1]],
                                                fi_conv[trig['MECA'][0] : trig['MECA'][1]],
                                                fi_conv[trig['CO2'][0] : trig['CO2'][1]],
                                                fi_conv[trig['FR_CV_2'][0] : trig['FR_CV_2'][1]],
                                                ), axis=0)

            baseline_coeff[fi,0] = np.mean(fi_conv_chunked)
            baseline_coeff[fi,1] = np.std(fi_conv_chunked)
            baseline_coeff[fi,2] = np.median(fi_conv_chunked)
            baseline_coeff[fi,3] = np.median(np.abs(x-np.median(fi_conv_chunked)))

        if debug:

            fig, axs = plt.subplots(ncols=2)
            axs[0].set_title('mean std')
            axs[0].plot(baseline_coeff[:,0], label='mean')
            axs[0].plot(baseline_coeff[:,1], label='std')
            axs[0].legend()
            axs[0].set_yscale('log')
            axs[1].set_title('median mad')
            axs[1].plot(baseline_coeff[:,2], label='median')
            axs[1].plot(baseline_coeff[:,3], label='mad')
            axs[1].legend()
            axs[1].set_yscale('log')
            plt.show()
    
        baseline_allchan[n_chan,:,:] = baseline_coeff

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(baseline_convolutions)(n_chan) for n_chan in range(np.size(data_allcond,0)))

    #### save baseline
    xr_dict = {'chan' : chan_list_eeg, 'frex' : range(frex.shape[0]), 'metrics' : ['mean', 'std', 'median', 'mad']}
    xr_baseline = xr.DataArray(baseline_allchan, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, sujet, 'baselines'))
    xr_baseline.to_netcdf(f'{sujet}_{odor_i}_baselines.nc')

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
            execute_function_in_slurm_bash('n3_precompute_baselines', 'compute_and_save_baseline', [sujet, odor_i])





