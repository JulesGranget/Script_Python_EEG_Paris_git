



import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

import frites

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False





########################################
######## PLI ISPC DFC FC ######## 
########################################


def get_pli_ispc_fc_dfc_trial(sujet, cond, odor_i, band_prep, band, freq):

    #### load data
    data = load_data_sujet(sujet, cond, odor_i)
    data = data[:len(chan_list_eeg),:]
    
    data_length = data.shape[-1]

    #### get params
    prms = get_params()

    wavelets = get_wavelets_fc(band_prep, freq)

    respfeatures_allcond = load_respfeatures(sujet)

    #### initiate res
    os.chdir(path_memmap)
    convolutions = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_convolutions.dat', dtype=np.complex128, mode='w+', shape=(len(chan_list_eeg), nfrex_fc, data_length))

    #### generate fake convolutions
    # convolutions = np.random.random(len(prms['chan_list_ieeg']) * nfrex_fc * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1]) * 1j
    # convolutions += np.random.random(len(prms['chan_list_ieeg']) * nfrex_fc * data.shape[1]).reshape(len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1]) 

    # convolutions = np.zeros((len(prms['chan_list_ieeg']), nfrex_fc, data.shape[1])) 

    print('CONV')

    #nchan = 0
    def convolution_x_wavelets_nchan(nchan_i, nchan):

        print_advancement(nchan_i, len(chan_list_eeg), steps=[25, 50, 75])
        
        nchan_conv = np.zeros((nfrex_fc, np.size(data,1)), dtype='complex')

        x = data[nchan_i,:]

        for fi in range(nfrex_fc):

            nchan_conv[fi,:] = scipy.signal.fftconvolve(x, wavelets[fi,:], 'same')

        convolutions[nchan_i,:,:] = nchan_conv

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(convolution_x_wavelets_nchan)(nchan_i, nchan) for nchan_i, nchan in enumerate(chan_list_eeg))

    #### free memory
    del data        

    #### verif conv
    if debug:
        plt.plot(convolutions[0,0,:])
        plt.show()

    #### compute index
    pairs_to_compute = []

    for pair_A in chan_list_eeg:

        for pair_B in chan_list_eeg:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    ######## FC / DFC ########

    os.chdir(path_memmap)
    res_fc_phase = np.memmap(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_phase.dat', dtype=np.float32, mode='w+', shape=(2, len(pairs_to_compute), len(phase_list), nfrex_fc))

    #pair_to_compute_i, pair_to_compute = 0, pairs_to_compute[0]
    def compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute):

        print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[25, 50, 75])

        pair_A, pair_B = pair_to_compute.split('-')[0], pair_to_compute.split('-')[-1]
        pair_A_i, pair_B_i = chan_list_eeg.index(pair_A), chan_list_eeg.index(pair_B)

        as1 = convolutions[pair_A_i,:,:]
        as2 = convolutions[pair_B_i,:,:]

        #### stretch data
        as1_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, as1, prms['srate'])[0]
        as2_stretch = stretch_data_tf(respfeatures_allcond[cond][odor_i], stretch_point_TF, as2, prms['srate'])[0]

        #phase = 'whole'
        for phase_i, phase in enumerate(phase_list):

            #### chunk
            if phase == 'whole':
                as1_stretch_chunk =  np.transpose(as1_stretch, (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch, (1, 0, 2)).reshape((nfrex_fc, -1))

            if phase == 'inspi':
                as1_stretch_chunk =  np.transpose(as1_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)], (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch[:,:,:int(stretch_point_TF*ratio_stretch_TF)], (1, 0, 2)).reshape((nfrex_fc, -1))

            if phase == 'expi':
                as1_stretch_chunk =  np.transpose(as1_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):], (1, 0, 2)).reshape((nfrex_fc, -1))
                as2_stretch_chunk =  np.transpose(as2_stretch[:,:,int(stretch_point_TF*ratio_stretch_TF):], (1, 0, 2)).reshape((nfrex_fc, -1))

            ##### collect "eulerized" phase angle differences
            cdd = np.exp(1j*(np.angle(as1_stretch_chunk)-np.angle(as2_stretch_chunk)))
            
            ##### compute ISPC and WPLI (and average over trials!)
            res_fc_phase[0, pair_to_compute_i, phase_i, :] = np.abs(np.mean(cdd, axis=1))
            # pli_dfc_i[slwin_values_i] = np.abs(np.mean(np.sign(np.imag(cdd))))
            res_fc_phase[1, pair_to_compute_i, phase_i, :] = np.abs( np.mean( np.imag(cdd), axis=1 ) ) / np.mean( np.abs( np.imag(cdd) ), axis=1 )

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_ispc_wpli_dfc)(pair_to_compute_i, pair_to_compute) for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute))

    if debug:
        for pair_to_compute_i, pair_to_compute in enumerate(pairs_to_compute):
            print_advancement(pair_to_compute_i, len(pairs_to_compute), steps=[10, 20, 50, 75])
            compute_ispc_wpli_dfc(pair_to_compute_i, pair_to_compute)

    res_fc_phase_export = res_fc_phase.copy()

    #### remove memmap
    os.chdir(path_memmap)
    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_phase.dat')
        del convolutions
    except:
        pass

    try:
        os.remove(f'{sujet}_{cond}_{band_prep}_{band}_{odor_i}_fc_convolutions.dat')
        del res_fc_phase
    except:
        pass

    return res_fc_phase_export








def get_wpli_ispc_fc_dfc(sujet, cond):

    #band_prep = 'wb'
    for band_prep in band_prep_list:
        #band, freq = 'theta', [4,8]
        for band, freq in freq_band_dict_FC[band_prep].items():

            #### verif computation
            if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_FC_wpli_ispc_{cond}_o_{band}_allpairs.nc')):
                print(f'ALREADY DONE FC {cond} {band}')
                return

            #### identify anat info
            pairs_to_compute = []

            for pair_A in chan_list_eeg:
                
                for pair_B in chan_list_eeg:

                    if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                        continue

                    pairs_to_compute.append(f'{pair_A}-{pair_B}')

            #odor_i = odor_list[0]
            for odor_i in odor_list:

                #### for dfc computation
                mat_fc = get_pli_ispc_fc_dfc_trial(sujet, cond, odor_i, band_prep, band, freq)

                if debug:
                    plt.plot(mat_fc[1,0,0,:], label='whole')
                    plt.plot(mat_fc[1,0,1,:], label='inspi')
                    plt.plot(mat_fc[1,0,2,:], label='expi')
                    plt.legend()
                    plt.show()

                #### export
                os.chdir(os.path.join(path_precompute, sujet, 'FC'))
                dict_xr = {'mat_type' : ['ispc', 'wpli'], 'pairs' : pairs_to_compute, 'phase' : ['whole', 'inspi', 'expi'], 'nfrex' : range(nfrex_fc)}
                xr_export = xr.DataArray(mat_fc, coords=dict_xr.values(), dims=dict_xr.keys())
                xr_export.to_netcdf(f'{sujet}_FC_wpli_ispc_{cond}_{odor_i}_{band}_allpairs.nc')








################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print('######## PRECOMPUTE DFC ########') 
        #cond = 'FR_CV_1'
        for cond in conditions:

            # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, monopol)
            execute_function_in_slurm_bash_mem_choice('n7_precompute_FC', 'get_wpli_ispc_fc_dfc', [sujet, cond], '35G')



