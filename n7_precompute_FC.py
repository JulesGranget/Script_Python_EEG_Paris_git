

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

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
######## MUTUAL INFORMATION ########
################################


def get_MI(sujet, stretch=False):

    #### verif computation
    compute_token = 0

    if stretch:
        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_MI_allpairs_stretch.nc')):
            print(f'ALREADY DONE MI {sujet} STRETCH')
            compute_token = 1

    else:
        if os.path.exists(os.path.join(path_precompute, sujet, 'FC', f'{sujet}_MI_allpairs.nc')):
            print(f'ALREADY DONE MI {sujet}')
            compute_token = 1

    if compute_token == 1:
        return

    #### identify anat info
    chan_list_MI = ['C3', 'Cz', 'C4', 'FC1', 'FC2']
    # chan_list_MI = chan_list_eeg

    pairs_to_compute = []

    for pair_A in chan_list_MI:
        
        for pair_B in chan_list_MI:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    #### compute
    cond_sel = ['FR_CV_1', 'CO2']

    if stretch:
        time_vec = np.arange(stretch_point_TF)
    else:
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

    os.chdir(path_memmap)
    MI_sujet = np.memmap(f'{sujet}_MI.dat', dtype=np.float32, mode='w+', shape=(len(pairs_to_compute), len(cond_sel), len(odor_list), time_vec.size))
        
    #pair_i, pair = 0, pairs_to_compute[0]
    def get_MI_pair(pair_i, pair):

        print_advancement(pair_i, len(pairs_to_compute), [25,50,75])

        #### LOAD DATA
        erp_data = {}

        A, B = pair.split('-')[0], pair.split('-')[1]
        chan_sel = {A : chan_list_eeg.index(A), B : chan_list_eeg.index(B)}

        cond_sel = ['FR_CV_1', 'CO2']

        respfeatures = load_respfeatures(sujet)

        #chan, chan_i = A, chan_list_eeg.index(A)
        for chan, chan_i in chan_sel.items():

            erp_data[chan] = {}

            #cond = 'CO2'
            for cond in cond_sel:

                erp_data[chan][cond] = {}

                #odor = odor_list[0]
                for odor in odor_list:

                    #### load
                    x_raw = load_data_sujet(sujet, cond, odor)[chan_i,:]

                    respfeatures_i = respfeatures[cond][odor]
                    inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                    #### ERP PREPROC
                    x = scipy.signal.detrend(x_raw, type='linear')
                    x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)
                    x = iirfilt(x, srate, lowcut=0.1, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)

                    #### CHUNK
                    if stretch:

                        time_vec = np.arange(stretch_point_TF)

                        x = zscore(x)

                        data_chunk, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_TF, x, srate)

                    else:

                        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

                        data_chunk = np.zeros((inspi_starts.shape[0], int(time_vec.size)))

                        for start_i, start_time in enumerate(inspi_starts):

                            t_start = int(start_time + ERP_time_vec[0]*srate)
                            t_stop = int(start_time + ERP_time_vec[1]*srate)

                            x_chunk = x[t_start: t_stop]

                            data_chunk[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()
                            # data_chunk[start_i, :] = (x_chunk - x_mean) / x_std

                    if debug:

                        for inspi_i, _ in enumerate(inspi_starts):

                            plt.plot(data_chunk[inspi_i, :], alpha=0.3)

                        plt.vlines(data_chunk[0, :].size/2, ymax=data_chunk.max(), ymin=data_chunk.min(), color='k')
                        plt.hlines([-3, 3], xmax=0, xmin=data_chunk[0, :].size, color='k')
                        plt.plot(data_chunk.mean(axis=0), color='r')
                        plt.title(f'{cond} {odor} : {data_chunk.shape[0]}')
                        plt.show()

                    erp_data[chan][cond][odor] = data_chunk
        
        #### initiate res
        
        #cond = 'CO2'
        for cond_i, cond in enumerate(cond_sel):

            #odor = odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                A_data = erp_data[A][cond][odor]
                B_data = erp_data[B][cond][odor]

                for i in range(time_vec.size):

                    MI_sujet[pair_i, cond_i, odor_i, i] = get_MI_2sig(A_data[:,i], B_data[:,i])

        if debug:

            plt.plot(MI_sujet[0,0,0,:])
            plt.show()

            fig, axs = plt.subplots(ncols=len(cond_sel), nrows=len(odor_list))

            for cond_i, cond in enumerate(cond_sel):

                #odor = odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    ax = axs[odor_i, cond_i]
                    ax.plot(MI_sujet[pair_i,cond_i, odor_i, :])
                    ax.set_title(f"{cond} {odor}")
                    ax.set_ylim(MI_sujet[pair_i,:,:,:].min(), MI_sujet[pair_i,:,:,:].max())

            plt.suptitle(sujet)
            plt.show()

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_MI_pair)(pair_i, pair) for pair_i, pair in enumerate(pairs_to_compute))

    #### export
    if stretch:
        MI_dict = {'pair' : pairs_to_compute, 'cond' : cond_sel, 'odor' : odor_list, 'time' : time_vec}
    else:
        MI_dict = {'pair' : pairs_to_compute, 'cond' : cond_sel, 'odor' : odor_list, 'phase' : time_vec}

    xr_MI = xr.DataArray(data=MI_sujet, dims=MI_dict.keys(), coords=MI_dict.values())
    
    os.chdir(os.path.join(path_precompute, sujet, 'FC'))

    if stretch:
        xr_MI.to_netcdf(f'{sujet}_MI_allpairs_stretch.nc')
    else:
        xr_MI.to_netcdf(f'{sujet}_MI_allpairs.nc')



def export_df_MI():

    stretch = False

    #### identify anat info
    chan_list_MI = ['C3', 'Cz', 'C4', 'FC1', 'FC2']
    # chan_list_MI = chan_list_eeg

    pairs_to_compute = []

    for pair_A in chan_list_MI:
        
        for pair_B in chan_list_MI:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')

    #### compute
    cond_sel = ['FR_CV_1', 'CO2']

    if stretch:
        MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_sel, 'odor' : odor_list, 'phase' : time_vec}
        time_vec = np.arange(stretch_point_TF)
    else:
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
        MI_dict = {'sujet' : sujet_list, 'pair' : pairs_to_compute, 'cond' : cond_sel, 'odor' : odor_list, 'time' : time_vec}

    MI_sujet = np.zeros((len(sujet_list), len(pairs_to_compute), len(cond_sel), len(odor_list), time_vec.size))

    for sujet_i, sujet in enumerate(sujet_list):

        os.chdir(os.path.join(path_precompute, sujet, 'FC'))

        if stretch:
            _xr_MI = xr.open_dataarray(f'{sujet}_MI_allpairs_stretch.nc')
        else:
            _xr_MI = xr.open_dataarray(f'{sujet}_MI_allpairs.nc')

        MI_sujet[sujet_i] = _xr_MI.values

    xr_MI = xr.DataArray(data=MI_sujet, dims=MI_dict.keys(), coords=MI_dict.values())

    phase_list = ['inspi', 'expi']
    
    MI_data = np.zeros((len(sujet_list), len(cond_sel), len(odor_list), len(phase_list), len(pairs_to_compute)))

    for cond_i, cond in enumerate(cond_sel):

        for odor_i, odor in enumerate(odor_list):
                    
            for sujet_i, sujet in enumerate(sujet_list):

                for phase_i, phase in enumerate(phase_list):

                    if phase == 'inspi':
                        mask_sel = time_vec[int(time_vec.size/2):]
                    elif phase == 'expi':
                        mask_sel = time_vec[:int(time_vec.size/2)]

                    for pair_i, pair in enumerate(pairs_to_compute):

                        MI_vec = xr_MI.loc[sujet, pair, cond, odor, mask_sel].values

                        MI_data[sujet_i, cond_i, odor_i, phase_i, pair_i] = MI_vec.mean()

    xr_MI = xr.DataArray(data=MI_data, dims=['sujet', 'cond', 'odor', 'phase', 'pair'], coords=[sujet_list, cond_sel, odor_list, phase_list, pairs_to_compute])

    df_MI = xr_MI.to_dataframe(name='MI').reset_index(drop=False)

    df_MI = pd.pivot_table(df_MI, values='MI', columns=['pair'], index=['sujet', 'cond', 'odor', 'phase']).reset_index(drop=False)

    os.chdir(os.path.join(path_precompute, 'allsujet', 'FC'))
    df_MI.to_excel('df_MI_allsujet.xlsx')





################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## MI ########

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)
        get_MI(sujet, stretch=False)
        get_MI(sujet, stretch=True)

    export_df_MI()











    ######## OTHER FC METRICS ########

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print('######## PRECOMPUTE DFC ########') 
        #cond = 'FR_CV_1'
        for cond in conditions:

            # get_wpli_ispc_fc_dfc(sujet, cond, band_prep, band, freq, monopol)
            execute_function_in_slurm_bash_mem_choice('n7_precompute_FC', 'get_wpli_ispc_fc_dfc', [sujet, cond], '35G')



