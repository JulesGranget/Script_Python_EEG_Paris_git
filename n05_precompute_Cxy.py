
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd


from n00_config_params import *
from n00bis_config_analysis_functions import *

import joblib

debug = False






################################################
######## CXY CYCLE FREQ SURROGATES ########
################################################



# Surrogate computation for a single channel
def compute_surrogates_coh_n_chan(n_chan, data_tmp, respi, mask_hzCxy, n_surrogates_coh, srate, hannw, nwind, noverlap, nfft):

    print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

    x = data_tmp[n_chan, :]
    surrogates_val_tmp = np.zeros((n_surrogates_coh, mask_hzCxy.sum()))

    for surr_i in range(n_surrogates_coh):
        
        x_shift = shuffle_Cxy(x)

        # Compute coherence
        hzCxy_tmp, Cxy = scipy.signal.coherence(x_shift, respi, fs=srate, window=hannw, nperseg=nwind, noverlap=noverlap, nfft=nfft)
        surrogates_val_tmp[surr_i, :] = Cxy[mask_hzCxy]

    return np.percentile(surrogates_val_tmp, percentile_coh, axis=0)



def precompute_surrogates_coh(sujet):

    save_path = os.path.join(path_precompute, 'allsujet', 'PSD_Coh')
    if os.path.exists(os.path.join(save_path, f"{sujet}_surr_Cxy.npy")):
        print(f'ALREADY COMPUTED {sujet}')
        return
    
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    # Precompute frequency bins and mask
    hzCxy = np.linspace(0, srate / 2, int(nfft / 2 + 1))
    mask_hzCxy = (hzCxy >= freq_surrogates[0]) & (hzCxy < freq_surrogates[1])
    hzCxy_respi = hzCxy[mask_hzCxy]

    # Initialize array to store surrogate results
    Cxy_surr_sujet = np.zeros((len(conditions), len(odor_list), len(chan_list), len(hzCxy_respi)))

    print(sujet)

    for cond_i, cond in enumerate(conditions):
        for odor_i, odor in enumerate(odor_list):
            print(cond, odor)

            # Load data once per condition and odor
            data_tmp = load_data_sujet(sujet, cond, odor)
            respi = data_tmp[chan_list.index('PRESS'), :]

            # Parallelized computation of surrogates
            surrogates_n_chan = joblib.Parallel(n_jobs=n_core, prefer='threads')(
                joblib.delayed(compute_surrogates_coh_n_chan)(n_chan, data_tmp, respi, mask_hzCxy, n_surrogates_coh, srate, hannw, nwind, noverlap, nfft) 
                for n_chan in range(data_tmp.shape[0])
            )

            # Store results
            for chan_i in range(data_tmp.shape[0]):
                Cxy_surr_sujet[cond_i, odor_i, chan_i, :] = surrogates_n_chan[chan_i]

    # Save results to disk
    os.chdir(save_path)
    np.save(f"{sujet}_surr_Cxy.npy", Cxy_surr_sujet)

    print('Done')


#sujet = sujet_list[0]
def precompute_surrogates_coh_permutations():

    save_path = os.path.join(path_precompute, 'allsujet', 'PSD_Coh')
    if os.path.exists(os.path.join(save_path, f"perm_intra_Cxy.nc")) and os.path.exists(os.path.join(save_path, f"surr_Cxy_perm_intra.nc")):
        print(f'ALREADY COMPUTED {sujet}')
        return
    
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    # Precompute frequency bins and mask
    hzCxy = np.linspace(0, srate / 2, int(nfft / 2 + 1))
    mask_hzCxy = (hzCxy >= freq_surrogates[0]) & (hzCxy < freq_surrogates[1])
    hzCxy_respi = hzCxy[mask_hzCxy]
    
    #### intra
    print('intra')
    cond_sel = ['MECA', 'CO2', 'FR_CV_2']
    odor_sel = ['o', '+', '-']

    xr_data = np.zeros((2, len(cond_sel), len(odor_sel), len(chan_list_eeg)))

    for cond_i, cond in enumerate(cond_sel):

        for odor_i, odor in enumerate(odor_sel):

            data_Cxy_baseline = np.zeros((len(sujet_list), len(chan_list_eeg)))
            data_Cxy_cond = np.zeros((len(sujet_list), len(chan_list_eeg)))

            for sujet_i, sujet in enumerate(sujet_list):

                print_advancement(sujet_i, len(sujet_list), steps=[25,50,75])

                respfeatures_allcond = load_respfeatures(sujet)

                median_resp_baseline = np.median(respfeatures_allcond['FR_CV_1'][odor]['cycle_freq'])
                mask_cxy_hzpxx_baseline = (hzCxy > (median_resp_baseline - around_respi_Cxy)) & (hzCxy < (median_resp_baseline + around_respi_Cxy))

                median_resp_cond = np.median(respfeatures_allcond[cond][odor]['cycle_freq'])
                mask_cxy_hzpxx_cond = (hzCxy > (median_resp_cond - around_respi_Cxy)) & (hzCxy < (median_resp_cond + around_respi_Cxy))

                data_baseline = load_data_sujet(sujet, 'FR_CV_1', odor)
                baseline_respi = data_baseline[chan_list.index('PRESS'), :]

                data_cond = load_data_sujet(sujet, cond, odor)
                cond_respi = data_baseline[chan_list.index('PRESS'), :]

                for chan_i, chan in enumerate(chan_list_eeg):

                    x_baseline = data_baseline[chan_i, :]
                    Cxy_baseline = scipy.signal.coherence(x_baseline, baseline_respi, fs=srate, window=hannw, 
                                                                                            nperseg=nwind, noverlap=noverlap, nfft=nfft)[1][mask_cxy_hzpxx_baseline]

                    y_cond = data_cond[chan_i, :]
                    Cxy_cond = scipy.signal.coherence(y_cond, cond_respi, fs=srate, window=hannw, 
                                                                                        nperseg=nwind, noverlap=noverlap, nfft=nfft)[1][mask_cxy_hzpxx_cond]
                    
                    data_Cxy_baseline[sujet_i, chan_i] = np.median(Cxy_baseline)
                    data_Cxy_cond[sujet_i, chan_i] = np.median(Cxy_cond)

            if debug:

                chan_i = 0
                plt.hist(data_Cxy_baseline[:,chan_i], label='baseline', bins=25)
                plt.hist(data_Cxy_cond[:,chan_i], label='cond', bins=25)
                plt.legend()
                plt.show()

            Cxy_diff = np.median(data_Cxy_cond - data_Cxy_baseline, axis=0)

            clusters = np.zeros((len(chan_list_eeg)))

            for chan_i, chan in enumerate(chan_list_eeg):
                
                clusters[chan_i] = get_permutation_2groups(data_Cxy_baseline[:, chan_i], data_Cxy_cond[:, chan_i], n_surrogates_coh, 
                                                           mode_diff='median', mode_generate_surr='minmax', percentile_thresh=[0.5, 99.5])

            if debug:

                ch_types = ['eeg'] * len(chan_list_eeg)
                info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
                info.set_montage('standard_1020')
                mask_params = dict(markersize=10, markerfacecolor='y')
                vlim = np.abs(Cxy_diff).max()

                fig, ax = plt.subplots()
                im, _ = mne.viz.plot_topomap(data=Cxy_diff, axes=ax, show=False, names=chan_list_eeg, pos=info, vlim=(-vlim, vlim),
                                                mask=clusters.astype('bool'), mask_params=mask_params, cmap='seismic')
                fig.colorbar(im, ax=ax, orientation='vertical')
                plt.title(odor)
                plt.show()

            xr_data[0, cond_i, odor_i, :], xr_data[1, cond_i, odor_i, :] = Cxy_diff, clusters

    xr_dict = {'type' : ['Cxy_diff', 'cluster'], 'cond' : cond_sel, 'odor' : odor_sel, 'chan' : chan_list_eeg}
    xr_intra = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    #### save intra
    os.chdir(save_path)
    xr_intra.to_netcdf('perm_intra_Cxy.nc')

    #### inter
    print('inter')

    cond_sel = conditions
    odor_sel = ['+', '-']
    xr_data = np.zeros((2, len(cond_sel), len(odor_sel), len(chan_list_eeg)))

    for cond_i, cond in enumerate(cond_sel):

        for odor_i, odor in enumerate(odor_sel):

            data_Cxy_baseline = np.zeros((len(sujet_list), len(chan_list_eeg)))
            data_Cxy_cond = np.zeros((len(sujet_list), len(chan_list_eeg)))

            for sujet_i, sujet in enumerate(sujet_list):

                print_advancement(sujet_i, len(sujet_list), steps=[25,50,75])

                respfeatures_allcond = load_respfeatures(sujet)

                median_resp_baseline = np.median(respfeatures_allcond[cond]['o']['cycle_freq'])
                mask_cxy_hzpxx_baseline = (hzCxy > (median_resp_baseline - around_respi_Cxy)) & (hzCxy < (median_resp_baseline + around_respi_Cxy))

                median_resp_cond = np.median(respfeatures_allcond[cond][odor]['cycle_freq'])
                mask_cxy_hzpxx_cond = (hzCxy > (median_resp_cond - around_respi_Cxy)) & (hzCxy < (median_resp_cond + around_respi_Cxy))

                data_baseline = load_data_sujet(sujet, cond, 'o')
                baseline_respi = data_baseline[chan_list.index('PRESS'), :]

                data_cond = load_data_sujet(sujet, cond, odor)
                cond_respi = data_baseline[chan_list.index('PRESS'), :]

                for chan_i, chan in enumerate(chan_list_eeg):

                    x_baseline = data_baseline[chan_i, :]
                    Cxy_baseline = scipy.signal.coherence(x_baseline, baseline_respi, fs=srate, window=hannw, 
                                                                                            nperseg=nwind, noverlap=noverlap, nfft=nfft)[1][mask_cxy_hzpxx_baseline]

                    y_cond = data_cond[chan_i, :]
                    Cxy_cond = scipy.signal.coherence(y_cond, cond_respi, fs=srate, window=hannw, 
                                                                                        nperseg=nwind, noverlap=noverlap, nfft=nfft)[1][mask_cxy_hzpxx_cond]
                    
                    data_Cxy_baseline[sujet_i, chan_i] = np.median(Cxy_baseline)
                    data_Cxy_cond[sujet_i, chan_i] = np.median(Cxy_cond)

                    
            if debug:

                plt.plot(hzCxy_respi, np.median(data_Cxy_baseline[:,0,:], axis=0), label='baseline')
                plt.plot(hzCxy_respi, np.median(data_Cxy_cond[:,0,:], axis=0), label='cond')
                plt.legend()
                plt.show()

            Cxy_diff = np.median(data_Cxy_cond - data_Cxy_baseline, axis=0)

            clusters = np.zeros((len(chan_list_eeg)))

            for chan_i, chan in enumerate(chan_list_eeg):
            
                clusters[chan_i] = get_permutation_2groups(data_Cxy_baseline[:, chan_i], data_Cxy_cond[:, chan_i], n_surrogates_coh, 
                                                           mode_diff='median', mode_generate_surr='minmax', percentile_thresh=[0.5, 99.5])
            if debug:

                ch_types = ['eeg'] * len(chan_list_eeg)
                info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
                info.set_montage('standard_1020')
                mask_params = dict(markersize=10, markerfacecolor='y')
                vlim = np.abs(Cxy_diff).max()

                fig, ax = plt.subplots()
                im, _ = mne.viz.plot_topomap(data=Cxy_diff, axes=ax, show=False, names=chan_list_eeg, pos=info, vlim=(-vlim, vlim),
                                                mask=clusters.astype('bool'), mask_params=mask_params, cmap='seismic')
                fig.colorbar(im, ax=ax, orientation='vertical')
                plt.title(odor)
                plt.show()

            xr_data[0,cond_i,odor_i,:], xr_data[1,cond_i,odor_i,:] = Cxy_diff, clusters

    xr_dict = {'type' : ['Cxy_diff', 'cluster'], 'cond' : cond_sel, 'odor' : odor_sel, 'chan' : chan_list_eeg}
    xr_inter = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    #### save inter
    os.chdir(save_path)
    xr_inter.to_netcdf('perm_inter_Cxy.nc')

    print('Done')






def precompute_surrogates_cyclefreq(sujet, band_prep, cond):
    
    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)

    #### load data
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        data_tmp = load_data_sujet(sujet, cond, odor_i)

        if os.path.exists(f'{sujet}_{cond}_{odor_i}_cyclefreq_{band_prep}.npy') == True :
            print(f'ALREADY COMPUTED {cond} {odor_i}')
            continue

        #### compute surrogates
        surrogates_n_chan = np.zeros((3, data_tmp.shape[0], stretch_point_surrogates))

        respfeatures_i = respfeatures_allcond[cond][odor_i]

        #n_chan = 0
        def compute_surrogates_cyclefreq_nchan(n_chan):

            print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

            x = data_tmp[n_chan,:]

            surrogates_val_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

            #surr_i = 0
            for surr_i in range(n_surrogates_cyclefreq):

                # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

                x_shift = shuffle_Cxy(x)
                #y_shift = shuffle_Cxy(y)

                x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_shift, srate)

                x_stretch_mean = np.mean(x_stretch, axis=0)

                surrogates_val_tmp[surr_i,:] = x_stretch_mean

            mean_surrogate_tmp = np.mean(surrogates_val_tmp, axis=0)
            surrogates_val_tmp_sorted = np.sort(surrogates_val_tmp, axis=0)
            percentile_i_up = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_up))
            percentile_i_dw = int(np.floor(n_surrogates_cyclefreq*percentile_cyclefreq_dw))

            up_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_up,:]
            dw_percentile_values_tmp = surrogates_val_tmp_sorted[percentile_i_dw,:]

            return mean_surrogate_tmp, up_percentile_values_tmp, dw_percentile_values_tmp

        compute_surrogates_cyclefreq_results = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(np.size(data_tmp,0)))

        #### fill results
        for n_chan in range(np.size(data_tmp,0)):

            surrogates_n_chan[0, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][0]
            surrogates_n_chan[1, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][1]
            surrogates_n_chan[2, n_chan, :] = compute_surrogates_cyclefreq_results[n_chan][2]
        
        #### save
        np.save(f'{sujet}_{cond}_{odor_i}_cyclefreq_{band_prep}.npy', surrogates_n_chan)

        print('done')






################################
######## MI / MVL ########
################################




#x = x_stretch_linear
def shuffle_windows(x):

    n_cycles_stretch = int( x.shape[0]/stretch_point_surrogates )

    shuffle_win = np.zeros(( n_cycles_stretch, stretch_point_surrogates ))

    for cycle_i in range(n_cycles_stretch):

        cut_i = np.random.randint(0, x.shape[0]-stretch_point_surrogates, 1)
        shuffle_win[cycle_i,:] = x[int(cut_i):int(cut_i+stretch_point_surrogates)]

    x_shuffled = np.mean(shuffle_win, axis=0)

    if debug:
        plt.plot(x_shuffled)
        plt.show()

    return x_shuffled




def precompute_MVL(sujet, band_prep, cond):

    print(cond)

    #### load params
    respfeatures_allcond = load_respfeatures(sujet)

    for odor_i in odor_list:

        respfeatures_i = respfeatures_allcond[cond][odor_i]

        #### load data
        os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))

        data_tmp = load_data_sujet(sujet, cond, odor_i)

        if os.path.exists(f'{sujet}_{cond}_{odor_i}_MVL_{band_prep}.npy') == True :
            print(f'ALREADY COMPUTED {cond} {odor_i}')
            continue

        #### compute surrogates
        #n_chan = 0
        def compute_surrogates_cyclefreq_nchan(n_chan):

            print_advancement(n_chan, data_tmp.shape[0], steps=[25, 50, 75])

            #### stretch
            x = data_tmp[n_chan,:]
            x_zscore = zscore(x)
            x_stretch, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_surrogates, x_zscore, srate)

            MVL_nchan = get_MVL(np.mean(x_stretch,axis=0)-np.mean(x_stretch,axis=0).min())

            x_stretch_linear = x_stretch.reshape(-1) 
            
            #### surrogates
            surrogates_stretch_tmp = np.zeros((n_surrogates_cyclefreq, stretch_point_surrogates))

            for surr_i in range(n_surrogates_cyclefreq):

                # print_advancement(surr_i, n_surrogates_cyclefreq, steps=[25, 50, 75])

                surrogates_stretch_tmp[surr_i,:] = shuffle_windows(x_stretch_linear)

            #### compute MVL
            MVL_surrogates_i = np.array([])

            for surr_i in range(n_surrogates_cyclefreq):

                x = surrogates_stretch_tmp[surr_i,:]
                
                MVL_surrogates_i = np.append(MVL_surrogates_i, get_MVL(x-x.min()))

            return MVL_nchan, MVL_surrogates_i

        compute_surrogates_MVL = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_surrogates_cyclefreq_nchan)(n_chan) for n_chan in range(data_tmp.shape[0]))

        #### fill results
        MVL_surrogates = np.zeros(( data_tmp.shape[0], n_surrogates_cyclefreq ))
        MVL_val = np.zeros(( data_tmp.shape[0] ))

        for n_chan in range(data_tmp.shape[0]):

            MVL_surrogates[n_chan,:] = compute_surrogates_MVL[n_chan][1]
            MVL_val[n_chan] = compute_surrogates_MVL[n_chan][0]

        #### verif
        if debug:
            n_chan = 0
            count, values, fig = plt.hist(MVL_surrogates[n_chan,:])
            plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 99), ymin=0, ymax=count.max())
            plt.vlines(np.percentile(MVL_surrogates[n_chan,:], 95), ymin=0, ymax=count.max())
            plt.vlines(MVL_val[n_chan], ymin=0, ymax=count.max(), color='r')
            plt.show()
        
        #### save
        np.save(f'{sujet}_{cond}_{odor_i}_MVL_{band_prep}.npy', MVL_surrogates)

        print('done')





################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    # precompute_surrogates_coh(sujet)
    execute_function_in_slurm_bash('n05_precompute_Cxy', 'precompute_surrogates_coh', [[sujet] for sujet in sujet_list])
    # sync_folders__push_to_crnldata()  

    # precompute_surrogates_coh(sujet)
    execute_function_in_slurm_bash('n05_precompute_Cxy', 'precompute_surrogates_coh_permutations', [])
    # sync_folders__push_to_crnldata()  






    #sujet = sujet_list[14]
    # for sujet in sujet_list:    

        #### compute and save
        # print(f'######## COMPUTE SURROGATES {sujet} ########')

        #cond = conditions[2]
        # for cond in conditions:

            # precompute_surrogates_cyclefreq(sujet, cond)
            # execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_surrogates_cyclefreq', [sujet, cond])

            # precompute_MVL(sujet, cond)
            # execute_function_in_slurm_bash('n5_precompute_surrogates', 'precompute_MVL', [sujet, cond])    
             
    


            






