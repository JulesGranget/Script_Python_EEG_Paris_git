
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
    if os.path.exists(os.path.join(save_path, f"perm_intra_Cxy.nc")) and os.path.exists(os.path.join(save_path, f"perm_inter_Cxy.nc")) and os.path.exists(os.path.join(save_path, f"perm_repnorep_Cxy.nc")):
        print(f'PERM Cxy ALREADY COMPUTED')
        return
    
    nwind, nfft, noverlap, hannw = get_params_spectral_analysis(srate)

    # Precompute frequency bins and mask
    hzCxy = np.linspace(0, srate / 2, int(nfft / 2 + 1))
    mask_hzCxy = (hzCxy >= freq_surrogates[0]) & (hzCxy < freq_surrogates[1])

    sujet_group_list = ['allsujet', 'rep', 'norep']

    #### get Cxy data
    print('get Cxy')
    data_Cxy = np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg)))

    for cond_i, cond in enumerate(conditions):

        for odor_i, odor in enumerate(odor_list):

            print(cond, odor)

            def process_subject(sujet_i, sujet):

                print_advancement(sujet_i, len(sujet_list), steps=[25,50,75])

                respfeatures_allcond = load_respfeatures(sujet)

                median_resp = np.median(respfeatures_allcond[cond][odor]['cycle_freq'])
                mask_cxy_hzpxx = (hzCxy > (median_resp - around_respi_Cxy)) & (hzCxy < (median_resp + around_respi_Cxy))

                # Data loading
                _data_Cxy = load_data_sujet(sujet, cond, odor)
                _respi = _data_Cxy[chan_list.index('PRESS'), :]

                _data_Cxy_allchan = np.zeros((len(chan_list_eeg)))

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    x = _data_Cxy[chan_i, :]
                    _hzpxx, _Cxy = scipy.signal.coherence(x, _respi, fs=srate, window=hannw, 
                                                                            nperseg=nwind, noverlap=noverlap, nfft=nfft)
                    
                    if debug:

                        plt.plot(_hzpxx, _Cxy)
                        plt.xlim(0,2)
                        plt.show()
                    
                    _data_Cxy_allchan[chan_i] = np.median(_Cxy[mask_cxy_hzpxx])
                    
                return _data_Cxy_allchan      

            # Parallel execution of the processing across subjects
            res = joblib.Parallel(n_jobs=n_core, prefer="processes")(joblib.delayed(process_subject)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))

            for sujet_i, sujet in enumerate(sujet_list):

                data_Cxy[sujet_i,cond_i,odor_i,:] = res[sujet_i]

    xr_data = np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg)))
    dict_Cxy_allsujet = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_Cxy_allsujet = xr.DataArray(data=data_Cxy, dims=dict_Cxy_allsujet.keys(), coords=dict_Cxy_allsujet)
    df_Cxy_allsujet = xr_Cxy_allsujet.to_dataframe(name='Cxy').reset_index(drop=False)

    mask_repnorep = [_sujet in sujet_best_list_rev for _sujet in df_Cxy_allsujet['sujet'].values]
    df_Cxy_allsujet['REP'] = mask_repnorep

    os.chdir(save_path)
    df_Cxy_allsujet.to_excel('Cxy_allsujet.xlsx')
    
    #### intra
    print('intra')
    cond_sel = ['MECA', 'CO2', 'FR_CV_2']
    odor_sel = ['o', '+', '-']

    xr_data = np.zeros((2, len(sujet_group_list), len(cond_sel), len(odor_sel), len(chan_list_eeg)))

    for cond_i, cond in enumerate(cond_sel):

        print(cond)

        for odor_i, odor in enumerate(odor_sel):

            #sujet_group_i, sujet_group = 0, sujet_group_list[0]
            for sujet_group_i, sujet_group in enumerate(sujet_group_list):

                data_Cxy_baseline = data_Cxy[:,0,odor_i,:]
                data_Cxy_cond = data_Cxy[:,cond_i+1,odor_i,:]

                if sujet_group == 'allsujet':
                    sujet_sel = np.arange(len(sujet_list))
                elif sujet_group == 'rep':
                    sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_best_list_rev]
                elif sujet_group == 'norep':
                    sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_no_respond_rev]
                
                data_Cxy_cond_chunk, data_Cxy_baseline_chunk = data_Cxy_cond[sujet_sel,:], data_Cxy_baseline[sujet_sel,:]
                Cxy_diff = np.median(data_Cxy_cond_chunk - data_Cxy_baseline_chunk, axis=0)

                clusters = np.zeros((len(chan_list_eeg)))

                #chan_i, chan = 11, chan_list_eeg[11]
                for chan_i, chan in enumerate(chan_list_eeg):
                
                    clusters[chan_i] = get_permutation_2groups(data_Cxy_baseline_chunk[:,chan_i], data_Cxy_cond_chunk[:,chan_i], n_surrogates_coh, 
                                                        mode_grouped=mode_grouped_ERP_STATS, mode_generate_surr=mode_generate_surr_ERP_STATS, 
                                                        stat_design='within', percentile_thresh=percentile_thresh_ERP_STATS)
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

                xr_data[0,sujet_group_i,cond_i,odor_i,:], xr_data[1,sujet_group_i,cond_i,odor_i,:] = Cxy_diff, clusters

    xr_dict = {'type' : ['Cxy_diff', 'cluster'], 'sujet_group' : sujet_group_list, 'cond' : cond_sel, 'odor' : odor_sel, 'chan' : chan_list_eeg}
    xr_intra = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    #### save
    os.chdir(save_path)
    xr_intra.to_netcdf('perm_intra_Cxy.nc')

    #### inter
    print('inter')

    cond_sel = conditions
    odor_sel = ['+', '-']
    xr_data = np.zeros((2, len(sujet_group_list), len(cond_sel), len(odor_sel), len(chan_list_eeg)))

    #cond_i, cond = 2, 'CO2'
    for cond_i, cond in enumerate(cond_sel):

        print(cond)

        #odor_i, odor = 0, odor_sel[0]
        for odor_i, odor in enumerate(odor_sel):

            #sujet_group_i, sujet_group = 0, sujet_group_list[0]
            for sujet_group_i, sujet_group in enumerate(sujet_group_list):

                data_Cxy_baseline = data_Cxy[:,cond_i,0,:]
                data_Cxy_cond = data_Cxy[:,cond_i,odor_i+1,:]

                if sujet_group == 'allsujet':
                    sujet_sel = np.arange(len(sujet_list))
                elif sujet_group == 'rep':
                    sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_best_list_rev]
                elif sujet_group == 'norep':
                    sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_no_respond_rev]
                
                data_Cxy_cond_chunk, data_Cxy_baseline_chunk = data_Cxy_cond[sujet_sel,:], data_Cxy_baseline[sujet_sel,:]
                Cxy_diff = np.median(data_Cxy_cond_chunk - data_Cxy_baseline_chunk, axis=0)

                clusters = np.zeros((len(chan_list_eeg)))

                #chan_i, chan = 11, chan_list_eeg[11]
                for chan_i, chan in enumerate(chan_list_eeg):
                
                    clusters[chan_i] = get_permutation_2groups(data_Cxy_baseline_chunk[:,chan_i], data_Cxy_cond_chunk[:,chan_i], n_surrogates_coh, 
                                                        mode_grouped=mode_grouped_ERP_STATS, mode_generate_surr=mode_generate_surr_ERP_STATS, 
                                                        stat_design='within', percentile_thresh=percentile_thresh_ERP_STATS)
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

                xr_data[0,sujet_group_i,cond_i,odor_i,:], xr_data[1,sujet_group_i,cond_i,odor_i,:] = Cxy_diff, clusters

    xr_dict = {'type' : ['Cxy_diff', 'cluster'], 'sujet_group' : sujet_group_list, 'cond' : cond_sel, 'odor' : odor_sel, 'chan' : chan_list_eeg}
    xr_inter = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    #### save
    os.chdir(save_path)
    xr_inter.to_netcdf('perm_inter_Cxy.nc')

    #### repnorep
    print('repnorep')

    cond_sel = conditions
    odor_sel = odor_list
    xr_data = np.zeros((2, len(cond_sel), len(odor_sel), len(chan_list_eeg)))

    #cond_i, cond = 2, 'CO2'
    for cond_i, cond in enumerate(cond_sel):

        for odor_i, odor in enumerate(odor_sel):

            print(cond, odor)

            sujet_sel_rep = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_best_list_rev]
            sujet_sel_norep = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_no_respond_rev]

            data_Cxy_baseline_chunk = data_Cxy[sujet_sel_norep,cond_i,odor_i,:]
            data_Cxy_cond_chunk = data_Cxy[sujet_sel_rep,cond_i,odor_i,:]
            
            Cxy_diff = np.median(data_Cxy_cond_chunk, axis=0) - np.median(data_Cxy_baseline_chunk, axis=0)

            clusters = np.zeros((len(chan_list_eeg)))

            #chan_i, chan = 11, chan_list_eeg[11]
            for chan_i, chan in enumerate(chan_list_eeg):
            
                clusters[chan_i] = get_permutation_2groups(data_Cxy_baseline_chunk[:,chan_i], data_Cxy_cond_chunk[:,chan_i], n_surrogates_coh, 
                                                        mode_grouped=mode_grouped_ERP_STATS, mode_generate_surr=mode_generate_surr_ERP_STATS, 
                                                        stat_design='between', percentile_thresh=percentile_thresh_ERP_STATS)
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
    xr_repnorep = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    #### save repnorep
    os.chdir(save_path)
    xr_repnorep.to_netcdf('perm_repnorep_Cxy.nc')

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

    precompute_surrogates_coh_permutations()

    # precompute_surrogates_coh(sujet)
    # execute_function_in_slurm_bash('n05_precompute_Cxy', 'precompute_surrogates_coh', [[sujet] for sujet in sujet_list])
    # sync_folders__push_to_crnldata()  

    # precompute_surrogates_coh_permutations()
    # execute_function_in_slurm_bash('n05_precompute_Cxy', 'precompute_surrogates_coh_permutations', [])
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
             
    


            






