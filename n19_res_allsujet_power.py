
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
# import cv2

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0ter_stats import *


debug = False







################################################
######## PLOT & SAVE PSD AND COH ########
################################################



def get_stats_topoplots(baseline_values, cond_values, chan_list_eeg):

    data = {'sujet' : [], 'cond' : [], 'chan' : [], 'value' : []}

    for sujet_i in range(baseline_values.shape[-1]):

        for chan_i, chan in enumerate(chan_list_eeg):

            data['sujet'].append(sujet_i)
            data['cond'].append('baseline')
            data['chan'].append(chan)
            data['value'].append(baseline_values[chan_i, sujet_i])

            data['sujet'].append(sujet_i)
            data['cond'].append('cond')
            data['chan'].append(chan)
            data['value'].append(cond_values[chan_i, sujet_i])
    
    df_stats = pd.DataFrame(data)

    mask_signi = np.array((), dtype='bool')

    for chan in chan_list_eeg:

        pval = get_stats_df(df=df_stats.query(f"chan == '{chan}'"), predictor='cond', outcome='value', subject='sujet', design='within')

        if pval < 0.05:
            mask_signi = np.append(mask_signi, True)
        else:
            mask_signi = np.append(mask_signi, False)

    return mask_signi



def get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet):

    source_path = os.getcwd()
    
    os.chdir(os.path.join(path_precompute, sujet, 'PSD_Coh'))
                
    with open(f'allcond_{sujet}_Pxx.pkl', 'rb') as f:
        Pxx_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_Cxy.pkl', 'rb') as f:
        Cxy_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_surrogates.pkl', 'rb') as f:
        surrogates_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_cyclefreq.pkl', 'rb') as f:
        cyclefreq_allcond = pickle.load(f)

    with open(f'allcond_{sujet}_MVL.pkl', 'rb') as f:
        MVL_allcond = pickle.load(f)

    os.chdir(source_path)

    return Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond

        


def compute_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allsujet():

    metric_to_load = ['Cxy', 'MVL', 'Cxy_surr', 'MVL_surr', 'Pxx_theta', 'Pxx_alpha', 'Pxx_beta', 'Pxx_l_gamma', 'Pxx_h_gamma']
    xr_data = np.zeros(( len(sujet_list), len(metric_to_load), len(conditions), len(odor_list), len(chan_list_eeg)))
    xr_dict = {'sujet' : sujet_list, 'metric' : metric_to_load, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_allsujet = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())  

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #### load data
        Pxx_allcond, Cxy_allcond, surrogates_allcond, cyclefreq_allcond, MVL_allcond = get_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allcond(sujet)
        prms = get_params()
        respfeatures_allcond = load_respfeatures(sujet)

        #### params
        hzPxx = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        hzCxy = np.linspace(0,srate/2,int(prms['nfft']/2+1))
        mask_hzCxy = (hzCxy>=freq_surrogates[0]) & (hzCxy<freq_surrogates[1])
        hzCxy = hzCxy[mask_hzCxy]

        #### reduce data
        for cond in conditions:

            for odor_i in odor_list:

                mean_resp = respfeatures_allcond[cond][odor_i]['cycle_freq'].mean()
                hzCxy_mask = (hzCxy > (mean_resp - around_respi_Cxy)) & (hzCxy < (mean_resp + around_respi_Cxy))

                Cxy_allchan_i = Cxy_allcond[cond][odor_i][:,hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy', cond, odor_i, :] = Cxy_allchan_i
                MVL_allchan_i = MVL_allcond[cond][odor_i]
                xr_allsujet.loc[sujet, 'MVL', cond, odor_i, :] = MVL_allchan_i

                Cxy_allchan_surr_i = surrogates_allcond['Cxy'][cond][odor_i][:len(chan_list_eeg),hzCxy_mask].mean(axis=1)
                xr_allsujet.loc[sujet, 'Cxy_surr', cond, odor_i, :] = np.array(Cxy_allchan_i > Cxy_allchan_surr_i)*1
                MVL_allchan_surr_i = np.array([np.percentile(surrogates_allcond['MVL'][cond][odor_i][nchan,:], 99) for nchan, _ in enumerate(chan_list_eeg)])
                xr_allsujet.loc[sujet, 'MVL_surr', cond, odor_i, :] = np.array(MVL_allchan_i > MVL_allchan_surr_i)*1

        for cond in conditions:

            for odor_i in odor_list:

                #band, freq = 'theta', [4, 8]
                for band, freq in freq_band_fc_analysis.items():

                    hzPxx_mask = (hzPxx >= freq[0]) & (hzPxx <= freq[-1])
                    Pxx_mean_i = Pxx_allcond[cond][odor_i][:,hzPxx_mask].mean(axis=1)
                    xr_allsujet.loc[sujet, f'Pxx_{band}', cond, odor_i, :] = Pxx_mean_i

    
    #### get stats
    metric_to_load = ['Pxx_theta', 'Pxx_alpha', 'Pxx_beta', 'Pxx_l_gamma', 'Pxx_h_gamma']
    allsujet_stats_data = np.zeros((2, len(metric_to_load), len(odor_list), len(conditions), len(chan_list_eeg)), dtype='bool')

    #stats_type = 'intra'
    for stats_type_i, stats_type in enumerate(['intra', 'inter']):
        
        #metric = 'Cxy'
        for metric_i, metric in enumerate(metric_to_load):

            #### intra
            if stats_type == 'intra':

                #odor = 'o'
                for odor_i, odor in enumerate(odor_list):

                    #cond = 'MECA'
                    for cond_i, cond in enumerate(conditions):

                        if cond == 'FR_CV_1':

                            continue

                        baseline_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,'FR_CV_1',odor,:,:].values
                        cond_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,odor,:,:].values

                        # print(stats_type,metric,cond,odor, flush=True)

                        allsujet_stats_data[stats_type_i, metric_i, odor_i, cond_i, :] = get_stats_topoplots(baseline_values, cond_values, chan_list_eeg)

            #### inter
            if stats_type == 'inter':

                #odor = 'o'
                for odor_i, odor in enumerate(odor_list):

                    if odor == 'o':
                        
                        continue

                    #cond = 'MECA'
                    for cond_i, cond in enumerate(conditions):

                        baseline_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,'o',:,:].values
                        cond_values = xr_allsujet.transpose('metric', 'cond', 'odor', 'chan', 'sujet').loc[metric,cond,odor,:,:].values

                        # print(stats_type,metric,cond,odor, flush=True)

                        allsujet_stats_data[stats_type_i, metric_i, odor_i, cond_i, :] = get_stats_topoplots(baseline_values, cond_values, chan_list_eeg)

    xr_dict = {'stats_type' : ['intra', 'inter'], 'metric' : ['theta', 'alpha', 'beta', 'l_gamma', 'h_gamma'], 'odor' : odor_list, 'cond' : conditions, 'chan' : chan_list_eeg}
    xr_allsujet_stats = xr.DataArray(allsujet_stats_data, dims=xr_dict.keys(), coords=xr_dict.values())  
    
    return xr_allsujet, xr_allsujet_stats




    

################################
######## TOPOPLOT ########
################################



def plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet, xr_allsujet_stats):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    sujet_respond = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_respond])

    #### scales
    scales_allband = {}

    #data_type = data_type_list[0]
    for data_type in xr_allsujet['metric'].values:

        scales_allband[data_type] = {}

        val = np.array([])

        for odor_i in odor_list:

            for cond in conditions:

                for sujet_group in ['allsujet', 'rep', 'no_rep']:

                    if data_type.find('surr') == -1:

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:,data_type, cond, odor_i].median('sujet').values
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond,data_type, cond, odor_i].median('sujet').values
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond,data_type, cond, odor_i].median('sujet').values
                    
                    if data_type.find('surr') != -1:

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:,data_type, cond, odor_i].sum('sujet').values / sujet_list.shape[0]
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond,data_type, cond, odor_i].sum('sujet').values / sujet_respond.shape[0]
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond,data_type, cond, odor_i].sum('sujet').values / sujet_no_respond.shape[0]
                    
                    val = np.append(val, data_sel)

        scales_allband[data_type]['min'] = val.min()
        scales_allband[data_type]['max'] = val.max()

    #### plot Cxy, MVL
    #sujet_group = 'allsujet'
    for sujet_group in ['allsujet', 'rep', 'no_rep']:

        #data_type = 'Cxy_surr'
        for data_type in xr_allsujet['metric'].values:

            if data_type.find('Pxx') == -1:

                vmin = np.round(scales_allband[data_type]['min'], 2)
                vmax = np.round(scales_allband[data_type]['max'], 2)

                fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
                plt.suptitle(f'{sujet_group}_{data_type}, vmin:{vmin} vmax:{vmax}')
                fig.set_figheight(10)
                fig.set_figwidth(10)

                #c, cond = 0, 'FR_CV_1'
                for c, cond in enumerate(conditions):

                    #r, odor_i = 0, odor_list[0]
                    for r, odor_i in enumerate(odor_list):

                        #### plot
                        ax = axs[r, c]

                        if r == 0:
                            ax.set_title(cond, fontweight='bold', rotation=0)
                        if c == 0:
                            ax.set_ylabel(f'{odor_i}')

                        if sujet_group == 'allsujet':
                            data_sel = xr_allsujet.loc[:, data_type, cond, odor_i].values
                        if sujet_group == 'rep':
                            data_sel = xr_allsujet.loc[sujet_respond, data_type, cond, odor_i].values
                        if sujet_group == 'no_rep':
                            data_sel = xr_allsujet.loc[sujet_no_respond, data_type, cond, odor_i].values

                        if data_type.find('surr') == -1:
                            data_sel = np.median(data_sel, axis=0)

                        elif data_type.find('surr') != -1:
                            if sujet_group == 'allsujet':
                                data_sel = np.sum(data_sel, axis=0) / sujet_list.shape[0]
                            if sujet_group == 'rep':
                                data_sel = np.sum(data_sel, axis=0) / sujet_respond.shape[0]
                            if sujet_group == 'no_rep':
                                data_sel = np.sum(data_sel, axis=0) / sujet_no_respond.shape[0]
                                             
                        mne.viz.plot_topomap(data_sel, info, axes=ax, vlim=(vmin, vmax), show=False)
                
                # plt.show() 

                #### save
                os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
                fig.savefig(f'{data_type}_{sujet_group}_{band_prep}_topo.jpeg', dpi=150)
                fig.clf()
                plt.close('all')
                gc.collect()

            elif data_type.find('Pxx') != -1:

                #### plot Pxx
                for stats_type in ['intra', 'inter']:

                    band = data_type[4:]

                    #### plot Pxx
                    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
                    plt.suptitle(f'{sujet_group}_{data_type}_{stats_type}_Pxx')
                    fig.set_figheight(10)
                    fig.set_figwidth(10)

                    #c, cond = 0, 'FR_CV_1'
                    for c, cond in enumerate(conditions):

                        #r, odor = 0, odor_list[0]
                        for r, odor in enumerate(odor_list):

                            #### plot
                            ax = axs[r, c]

                            if r == 0:
                                ax.set_title(cond, fontweight='bold', rotation=0)
                            if c == 0:
                                ax.set_ylabel(f'{odor}')

                            mask = xr_allsujet_stats.loc[stats_type, band, odor, cond,:].values
                            mask_params = dict(markersize=5, markerfacecolor='y')

                            if sujet_group == 'allsujet':
                                data_sel = xr_allsujet.loc[:, data_type, cond, odor_i].median('sujet').values
                            if sujet_group == 'rep':
                                data_sel = xr_allsujet.loc[sujet_respond, data_type, cond, odor_i].median('sujet').values
                            if sujet_group == 'no_rep':
                                data_sel = xr_allsujet.loc[sujet_no_respond, data_type, cond, odor_i].median('sujet').values
                            
                            mne.viz.plot_topomap(data_sel, info, axes=ax, 
                                                vlim=(scales_allband[data_type]['min'], scales_allband[data_type]['max']), 
                                                mask=mask, mask_params=mask_params, show=False)

                    # plt.show() 

                    #### save
                    os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
                    fig.savefig(f'{data_type}_{sujet_group}_{band_prep}_{stats_type}_topo.jpeg', dpi=150)
                    fig.clf()
                    plt.close('all')
                    gc.collect()







########################################
######## PLOT & SAVE TF & ITPC ########
########################################

#tf = tf_plot
def get_tf_stats_no_cluster_thresh(tf, pixel_based_distrib):

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

    return tf_thresh


#tf = tf_plot
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

    #### thresh cluster
    tf_thresh = tf_thresh.astype('uint8')
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(tf_thresh)
    #### nb_blobs, im_with_separated_blobs, stats = nb clusters, clusters image with labeled clusters, info on clusters
    sizes = stats[1:, -1]
    nb_blobs -= 1
    min_size = np.percentile(sizes,tf_stats_percentile_cluster)  

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





def compute_TF_allsujet():

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:

        if tf_mode == 'ITPC':
            continue

        sujet_best_list = np.array(['BD12', 'CM32', 'FA11', 'GM16', 'HJ31', 'JR17', 'MA33',
                            'MN26', 'PD01', 'SC29', 'TA09', 'TJ24', 'TM19', 'VN03','ZV21'])
        sujet_best_list_rev = np.array(['12BD', '32CM', '11FA', '16GM', '31HJ', '17JR', '33MA',
                                    '26MN', '01PD', '29SC', '09TA', '24TJ', '19TM', '03VN','21ZV'])

        # sujet_best_list_rev = []
        # for sujet in sujet_best_list:
        #     sujet_best_list_rev.append(f'{sujet[2:]}{sujet[:2]}')

        # sujet_best_list = sujet_best_list_rev.copy()
        sujet_no_best_list = [sujet for sujet in sujet_list if sujet not in sujet_best_list]

        print('COMPUTE', flush=True)

        if os.path.exists(os.path.join(path_precompute, 'allsujet', tf_mode, f'allsujet_{tf_mode}.nc')):
            print('ALREADY COMPUTED', flush=True)
            return

        else:

            #### generate xr
            os.chdir(path_memmap)
            group_sujet = ['allsujet', 'rep', 'no_rep']
            data_xr = np.memmap(f'allsujet_tf_reduction.dat', dtype=np.float64, mode='w+', shape=(len(chan_list_eeg), len(group_sujet), len(conditions), len(odor_list), nfrex, stretch_point_TF))

            #nchan = 0
            def compute_TF_ITPC(nchan, tf_mode):

                print(f'#### chan{nchan}', flush=True)

                tf_median = np.zeros((len(sujet_list),len(conditions),len(odor_list),nfrex,stretch_point_TF))

                #sujet_i, sujet = 0, sujet_list[0]
                for sujet_i, sujet in enumerate(sujet_list):

                    print_advancement(sujet_i, len(sujet_list), steps=[25, 50, 75])

                    #cond_i, cond = 0, conditions[0]
                    for cond_i, cond in enumerate(conditions):
                        #odor_i, odor = 0, odor_list[0]
                        for odor_i, odor in enumerate(odor_list):

                            os.chdir(os.path.join(path_precompute, sujet, tf_mode))
                            tf_median[sujet_i, cond_i, odor_i, :, :] = np.median(np.load(f'{sujet}_tf_conv_{cond}_{odor}.npy')[nchan,:,:,:], axis=0)

                for sujet_group_i, sujet_group in enumerate(group_sujet):

                    if sujet_group == 'allsujet':  
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median, axis=0)
                    if sujet_group == 'rep':
                        sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_best_list]
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median[sujet_sel,:], axis=0)
                    if sujet_group == 'no_rep':  
                        sujet_sel = [sujet_i for sujet_i, sujet in enumerate(sujet_list) if sujet in sujet_no_best_list]
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median, axis=0)

                if debug:

                    os.chdir(path_general)
                    time = range(stretch_point_TF)
                    tf_plot = np.median(tf_median, axis=0)[0,0,:,:]
                    plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap=plt.get_cmap('seismic'))
                    plt.yscale('log')
                    plt.savefig('test.jpg')

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_TF_ITPC)(nchan, tf_mode) for nchan, nchan_name in enumerate(chan_list_eeg))

            dict_xr = {'nchan' : chan_list_eeg, 'group_sujet' : group_sujet, 'cond' : conditions, 'odor' : odor_list, 'nfrex' : np.arange(0, nfrex), 'times' : np.arange(0, stretch_point_TF)}
            xr_allsujet = xr.DataArray(data_xr, coords=dict_xr.values(), dims=dict_xr.keys())

            os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
            xr_allsujet.to_netcdf(f'allsujet_{tf_mode}.nc')

            os.chdir(path_memmap)
            os.remove(f'allsujet_tf_reduction.dat')

    print('done', flush=True)


def compilation_compute_TF_ITPC():

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:
        
        if tf_mode == 'ITPC':
            continue

        print('COMPUTE', flush=True)

        os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
        xr_allsujet = xr.open_dataarray(f'allsujet_{tf_mode}.nc')

        group_list = ['allsujet', 'rep', 'no_rep']
        stats_plot = False
    
        print('PLOT', flush=True)

        #### scale
        vmin = np.array([])
        vmax = np.array([])

        for nchan, nchan_name in enumerate(chan_list_eeg):
        
            vals = np.array([])

            for cond in conditions:

                for odor_i in odor_list:

                    vals = np.append(vals, xr_allsujet.loc[nchan_name, 'allsujet', cond, odor_i, :, :].values.reshape(-1))

            median_diff = np.percentile(np.abs(vals - np.median(vals)), tf_plot_percentile_scale)

            vmin = np.append(vmin, np.median(vals) - median_diff)
            vmax = np.append(vmax, np.median(vals) + median_diff)

            del vals

        #### inspect stats
        if debug:

            tf_stats_type = 'intra'
            n_chan, chan_name = 0, chan_list_eeg[0]
            r, odor_i = 0, odor_list[0]
            c, cond = 1, conditions[1] 
            _vmin, _vmax = vmin[n_chan], vmax[n_chan]

            tf_plot = xr_allsujet.loc[nchan_name, cond, odor_i, :, :].values
            time = xr_allsujet['times'].values

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
            pixel_based_distrib = np.load(f'allsujet_{tf_mode.lower()}_STATS_nchan{nchan}_{cond}_{odor_i}_intra.npy')[n_chan]

            plt.pcolormesh(time, frex, tf_plot, vmin=_vmin, vmax=_vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')
            plt.contour(time, frex, get_tf_stats_no_cluster_thresh(tf_plot, pixel_based_distrib), levels=0, colors='g')
            plt.show()

            plt.pcolormesh(time, frex, tf_plot, vmin=_vmin, vmax=_vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')
            plt.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')
            plt.show()


            #wavelet_i = 0
            for wavelet_i in range(tf_plot.shape[0]):

                plt.plot(tf_plot[wavelet_i,:], color='b')
                plt.hlines([pixel_based_distrib[wavelet_i,0], pixel_based_distrib[wavelet_i,1]], xmin=0, xmax=tf_plot.shape[-1] ,color='r')

                plt.title(f'{np.round(wavelet_i/tf_plot.shape[0],2)}')

                plt.show()


        #### plot
        os.chdir(os.path.join(path_results, 'allplot', tf_mode))

        #tf_stats_type = 'intra'
        for tf_stats_type in ['inter', 'intra']:

            #group = group_list[0]
            for group in group_list:

                print(tf_stats_type, flush=True)

                #n_chan, chan_name = 0, chan_list_eeg[0]
                for n_chan, chan_name in enumerate(chan_list_eeg):

                    print_advancement(n_chan, len(chan_list_eeg), steps=[25, 50, 75])

                    #### plot
                    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

                    plt.suptitle(f'allsujet_{chan_name}')

                    fig.set_figheight(10)
                    fig.set_figwidth(15)

                    time = range(stretch_point_TF)

                    #r, odor_i = 0, odor_list[0]
                    for r, odor_i in enumerate(odor_list):

                        #c, cond = 1, conditions[1]
                        for c, cond in enumerate(conditions):

                            tf_plot = xr_allsujet.loc[chan_name, group, cond, odor_i, :, :].values
                        
                            ax = axs[r,c]

                            if r == 0 :
                                ax.set_title(cond, fontweight='bold', rotation=0)

                            if c == 0:
                                ax.set_ylabel(odor_i)

                            ax.pcolormesh(time, frex, tf_plot, vmin=vmin[nchan], vmax=vmax[nchan], shading='gouraud', cmap=plt.get_cmap('seismic'))
                            ax.set_yscale('log')

                            if stats_plot:

                                if tf_mode == 'TF' and cond != 'FR_CV_1' and tf_stats_type == 'intra':
                                    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                    pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_intra.npy'), axis=1)
                                    
                                    if get_tf_stats(tf_plot.values, pixel_based_distrib).sum() != 0:
                                        ax.contour(time, frex, get_tf_stats(tf_plot.values, pixel_based_distrib), levels=0, colors='g')

                                if tf_mode == 'TF' and odor_i != 'o' and tf_stats_type == 'inter':
                                    os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                    pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{nchan}_{cond}_{odor_i}_inter.npy'), axis=1)
                                    
                                    if get_tf_stats(tf_plot.values, pixel_based_distrib).sum() != 0:
                                        ax.contour(time, frex, get_tf_stats(tf_plot.values, pixel_based_distrib), levels=0, colors='g')

                            ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[-1], colors='g')
                            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

                    #plt.show()

                    os.chdir(os.path.join(path_results, 'allplot', tf_mode))

                    #### save
                    if tf_stats_type == 'inter':
                        fig.savefig(f'allsujet_{chan_name}_{group}_inter.jpeg', dpi=150)
                    if tf_stats_type == 'intra':
                        fig.savefig(f'allsujet_{chan_name}_{group}_intra.jpeg', dpi=150)
                        
                    fig.clf()
                    plt.close('all')
                    gc.collect()






########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL():

    xr_allsujet, xr_allsujet_stats = compute_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allsujet()

    print('######## PLOT & SAVE TOPOPLOT ########', flush=True)
    plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet, xr_allsujet_stats)

    print('done', flush=True)






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    band_prep = 'wb'

    #### Pxx Cxy CycleFreq
    # compilation_compute_Pxx_Cxy_Cyclefreq_MVL()
    # execute_function_in_slurm_bash_mem_choice('n19_res_allsujet_power', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [nchan, nchan_name, band_prep], 15)

    #### TF & ITPC
    execute_function_in_slurm_bash_mem_choice('n19_res_allsujet_power', 'compute_TF_allsujet', [], 15)

    compilation_compute_TF_ITPC()
    # execute_function_in_slurm_bash_mem_choice('n19_res_allsujet_power', 'compilation_compute_TF_ITPC', [nchan, nchan_name, band_prep], 15)


