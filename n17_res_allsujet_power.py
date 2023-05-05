
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

import pickle
import gc

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False







################################################
######## PLOT & SAVE PSD AND COH ########
################################################




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
    xr_data = np.zeros(( len(sujet_list_compute), len(metric_to_load), len(conditions), len(odor_list), len(chan_list_eeg)))
    xr_dict = {'sujet' : sujet_list_compute, 'metric' : metric_to_load, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg}
    xr_allsujet = xr.DataArray(xr_data, dims=xr_dict.keys(), coords=xr_dict.values())  

    #sujet = sujet_list_compute[0]
    for sujet in sujet_list_compute:

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

    xr_allsujet = xr_allsujet.median(axis=0)
    
    return xr_allsujet




    

################################
######## TOPOPLOT ########
################################



def plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet):

    #### create montage
    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### scales
    scales_cond = {}
    scales_odor = {}
    scales_allband = {}

    data_type_list = [f'Pxx_{band}' for band in freq_band_fc_analysis.keys()] + ['MVL']

    #data_type = data_type_list[0]
    for data_type in data_type_list:

        if data_type == 'MVL':

            scales_cond[data_type] = {}
            scales_odor[data_type] = {}

            for odor_i in odor_list:

                scales_cond[data_type][odor_i] = {}

                val = np.array([])

                for cond in conditions:

                    val = np.append(val, xr_allsujet.loc[data_type, cond, odor_i])

                scales_cond[data_type][odor_i]['min'] = val.min()
                scales_cond[data_type][odor_i]['max'] = val.max()

            for cond in conditions:

                scales_odor[data_type][cond] = {}

                val = np.array([])

                for odor_i in odor_list:

                    val = np.append(val, xr_allsujet.loc[data_type, cond, odor_i])

                scales_odor[data_type][cond]['min'] = val.min()
                scales_odor[data_type][cond]['max'] = val.max()

        scales_allband[data_type] = {}

        val = np.array([])

        for odor_i in odor_list:

            for cond in conditions:

                val = np.append(val, xr_allsujet.loc[data_type, cond, odor_i])

        scales_allband[data_type]['min'] = val.min()
        scales_allband[data_type]['max'] = val.max()

    #### plot Cxy, MVL
    for data_type in ['raw', 'stats']:

        for metric_type in ['Cxy', 'MVL']:

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
            plt.suptitle(f'allsujet_{metric_type}_{data_type}')
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
                    
                    if metric_type == 'Cxy' and data_type == 'stats':
                        mne.viz.plot_topomap(xr_allsujet.loc[f'Cxy_surr', cond, odor_i], info, axes=ax, vmin=0, 
                                            vmax=1, show=False)
                        
                    if metric_type == 'Cxy' and data_type == 'raw':
                        mne.viz.plot_topomap(xr_allsujet.loc[f'Cxy', cond, odor_i], info, axes=ax, show=False)

                    if metric_type == 'MVL' and data_type == 'stats':
                        mne.viz.plot_topomap(xr_allsujet.loc[f'MVL_surr', cond, odor_i], info, axes=ax, show=False)
                        
                    if metric_type == 'MVL' and data_type == 'raw':
                        vmin = np.round(scales_allband['MVL']['min'], 2)
                        vmax = np.round(scales_allband['MVL']['max'], 2)
                        mne.viz.plot_topomap(xr_allsujet.loc[f'MVL', cond, odor_i], info, axes=ax, vmin=vmin, vmax=vmax, show=False)

            # plt.show() 

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
            fig.savefig(f'allsujet_{metric_type}_{data_type}_{band_prep}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()


    #band, freq = 'theta', [4, 8]
    for band, freq in freq_band_fc_analysis.items():

        #### plot Pxx
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'allsujet_{band}_Pxx')
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
                
                mne.viz.plot_topomap(xr_allsujet.loc[f'Pxx_{band}', cond, odor_i], info, axes=ax, vlim=(scales_allband[f'Pxx_{band}']['min'], 
                                     scales_allband[f'Pxx_{band}']['max']), show=False)

        # plt.show() 

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'PSD_Coh', 'topoplot'))
        fig.savefig(f'allsujet_Pxx_{band}_{band_prep}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()







########################################
######## PLOT & SAVE TF & ITPC ########
########################################

#tf = tf_plot
def get_tf_stats(tf, pixel_based_distrib):

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





    


def compilation_compute_TF_ITPC(sujet_list_compute):

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:

        if tf_mode == 'ITPC':
            continue

        if os.path.exists(os.path.join(path_precompute, 'allsujet', tf_mode, f'allsujet_{tf_mode}.nc')):
            os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
            xr_allsujet = xr.open_dataarray(f'allsujet_{tf_mode}.nc')

        else:

            #### generate xr
            os.chdir(path_memmap)
            data_xr = np.memmap(f'allsujet_tf_reduction.dat', dtype=np.float64, mode='w+', shape=(len(chan_list_eeg), len(conditions), len(odor_list), nfrex, stretch_point_TF))

            #nchan = 0
            def compute_TF_ITPC(nchan, tf_mode):

                print_advancement(nchan, len(chan_list_eeg), steps=[25, 50, 75])

                tf_median = np.zeros((len(sujet_list_compute),len(conditions),len(odor_list),nfrex,stretch_point_TF))

                #sujet_i, sujet = 0, sujet_list_compute[0]
                for sujet_i, sujet in enumerate(sujet_list_compute):

                    print_advancement(sujet_i, len(sujet_list_compute), steps=[25, 50, 75])

                    #cond_i, cond = 0, conditions[0]
                    for cond_i, cond in enumerate(conditions):
                        #odor_i, odor = 0, odor_list[0]
                        for odor_i, odor in enumerate(odor_list):

                            os.chdir(os.path.join(path_precompute, sujet, tf_mode))
                            tf_median[sujet_i, cond_i, odor_i, :, :] = np.median(np.load(f'{sujet}_tf_conv_{cond}_{odor}.npy')[nchan,:,:,:], axis=0)

                data_xr[nchan,:,:,:,:] = np.median(tf_median, axis=0)

                if debug:

                    os.chdir(path_general)
                    time = range(stretch_point_TF)
                    tf_plot = np.median(tf_median, axis=0)[0,0,:,:]
                    plt.pcolormesh(time, frex, tf_plot, shading='gouraud', cmap=plt.get_cmap('seismic'))
                    plt.yscale('log')
                    plt.savefig('test.jpg')

            joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(compute_TF_ITPC)(nchan, tf_mode) for nchan, nchan_name in enumerate(chan_list_eeg))

            dict_xr = {'nchan' : chan_list_eeg, 'cond' : conditions, 'odor' : odor_list, 'nfrex' : np.arange(0, nfrex), 'times' : np.arange(0, stretch_point_TF)}
            xr_allsujet = xr.DataArray(data_xr, coords=dict_xr.values(), dims=dict_xr.keys())

            os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))
            xr_allsujet.to_netcdf(f'allsujet_{tf_mode}.nc')

            os.chdir(path_memmap)
            os.remove(f'allsujet_tf_reduction.dat')
    
        #### scale
        vmin = np.array([])
        vmax = np.array([])

        for nchan, nchan_name in enumerate(chan_list_eeg):
        
            vals = np.array([])

            for cond in conditions:

                for odor_i in odor_list:

                    vals = np.append(vals, xr_allsujet.loc[nchan_name, cond, odor_i, :, :].values.reshape(-1))

            median_diff = np.percentile(np.abs(vals - np.median(vals)), 100-tf_plot_percentile_scale)

            vmin = np.append(vmin, np.median(vals) - median_diff)
            vmax = np.append(vmax, np.median(vals) + median_diff)

            del vals

        #### inspect stats
        if debug:

            tf_stats_type = 'intra'
            n_chan, chan_name = 0, chan_list_eeg[0]
            r, odor_i = 0, odor_list[0]
            c, cond = 1, conditions[1]

            tf_plot = xr_allsujet.loc[nchan_name, cond, odor_i, :, :]

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
            pixel_based_distrib = np.load(f'allsujet_{tf_mode.lower()}_STATS_{cond}_{odor_i}_intra.npy')[n_chan]

            plt.pcolormesh(time, frex, tf_plot, vmin=vmin, vmax=vmax, shading='gouraud', cmap=plt.get_cmap('seismic'))
            plt.yscale('log')
            plt.contour(time, frex, get_tf_stats(tf_plot, pixel_based_distrib), levels=0, colors='g')

            plt.show()

            plt.pcolormesh(get_tf_stats(tf_plot, pixel_based_distrib))
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

                        tf_plot = xr_allsujet.loc[chan_name, cond, odor_i, :, :]
                    
                        ax = axs[r,c]

                        if r == 0 :
                            ax.set_title(cond, fontweight='bold', rotation=0)

                        if c == 0:
                            ax.set_ylabel(odor_i)

                        ax.pcolormesh(time, frex, tf_plot, vmin=vmin[nchan], vmax=vmax[nchan], shading='gouraud', cmap=plt.get_cmap('seismic'))
                        ax.set_yscale('log')

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

                        ax.vlines(ratio_stretch_TF*stretch_point_TF, ymin=frex[0], ymax=frex[1], colors='g')
                        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])

                #plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode))

                #### save
                if tf_stats_type == 'inter':
                    fig.savefig(f'allsujet_{chan_name}_inter_{band_prep}.jpeg', dpi=150)
                if tf_stats_type == 'intra':
                    fig.savefig(f'allsujet_{chan_name}_intra_{band_prep}.jpeg', dpi=150)
                    
                fig.clf()
                plt.close('all')
                gc.collect()






########################################
######## COMPILATION FUNCTION ########
########################################

def compilation_compute_Pxx_Cxy_Cyclefreq_MVL():

    xr_allsujet = compute_Pxx_Cxy_Cyclefreq_MVL_Surrogates_allsujet()

    print('######## PLOT & SAVE TOPOPLOT ########')
    plot_save_PSD_Cxy_CF_MVL_TOPOPLOT(xr_allsujet)

    print('done')






################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    band_prep = 'wb'
    sujet_list_compute = sujet_list

    #### Pxx Cxy CycleFreq
    compilation_compute_Pxx_Cxy_Cyclefreq_MVL(sujet_list_compute)
    # execute_function_in_slurm_bash_mem_choice('n9_res_power', 'compilation_compute_Pxx_Cxy_Cyclefreq_MVL', [nchan, nchan_name, band_prep], 15)

    #### TF & ITPC
    compilation_compute_TF_ITPC(sujet_list_compute)
    # execute_function_in_slurm_bash_mem_choice('n9_res_power', 'compilation_compute_TF_ITPC', [nchan, nchan_name, band_prep], 15)


