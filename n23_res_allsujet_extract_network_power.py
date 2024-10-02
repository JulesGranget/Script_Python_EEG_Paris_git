
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
from sklearn import linear_model
import gc
import xarray as xr
import seaborn as sns
import pickle
import cv2

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0ter_stats import *
from n21bis_res_allsujet_ERP import *

debug = False




########################################
######## STATS FUNCTIONS ########
########################################

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


def generate_df_extract_power():

    if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_pxx.xlsx')):

        os.chdir(os.path.join(path_results, 'allplot', 'df'))
        df_pxx = pd.read_excel('df_pxx.xlsx')

        return df_pxx

    else: 

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        xr_allsujet = xr.open_dataarray(f'allsujet_TF.nc')

        band_list = {'theta' : [4,8], 'alpha' : [8,12], 'beta' : [12,30], 'gamma' : [50,120]}
        phase_list = {'all' : xr_allsujet['times'].values.astype('int'), 'inspi' : np.arange(xr_allsujet['times'].shape[0]/2).astype('int'), 'expi' : np.arange(xr_allsujet['times'].shape[0]/2, xr_allsujet['times'].shape[0]).astype('int')}
        nfrex_array = np.arange(xr_allsujet['nfrex'].shape[0])

        mask_params = dict(markersize=15, markerfacecolor='y')

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        df_pxx = pd.DataFrame({'sujet' : [], 'nchan' : [], 'cond' : [], 'odor' : [], 'band' : [], 'phase' : [], 'Pxx' : []})

        for sujet in sujet_list:

            print(sujet)

            for nchan_i, nchan in enumerate(xr_allsujet['nchan'].values):

                for cond in xr_allsujet['cond'].values:

                    for odor in xr_allsujet['odor'].values:

                        os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                        _tf = np.median(np.load(f'{sujet}_tf_conv_{cond}_{odor}.npy')[nchan_i,:,:,:], axis=0)

                        for band in band_list.keys():

                            for phase in phase_list.keys():

                                mask_band = nfrex_array[(nfrex_array <= band_list[band][1]) & (nfrex_array >= band_list[band][0])]

                                _pxx = np.median(_tf[mask_band, :][:, phase_list[phase]])

                                df_pxx = pd.concat([df_pxx, pd.DataFrame({'sujet' : [sujet], 'nchan' : [nchan], 'cond' : [cond], 'odor' : [odor], 'band' : [band], 'phase' : [phase], 'Pxx' : [_pxx]})])

        group_data = []

        for row_i in range(df_pxx.shape[0]):

            if df_pxx.iloc[row_i,:]['sujet'] in sujet_best_list_rev:

                group_data.append('rep')

            if df_pxx.iloc[row_i,:]['sujet'] not in sujet_best_list_rev:

                group_data.append('no_rep')

        df_pxx['group'] = group_data

        os.chdir(os.path.join(path_results, 'allplot', 'df'))
        df_pxx.to_excel('df_pxx.xlsx')

        return df_pxx




def generate_df_extract_sum():

    if os.path.exists(os.path.join(path_results, 'allplot', 'df', 'df_tf_sum.xlsx')):

        os.chdir(os.path.join(path_results, 'allplot', 'df'))
        df_pxx = pd.read_excel('df_tf_sum.xlsx')

        return df_pxx

    else: 

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        xr_allsujet = xr.open_dataarray(f'allsujet_TF.nc')

        group_list = ['allsujet', 'rep', 'no_rep']
        time = np.arange(stretch_point_TF)

        tf_mode = 'TF'

        band_list = {'theta' : [4, 8], 'alpha' : [8, 12], 'beta' : [12, 30], 'gamma' : [50, 150]}
        respi_phase_list = {'all' : time, 'inspi' : np.arange(time.shape[0]/2).astype('int'), 'expi' : np.arange(time.shape[0]/2, time.shape[0]).astype('int')}

        stats_pxx_dict = {'stat_type' : ['inter', 'intra'], 'group' : group_list, 'band' : list(band_list), 'cond' : conditions, 'odor' : odor_list, 'phase' : list(respi_phase_list), 'chan' : chan_list_eeg}
        stats_pxx_data = np.zeros((2, len(group_list), len(band_list), len(conditions), len(odor_list), len(respi_phase_list), len(chan_list_eeg)))

        #tf_stats_type = 'inter'
        for tf_stats_type_i, tf_stats_type in enumerate(['inter', 'intra']):

            #group = group_list[0]
            for group_i, group in enumerate(group_list):

                #odor_i, odor = 1, odor_list[1]
                for odor_i, odor in enumerate(odor_list):

                    print(tf_stats_type, group, odor)

                    #cond_i, cond = 1, conditions[1]
                    for cond_i, cond in enumerate(conditions):

                        #n_chan, chan_name = 0, chan_list_eeg[0]
                        for n_chan, chan_name in enumerate(chan_list_eeg):

                            tf_plot = xr_allsujet.loc[chan_name, group, cond, odor, :, :].values

                            if cond != 'FR_CV_1' and tf_stats_type == 'intra':
                                os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_intra.npy'), axis=1)
                                
                            elif odor != 'o' and tf_stats_type == 'inter':
                                os.chdir(os.path.join(path_precompute, 'allsujet', tf_mode))

                                pixel_based_distrib = np.median(np.load(f'allsujet_tf_STATS_nchan{n_chan}_{cond}_{odor}_inter.npy'), axis=1)

                            else:

                                continue
                                
                            tf_stats = get_tf_stats(tf_plot, pixel_based_distrib).astype('bool')

                            #band_i, band = 0, band_list[0]
                            for band_i, band in enumerate(band_list.keys()):

                                #phase_i, phase = 0, 'all'
                                for phase_i, phase in enumerate(respi_phase_list.keys()):

                                    tf_stats_chunk = tf_stats[band_list[band][0]:band_list[band][1],respi_phase_list[phase]]
                                    tf_plot_chunk = tf_plot[band_list[band][0]:band_list[band][1],respi_phase_list[phase]]

                                    stats_pxx_data[tf_stats_type_i, group_i, band_i, cond_i, odor_i, phase_i, n_chan] = tf_plot_chunk[tf_stats_chunk].sum()

                                    
        xr_stats_pxx = xr.DataArray(data=stats_pxx_data, dims=stats_pxx_dict.keys(), coords=stats_pxx_dict.values())
        df_stats_pxx = xr_stats_pxx.to_dataframe(name='Pxx').reset_index(drop=False)

        os.chdir(os.path.join(path_results, 'allplot', 'df'))
        df_stats_pxx.to_excel('df_tf_sum.xlsx')




def plot_network_TF_IE():

    tf_mode = 'TF'

    print('COMPUTE', flush=True)

    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    df_pxx = pd.read_excel('df_pxx.xlsx')

    group_list = ['allsujet', 'rep', 'no_rep']
    band_list = ['theta', 'alpha', 'gamma']
    respi_phase_list = ['all', 'inspi', 'expi']
    time = np.arange(stretch_point_TF)
    chan_list_lobes = {'allregion' : chan_list,
                    'frontal' : ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2'], 
                   'parietal': ['C3', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4']}

    #### scale allsujet
    ylim = {}

    for band_i, band in enumerate(band_list):

        ylim[band] = {}

        _data = df_pxx.query(f"phase != 'all' and band == '{band}'")['Pxx'].values

        ylim[band]['min'] = -1.5*_data.std()
        ylim[band]['max'] = 1.5*_data.std()


    #### plot allsujet
    pd.DataFrame.iteritems = pd.DataFrame.items

    for region in chan_list_lobes.keys():

        for group in group_list:

            if group == 'allsujet':
                group_sel = ['rep', 'no_rep']
            else:
                group_sel = [group]

            for band_i, band in enumerate(band_list):

                print(group, band)

                df_plot = df_pxx.query(f"phase != 'all' and group in {group_sel} and band == '{band}' and nchan in {chan_list_lobes[region]}")    

                g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'odor', palette='flare')
                plt.legend()
                plt.suptitle(f"{region} {group} {band}")
                plt.tight_layout()
                plt.ylim(ylim[band]['min'],ylim[band]['max'])
                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'figure_stats_pxx_tf'))

                plt.savefig(f'{region}_{group}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()


    




def plot_network_TF_IE_sum():

    tf_mode = 'TF'

    print('COMPUTE', flush=True)

    os.chdir(os.path.join(path_results, 'allplot', 'df'))
    df_pxx = pd.read_excel('df_tf_sum.xlsx')

    group_list = ['allsujet', 'rep', 'no_rep']
    band_list = ['theta', 'alpha', 'gamma']
    respi_phase_list = ['all', 'inspi', 'expi']
    time = np.arange(stretch_point_TF)
    chan_list_lobes = {'allregion' : chan_list,
                    'frontal' : ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2'], 
                   'parietal': ['C3', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4']}
    
    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### sum for nchan
    df_pxx_region = pd.DataFrame({'region' : [], 'stat_type' : [], 'group' : [], 'band' : [], 'cond' : [], 'odor' : [], 'phase' : [], 'Pxx' : []})

    for region in chan_list_lobes.keys():

        region_sel = chan_list_lobes[region]

        print(region)

        for stat_type in df_pxx['stat_type'].unique():

            for group in df_pxx['group'].unique():

                for band in df_pxx['band'].unique():

                    for cond in df_pxx['cond'].unique():

                        for odor in df_pxx['odor'].unique():

                            for phase in df_pxx['phase'].unique():

                                _sum = df_pxx.query(f"stat_type == '{stat_type}' and group == '{group}' and band == '{band}' and cond == '{cond}' and odor == '{odor}' and phase == '{phase}' and chan in {region_sel}")['Pxx'].sum()
                                df_pxx_region = pd.concat([df_pxx_region, pd.DataFrame({'region' : [region], 'stat_type' : [stat_type], 'group' : [group], 'band' : [band], 'cond' : [cond], 'odor' : [odor], 'phase' : [phase], 'Pxx' : [_sum]})])


    #### scale allsujet
    # ylim = {}

    # for band_i, band in enumerate(band_list):

    #     ylim[band] = {}

    #     _data = df_pxx.query(f"phase != 'all' and band == '{band}'")['Pxx'].values

    #     ylim[band]['min'] = -1.5*_data.std()
    #     ylim[band]['max'] = 1.5*_data.std()

    ylim = {}

    for stat_type in df_pxx['stat_type'].unique():

        ylim[stat_type] = {}

        for band_i, band in enumerate(band_list):

            ylim[stat_type][band] = {}

            for region in chan_list_lobes.keys():

                ylim[stat_type][band][region] = {}

                for group in ['allsujet', 'repnorep']:

                    ylim[stat_type][band][region][group] = {}

                    if group == 'allsujet':
                        group_sel = ['allsujet']
                    if group == 'repnorep':
                        group_sel = ['rep', 'no_rep']

                    _data = df_pxx_region.query(f"stat_type == '{stat_type}' and region == '{region}' and phase != 'all' and band == '{band}' and group in {group_sel}")['Pxx'].values

                    ylim[stat_type][band][region][group]['min'] = _data.min() - 0.1*_data.min()
                    ylim[stat_type][band][region][group]['max'] = _data.max() + 0.1*_data.max()
                    ylim[stat_type][band][region][group]['vlim'] = np.absolute([_data.min(), _data.max()]).max()

    #### plot allsujet intra
    pd.DataFrame.iteritems = pd.DataFrame.items

    stat_type = 'intra'

    for region in chan_list_lobes.keys():

        for group in group_list:

            #band_i, band = 0, 'theta'
            for band_i, band in enumerate(band_list):

                if group == 'allsujet':
                    group_sel = 'allsujet'
                else:
                    group_sel = 'repnorep'

                print(group, band)

                df_plot = df_pxx_region.query(f"cond in ['MECA', 'CO2', 'FR_CV_2'] and stat_type == '{stat_type}' and phase != 'all' and group == '{group}' and band == '{band}' and region == '{region}'")

                g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'odor', palette='flare')
                plt.legend()
                plt.suptitle(f"{region} {group} {band}")
                plt.tight_layout()

                # plt.ylim(ylim[stat_type][band][region][group_sel]['min'],ylim[stat_type][band][region][group_sel]['max'])

                plt.ylim(-ylim[stat_type][band][region][group_sel]['vlim'],ylim[stat_type][band][region][group_sel]['vlim'])

                if df_plot['Pxx'].max() > ylim[stat_type][band][region][group_sel]['vlim'] or df_plot['Pxx'].min() < -ylim[stat_type][band][region][group_sel]['vlim']:
                    raise ValueError()

                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'figure_stats_sum_tf'))

                plt.savefig(f'{stat_type}_{region}_{group}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()
    
    #### plot allsujet inter
    pd.DataFrame.iteritems = pd.DataFrame.items

    stat_type = 'inter'

    for region in chan_list_lobes.keys():

        for group in group_list:

            #band_i, band = 0, 'theta'
            for band_i, band in enumerate(band_list):

                if group == 'allsujet':
                    group_sel = 'allsujet'
                else:
                    group_sel = 'repnorep'

                print(group, band)

                df_plot = df_pxx_region.query(f"odor in ['+', '-'] and stat_type == '{stat_type}' and phase != 'all' and group == '{group}' and band == '{band}' and region == '{region}'")

                g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'odor', palette='flare')
                plt.legend()
                plt.suptitle(f"{region} {group} {band}")
                plt.tight_layout()

                # plt.ylim(ylim[stat_type][band][region][group_sel]['min'],ylim[stat_type][band][region][group_sel]['max'])

                plt.ylim(-ylim[stat_type][band][region][group_sel]['vlim'],ylim[stat_type][band][region][group_sel]['vlim'])

                if df_plot['Pxx'].max() > ylim[stat_type][band][region][group_sel]['vlim'] or df_plot['Pxx'].min() < -ylim[stat_type][band][region][group_sel]['vlim']:
                    raise ValueError()

                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'figure_stats_sum_tf'))

                plt.savefig(f'{stat_type}_{region}_{group}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()

    #### plot repnorep inter
    pd.DataFrame.iteritems = pd.DataFrame.items

    stat_type = 'inter'

    for region in chan_list_lobes.keys():

        for cond in conditions:

            #band_i, band = 0, 'theta'
            for band_i, band in enumerate(band_list):

                group_sel = 'repnorep'

                print(group, band)

                df_plot = df_pxx_region.query(f"cond == '{cond}' and odor in ['+', '-'] and stat_type == 'inter' and phase != 'all' and group in ['rep', 'no_rep'] and band == '{band}' and region == '{region}'")

                g = sns.FacetGrid(df_plot, col="odor", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'group', palette='flare')
                plt.legend()
                plt.suptitle(f"{region} {band} {cond}")
                plt.tight_layout()

                # plt.ylim(ylim[stat_type][band][region][group_sel]['min'],ylim[stat_type][band][region][group_sel]['max'])

                plt.ylim(-ylim[stat_type][band][region][group_sel]['vlim'],ylim[stat_type][band][region][group_sel]['vlim'])

                if df_plot['Pxx'].max() > ylim[stat_type][band][region][group_sel]['vlim'] or df_plot['Pxx'].min() < -ylim[stat_type][band][region][group_sel]['vlim']:
                    raise ValueError()

                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'figure_stats_sum_tf'))

                plt.savefig(f'repnorep_{stat_type}_{region}_{cond}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()

    #### plot repnorep intra
    pd.DataFrame.iteritems = pd.DataFrame.items

    stat_type = 'intra'

    for region in chan_list_lobes.keys():

        for odor in odor_list:

            #band_i, band = 0, 'theta'
            for band_i, band in enumerate(band_list):

                group_sel = 'repnorep'

                print(group, band)

                df_plot = df_pxx_region.query(f"odor == '{odor}' and stat_type == '{stat_type}' and phase != 'all' and group in ['rep', 'no_rep'] and band == '{band}' and region == '{region}'")

                g = sns.FacetGrid(df_plot, col="cond", height=5, aspect=1)
                g.map(sns.barplot, "phase", "Pxx", 'group', palette='flare')
                plt.legend()
                plt.suptitle(f"{region} {band} {odor}")
                plt.tight_layout()

                # plt.ylim(ylim[stat_type][band][region][group_sel]['min'],ylim[stat_type][band][region][group_sel]['max'])

                plt.ylim(-ylim[stat_type][band][region][group_sel]['vlim'],ylim[stat_type][band][region][group_sel]['vlim'])

                if df_plot['Pxx'].max() > ylim[stat_type][band][region][group_sel]['vlim'] or df_plot['Pxx'].min() < -ylim[stat_type][band][region][group_sel]['vlim']:
                    raise ValueError()

                # plt.show()

                os.chdir(os.path.join(path_results, 'allplot', tf_mode, 'Pxx', 'figure_stats_sum_tf'))

                plt.savefig(f'repnorep_{stat_type}_{region}_{odor}_{band}.jpeg', dpi=150)
                    
                plt.close('all')
                gc.collect()

    #### plot topo
    ylim = {}

    for stat_type in df_pxx['stat_type'].unique():

        ylim[stat_type] = {}

        for band_i, band in enumerate(band_list):

            ylim[stat_type][band] = {}

            for group in ['allsujet', 'repnorep']:

                ylim[stat_type][band][group] = {}

                if group == 'allsujet':
                    group_sel = ['allsujet']
                if group == 'repnorep':
                    group_sel = ['rep', 'no_rep']

                _data = df_pxx.query(f"stat_type == '{stat_type}' and phase != 'all' and band == '{band}' and group in {group_sel}")['Pxx'].values

                ylim[stat_type][band][group]['min'] = _data.min() - 0.1*_data.min()
                ylim[stat_type][band][group]['max'] = _data.max() + 0.1*_data.max()
                ylim[stat_type][band][group]['vlim'] = np.absolute([_data.min(), _data.max()]).max()

    
    #### plot allsujet intra
    stat_type = 'intra'
    group = 'allsujet'

    for band in band_list:
        
        for odor_i, odor in enumerate(['o', '+', '-']):

            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

                for phase_i, phase in enumerate(['inspi', 'expi']):

                    print(stat_type, phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        data_topoplot[chan_i] = df_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{stat_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                        # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                        # mask_signi[chan_i] = _p.astype('bool')

                        if data_topoplot[chan_i] != 0:
                            mask_signi[chan_i] = True
                        else:
                            mask_signi[chan_i] = False
                            

                    ax = axs[cond_i, phase_i]

                    ax.set_title(f"{cond} {phase} {group}")

                    # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                    #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim))
                    
                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask_params=mask_params, vlim=(-ylim[stat_type][band]['allsujet']['vlim'], ylim[stat_type][band]['allsujet']['vlim']))

            plt.tight_layout()

            plt.suptitle(f"{odor} {band} {stat_type} {np.round(ylim[stat_type][band]['allsujet']['vlim'],2)}")

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
            fig.savefig(f"SUM_allsujet_{stat_type}_{band}_{odor}.jpeg")

            plt.close('all')
            
            # plt.show()

    #### plot allsujet inter
    stat_type = 'inter'
    group = 'allsujet'

    for band in band_list:
        
        for cond_i, cond in enumerate(conditions):

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

            for odor_i, odor in enumerate(['+', '-']):    

                for phase_i, phase in enumerate(['inspi', 'expi', 'all']):

                    print(stat_type, phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        data_topoplot[chan_i] = df_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{stat_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                        # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                        # mask_signi[chan_i] = _p.astype('bool')

                        if data_topoplot[chan_i] != 0:
                            mask_signi[chan_i] = True
                        else:
                            mask_signi[chan_i] = False                            

                    ax = axs[odor_i, phase_i]

                    ax.set_title(f"{odor} {phase} {group}")

                    # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                    #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim))
                    
                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask_params=mask_params, vlim=(-ylim[stat_type][band]['allsujet']['vlim'], ylim[stat_type][band]['allsujet']['vlim']))                    

            plt.tight_layout()

            plt.suptitle(f"{cond} {band} {stat_type} {np.round(ylim[stat_type][band]['allsujet']['vlim'],2)}")

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
            fig.savefig(f"SUM_allsujet_{stat_type}_{band}_{cond}.jpeg")

            plt.close('all')
            
            # plt.show()

    for band in band_list:
        
        for cond_i, cond in enumerate(conditions):

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,15))

            for odor_i, odor in enumerate(['+', '-']):    

                for phase_i, phase in enumerate(['all']):

                    print(stat_type, phase, odor, cond)

                    mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                    data_topoplot = np.zeros((len(chan_list_eeg)))

                    #chan_i, chan = 0, chan_list_eeg[0]
                    for chan_i, chan in enumerate(chan_list_eeg):

                        data_topoplot[chan_i] = df_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{stat_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                        # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                        # mask_signi[chan_i] = _p.astype('bool')

                        if data_topoplot[chan_i] != 0:
                            mask_signi[chan_i] = True
                        else:
                            mask_signi[chan_i] = False                            

                    ax = axs[odor_i]

                    ax.set_title(f"{odor} {phase} {group}")

                    # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                    #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim))
                    
                    mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                    mask_params=mask_params, vlim=(-ylim[stat_type][band]['allsujet']['vlim'], ylim[stat_type][band]['allsujet']['vlim']))                    

            plt.tight_layout()

            plt.suptitle(f"{cond} {band} {stat_type} {np.round(ylim[stat_type][band]['allsujet']['vlim'],2)}")

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
            fig.savefig(f"SUM_allsujet_{stat_type}_{band}_{cond}_allphase.jpeg")

            plt.close('all')
            
            # plt.show()
    
    #### plot rep norep intra
    stat_type = 'intra'

    for band in band_list:

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):    

            fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15,15))

            for odor_i, odor in enumerate(['o', '+', '-']):

                x_i = -1

                for phase_i, phase in enumerate(['inspi', 'expi']):

                    for group in ['rep', 'no_rep']:

                        x_i += 1

                        print('intra', phase, odor, cond)

                        mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                        data_topoplot = np.zeros((len(chan_list_eeg)))

                        #chan_i, chan = 0, chan_list_eeg[0]
                        for chan_i, chan in enumerate(chan_list_eeg):

                            data_topoplot[chan_i] = df_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{stat_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                            # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                            # mask_signi[chan_i] = _p.astype('bool')

                            if data_topoplot[chan_i] != 0:
                                mask_signi[chan_i] = True
                            else:
                                mask_signi[chan_i] = False
                                

                        ax = axs[odor_i, x_i]

                        ax.set_title(f"{odor} {phase} {group}")

                        # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                        #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim))
                        
                        mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        mask_params=mask_params, vlim=(-ylim[stat_type][band]['repnorep']['vlim'], ylim[stat_type][band]['repnorep']['vlim']))

            plt.tight_layout()

            plt.suptitle(f"{cond} {band} {stat_type} {np.round(ylim[stat_type][band]['repnorep']['vlim'],2)}")

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
            fig.savefig(f"SUM_repnorep_{stat_type}_{band}_{cond}.jpeg")

            plt.close('all')
            
            # plt.show()

    #### plot repnorep inter
    stat_type = 'inter'

    for band in band_list:

        for odor_i, odor in enumerate(['+', '-']):    

            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15,15))

            for cond_i, cond in enumerate(conditions):

                x_i = -1

                for phase_i, phase in enumerate(['inspi', 'expi']):

                    for group in ['rep', 'no_rep']:

                        x_i += 1

                        print('intra', phase, odor, cond)

                        mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')
                        data_topoplot = np.zeros((len(chan_list_eeg)))

                        #chan_i, chan = 0, chan_list_eeg[0]
                        for chan_i, chan in enumerate(chan_list_eeg):

                            data_topoplot[chan_i] = df_pxx.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{stat_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]

                            # _p = df_stats_pxx_p.query(f"odor == '{odor}' and phase == '{phase}' and group == '{group}' and stat_type == '{tf_stats_type}' and cond == '{cond}' and band == '{band}' and chan == '{chan}'")['Pxx'].values[0]
                            # mask_signi[chan_i] = _p.astype('bool')

                            if data_topoplot[chan_i] != 0:
                                mask_signi[chan_i] = True
                            else:
                                mask_signi[chan_i] = False
                                

                        ax = axs[cond_i, x_i]

                        ax.set_title(f"{cond} {phase} {group}")

                        # mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                        #                 mask=mask_signi, mask_params=mask_params, vlim=(0, vlim))
                        
                        mne.viz.plot_topomap(data=data_topoplot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                        mask_params=mask_params, vlim=(-ylim[stat_type][band]['repnorep']['vlim'], ylim[stat_type][band]['repnorep']['vlim']))

            plt.tight_layout()

            plt.suptitle(f"{odor} {band} {stat_type} {np.round(ylim[stat_type][band]['repnorep']['vlim'],2)}")

            os.chdir(os.path.join(path_results, 'allplot', 'TF', 'Pxx', 'topoplot'))
            fig.savefig(f"SUM_repnorep_{stat_type}_{band}_{odor}.jpeg")

            plt.close('all')
            
            # plt.show()







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    generate_df_extract_power()

    generate_df_extract_sum()

    plot_network_TF_IE()

    plot_network_TF_IE_sum()


