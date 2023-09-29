
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

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0ter_stats import *
from n21bis_res_allsujet_PPI import *

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test

debug = False





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    #### ERP
    # cond_erp = ['VS', 'MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data, xr_data_sem = compute_ERP()
    df_stats_interintra = get_df_stats(xr_data)
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data)
    cluster_stats, cluster_stats_rep_norep = get_cluster_stats(xr_data)

    xr_PPI_count = get_PPI_count(xr_data)
    
    # shuffle_way = 'inter_cond'
    # shuffle_way = 'intra_cond'
    shuffle_way = 'linear_based'
    xr_surr = compute_surr_ERP(xr_data, shuffle_way)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr, xr_PPI_count)

    plot_ERP_response_profile(xr_data, xr_data_sem)

    plot_ERP_diff(xr_data, df_stats_interintra, cluster_stats)

    plot_ERP_rep_norep(xr_data, cluster_stats_rep_norep)

    plot_erp_response_stats(xr_data)

    #### plot PPI across subject
    plot_PPI_proportion(xr_PPI_count)

    #### respi
    plot_mean_respi()

    #### manual evaluation
    generate_ppi_evaluation(xr_data)

    #### reg 
    print(f'#### plot discomfort / slope ####', flush=True)
    plot_slope_versus_discomfort(xr_data, xr_lm_data)







################################
######## EXECUTE ########
################################




    if debug:

        sujet_no_respond_erp_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond_erp])

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        mat_adjacency = mne.channels.find_ch_adjacency(info, 'eeg')

        data_baseline = xr_data.loc[sujet_best_list_rev, 'CO2', '-', 'Cz', :].values
        data_cond = xr_data.loc[sujet_no_respond_erp_rev, 'CO2', '-', 'Cz', :].values

        data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_2', '-', 'Cz', :].values
        data_cond = xr_data.loc[sujet_no_respond_erp_rev, 'FR_CV_2', '-', 'Cz', :].values

        data_baseline = xr_data.loc[:, 'CO2', 'o', 'Fp1', :].values
        data_cond = xr_data.loc[:, 'CO2', '-', 'Fp1', :].values
                                
        T_obs, clusters_none, cluster_p_values_none, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=None,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )

        T_obs, clusters_thresh, cluster_p_values_thresh, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=2,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )

        threshold_tfce = dict(start=2, step=0.5)
        T_obs, clusters_tfce, cluster_p_values_tfce, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=threshold_tfce,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )

        cluster_type_list = ['none', 'thresh', 'TFCE']

        times = xr_data['time'].values

        fig, axs = plt.subplots(ncols=len(cluster_type_list))

        #cluster_type_i, cluster_type = 0, cluster_type[0]
        for cluster_type_i, cluster_type in enumerate(cluster_type_list):

            ax = axs[cluster_type_i]

            ax.set_title(cluster_type)

            ax.plot(times, data_baseline.mean(axis=0))
            ax.plot(times, data_cond.mean(axis=0))

            if cluster_type == 'none':
                for i_c, c in enumerate(clusters_none):
                    c = c[0]
                    if cluster_p_values_none[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            if cluster_type == 'thresh':
                for i_c, c in enumerate(clusters_thresh):
                    c = c[0]
                    if cluster_p_values_thresh[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            if cluster_type == 'TFCE':
                for i_c, c in enumerate(clusters_tfce):
                    if cluster_p_values_tfce[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

        plt.show()

        for sujet_i in range(data_cond.shape[0]):

            plt.plot(times, data_baseline[sujet_i,:], alpha=0.5, color='m')
            plt.plot(times, data_cond[sujet_i,:], alpha=0.5, color='c')

        plt.plot(times, data_baseline.mean(axis=0), color='r', label='o')
        plt.plot(times, data_cond.mean(axis=0), color='b', label='-')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

        #### manual shift
        data_baseline
        data_cond

        ttest_vec = np.zeros((data_cond.shape[-1]))
        for time_i in range(data_cond.shape[-1]): 
            ttest_vec[time_i] = pg.ttest(data_baseline[:,time_i], data_cond[:,time_i], paired=True, alternative='two-sided', correction='auto', confidence=0.95)['p-val'].values[0]

        T_obs, clusters_none, cluster_p_values_none, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=None,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )
        
        ttest_surr = np.zeros((n_surr, data_cond.shape[-1]))

        n_trials_baselines = data_baseline.shape[0]
        n_trials_cond = data_cond.shape[0]
        n_trials_min = np.array([n_trials_baselines, n_trials_cond]).min()

        n_surr = 100
        data_shuffle = np.concatenate((data_baseline, data_cond), axis=0)
        n_trial_tot = data_shuffle.shape[0]

        ttest_vec_shuffle = np.zeros((n_surr, data_shuffle.shape[-1]))

        for surr_i in range(n_surr):

            print_advancement(surr_i, n_surr, steps=[25, 50, 75])

            #### shuffle
            random_sel = np.random.choice(n_trial_tot, size=n_trial_tot, replace=False)
            data_shuffle_baseline = data_shuffle[random_sel[:n_trials_min]]
            data_shuffle_cond = data_shuffle[random_sel[n_trials_min:n_trials_min*2]]

            for time_i in range(data_shuffle.shape[-1]): 
                ttest_vec_shuffle[surr_i, time_i] = pg.ttest(data_shuffle_baseline[:,time_i], data_shuffle_cond[:,time_i], paired=False, alternative='two-sided', correction='auto', confidence=0.95)['p-val'].values[0]

            if debug:
                plt.plot(ttest_vec_shuffle, label='shuffle')
                plt.plot(ttest_vec, label='observed')
                plt.plot(T_obs, label='permut')
                plt.legend()
                plt.show()

            min_shape = np.array([n_trials_baselines, n_trials_cond]).min()
            draw_indicator = np.random.randint(low=0, high=2, size=min_shape)
            sel_baseline = np.random.choice(n_trials_baselines, size=(draw_indicator == 1).sum(), replace=False)
            sel_cond = np.random.choice(n_trials_cond, size=(draw_indicator == 0).sum(), replace=False)

            #### extract max min
            tf_shuffle[:len(sel_baseline),:,:] = tf_stretch_baselines[nchan, sel_baseline, :, :]
            tf_shuffle[len(sel_baseline):,:,:] = tf_stretch_cond[nchan, sel_cond, :, :]

            _min, _max = np.median(tf_shuffle, axis=0).min(axis=1), np.median(tf_shuffle, axis=0).max(axis=1)
            # _min, _max = np.percentile(np.median(tf_shuffle, axis=0), 1, axis=1), np.percentile(np.median(tf_shuffle, axis=0), 99, axis=1)
            
            pixel_based_distrib_i[:, surrogates_i, 0] = _min
            pixel_based_distrib_i[:, surrogates_i, 1] = _max

        plt.plot(ttest_vec)
        plt.show()



        cluster_type_list = range(1,10)

        cluster_res = []

        for thresh in cluster_type_list:
                                
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [data_baseline, data_cond],
                n_permutations=1000,
                threshold=thresh,
                tail=0,
                n_jobs=4,
                out_type="mask",
                verbose=None
            )

            cluster_res.append((clusters, cluster_p_values))


        times = xr_data['time'].values

        fig, axs = plt.subplots(ncols=len(cluster_type_list))

        #cluster_type_i, cluster_type = 0, cluster_type[0]
        for cluster_type_i, cluster_type in enumerate(cluster_type_list):

            ax = axs[cluster_type_i]

            ax.set_title(cluster_type)

            ax.plot(times,data_baseline.mean(axis=0))
            ax.plot(times,data_cond.mean(axis=0))

            clusters, cluster_p_values = cluster_res[cluster_type_i]

            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                else:
                    ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

        plt.show()

        n_conditions = 2
        n_observations = data_cond.shape[0]
        pval = 0.05  # arbitrary
        dfn = n_conditions - 1  # degrees of freedom numerator
        dfd = n_observations - 2  # degrees of freedom denominator
        thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

        plt.plot(times, T_obs)
        plt.hlines(thresh, xmin=times.min(), xmax=times.max(), color='r')
        plt.show()

        clusters = np.where(np.diff((T_obs > thresh)*1) != 0)[0].reshape(-1,2)

        fig, ax = plt.subplots()
        ax.plot(times, data_baseline.mean(axis=0), color='r', label='o')
        ax.plot(times, data_cond.mean(axis=0), color='b', label='-')
        ax.invert_yaxis()

        for c_i in range(clusters.shape[0]):

            h = ax.axvspan(times[clusters[c_i,0]], times[clusters[c_i,1]], color="r", alpha=0.3)
            
        plt.legend()
        plt.show()

        for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                else:
                    ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)



        data_baseline = xr_data.loc[:, 'FR_CV_1', '-', :, :].values
        data_cond = xr_data.loc[:, 'CO2', '-', :, :].values
                                
        T_obs, clusters_none, cluster_p_values_none, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=None,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )

        T_obs, clusters_thresh, cluster_p_values_thresh, H0 = permutation_cluster_test(
            [data_baseline, data_cond],
            n_permutations=1000,
            threshold=1,
            tail=0,
            n_jobs=4,
            out_type="mask",
            verbose=None
        )


        cluster_type_list = ['none', 'thresh']

        times = xr_data['time'].values

        fig, axs = plt.subplots(ncols=len(cluster_type_list))

        #cluster_type_i, cluster_type = 0, cluster_type[0]
        for cluster_type_i, cluster_type in enumerate(cluster_type_list):

            ax = axs[cluster_type_i]

            ax.set_title(cluster_type)

            ax.plot(times,data_baseline.mean(axis=0) - data_cond.mean(axis=0))

            if cluster_type == 'none':
                for i_c, c in enumerate(clusters_none):
                    c = c[0]
                    if cluster_p_values_none[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            if cluster_type == 'thresh':
                for i_c, c in enumerate(clusters_thresh):
                    c = c[0]
                    if cluster_p_values_thresh[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            if cluster_type == 'TFCE':
                for i_c, c in enumerate(clusters_tfce):
                    if cluster_p_values_tfce[i_c] <= 0.05:
                        h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    # else:
                    #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

        plt.show()





        cluster_type_list = np.arange(0,5,0.5)

        cluster_res = []

        for thresh in cluster_type_list:
                                
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [data_baseline, data_cond],
                n_permutations=1000,
                threshold=thresh,
                tail=0,
                n_jobs=4,
                out_type="mask",
                verbose=None
            )

            cluster_res.append((clusters, cluster_p_values))


        times = xr_data['time'].values

        fig, axs = plt.subplots(ncols=len(cluster_type_list))

        #cluster_type_i, cluster_type = 0, cluster_type[0]
        for cluster_type_i, cluster_type in enumerate(cluster_type_list):

            ax = axs[cluster_type_i]

            ax.set_title(cluster_type)

            ax.plot(times,data_baseline.mean(axis=0))
            ax.plot(times,data_cond.mean(axis=0))

            clusters, cluster_p_values = cluster_res[cluster_type_i]

            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                # else:
                #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

        plt.show()






        data_baseline = np.abs(xr_data.loc[sujet_best_list_rev, 'CO2', '-', :, :].max('time').values - xr_data.loc[sujet_best_list_rev, 'CO2', '-', :, :].min('time').values)
        data_cond = np.abs(xr_data.loc[sujet_no_respond_erp_rev, 'CO2', '-', :, :].max('time').values - xr_data.loc[sujet_no_respond_erp_rev, 'CO2', '-', :, :].min('time').values)

        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                [data_baseline, data_cond],
                n_permutations=1000,
                threshold=None,
                seed=None,
                tail=0,
                n_jobs=4,
                out_type="mask",
                adjacency=mat_adjacency[0],
                verbose=None
            )
            


            
        mask_signi = clusters[0]

        mask_params = dict(markersize=15, markerfacecolor='y')

        fig, axs = plt.subplots(ncols=2, figsize = (10,5))

        min, max = np.concatenate((data_baseline, data_cond), axis=0).min(), np.concatenate((data_baseline, data_cond), axis=0).max()

        for cond_i, cond in enumerate(['baseline', 'cond']):

            ax = axs[cond_i]

            if cond == 'baseline':
                data_plot = np.median(data_baseline, axis=0)

            else:
                data_plot = np.median(data_cond, axis=0)

            im, cn = mne.viz.plot_topomap(data=data_plot, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(min,max), colorbar=True)
        
        ax_x_start = 1
        ax_x_width = 0.04
        ax_y_start = 0.1
        ax_y_height = 0.9
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('Coherence',fontsize=15)
        
        plt.show()



        evokedArr =  mne.EvokedArray(xr_data.loc[sujet_best_list_rev, 'CO2', '-', :, :].median('sujet').values, info) 
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        evokedArr.set_montage(ten_twenty_montage)


        fig, ax = plt.subplots(1, 4, gridspec_kw=dict(width_ratios=[5, 1, 5, 1]))

        evokedArr.plot_topomap(times=0, time_unit='s', time_format=None,
                            axes = ax[:2], cmap='Spectral_r', colorbar=True,
                            show=False)
        ax[0].set_title('Delta')

        evokedArr.data *= 10  # to make clear that the colorbars are different
        evokedArr.plot_topomap(times=0, time_unit='s', time_format=None,
                            axes = ax[2:], cmap='Spectral_r', colorbar = True,
                            show=False)
        ax[2].set_title('Theta')
        plt.show()

