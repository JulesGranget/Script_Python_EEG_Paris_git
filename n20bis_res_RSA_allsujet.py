

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import seaborn as sns
import pickle
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *


debug = False










########################################
######## LOAD DATA ########
########################################



def get_xr_data_RSA():

    time_vec = np.arange(500)

    xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'time' : time_vec}
    xr_data_raw = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())
    xr_data = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

    for sujet_i, sujet in enumerate(sujet_list):

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
                    
        with open(f'{sujet}_RSA.pkl', 'rb') as f:
            RSA_allcond = pickle.load(f)

        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(conditions):

                xr_data_raw[sujet_i, cond_i, odor_i, :] = RSA_allcond[odor][cond].mean(axis=0)

    #### zscore
    for sujet_i, sujet in enumerate(sujet_list):

        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(conditions):

                xr_data[sujet_i, cond_i, odor_i, :] = (xr_data_raw[sujet_i, cond_i, odor_i, :] - xr_data_raw[:, cond_i, odor_i, :].values.mean()) / xr_data_raw[:, cond_i, odor_i, :].values.std()

    return xr_data





########################################
######## COMPUTE RSA PERM ########
########################################


def get_df_stats(xr_data):

    if os.path.exists(os.path.join(path_results, 'allplot', 'RSA', 'df_stats_all_intra.xlsx')):

        os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

        print('ALREADY COMPUTED', flush=True)

        df_stats_all_intra = pd.read_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter = pd.read_excel('df_stats_all_inter.xlsx')
        df_stats_all_repnorep = pd.read_excel('df_stats_all_repnorep.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter, 'repnorep' : df_stats_all_repnorep}

    else:

        sujet_group = ['allsujet', 'rep', 'non_rep']

        #### generate df
        df_min = xr_data.min('time').to_dataframe(name='val').reset_index(drop=False)
        df_max = xr_data.max('time').to_dataframe(name='val').reset_index(drop=False)

        df_minmax = df_min.copy()
        df_minmax['val'] = np.abs(df_max['val'].values - df_min['val'].values)

        #### plot
        os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

        #### inter
        predictor = 'odor'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"cond == '{cond}'")
                    df_stats = df_minmax.query(f"cond == '{cond}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and sujet in {sujet_best_list_rev.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and sujet in {sujet_no_respond_rev.tolist()}")
    
                if group_i + cond_i == 0:
                    df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                    df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                    df_stats_all.insert(0, 'cond', np.array([cond]*df_stats_all.shape[0]))

                else:
                    _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                    _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                    _df_stats_all.insert(0, 'cond', np.array([cond]*_df_stats_all.shape[0]))
                    df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['inter'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_inter = df_stats_all.copy()

        #### intra
        predictor = 'cond'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"odor == '{odor}'")
                    df_stats = df_minmax.query(f"odor == '{odor}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and sujet in {sujet_best_list_rev.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and sujet in {sujet_no_respond_rev.tolist()}")
    
                if group_i + odor_i == 0:
                    df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                    df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                    df_stats_all.insert(0, 'odor', np.array([odor]*df_stats_all.shape[0]))

                else:
                    _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                    _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                    _df_stats_all.insert(0, 'odor', np.array([odor]*_df_stats_all.shape[0]))
                    df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['intra'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_intra = df_stats_all.copy()
        
        #### repnorep
        predictor = 'rep'
        outcome = 'val'

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                df_stats_rep = df_minmax.query(f"odor == '{odor}' and sujet in {sujet_best_list_rev.tolist()} and cond == '{cond}'")
                df_stats_rep['rep'] = [True]*df_stats_rep.shape[0]
                df_stats_norep = df_minmax.query(f"odor == '{odor}' and sujet in {sujet_no_respond_rev.tolist()} and cond == '{cond}'")
                df_stats_norep['rep'] = [False]*df_stats_norep.shape[0]

                df_stats = pd.concat([df_stats_rep, df_stats_norep])
    
                if cond_i + odor_i == 0:
                    df_stats_all = get_df_stats_pre(df_stats, predictor, outcome, subject='sujet', design='between', transform=False, verbose=False)[['test', 'alternative', 'p-val']]
                    df_stats_all.insert(0, 'cond', np.array([cond]*df_stats_all.shape[0]))
                    df_stats_all.insert(0, 'odor', np.array([odor]*df_stats_all.shape[0]))

                else:
                    _df_stats_all = get_df_stats_pre(df_stats, predictor, outcome, subject='sujet', design='between', transform=False, verbose=False)[['test', 'alternative', 'p-val']]
                    _df_stats_all.insert(0, 'cond', np.array([cond]*_df_stats_all.shape[0]))
                    _df_stats_all.insert(0, 'odor', np.array([odor]*_df_stats_all.shape[0]))
                    df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_repnorep = df_stats_all.copy()

        #### export
        df_stats_all_intra = df_stats_all_intra.query(f"A == 'FR_CV_1' or B == 'FR_CV_1'")
        df_stats_all_inter = df_stats_all_inter.query(f"A == 'o' or B == 'o'")

        os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

        df_stats_all_intra.to_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter.to_excel('df_stats_all_inter.xlsx')
        df_stats_all_repnorep.to_excel('df_stats_all_repnorep.xlsx')

        df_stats_all_intra.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'FR_CV_1' or B == 'FR_CV_1'").to_excel('df_stats_all_intra_signi.xlsx')
        df_stats_all_inter.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'o' or B == 'o'").to_excel('df_stats_all_inter_signi.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter, 'repnorep' : df_stats_all_repnorep}

    return df_stats_all





########################################
######## COMPUTE RSA PERM ########
########################################



def compute_cluster_RSA(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'RSA', 'RSA_cluster_stats_manual_perm.pkl')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'RSA'))

        print('ALREADY COMPUTED', flush=True)

        with open('RSA_cluster_stats_manual_perm.pkl', 'rb') as fp:
            cluster_stats = pickle.load(fp)

        with open('RSA_cluster_stats_rep_norep_manual_perm.pkl', 'rb') as fp:
            cluster_stats_rep_norep = pickle.load(fp)

    else:

        time_vec = np.arange(500)

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        conditions_diff = ['MECA', 'CO2', 'FR_CV_2']
        odor_diff = ['+', '-']
        sujet_group = ['allsujet', 'rep', 'non_rep']

        cluster_stats = {}

        cluster_stats['intra'] = {}

        #group = sujet_group[0]
        for group in sujet_group:

            cluster_stats['intra'][group] = {}

            print(group)

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                cluster_stats['intra'][group][odor] = {}

                #cond = conditions_diff[0]
                for cond in conditions_diff:

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :].values
                        data_cond = xr_data.loc[:, cond, odor, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :].values

                    if debug:

                        plt.plot(data_baseline.mean(axis=0))
                        plt.plot(data_cond.mean(axis=0))
                        plt.show()

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

                    cluster_stats['intra'][group][odor][cond] = mask

        cluster_stats['inter'] = {}

        for group in sujet_group:

            cluster_stats['inter'][group] = {}

            print(group)

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                cluster_stats['inter'][group][cond] = {}

                for odor in odor_diff:

                    cluster_stats['inter'][group][cond][odor] = {}

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, cond, 'o', :].values
                        data_cond = xr_data.loc[:, cond, odor, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :].values

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

                    cluster_stats['inter'][group][cond][odor] = mask

        os.chdir(os.path.join(path_precompute, 'allsujet', 'RSA'))
        
        with open('RSA_cluster_stats_manual_perm.pkl', 'wb') as fp:
            pickle.dump(cluster_stats, fp)

        cluster_stats_rep_norep = {}

        #odor_i, odor = 2, odor_list[2]
        for odor_i, odor in enumerate(odor_list):

            cluster_stats_rep_norep[odor] = {}

            for cond in conditions:

                data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, :].values
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :].values

                mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate)

                cluster_stats_rep_norep[odor][cond] = mask

        os.chdir(os.path.join(path_precompute, 'allsujet', 'RSA'))
        
        with open('RSA_cluster_stats_rep_norep_manual_perm.pkl', 'wb') as fp:
            pickle.dump(cluster_stats_rep_norep, fp)

    return cluster_stats, cluster_stats_rep_norep









################################
######## PLOT RSA ########
################################


def plot_RSA_diff(xr_data, cluster_stats, cluster_stats_rep_norep, df_stats_all):

    time_vec = np.arange(500)
    
    sujet_group = ['allsujet', 'rep', 'non_rep']

    time_vec = np.arange(xr_data.shape[-1])

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']
    odor_list_diff = ['+', '-']

    ######## SUMMARY NCHAN INTRA ########

    print('PLOT SUMMARY RSA INTRA')

    os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

    #group = sujet_group[0]
    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_diff))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions[2]
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                if group == 'allsujet':
                    data_stretch = xr_data.loc[:, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[:, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[:, cond, odor, :]['sujet'].shape[0]
                elif group == 'rep':
                    data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_best_list_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[sujet_best_list_rev, cond, odor, :]['sujet'].shape[0]
                elif group == 'non_rep':
                    data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_no_respond_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[sujet_no_respond_rev, cond, odor, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

                scales_val['min'].append((data_stretch - sem).min())
                scales_val['max'].append((data_stretch + sem).max())

        scales_val['min'] = np.array(scales_val['min']).min()
        scales_val['max'] = np.array(scales_val['max']).max()

        plt.suptitle(f'{group} {n_sujet} intra')

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                if group == 'allsujet':
                    data_stretch = xr_data.loc[:, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[:, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, :].shape[0])
                    baseline = xr_data.loc[:, 'FR_CV_1', odor, :].mean('sujet').values
                    sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, :].shape[0])
                elif group == 'rep':
                    data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_best_list_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, :].shape[0])
                    baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :].mean('sujet').values
                    sem_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :].shape[0])
                elif group == 'non_rep':
                    data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_no_respond_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, :].shape[0])
                    baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :].mean('sujet').values
                    sem_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :].shape[0])

                ax = axs[odor_i, cond_i]

                pval_pre = df_stats_all['intra'].query(f"group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['pre_test_pval'].values[0]
                pval_post = df_stats_all['intra'].query(f"group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['p_unc'].values[0]
                if pval_pre <= 0.05 and pval_post <= 0.05: 
                    pval_title = pval_stars(pval_post)
                else:
                    pval_title = 'ns'

                ax.set_title(f"{cond} {pval_title}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                ax.set_ylim(scales_val['min'], scales_val['max'])

                ax.plot(time_vec, data_stretch, color='r')
                ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                ax.plot(time_vec, baseline, label='FR_CV_1', color='b')
                ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                clusters = cluster_stats['intra'][group][odor][cond]
                ax.fill_between(time_vec, scales_val['min'], scales_val['max'], where=clusters.astype('int'), alpha=0.3, color='r')

                ax.vlines(time_vec.size/2, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        fig.savefig(f'intra_{group}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()




    ######## SUMMARY NCHAN INTER ########

    print('PLOT SUMMARY RSA INTER')

    os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

    #group = sujet_group[0]
    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list_diff), ncols=len(conditions))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                if group == 'allsujet':
                    data_stretch = xr_data.loc[:, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[:, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[:, cond, odor, :]['sujet'].shape[0]
                elif group == 'rep':
                    data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_best_list_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[sujet_best_list_rev, cond, odor, :]['sujet'].shape[0]
                elif group == 'non_rep':
                    data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_no_respond_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, :].shape[0])
                    n_sujet = xr_data.loc[sujet_no_respond_rev, cond, odor, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

                scales_val['min'].append((data_stretch - sem).min())
                scales_val['max'].append((data_stretch + sem).max())

        scales_val['min'] = np.array(scales_val['min']).min()
        scales_val['max'] = np.array(scales_val['max']).max()

        plt.suptitle(f'{group} {n_sujet} inter')

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list_diff):

                if group == 'allsujet':
                    data_stretch = xr_data.loc[:, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[:, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, :].shape[0])
                    baseline = xr_data.loc[:, cond, 'o', :].mean('sujet').values
                    sem_baseline = xr_data.loc[:, cond, 'o', :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, 'o', :].shape[0])
                elif group == 'rep':
                    data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_best_list_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, :].shape[0])
                    baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', :].mean('sujet').values
                    sem_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, 'o', :].shape[0])
                elif group == 'non_rep':
                    data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
                    sem = xr_data.loc[sujet_no_respond_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, :].shape[0])
                    baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', :].mean('sujet').values
                    sem_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, 'o', :].shape[0])

                ax = axs[odor_i, cond_i]

                pval_pre = df_stats_all['inter'].query(f"group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['pre_test_pval'].values[0]
                pval_post = df_stats_all['inter'].query(f"group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['p_unc'].values[0]
                if pval_pre <= 0.05 and pval_post <= 0.05: 
                    pval_title = pval_stars(pval_post)
                else:
                    pval_title = 'ns'

                ax.set_title(f"{cond} {pval_title}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                ax.set_ylim(scales_val['min'], scales_val['max'])

                ax.plot(time_vec, data_stretch, color='r')
                ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                ax.plot(time_vec, baseline, label='o', color='b')
                ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                clusters = cluster_stats['inter'][group][cond][odor]
                ax.fill_between(time_vec, scales_val['min'], scales_val['max'], where=clusters.astype('int'), alpha=0.3, color='r')

                ax.vlines(time_vec.size/2, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        fig.savefig(f'inter_{group}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()

    ######## RSA REP NOREP ########

    print('PLOT SUMMARY RSA REP NOREP')

    os.chdir(os.path.join(path_results, 'allplot', 'RSA'))

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    scales_val = {'min' : [], 'max' : []}

    #cond_i, cond = 2, conditions[2]
    for cond_i, cond in enumerate(conditions):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
            n_sujet_rep = xr_data.loc[sujet_best_list_rev, cond, odor, :]['sujet'].shape[0]

            scales_val['min'].append(data_stretch.min())
            scales_val['max'].append(data_stretch.max())

            data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
            n_sujet_norep = xr_data.loc[sujet_no_respond_rev, cond, odor, :]['sujet'].shape[0]

            scales_val['min'].append(data_stretch.min())
            scales_val['max'].append(data_stretch.max())

    scales_val['min'] = np.array(scales_val['min']).min()
    scales_val['max'] = np.array(scales_val['max']).max()

    plt.suptitle(f'rep:{n_sujet_rep} no_rep:{n_sujet_norep}')

    #cond_i, cond = 2, conditions[2]
    for cond_i, cond in enumerate(conditions):

        #odor_i, odor = 1, odor_list[1]
        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i, cond_i]

            if cond_i == 0:
                ax.set_ylabel(odor)

            data_stretch_rep = xr_data.loc[sujet_best_list_rev, cond, odor, :].mean('sujet').values
            sem_rep = xr_data.loc[sujet_best_list_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, :].shape[0])

            data_stretch_norep = xr_data.loc[sujet_no_respond_rev, cond, odor, :].mean('sujet').values
            sem_norep = xr_data.loc[sujet_no_respond_rev, cond, odor, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, :].shape[0])

            pval_pre = df_stats_all['repnorep'].query(f"odor == '{odor}' and cond == '{cond}'")['p-val'].values[0]
            if pval_pre <= 0.05: 
                pval_title = pval_stars(pval_pre)
            else:
                pval_title = 'ns'

            ax.set_title(f"{cond} {pval_title}")

            ax.set_ylim(scales_val['min'], scales_val['max'])

            ax.plot(time_vec, data_stretch_rep, label='rep', color='r')
            ax.fill_between(time_vec, data_stretch_rep+sem_rep, data_stretch_rep-sem_rep, alpha=0.25, color='m')

            ax.plot(time_vec, data_stretch_norep, label='no_rep', color='b')
            ax.fill_between(time_vec, data_stretch_norep+sem_norep, data_stretch_norep-sem_norep, alpha=0.25, color='c')

            clusters = cluster_stats_rep_norep[odor][cond]
            ax.fill_between(time_vec, scales_val['min'], scales_val['max'], where=clusters.astype('int'), alpha=0.3, color='r')

            ax.vlines(time_vec.size/2, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

    fig.tight_layout()
    plt.legend()

    # plt.show()

    #### save
    os.chdir(os.path.join(path_results, 'allplot', 'RSA'))
    fig.savefig(f'rep_norep.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()

    ######## HISTOGRAM GROUP ########

    #group = sujet_group[0]
    for group in sujet_group:

        #### generate df
        if group == 'allsujet':
            xr_data_sel = xr_data.loc[:, :, :, :]
        elif group == 'rep':
            xr_data_sel = xr_data.loc[sujet_best_list_rev, :, :, :]
        elif group == 'non_rep':
            xr_data_sel = xr_data.loc[sujet_no_respond_rev, :, :, :]

        df_min = xr_data_sel.min('time').to_dataframe(name='val').reset_index(drop=False)
        df_max = xr_data_sel.max('time').to_dataframe(name='val').reset_index(drop=False)

        df_minmax = df_min.copy()
        df_minmax['val'] = np.abs(df_max['val'].values - df_min['val'].values)

        sns.catplot(data=df_minmax, kind='bar', x="cond", y="val", hue="odor").set(title=f'RSA {group}')
        sns.set(rc={'figure.figsize':(10,7)})
        # plt.show()

        #### save
        plt.savefig(f'hist_{group}.jpeg', dpi=150)

        plt.close('all')
        gc.collect()

    ######## HISTOGRAM REPNOREP ########

    df_min = xr_data.min('time').to_dataframe(name='val').reset_index(drop=False)
    df_max = xr_data.max('time').to_dataframe(name='val').reset_index(drop=False)

    df_minmax = df_min.copy()
    df_minmax['val'] = np.abs(df_max['val'].values - df_min['val'].values)

    sel_list_append = []

    for row_i in range(df_minmax.shape[0]):

        if df_minmax.iloc[row_i]['sujet'] in sujet_best_list_rev:
            sel_list_append.append('YES')
        else:
            sel_list_append.append('NO')

    df_minmax['select_best'] = sel_list_append

    for odor in odor_list:

        sns.catplot(data=df_minmax.query(f"odor == '{odor}'"), kind='bar', x="cond", y="val", hue="select_best").set(title=f'repnorep {odor}')
        sns.set(rc={'figure.figsize':(10,7)})
        # plt.show()

        #### save
        plt.savefig(f'hist_repnorep_{odor}.jpeg', dpi=150)

        plt.close('all')
        gc.collect()


################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    xr_data = get_xr_data_RSA()
    df_stats_all = get_df_stats(xr_data)
    cluster_stats, cluster_stats_rep_norep = compute_cluster_RSA(xr_data)
    plot_RSA_diff(xr_data, cluster_stats, cluster_stats_rep_norep, df_stats_all)


