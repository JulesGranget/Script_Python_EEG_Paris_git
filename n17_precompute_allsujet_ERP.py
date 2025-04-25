

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False




def compute_normalization_params():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_normalization_params.nc')):
        print('allready computed')
        return

    xr_data = np.zeros((len(sujet_list), len(chan_list_eeg), 4))
    xr_dict = {'sujet' : sujet_list, 'chan_list' : chan_list_eeg, 'params' : ['mean', 'std', 'median', 'mad']}

    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)

        _respi = np.empty((len(chan_list_eeg),1))

        for odor in odor_list:

            for cond in conditions:

                data = load_data_sujet(sujet, cond, odor)[:len(chan_list_eeg),:]
                data = scipy.signal.detrend(data, type='linear', axis=1)
                data = iirfilt(data, srate, lowcut=0.05, highcut=None, axis=1)
                data = iirfilt(data, srate, lowcut=None, highcut=45, axis=1)
                _respi = np.concat([_respi, data], axis=1)

        xr_data[sujet_i,:,0], xr_data[sujet_i,:,1], xr_data[sujet_i,:,2], xr_data[sujet_i,:,3] = _respi.mean(axis=1), _respi.std(axis=1), np.median(_respi, axis=1), scipy.stats.median_abs_deviation(_respi, axis=1)


    xr_normal_params = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
    xr_normal_params.to_netcdf('allsujet_ERP_normalization_params.nc')




def compute_ERP():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_data.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data = xr.open_dataarray('allsujet_ERP_data.nc')
        xr_data_sem = xr.open_dataarray('allsujet_ERP_data_sem.nc')

    else:

        # t_start_PPI = PPI_time_vec[0]
        # t_stop_PPI = PPI_time_vec[1]

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

        xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec}
        xr_data = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_dict_sem = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec}
        xr_data_sem = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        df_sd_cleaning = {'sujet' : [], 'nchan' : [], 'cond' : [], 'odor' : [], 'n_removed' : [], 'n_keep' : [], '%_removed' : []}

        #sujet_i, sujet = 5, sujet_list[5]
        for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = 0, chan_list_eeg[0]
            for nchan_i, nchan in enumerate(chan_list_eeg):

                erp_data = {}

                #cond = 'CO2'
                for cond in conditions:

                    erp_data[cond] = {}

                    #odor = odor_list[0]
                    for odor in odor_list:

                        #### load
                        data = load_data_sujet(sujet, cond, odor)
                        respi = data[-3,:] 
                        data = data[:len(chan_list_eeg),:]

                        respfeatures_i = respfeatures[cond][odor]
                        inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                        #### chunk
                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                        #### low pass 45Hz
                        x = data[nchan_i,:]
                        x = scipy.signal.detrend(x, type='linear')
                        x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)
                        x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)

                        for start_i, start_time in enumerate(inspi_starts):

                            t_start = int(start_time + t_start_PPI*srate)
                            t_stop = int(start_time + t_stop_PPI*srate)

                            x_chunk = x[t_start: t_stop]

                            data_stretch[start_i, :] = (x_chunk - x_chunk.mean()) / x_chunk.std()
                            # data_stretch[start_i, :] = (x_chunk - x_mean) / x_std

                        data_stretch_mask = (data_stretch >= 3) | (data_stretch <= -3)
                        df_sd_cleaning['sujet'].append(sujet)
                        df_sd_cleaning['nchan'].append(nchan)
                        df_sd_cleaning['cond'].append(cond)
                        df_sd_cleaning['odor'].append(odor)
                        df_sd_cleaning['n_removed'].append((data_stretch_mask.sum(axis=1) != 0).sum())
                        df_sd_cleaning['n_keep'].append((data_stretch_mask.sum(axis=1) == 0).sum())
                        df_sd_cleaning['%_removed'].append((data_stretch_mask.sum(axis=1) != 0).sum() * 100 / data_stretch.shape[0])

                        if debug:

                            time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                            for inspi_i, _ in enumerate(inspi_starts):

                                plt.plot(time_vec, data_stretch[inspi_i, :], alpha=0.3)

                            plt.vlines(0, ymax=data_stretch.max(), ymin=data_stretch.min(), color='k')
                            plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                            plt.plot(time_vec, data_stretch.mean(axis=0), color='r')
                            plt.title(f'{cond} {odor} : {data_stretch.shape[0]}')
                            plt.show()

                        # if debug:

                        #     time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                        #     for inspi_i, _ in enumerate(inspi_starts):

                        #         plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                        #     plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                        #     plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                        #     plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                        #     plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                        #     plt.show()

                        erp_data[cond][odor] = data_stretch

                #### regroup FR_CV
                # erp_data['VS'] = {}

                # for odor in odor_list:

                #     erp_data['VS'][odor] = np.concatenate([erp_data['FR_CV_1'][odor], erp_data['FR_CV_2'][odor]], axis=0)

                #### load
                for cond in conditions:

                    #odor = odor_list[0]
                    for odor in odor_list:
                        
                        xr_data.loc[sujet, cond, odor, nchan, :] = erp_data[cond][odor].mean(axis=0)
                        xr_data_sem.loc[sujet, cond, odor, nchan, :] = erp_data[cond][odor].std(axis=0) / np.sqrt(erp_data[cond][odor].shape[0])

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_data.to_netcdf('allsujet_ERP_data.nc')
        xr_data_sem.to_netcdf('allsujet_ERP_data_sem.nc')

        df_sd_cleaning = pd.DataFrame(df_sd_cleaning)
        df_sd_cleaning.to_excel('df_sd_cleaning.xlsx')

    return xr_data, xr_data_sem
            

def compute_ERP_stretch(normalization='rscore'):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_data_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data_stretch = xr.open_dataarray('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch = xr.open_dataarray('allsujet_ERP_data_sem_stretch.nc')

    else:

        cond_list = ['FR_CV_1', 'MECA', 'CO2']

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_normal_params = xr.open_dataarray('allsujet_ERP_normalization_params.nc')

        xr_dict_stretch = {'sujet' : sujet_list, 'cond' : cond_list, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'phase' : np.arange(stretch_point_TF)}
        
        os.chdir(path_memmap)
        data_stretch_ERP = np.memmap(f'data_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(odor_list),len(chan_list_eeg), stretch_point_TF))
        data_sem_stretch_ERP = np.memmap(f'data_sem_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(cond_list), len(odor_list),len(chan_list_eeg), stretch_point_TF))

        #sujet_i, sujet = 0, sujet_list[0]
        def get_stretch_data_for_ERP(sujet_i, sujet):

        #sujet_i, sujet = np.where(sujet_list == '12BD')[0][0], '12BD'
        # for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #cond_i, cond = 2, 'CO2'
            for cond_i, cond in enumerate(cond_list):
                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):
                        
                    data = load_data_sujet(sujet, cond, odor)[:len(chan_list_eeg),:]
                    data = scipy.signal.detrend(data, type='linear' ,axis=1)
                    data = iirfilt(data, srate, lowcut=0.05, highcut=None, axis=1)
                    data = iirfilt(data, srate, lowcut=None, highcut=45, axis=1)

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond][odor], stretch_point_TF, data[nchan_i,:], srate)

                        if normalization == 'zscore':
                            data_stretch_norm = (data_stretch - xr_normal_params.loc[sujet, nchan, 'mean'].values) / xr_normal_params.loc[sujet, nchan, 'std'].values
                        if normalization == 'rscore':
                            data_stretch_norm = (data_stretch - xr_normal_params.loc[sujet, nchan, 'median'].values) * 0.6745 / xr_normal_params.loc[sujet, nchan, 'mad'].values

                        if debug:

                            plt.plot(data, label='raw')
                            plt.plot(scipy.signal.detrend(data, type='linear'), label='detrend')
                            plt.legend()
                            plt.show()

                            plt.hist(data, label='raw', bins=100)
                            plt.hist(scipy.signal.detrend(data, type='linear'), label='detrend', bins=100)
                            plt.legend()
                            plt.show()

                            plt.plot(data_stretch_norm[0,:], label='raw')
                            plt.plot(scipy.stats.zscore(data_stretch_norm, axis=1)[0,:], label='stretch')
                            plt.legend()
                            plt.show()

                            fig, ax = plt.subplots()

                            for inspi_i in range(data_stretch_norm.shape[0]):

                                ax.plot(np.arange(stretch_point_TF), data_stretch_norm[inspi_i, :], alpha=0.3)

                            plt.vlines(stretch_point_TF/2, ymax=data_stretch_norm.max(), ymin=data_stretch_norm.min(), color='k')
                            ax.plot(np.arange(stretch_point_TF), data_stretch_norm.mean(axis=0), color='r')
                            plt.title(f'{cond} : {data_stretch_norm.shape[0]}')
                            ax.invert_yaxis()
                            plt.show()

                        if normalization == 'zscore':

                            data_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = data_stretch_norm.mean(axis=0)
                            data_sem_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = data_stretch_norm.std(axis=0) / np.sqrt(data_stretch_norm.shape[0])

                        elif normalization == 'rscore':

                            data_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = np.median(data_stretch_norm, axis=0)
                            data_sem_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = scipy.stats.median_abs_deviation(data_stretch_norm, axis=0) / np.sqrt(data_stretch_norm.shape[0])

                        if debug:

                            data_stretch_zscore = (data_stretch - xr_normal_params.loc[sujet, nchan, 'mean'].values) / xr_normal_params.loc[sujet, nchan, 'std'].values
                            data_stretch_rscore = (data_stretch - xr_normal_params.loc[sujet, nchan, 'median'].values) * 0.6745 / xr_normal_params.loc[sujet, nchan, 'mad'].values

                            plt.plot(np.arange(stretch_point_TF), np.median(data_stretch, axis=0), label='median_nonorm', color='y')
                            plt.plot(np.arange(stretch_point_TF), np.mean(data_stretch, axis=0), label='mean_nonorm', color='k')
                            plt.plot(np.arange(stretch_point_TF), np.median(data_stretch_rscore, axis=0), label='median', color='b')
                            plt.fill_between(np.arange(stretch_point_TF), np.median(data_stretch_rscore, axis=0) - (scipy.stats.median_abs_deviation(data_stretch_rscore, axis=0) / np.sqrt(data_stretch_rscore.shape[0])), 
                                             np.median(data_stretch_rscore, axis=0) + (scipy.stats.median_abs_deviation(data_stretch_rscore, axis=0) / np.sqrt(data_stretch_rscore.shape[0])), alpha=0.3, color='c')
                            plt.plot(np.arange(stretch_point_TF), np.mean(data_stretch_zscore, axis=0), label='mean', color='r')
                            plt.fill_between(np.arange(stretch_point_TF), np.mean(data_stretch_zscore, axis=0) - (np.std(data_stretch_zscore, axis=0) / np.sqrt(data_stretch_zscore.shape[0])), 
                                             np.mean(data_stretch_zscore, axis=0) + (np.std(data_stretch_zscore, axis=0) / np.sqrt(data_stretch_zscore.shape[0])), alpha=0.3, color='m')
                            plt.legend()
                            plt.show()

                        # inverse to have inspi on the right and expi on the left
                        # data_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_load[int(stretch_point_TF/2):], data_stretch_load[:int(stretch_point_TF/2)]))
                        # data_sem_stretch_ERP[sujet_i, cond_i, nchan_i, :] = np.hstack((data_stretch_sem_load[int(stretch_point_TF/2):], data_stretch_sem_load[:int(stretch_point_TF/2)]))

        #### parallelize
        # joblib.Parallel(n_jobs = n_core, prefer = 'threads')(joblib.delayed(get_stretch_data_for_ERP)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))
        for sujet_i, sujet in enumerate(sujet_list):
            get_stretch_data_for_ERP(sujet_i, sujet)

        #### load data in xr
        xr_data_stretch = xr.DataArray(data=data_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())
        xr_data_sem_stretch = xr.DataArray(data=data_sem_stretch_ERP, dims=xr_dict_stretch.keys(), coords=xr_dict_stretch.values())

        os.chdir(path_memmap)
        try:
            os.remove(f'data_stretch_ERP.dat')
            del data_stretch_ERP
        except:
            pass

        os.chdir(path_memmap)
        try:
            os.remove(f'data_sem_stretch_ERP.dat')
            del data_sem_stretch_ERP
        except:
            pass

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_data_stretch.to_netcdf('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch.to_netcdf('allsujet_ERP_data_sem_stretch.nc')

    return xr_data_stretch, xr_data_sem_stretch





def get_cluster_stats_manual_prem1d(stretch=True):

    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra_perm1d_stretch.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('PERM ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm1d_stretch.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm1d_stretch.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm1d_stretch.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep
        
    elif stretch == False:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra_perm1d.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('PERM ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm1d.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm1d.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm1d.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep    

    #### intra
    cond_sel = ['MECA', 'CO2']
    odor_sel = ['o', '+', '-']
    sujet_group = ['allsujet', 'rep', 'non_rep']

    if stretch:
        time_vec = np.arange(stretch_point_TF)
    else:      
        time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)

    os.chdir(path_memmap)
    cluster_stats_intra = np.memmap(f'cluster_stats_intra_perm1d.dat', dtype=np.float64, mode='w+', shape=(len(sujet_group), len(chan_list_eeg), len(odor_sel), len(cond_sel), time_vec.shape[0]))

    def _get_cluster_stat_group(group_i, group):
    #group = sujet_group[0]
    # for group in sujet_group:

        print(f'intra {group}')

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch(normalization='rscore')
        else:
            xr_data, xr_data_sem = compute_ERP()

        #nchan = 'C3'
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #odor_i, odor = 0, odor_sel[0]
            for odor_i, odor in enumerate(odor_sel):

                #cond = cond_sel[0]
                for cond_i, cond in enumerate(cond_sel):

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate,
                                                      stat_design='within', mode_grouped=mode_grouped_ERP_STATS, 
                                                      mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                      mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)
                    # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                    if debug:

                        plt.plot(data_baseline.mean(axis=0))
                        plt.plot(data_cond.mean(axis=0))

                        plt.show()

                    cluster_stats_intra[group_i, nchan_i, odor_i, cond_i, :] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_group)(group_i, group) for group_i, group in enumerate(sujet_group))

    xr_dict = {'group' : sujet_group, 'chan' : chan_list_eeg, 'odor' : odor_list, 'cond' : cond_sel, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_intra, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_intra_perm1d_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_intra_perm1d.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_intra_perm1d.dat')
        del cluster_stats_intra
    except:
        pass

    #### inter
    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    odor_sel = ['+', '-']
    sujet_group = ['allsujet', 'rep', 'non_rep']

    os.chdir(path_memmap)
    cluster_stats_inter = np.memmap(f'cluster_stats_inter_perm1d.dat', dtype=np.float64, mode='w+', shape=(len(sujet_group), len(chan_list_eeg), len(odor_sel), len(cond_sel), time_vec.shape[0]))

    def _get_cluster_stat_group(group_i, group):
    # for group in sujet_group:

        print(f'inter {group}')

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch()
        else:
            xr_data, xr_data_sem = compute_ERP()

        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 2, cond_sel[2]
            for cond_i, cond in enumerate(cond_sel):

                for odor_i, odor in enumerate(odor_sel):

                    if group == 'allsujet':
                        data_baseline = xr_data.loc[:, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values
                    elif group == 'rep':
                        data_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    elif group == 'non_rep':
                        data_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate,
                                                      stat_design='within', mode_grouped=mode_grouped_ERP_STATS, 
                                                      mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                      mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)
                    # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(np.arange(data_baseline.shape[1]), data_baseline.mean(axis=0))
                        ax.plot(np.arange(data_baseline.shape[1]), data_cond.mean(axis=0))
                        ax.fill_between(np.arange(data_baseline.shape[1]), -0.5, 0.5, where=mask.astype('int'), alpha=0.3, color='r')
                        plt.show()

                    cluster_stats_inter[group_i, nchan_i, odor_i, cond_i, :] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_group)(group_i, group) for group_i, group in enumerate(sujet_group))

    xr_dict = {'group' : sujet_group, 'chan' : chan_list_eeg, 'odor' : odor_sel, 'cond' : cond_sel, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_inter, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_inter_perm1d_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_inter_perm1d.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_inter_perm1d.dat')
        del cluster_stats_inter
    except:
        pass

    #### rep norep
    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    odor_sel = odor_list

    os.chdir(path_memmap)
    cluster_stats_rep_norep = np.memmap(f'cluster_stats_rep_norep_perm1d.dat', dtype=np.float64, mode='w+', shape=(len(chan_list_eeg), len(odor_list), len(cond_sel), time_vec.shape[0]))

    def _get_cluster_stat_rep_norep_chan(chan_i, chan):

        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch()
        else:
            xr_data, xr_data_sem = compute_ERP()

        print(chan)

        #odor_i, odor = 2, odor_list[2]
        for odor_i, odor in enumerate(odor_list):

            for cond_i, cond in enumerate(cond_sel):

                data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, chan, :].values
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, chan, :].values

                mask = get_permutation_cluster_1d(data_baseline, data_cond, ERP_n_surrogate,
                                                      stat_design='between', mode_grouped=mode_grouped_ERP_STATS, 
                                                      mode_generate_surr=mode_generate_surr_ERP_STATS, percentile_thresh=percentile_thresh_ERP_STATS,
                                                      mode_select_thresh=mode_select_thresh_ERP_STATS, size_thresh_alpha=size_thresh_alpha_ERP_STATS)
                # mask = get_permutation_cluster_1d_DEBUG(data_baseline, data_cond, ERP_n_surrogate)

                cluster_stats_rep_norep[chan_i, odor_i, cond_i ,:] = mask

    joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(_get_cluster_stat_rep_norep_chan)(chan_i, chan) for chan_i, chan in enumerate(chan_list_eeg))

    xr_dict = {'chan' : chan_list_eeg, 'odor' : odor_list, 'cond' : cond_sel, 'time' : time_vec}
    xr_cluster = xr.DataArray(data=cluster_stats_rep_norep, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_rep_norep_perm1d_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_rep_norep_perm1d.nc')

    os.chdir(path_memmap)
    try:
        os.remove(f'cluster_stats_rep_norep_perm1d.dat')
        del cluster_stats_rep_norep
    except:
        pass
    
    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm1d_stretch.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm1d_stretch.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm1d_stretch.nc')
    else:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm1d.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm1d.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm1d.nc')

    return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep




def get_cluster_stats_manual_prem2g(stretch=True):

    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra_perm2g_stretch.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('PERM ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm2g_stretch.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm2g_stretch.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm2g_stretch.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep
        
    elif stretch == False:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats_intra_perm2g.nc')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

            print('PERM ALREADY COMPUTED', flush=True)

            cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm2g.nc')
            cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm2g.nc')
            cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm2g.nc')

            return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep    
        
    #### params
    phase_list = ['inspi', 'expi']
    phase_vec_list = {'inspi' : np.arange(stretch_point_TF/2).astype('int'), 'expi' : np.arange(stretch_point_TF/2, stretch_point_TF).astype('int')}

    #### intra
    cond_sel = ['MECA', 'CO2']
    odor_sel = ['o', '+', '-']
    sujet_group = ['allsujet', 'rep', 'non_rep']

    def process_group(group_i, group, phase_list, chan_list_eeg, odor_sel, cond_sel, 
                  sujet_best_list_rev, sujet_no_respond_rev, phase_vec_list, ERP_n_surrogate, debug, stretch):
        
        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch(normalization='rscore')
        else:
            xr_data, xr_data_sem = compute_ERP()
    
        print(f'intra {group}')

        _cluster_stats_intra = np.zeros((len(phase_list), len(chan_list_eeg), len(odor_sel), len(cond_sel)))

        for phase_i, phase in enumerate(phase_list):
            for nchan_i, nchan in enumerate(chan_list_eeg):
                for odor_i, odor in enumerate(odor_sel):
                    for cond_i, cond in enumerate(cond_sel):

                        if group == 'allsujet':
                            data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[:, cond, odor, nchan, phase_vec_list[phase]].median('phase').values
                        elif group == 'rep':
                            data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values
                        elif group == 'non_rep':
                            data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values

                        mask = get_permutation_2groups(
                            data_baseline, data_cond, ERP_n_surrogate, stat_design='within', mode_grouped='median', 
                            mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5]
                        )

                        if debug:
                            plt.plot(data_baseline.mean(axis=0))
                            plt.plot(data_cond.mean(axis=0))
                            plt.show()

                        _cluster_stats_intra[phase_i, nchan_i, odor_i, cond_i] = mask

        return _cluster_stats_intra

    # Parallel execution of processing across groups
    res = joblib.Parallel(n_jobs=n_core, prefer="processes")(joblib.delayed(process_group)(
            group_i, group, phase_list, chan_list_eeg, odor_sel, cond_sel, 
            sujet_best_list_rev, sujet_no_respond_rev, phase_vec_list, ERP_n_surrogate, debug, stretch) for group_i, group in enumerate(sujet_group))
    
    cluster_stats_intra = np.zeros((len(sujet_group), len(phase_list), len(chan_list_eeg), len(odor_sel), len(cond_sel)))

    for group_i, group in enumerate(sujet_group):

        cluster_stats_intra[group_i] = res[group_i]

    xr_dict = {'group' : sujet_group, 'phase' : phase_list, 'chan' : chan_list_eeg, 'odor' : odor_sel, 'cond' : cond_sel}
    xr_cluster = xr.DataArray(data=cluster_stats_intra, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_intra_perm2g_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_intra_perm2g.nc')


    #### inter
    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    odor_sel = ['+', '-']
    sujet_group = ['allsujet', 'rep', 'non_rep']

    def process_group(group_i, group, phase_list, chan_list_eeg, odor_sel, cond_sel, 
                  sujet_best_list_rev, sujet_no_respond_rev, phase_vec_list, ERP_n_surrogate, debug, stretch):
        
        if stretch:
            xr_data, xr_data_sem = compute_ERP_stretch(normalization='rscore')
        else:
            xr_data, xr_data_sem = compute_ERP()
    
        print(f'inter {group}')

        _cluster_stats_inter = np.zeros((len(phase_list), len(chan_list_eeg), len(odor_sel), len(cond_sel)))

        for phase_i, phase in enumerate(phase_list):
            for nchan_i, nchan in enumerate(chan_list_eeg):
                for odor_i, odor in enumerate(odor_sel):
                    for cond_i, cond in enumerate(cond_sel):

                        if group == 'allsujet':
                            data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[:, cond, odor, nchan, phase_vec_list[phase]].median('phase').values
                        elif group == 'rep':
                            data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values
                        elif group == 'non_rep':
                            data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, phase_vec_list[phase]].median('phase').values
                            data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values

                        mask = get_permutation_2groups(
                            data_baseline, data_cond, ERP_n_surrogate, stat_design='within', mode_grouped='median', 
                            mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5]
                        )

                        if debug:
                            plt.plot(data_baseline.mean(axis=0))
                            plt.plot(data_cond.mean(axis=0))
                            plt.show()

                        _cluster_stats_inter[phase_i, nchan_i, odor_i, cond_i] = mask

        return _cluster_stats_inter

    # Parallel execution of processing across groups
    res = joblib.Parallel(n_jobs=n_core, prefer="processes")(joblib.delayed(process_group)(
            group_i, group, phase_list, chan_list_eeg, odor_sel, cond_sel, 
            sujet_best_list_rev, sujet_no_respond_rev, phase_vec_list, ERP_n_surrogate, debug, stretch) for group_i, group in enumerate(sujet_group))
    
    cluster_stats_inter = np.zeros((len(sujet_group), len(phase_list), len(chan_list_eeg), len(odor_sel), len(cond_sel)))

    for group_i, group in enumerate(sujet_group):

        cluster_stats_inter[group_i] = res[group_i]

    xr_dict = {'group' : sujet_group, 'phase' : phase_list, 'chan' : chan_list_eeg, 'odor' : odor_sel, 'cond' : cond_sel}
    xr_cluster = xr.DataArray(data=cluster_stats_inter, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_inter_perm2g_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_inter_perm2g.nc')

    #### rep norep
    cond_sel = ['FR_CV_1', 'MECA', 'CO2']
    odor_sel = odor_list

    print('repnorep')

    cluster_stats_repnorep = np.zeros((len(phase_list), len(chan_list_eeg), len(odor_sel), len(cond_sel)))
        
    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch(normalization='rscore')
    else:
        xr_data, xr_data_sem = compute_ERP()

    for phase_i, phase in enumerate(phase_list):
        for nchan_i, nchan in enumerate(chan_list_eeg):
            for odor_i, odor in enumerate(odor_sel):
                for cond_i, cond in enumerate(cond_sel):

                    data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, phase_vec_list[phase]].median('phase').values

                    mask = get_permutation_2groups(
                        data_baseline, data_cond, ERP_n_surrogate, stat_design='between', mode_grouped='median', 
                        mode_generate_surr='percentile', percentile_thresh=[0.5, 99.5]
                    )

                    if debug:
                        plt.plot(data_baseline.mean(axis=0))
                        plt.plot(data_cond.mean(axis=0))
                        plt.show()

                    cluster_stats_repnorep[phase_i, nchan_i, odor_i, cond_i] = mask

    xr_dict = {'phase' : phase_list, 'chan' : chan_list_eeg, 'odor' : odor_list, 'cond' : cond_sel}
    xr_cluster = xr.DataArray(data=cluster_stats_repnorep, dims=xr_dict.keys(), coords=xr_dict.values())

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        xr_cluster.to_netcdf('cluster_stats_rep_norep_perm2g_stretch.nc')
    else:
        xr_cluster.to_netcdf('cluster_stats_rep_norep_perm2g.nc')
    
    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    if stretch:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm2g_stretch.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm2g_stretch.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm2g_stretch.nc')
    else:
        cluster_stats_intra = xr.open_dataarray('cluster_stats_intra_perm2g.nc')
        cluster_stats_inter = xr.open_dataarray('cluster_stats_inter_perm2g.nc')
        cluster_stats_rep_norep = xr.open_dataarray('cluster_stats_rep_norep_perm2g.nc')

    return cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep






################################
######## EXECUTE ########
################################

if __name__ == '__main__':
    
    compute_normalization_params()
    compute_ERP_stretch(normalization='rscore')
    get_cluster_stats_manual_prem1d(stretch=True)
    get_cluster_stats_manual_prem2g(stretch=True)






