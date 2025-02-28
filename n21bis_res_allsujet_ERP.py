
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

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n08_precompute_ERP import *

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test
from mne.stats import spatio_temporal_cluster_1samp_test

debug = False








################################
######## ERP ANALYSIS ########
################################


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
            

def explore_sd_cleaning_preocmpute():

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
    df_sd_cleaning = pd.read_excel('df_sd_cleaning.xlsx')
    df_sd_cleaning['n_tot'] = df_sd_cleaning['n_removed'] + df_sd_cleaning['n_keep'] 

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'erp_cleaning_methods'))

    for cond in ['FR_CV_1', 'CO2']:

        df_plot = df_sd_cleaning.query(f"cond == '{cond}'") 
        sns.displot(data=df_plot, x="n_removed", hue='odor', kind='kde')
        plt.xlim(0,150)
        plt.title(cond)
        plt.tight_layout()
        plt.savefig(f'{cond}_n_removed.png')
        plt.close('all')

        df_plot = df_sd_cleaning.query(f"cond == '{cond}'") 
        sns.displot(data=df_plot, x="n_keep", hue='odor', kind='kde')
        plt.xlim(0,150)
        plt.title(cond)
        plt.tight_layout()
        plt.savefig(f'{cond}_n_keep.png')
        plt.close('all')

        df_plot = df_sd_cleaning.query(f"cond == '{cond}'") 
        sns.displot(data=df_plot, x="%_removed", hue='odor', kind='kde')
        plt.xlim(0,150)
        plt.title(cond)
        plt.tight_layout()
        plt.savefig(f'{cond}_%_removed.png')
        plt.close('all')

        df_plot = df_sd_cleaning.query(f"cond == '{cond}'") 
        sns.displot(data=df_plot, x="n_tot", hue='odor', kind='kde')
        plt.xlim(0,150)
        plt.title(cond)
        plt.tight_layout()
        plt.savefig(f'{cond}_n_tot.png')
        plt.close('all')



    
    










def compute_ERP_stretch():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_data_stretch.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_data_stretch = xr.open_dataarray('allsujet_ERP_data_stretch.nc')
        xr_data_sem_stretch = xr.open_dataarray('allsujet_ERP_data_sem_stretch.nc')

    else:

        xr_dict_stretch = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'phase' : np.arange(stretch_point_TF)}
        
        os.chdir(path_memmap)
        data_stretch_ERP = np.memmap(f'data_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), stretch_point_TF))
        data_sem_stretch_ERP = np.memmap(f'data_sem_stretch_ERP.dat', dtype=np.float64, mode='w+', shape=(len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), stretch_point_TF))

        def get_stretch_data_for_ERP(sujet_i, sujet):

        #sujet_i, sujet = np.where(sujet_list == '12BD')[0][0], '12BD'
        # for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            respfeatures = load_respfeatures(sujet)

            #nchan_i, nchan = chan_list_eeg.index('FC1'), 'FC1'
            for nchan_i, nchan in enumerate(chan_list_eeg):

                #cond = 'CO2'
                for cond_i, cond in enumerate(conditions):

                    #odor = odor_list[0]
                    for odor_i, odor in enumerate(odor_list):

                        #### load
                        data = load_data_sujet(sujet, cond, odor)
                        respi = data[-3,:] 

                        x_raw = data[nchan_i,:]
                        x = scipy.signal.detrend(x_raw, type='linear')
                        x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)
                        x = iirfilt(x, srate, lowcut=0.05, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0)
                        x = zscore(x)

                        data_stretch, mean_inspi_ratio = stretch_data(respfeatures[cond][odor], stretch_point_TF, x, srate)

                        # data_stretch = zscore_mat(data_stretch)

                        if debug:

                            plt.plot(zscore(x_raw), label='raw')
                            plt.plot(x, label='cleaned')
                            plt.legend()
                            plt.show()

                            plt.plot(data[nchan_i,:], label='raw')
                            plt.plot(scipy.signal.detrend(data[nchan_i,:], type='linear'), label='detrend')
                            plt.legend()
                            plt.show()

                            fig, ax = plt.subplots()

                            for inspi_i in range(data_stretch.shape[0]):

                                ax.plot(np.arange(stretch_point_TF), data_stretch[inspi_i, :], alpha=0.3)

                            plt.vlines(stretch_point_TF/2, ymax=data_stretch.max(), ymin=data_stretch.min(), color='k')
                            ax.plot(np.arange(stretch_point_TF), data_stretch.mean(axis=0), color='r')
                            plt.title(f'{cond} {odor} : {data_stretch.shape[0]}')
                            ax.invert_yaxis()
                            plt.show()

                        # inverse to have inspi on the right and expi on the left
                        data_stretch_load = data_stretch.mean(axis=0)
                        data_stretch_sem_load = data_stretch.std(axis=0) / np.sqrt(data_stretch.shape[0])

                        data_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = np.hstack((data_stretch_load[int(stretch_point_TF/2):], data_stretch_load[:int(stretch_point_TF/2)]))
                        data_sem_stretch_ERP[sujet_i, cond_i, odor_i, nchan_i, :] = np.hstack((data_stretch_sem_load[int(stretch_point_TF/2):], data_stretch_sem_load[:int(stretch_point_TF/2)]))

        #### parallelize
        joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_stretch_data_for_ERP)(sujet_i, sujet) for sujet_i, sujet in enumerate(sujet_list))

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



def compute_lm_on_ERP(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'allsujet_ERP_lm_data.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_lm_data = xr.open_dataarray('allsujet_ERP_lm_data.nc')
        xr_lm_pred = xr.open_dataarray('allsujet_ERP_lm_pred.nc')
        xr_lm_pred_coeff = xr.open_dataarray('allsujet_ERP_lm_pred_coeff.nc')

        return xr_lm_data, xr_lm_pred, xr_lm_pred_coeff

    else:

        # t_start_PPI = PPI_time_vec[0]
        # t_stop_PPI = PPI_time_vec[1]

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

        sujet_group = ['allsujet', 'rep', 'non_rep']
        data_type = ['coeff', 'pval', 'slope', 'd_int']

        xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
        xr_lm_data = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

        time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
        xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_vec_lm}
        xr_lm_pred = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), len(time_vec_lm))), dims=xr_dict.keys(), coords=xr_dict.values())

        xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'type' : data_type}
        xr_lm_pred_coeff = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), len(data_type))), dims=xr_dict.keys(), coords=xr_dict.values())

        #sujet_i, sujet = 22, sujet_list[22]
        for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)

            #cond = 'MECA'
            for cond in conditions:

                #odor = odor_list[0]
                for odor in odor_list:

                    for nchan_i, nchan in enumerate(chan_list_eeg):
                    
                        data = xr_data.loc[sujet, cond, odor, nchan, :].values
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)
                        Y = data[time_vec_mask].reshape(-1,1)
                        X = time_vec[time_vec_mask].reshape(-1,1)

                        lm = linear_model.LinearRegression()

                        lm.fit(X, Y)

                        Y_pred = lm.predict(X)

                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                        xr_lm_data.loc[sujet, cond, odor, nchan, 'coeff'] = np.round(r_value**2, 5)
                        xr_lm_data.loc[sujet, cond, odor, nchan, 'pval'] = np.round(p_value, 5)
                        xr_lm_data.loc[sujet, cond, odor, nchan, 'slope'] = np.round(slope, 5)

                        #### get derivee integral
                        diff_data = np.insert(np.diff(data), 0, np.median(np.diff(data)))
                        diff_integral = np.trapz(diff_data[time_vec_mask])
                        
                        xr_lm_data.loc[sujet, cond, odor, nchan, 'd_int'] = np.round(diff_integral, 5)

                        #### verif
                        if debug:
                
                            plt.plot(X, Y)
                            plt.plot(X, Y_pred, color="b", linewidth=3)

                            plt.show()

        #### lm pred for mean
        #group = sujet_group[0]
        for group in sujet_group:

            #cond = 'MECA'
            for cond in conditions:

                #odor = odor_list[0]
                for odor in odor_list:

                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        if group == 'allsujet':
                            data = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        elif group == 'rep':
                            data = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
                        elif group == 'non_rep':
                            data = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, data.shape[0])
                        time_vec_mask = (time_vec >= PPI_lm_start) & (time_vec <= PPI_lm_stop)
                        Y = data[time_vec_mask].reshape(-1,1)
                        X = time_vec[time_vec_mask].reshape(-1,1)

                        lm = linear_model.LinearRegression()

                        lm.fit(X, Y)

                        Y_pred = lm.predict(X)

                        xr_lm_pred.loc[group, cond, odor, nchan, :] = Y_pred.reshape(-1)

                        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X.reshape(-1), Y.reshape(-1))

                        xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'coeff'] = np.round(r_value**2, 5)
                        xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'pval'] = np.round(p_value, 5)
                        xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'] = np.round(slope, 5)

                        #### verif
                        if debug:
                
                            plt.plot(X, Y)
                            plt.plot(X, Y_pred, color="b", linewidth=3)

                            plt.show()

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_lm_data.to_netcdf('allsujet_ERP_lm_data.nc')
        xr_lm_pred.to_netcdf('allsujet_ERP_lm_pred.nc')
        xr_lm_pred_coeff.to_netcdf('allsujet_ERP_lm_pred_coeff.nc')

        return xr_lm_data, xr_lm_pred, xr_lm_pred_coeff






def shuffle_data_ERP(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle.mean(0)




def shuffle_data_ERP_linear_based(data):

    ERP_shuffle = np.zeros(data.shape)

    for ERP_i in range(data.shape[0]):

        cut = np.random.randint(0, data.shape[1], 1)[0]
        ERP_shuffle[ERP_i,:data[ERP_i,cut:].shape[0]] = data[ERP_i,cut:]
        ERP_shuffle[ERP_i,data[ERP_i,cut:].shape[0]:] = data[ERP_i,:cut]

    return ERP_shuffle





#baseline_values, cond_values = xr_lm_data.loc[:, 'CO2', 'o', :, 'slope'].values, xr_lm_data.loc[:, 'CO2', '-', :, 'slope'].values
def get_stats_topoplots(baseline_values, cond_values, chan_list_eeg):

    data = {'sujet' : [], 'cond' : [], 'chan' : [], 'value' : []}

    for sujet_i in range(baseline_values.shape[0]):

        for chan_i, chan in enumerate(chan_list_eeg):

            data['sujet'].append(sujet_i)
            data['cond'].append('baseline')
            data['chan'].append(chan)
            data['value'].append(baseline_values[sujet_i, chan_i])

            data['sujet'].append(sujet_i)
            data['cond'].append('cond')
            data['chan'].append(chan)
            data['value'].append(cond_values[sujet_i, chan_i])
    
    df_stats = pd.DataFrame(data)

    mask_signi = np.array((), dtype='bool')

    for chan in chan_list_eeg:

        pval = get_stats_df(df=df_stats.query(f"chan == '{chan}'"), predictor='cond', outcome='value', subject='sujet', design='within')

        if pval < 0.05:
            mask_signi = np.append(mask_signi, True)
        else:
            mask_signi = np.append(mask_signi, False)

    if debug:

        plt.hist(df_stats.query(f"chan == '{chan}' and cond == 'cond'")['value'].values, bins=50)
        plt.hist(df_stats.query(f"chan == '{chan}' and cond == 'baseline'")['value'].values, bins=50)
        plt.show()

    return mask_signi







def compute_surr_ERP(xr_data, shuffle_way):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'allsujet_ERP_surrogates_{shuffle_way}.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print()

        print('ALREADY COMPUTED', flush=True)

        xr_surr = xr.open_dataarray(f'allsujet_ERP_surrogates_{shuffle_way}.nc')

    else:

        #### get data
        # t_start_PPI = PPI_time_vec[0]
        # t_stop_PPI = PPI_time_vec[1]

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

        xr_dict = {'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'percentile_type' : ['up', 'down'], 'time' : time_vec}
        xr_surr = xr.DataArray(data=np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg), 2, time_vec.shape[0])), dims=xr_dict.keys(), coords=xr_dict.values())

        if shuffle_way == 'linear_based':

            sujet_group = ['allsujet', 'rep', 'non_rep']

            xr_dict = {'group' : sujet_group, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'percentile_type' : ['up', 'down']}
            xr_surr = xr.DataArray(data=np.zeros((len(sujet_group), len(conditions), len(odor_list), len(chan_list_eeg), 2)), dims=xr_dict.keys(), coords=xr_dict.values())

            sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
            sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

            for group in sujet_group:

                #cond = 'CO2'
                for cond in conditions:

                    #odor = odor_list[0]
                    for odor in odor_list:

                        print(group, cond, odor)

                        #nchan_i, nchan = 0, chan_list_eeg[0]
                        for nchan_i, nchan in enumerate(chan_list_eeg):
                        
                            if group == 'allsujet':
                                data_cond = xr_data.loc[:,cond,odor,nchan,:].values
                            if group == 'rep':
                                data_cond = xr_data.loc[sujet_best_list,cond,odor,nchan,:].values
                            if group == 'no_rep':
                                data_cond = xr_data.loc[sujet_no_respond,cond,odor,nchan,:].values

                            _ERP_surr = np.zeros((ERP_n_surrogate))

                            for surr_i in range(ERP_n_surrogate):

                                surr_i_mean = shuffle_data_ERP_linear_based(data_cond).mean(axis=0)

                                Y = surr_i_mean[time_vec_mask]
                                X = time_vec[time_vec_mask]

                                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
                                
                                _ERP_surr[surr_i] = slope

                            if debug:

                                Y = data_cond.mean(0)[time_vec_mask]
                                X = time_vec[time_vec_mask]
                                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                plt.plot(time_vec, data_cond.mean(axis=0))
                                plt.gca().invert_yaxis()
                                plt.title(nchan)
                                plt.show()

                                count, hist, _ = plt.hist(_ERP_surr, bins=50)
                                plt.gca().invert_xaxis()
                                plt.vlines(slope, ymin=0, ymax=count.max(), color='r', label='raw')
                                plt.vlines(np.percentile(_ERP_surr, 5), ymin=0, ymax=count.max(), color='b', label='99')
                                plt.legend()
                                plt.title(nchan)
                                plt.show()

                            xr_surr.loc[group, cond, odor, nchan, 'up'] = np.percentile(_ERP_surr, 95, axis=0)
                            xr_surr.loc[group, cond, odor, nchan, 'down'] = np.percentile(_ERP_surr, 5, axis=0)

        else:
            
            for sujet in sujet_list:

                print(sujet)

                respfeatures = load_respfeatures(sujet)

                data_chunk_allcond = {}

                t_start_PPI = PPI_time_vec[0]
                t_stop_PPI = PPI_time_vec[1]

                #cond = 'MECA'
                for cond in conditions:

                    data_chunk_allcond[cond] = {}

                    #odor = odor_list[0]
                    for odor in odor_list:

                        data_chunk_allcond[cond][odor] = {}

                        #### load
                        data = load_data_sujet(sujet, cond, odor)
                        data = data[:len(chan_list_eeg),:]

                        respfeatures_i = respfeatures[cond][odor]
                        inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                        #### low pass 45Hz
                        for chan_i, chan in enumerate(chan_list_eeg):
                            data[chan_i,:] = iirfilt(data[chan_i,:], srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                        #nchan_i, nchan = 0, chan_list_eeg[0]
                        for nchan_i, nchan in enumerate(chan_list_eeg):

                            #### chunk
                            stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                            data_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                            #### low pass 45Hz
                            x = data[nchan_i,:]
                            x = iirfilt(x, srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                            x_mean, x_std = x.mean(), x.std()
                            microV_SD = int(x_std*1e6)

                            for start_i, start_time in enumerate(inspi_starts):

                                t_start = int(start_time + t_start_PPI*srate)
                                t_stop = int(start_time + t_stop_PPI*srate)

                                data_stretch[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                            if debug:

                                time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                for inspi_i, _ in enumerate(inspi_starts):

                                    plt.plot(time_vec, data_stretch[inspi_i, :], alpha=0.3)

                                plt.vlines(0, ymax=data_stretch.max(), ymin=data_stretch.min(), color='k')
                                plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                plt.plot(time_vec, data_stretch.mean(axis=0), color='r')
                                plt.title(f'{cond} {odor} : {data_stretch.shape[0]}, 3SD : {microV_SD}')
                                plt.show()

                            #### clean
                            data_stretch_clean = data_stretch[~((data_stretch >= 3) | (data_stretch <= -3)).any(axis=1),:]

                            if debug:

                                time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                for inspi_i, _ in enumerate(inspi_starts):

                                    plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                                plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                                plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                                plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                                plt.show()

                            data_chunk_allcond[cond][odor][nchan] = data_stretch_clean

                #### regroup FR_CV
                data_chunk_allcond['VS'] = {}

                for odor in odor_list:

                    data_chunk_allcond['VS'][odor] = {}

                    #### low pass 45Hz
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)

                data_chunk_allcond['FR_CV_1'] = {}
                data_chunk_allcond['FR_CV_2'] = {}

                #### compute ERP surr
                if shuffle_way == 'intra_cond':

                    cond = 'VS'

                    #odor = odor_list[0]
                    for odor in odor_list:

                        #nchan_i, nchan = 0, chan_list_eeg[0]
                        for nchan_i, nchan in enumerate(chan_list_eeg):
                        
                            data = data_chunk_allcond[cond][odor][nchan]

                            _ERP_surr = np.zeros((ERP_n_surrogate, data.shape[1]))

                            for surr_i in range(ERP_n_surrogate):

                                _ERP_surr[surr_i,:] = shuffle_data_ERP(data).mean(axis=0)

                            # up_percentile = np.std(_ERP_surr[:,:], axis=0)*3
                            # down_percentile = np.percentile(_ERP_surr, 99, axis=0)

                            up_percentile = np.std(_ERP_surr, axis=0)*3
                            down_percentile = -np.std(_ERP_surr, axis=0)*3

                            xr_surr.loc[sujet, cond, odor, nchan, 'up', :] = up_percentile
                            xr_surr.loc[sujet, cond, odor, nchan, 'down', :] = down_percentile

                if shuffle_way == 'inter_cond':

                    ERP_surr = {}

                    #cond = 'MECA'
                    for cond in conditions:

                        if cond == 'VS':
                            continue

                        ERP_surr[cond] = {}

                        #odor = odor_list[0]
                        for odor in odor_list:

                            ERP_surr[cond][odor] = {}

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data_cond = data_chunk_allcond[cond][odor][nchan]
                                data_baseline = data_chunk_allcond['VS'][odor][nchan]

                                min_shape = np.array([data_cond.shape[0], data_baseline.shape[0]]).min()

                                _ERP_surr = np.zeros((ERP_n_surrogate, data_cond.shape[1]))

                                for surr_i in range(ERP_n_surrogate):

                                    balance_selection_baseline = np.random.choice(np.arange(data_baseline.shape[0]), min_shape, replace=False)
                                    balance_selection_cond = np.random.choice(np.arange(data_cond.shape[0]), min_shape, replace=False)

                                    rand_selection_shuffle = np.random.randint(low=0, high=2, size=min_shape)
                                    rand_selection_shuffle_i = np.where(rand_selection_shuffle == 1)[0]

                                    data_cond_shuffle = data_cond[balance_selection_cond,:]
                                    data_baseline_shuffle = data_baseline[balance_selection_baseline,:]
                                    
                                    data_cond_shuffle[rand_selection_shuffle_i,:] = data_baseline[balance_selection_baseline,:][rand_selection_shuffle_i,:]
                                    data_baseline_shuffle[rand_selection_shuffle_i,:] = data_cond[balance_selection_cond,:][rand_selection_shuffle_i,:]

                                    _ERP_surr[surr_i,:] = data_cond_shuffle.mean(axis=0)

                                up_percentile = np.percentile(_ERP_surr, 98, axis=0)
                                down_percentile = np.percentile(_ERP_surr, 2, axis=0)

                                ERP_surr[cond][odor][nchan] = np.vstack((up_percentile, down_percentile))

                    #cond = 'MECA'
                    for cond in conditions:

                        #odor = odor_list[0]
                        for odor in odor_list:

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data = data_chunk_allcond[cond][odor][nchan]

                                _ERP_surr = np.zeros((ERP_n_surrogate, data.shape[1]))

                                for surr_i in range(ERP_n_surrogate):

                                    _ERP_surr[surr_i,:] = shuffle_data_ERP(data)

                                xr_surr.loc[sujet, cond, odor, nchan, 'up', :] = np.percentile(_ERP_surr, 98, axis=0)
                                xr_surr.loc[sujet, cond, odor, nchan, 'down', :] = np.percentile(_ERP_surr, 2, axis=0)

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_surr.to_netcdf(f'allsujet_ERP_surrogates_{shuffle_way}.nc')

    return xr_surr



def get_PPI_count(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'PPI_count_linear_based.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        xr_PPI_count = xr.open_dataarray(f'PPI_count_linear_based.nc')

    else:

        t_start_PPI = PPI_time_vec[0]
        t_stop_PPI = PPI_time_vec[1]

        PPI_lm_start = PPI_lm_time[0]
        PPI_lm_stop = PPI_lm_time[1] 

        time_vec = xr_data['time'].values
        time_vec_mask = (time_vec > PPI_lm_start) & (time_vec < PPI_lm_stop)

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
        df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

        examinateur_list = ['JG', 'MCN', 'TS']

        xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
        xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

        #examinateur = examinateur_list[2]
        for examinateur in examinateur_list:

            if examinateur == 'JG':

                for sujet in sujet_list:

                    respfeatures = load_respfeatures(sujet)

                    data_chunk_allcond = {}
                    data_value_microV = {}

                    t_start_PPI = ERP_time_vec[0]
                    t_stop_PPI = ERP_time_vec[-1]

                    #cond = 'FR_CV_1'
                    for cond in conditions:

                        data_chunk_allcond[cond] = {}
                        data_value_microV[cond] = {}

                        #odor = odor_list[0]
                        for odor in odor_list:

                            print('compute erp')
                            print(sujet, cond, odor)

                            data_chunk_allcond[cond][odor] = {}
                            data_value_microV[cond][odor] = {}

                            #### load
                            data = load_data_sujet(sujet, cond, odor)
                            data = data[:len(chan_list_eeg),:]

                            respfeatures_i = respfeatures[cond][odor]
                            inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                            #### low pass 45Hz
                            for chan_i, chan in enumerate(chan_list_eeg):
                                data[chan_i,:] = iirfilt(data[chan_i,:], srate, lowcut=None, highcut=45, order=4, ftype='butter', verbose=False, show=False, axis=0)

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):

                                #### chunk
                                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                                data_chunk = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                                x = data[nchan_i,:]

                                x_mean, x_std = x.mean(), x.std()
                                microV_SD = int(x_std*1e6)

                                for start_i, start_time in enumerate(inspi_starts):

                                    t_start = int(start_time + t_start_PPI*srate)
                                    t_stop = int(start_time + t_stop_PPI*srate)

                                    data_chunk[start_i, :] = (x[t_start: t_stop] - x_mean) / x_std

                                if debug:

                                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                    for inspi_i, _ in enumerate(inspi_starts):

                                        plt.plot(time_vec, data_chunk[inspi_i, :], alpha=0.3)

                                    plt.vlines(0, ymax=data_chunk.max(), ymin=data_chunk.min(), color='k')
                                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                    plt.plot(time_vec, data_chunk.mean(axis=0), color='r')
                                    plt.title(f'{cond} {odor} : {data_chunk.shape[0]}, 3SD : {microV_SD}')
                                    plt.show()

                                #### clean
                                data_stretch_clean = data_chunk[~((data_chunk >= 3) | (data_chunk <= -3)).any(axis=1),:]

                                if debug:

                                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                                    for inspi_i, _ in enumerate(inspi_starts):

                                        plt.plot(time_vec, data_stretch_clean[inspi_i, :], alpha=0.3)

                                    plt.vlines(0, ymax=data_stretch_clean.max(), ymin=data_stretch_clean.min(), color='k')
                                    plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                                    plt.plot(time_vec, data_stretch_clean.mean(axis=0), color='r')
                                    plt.title(f'{cond} {odor} : {data_stretch_clean.shape[0]}')
                                    plt.show()

                                data_chunk_allcond[cond][odor][nchan] = data_stretch_clean
                                data_value_microV[cond][odor][nchan] = microV_SD

                    #### regroup FR_CV
                    # data_chunk_allcond['VS'] = {}
                    # data_value_microV['VS'] = {}

                    # for odor in odor_list:

                    #     data_chunk_allcond['VS'][odor] = {}
                    #     data_value_microV['VS'][odor] = {}

                    #     #### low pass 45Hz
                    #     for nchan_i, nchan in enumerate(chan_list_eeg):

                    #         data_chunk_allcond['VS'][odor][nchan] = np.concatenate([data_chunk_allcond['FR_CV_1'][odor][nchan], data_chunk_allcond['FR_CV_2'][odor][nchan]], axis=0)
                    #         data_value_microV['VS'][odor][nchan] = data_value_microV['FR_CV_1'][odor][nchan] + data_value_microV['FR_CV_2'][odor][nchan] / 2

                    # data_chunk_allcond['FR_CV_1'] = {}
                    # data_chunk_allcond['FR_CV_2'] = {}

                    # data_value_microV['FR_CV_1'] = {}
                    # data_value_microV['FR_CV_2'] = {}

                    #cond = 'CO2'
                    for cond in conditions:

                        #odor = odor_list[0]
                        for odor in odor_list:

                            print('compute surr')
                            print(sujet, cond, odor)

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):
                            
                                data_cond = data_chunk_allcond[cond][odor][nchan]

                                Y = data_cond.mean(axis=0)[time_vec_mask]
                                X = time_vec[time_vec_mask]

                                slope_observed, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                _ERP_surr = np.zeros((ERP_n_surrogate))

                                for surr_i in range(ERP_n_surrogate):

                                    ERP_shuffle = np.zeros(data_cond.shape)

                                    for erp_i in range(data_cond.shape[0]):

                                        cut = np.random.randint(0, data_cond.shape[1], 1)[0]

                                        ERP_shuffle[erp_i,:data_cond[:,cut:].shape[1]] = data_cond[erp_i,cut:]
                                        ERP_shuffle[erp_i,data_cond[:,cut:].shape[1]:] = data_cond[erp_i,:cut]
                                            
                                    surr_i_mean = ERP_shuffle.mean(axis=0)

                                    Y = surr_i_mean[time_vec_mask]
                                    X = time_vec[time_vec_mask]

                                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
                                    
                                    _ERP_surr[surr_i] = slope

                                if debug:

                                    Y = data_cond.mean(0)[time_vec_mask]
                                    X = time_vec[time_vec_mask]
                                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)

                                    plt.plot(time_vec, data_cond.mean(axis=0))
                                    plt.gca().invert_yaxis()
                                    plt.title(nchan)
                                    plt.show()

                                    count, hist, _ = plt.hist(_ERP_surr, bins=50)
                                    plt.gca().invert_xaxis()
                                    plt.vlines(slope, ymin=0, ymax=count.max(), color='r', label='raw')
                                    plt.vlines(np.percentile(_ERP_surr, 5), ymin=0, ymax=count.max(), color='b', label='99')
                                    plt.legend()
                                    plt.title(nchan)
                                    plt.show()

                                if slope_observed < np.percentile(_ERP_surr, 5):

                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = 1

            else:

                #sujet = sujet_list[0]
                for sujet in sujet_list:

                    if sujet in ['28NT']:
                        continue

                    #cond = 'CO2'
                    for cond in conditions:

                        if cond in ['FR_CV_1', 'FR_CV_2']:
                            continue

                        #odor = odor_list[0]
                        for odor in odor_list:

                            #nchan_i, nchan = 0, chan_list_eeg[0]
                            for nchan_i, nchan in enumerate(chan_list_eeg):

                                if nchan in ['Cz', 'Fz']:

                                    _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                                    xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_PPI_count.to_netcdf(f'PPI_count_linear_based.nc')

    return xr_PPI_count






################################
######## ERP PLOT ########
################################



def plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr, xr_PPI_count):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    # t_start_PPI = PPI_time_vec[0]
    # t_stop_PPI = PPI_time_vec[1]

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])

    ######## SUMMARY NCHAN ########

    print('PLOT SUMMARY ERP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary'))

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            scales_val = {'min' : [], 'max' : []}

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[:, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_best_list, cond, odor, nchan, :]['sujet'].shape[0]
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        n_sujet = xr_data.loc[sujet_no_respond, cond, odor, nchan, :]['sujet'].shape[0]
                    scales_val['min'].append(data_stretch.min())
                    scales_val['max'].append(data_stretch.max())

            scales_val['min'] = np.array(scales_val['min']).min()
            scales_val['max'] = np.array(scales_val['max']).max()

            plt.suptitle(f'{nchan} {group} {n_sujet}')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    data_surr_down = xr_surr.loc[group, cond, odor, nchan, 'down'].values

                    ax = axs[odor_i, cond_i]

                    # if cond != 'VS':

                    #     if (data_stretch[mask_time_PPI] < data_surr_down[mask_time_PPI]).any() and (data_stretch[mask_time_PPI] > data_surr_up[mask_time_PPI]).any():
                            
                    #         PPI_presence = 1

                    #         if debug:

                    #             plt.plot(data_stretch[mask_time_PPI])
                    #             plt.plot(data_surr_down[mask_time_PPI])
                    #             plt.plot(data_surr_up[mask_time_PPI])
                    #             plt.show()

                    #     else:

                    #         PPI_presence = 0

                    if xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values < data_surr_down:
                        
                        PPI_presence = 1

                        if debug:

                            plt.plot(data_stretch[mask_time_PPI])
                            plt.plot(data_surr_down[mask_time_PPI])
                            plt.show()

                            for nchan in chan_list_eeg:

                                plt.plot(time_vec, xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values)
                                plt.title(nchan)
                                plt.gca().invert_yaxis()
                                plt.show()

                                count, bins, _ = plt.hist(xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', time_vec[0]], bins=50)
                                plt.vlines(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values, ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.median(xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', time_vec[0]]), ymin=count.min(), ymax=count.max(), color='g')
                                plt.show()

                            for nchan in chan_list_eeg:

                                for sujet in sujet_no_respond:

                                    plt.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :].values, alpha=0.5)

                                plt.plot(time_vec, xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values, alpha=1)
                                plt.title(nchan)
                                plt.gca().invert_yaxis()
                                plt.show()     

                    else:

                        PPI_presence = 0

                    if cond_i ==0:
                        ax.set_ylabel(odor)

                    ax.set_title(f"{cond} \n slope:{np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'slope'].values, 3)} / r2:{np.round(xr_lm_pred_coeff.loc[group, cond, odor, nchan, 'coeff'].values, 3)} \n PPI:{PPI_presence}", fontweight='bold')

                    ax.set_ylim(allplot_erp_ylim[0], allplot_erp_ylim[1])

                    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                    ax.plot(time_vec, data_stretch)
                    # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
                    # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

                    # if cond != 'VS':
                    #     ax.plot(time_vec, data_surr_up, color='k', linestyle='--')
                    #     ax.plot(time_vec, data_surr_down, color='k', linestyle='--')

                    time_vec_lm = np.arange(PPI_lm_start, PPI_lm_stop, 1/srate)
                    ax.plot(time_vec_lm, xr_lm_pred.loc[group, cond, odor, nchan, :], color='r', linewidth=3)

                    ax.invert_yaxis()

                    ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

            fig.tight_layout()

            # plt.show()

            #### save
            fig.savefig(f'{nchan}_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

    ######## get mask topo ########
    stats_group_list = ['inter', 'intra']
    metric_list_stats = ['slope', 'd_int']

    topo_stats_dict = {'stats_group' : stats_group_list, 'group' : sujet_group, 'metric' : metric_list_stats, 'cond' : conditions,  'odor' : odor_list, 'nchan' : chan_list_eeg}
    topo_stats_data = np.zeros((len(stats_group_list), len(sujet_group), len(metric_list_stats), len(conditions), len(odor_list), len(chan_list_eeg)), dtype=bool)
    xr_topo_stats = xr.DataArray(topo_stats_data, dims=topo_stats_dict.keys(), coords=topo_stats_dict.values())

    for stats_group in stats_group_list:

        if stats_group == 'inter':

            for group in sujet_group:

                for metric in metric_list_stats:

                    for cond in conditions:

                        for odor in odor_list:

                            if odor != 'o':

                                if group == 'allsujet':
                                    data_baseline = xr_lm_data.loc[:, cond, 'o', :, metric].values
                                    data_cond = xr_lm_data.loc[:, cond, odor, :, metric].values

                                if group == 'rep':
                                    data_baseline = xr_lm_data.loc[sujet_best_list, cond, 'o', :, metric].values
                                    data_cond = xr_lm_data.loc[sujet_best_list, cond, odor, :, metric].values
                    
                                if group == 'non_rep':
                                    data_baseline = xr_lm_data.loc[sujet_no_respond_rev, cond, 'o', :, metric].values
                                    data_cond = xr_lm_data.loc[sujet_no_respond_rev, cond, odor, :, metric].values
                    
                                xr_topo_stats.loc[stats_group, group, metric, cond, odor, :] = get_stats_topoplots(data_baseline, data_cond, chan_list_eeg)

        if stats_group == 'intra':

            for group in sujet_group:

                for metric in metric_list_stats:

                    for odor in odor_list:

                        for cond in conditions:

                            if cond != 'FR_CV_1':

                                if group == 'allsujet':
                                    data_baseline = xr_lm_data.loc[:, 'FR_CV_1', odor, :, metric].values
                                    data_cond = xr_lm_data.loc[:, cond, odor, :, metric].values

                                if group == 'rep':
                                    data_baseline = xr_lm_data.loc[sujet_best_list, 'FR_CV_1', odor, :, metric].values
                                    data_cond = xr_lm_data.loc[sujet_best_list, cond, odor, :, metric].values
                    
                                if group == 'non_rep':
                                    data_baseline = xr_lm_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, metric].values
                                    data_cond = xr_lm_data.loc[sujet_no_respond_rev, cond, odor, :, metric].values
                    
                                xr_topo_stats.loc[stats_group, group, metric, cond, odor, :] = get_stats_topoplots(data_baseline, data_cond, chan_list_eeg)

    ######## TOPOPLOT SLOPE ########

    print('TOPOPLOT SLOPE')

    #### scales
    val = np.array([])

    for odor in odor_list:

        for cond in conditions:

            val = np.append(val, xr_lm_pred_coeff.loc['allsujet', cond, odor, :, 'slope'])

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    for stats_group in stats_group_list:
    
        #group = sujet_group[0]
        for group in sujet_group:

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

            if group == 'allsujet':
                plt.suptitle(f'{stats_group} {group} slope (s{xr_data.shape[0]})')
            if group == 'rep':
                plt.suptitle(f'{stats_group} {group} slope (s{len(sujet_best_list)})')
            if group == 'non_rep':
                plt.suptitle(f'{stats_group} {group} slope (s{len(sujet_no_respond)})')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #c, cond = 0, 'FR_CV_1'
            for c, cond in enumerate(conditions):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    data_plot = xr_lm_pred_coeff.loc[group, cond, odor, :, 'slope'].values*-1
                    # data_plot = xr_lm_data.loc[:, cond, odor, :, 'slope'].mean('sujet').values*-1

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False, 
                                        mask=xr_topo_stats.loc[stats_group, group, 'slope', cond, odor, :], mask_params=dict(markersize=5, markerfacecolor='y'))

            plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
            fig.savefig(f'slope_{group}_{stats_group}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()

    #### for cond
    for c, cond in enumerate(conditions):

        #### scales
        val = np.array([])
            
        for sujet_type_i, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list, cond, odor, :, 'slope'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'slope'].mean('sujet').values*-1

                val = np.append(val, data_plot)

        scale_min = val.min()
        scale_max = val.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)
        plt.suptitle(f'slope')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list)})', fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond)})', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list, cond, odor, :, 'slope'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'slope'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'slope_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    ######## TOPOPLOT D_INT ########

    print('TOPOPLOT D INT')

    #### scales
    val = np.array([])

    for odor_i in odor_list:

        for cond in conditions:

            val = np.append(val, xr_lm_data.loc[:, cond, odor, :, 'd_int'].mean('sujet'))

    scale_min = val.min()
    scale_max = val.max()

    #### plot
    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        if group == 'allsujet':
            plt.suptitle(f'{group} d_int (s{xr_data.shape[0]})')
        if group == 'rep':
            plt.suptitle(f'{group} d_int (s{len(sujet_best_list)})')
        if group == 'non_rep':
            plt.suptitle(f'{group} d_int (s{len(sujet_no_respond)})')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if group == 'allsujet':
                    data_plot = xr_lm_data.loc[:, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if group == 'rep':
                    data_plot = xr_lm_data.loc[sujet_best_list, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if group == 'non_rep':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False, 
                                        mask=xr_topo_stats.loc[stats_group, group, 'd_int', cond, odor, :], mask_params=dict(markersize=5, markerfacecolor='y'))

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'd_int_{group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### for cond
    for c, cond in enumerate(conditions):

        #### scales
        val = np.array([])
            
        for sujet_type_i, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                val = np.append(val, data_plot)

        scale_min = val.min()
        scale_max = val.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)

        plt.suptitle(f'd_int')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list)})', fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond)})', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if sujet_type == 'respond':
                    data_plot = xr_lm_data.loc[sujet_best_list, cond, odor, :, 'd_int'].mean('sujet').values*-1
                if sujet_type == 'no_respond':
                    data_plot = xr_lm_data.loc[sujet_no_respond, cond, odor, :, 'd_int'].mean('sujet').values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'd_int_{cond}_topo_respond.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    ######## TOPOPLOT PPI ########

    print('TOPOPLOT PPI')

    #### plot allsujet

    scale = {}

    for group in sujet_group:

        _max = []

        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):
                
                if group == 'allsujet':
                    PPI_count = xr_PPI_count.loc['JG', :, cond, odor, :].sum('sujet')
                if group == 'rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list, cond, odor, :].sum('sujet')
                if group == 'non_rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')

                _max.append(PPI_count.max())

        scale[group] = np.array(_max).max()

    for group in sujet_group:

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        if group == 'allsujet':
            plt.suptitle(f'PPI {group} (s{xr_data.shape[0]})')
        if group == 'rep':
            plt.suptitle(f'PPI {group} (s{len(sujet_best_list)})')
        if group == 'non_rep':
            plt.suptitle(f'PPI {group} (s{len(sujet_no_respond)})')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{cond}', fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                if group == 'allsujet':
                    PPI_count = xr_PPI_count.loc['JG', :, cond, odor, :].sum('sujet')
                    vmax = xr_data.shape[0]
                if group == 'rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list, cond, odor, :].sum('sujet')
                    vmax = 15
                if group == 'non_rep':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')
                    vmax = 15
                
                mne.viz.plot_topomap(PPI_count, info, vlim=(0, vmax), axes=ax, show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'PPI_{group}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### PPI for cond

    for c, cond in enumerate(conditions):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)

        plt.suptitle(f'PPI {cond}')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, sujet_type in enumerate(['respond', 'no_respond']):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    if sujet_type == 'respond':
                        ax.set_title(f"{sujet_type} (s{len(sujet_best_list)})", fontweight='bold', rotation=0)
                    if sujet_type == 'no_respond':
                        ax.set_title(f"{sujet_type} (s{len(sujet_no_respond)})", fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')

                if sujet_type == 'respond':
                    PPI_count = xr_PPI_count.loc['JG', sujet_best_list, cond, odor, :].sum('sujet')
                if sujet_type == 'no_respond':
                    PPI_count = xr_PPI_count.loc['JG', sujet_no_respond, cond, odor, :].sum('sujet')
                
                mne.viz.plot_topomap(PPI_count, info, vlim=(0, 15), axes=ax, show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary'))
        fig.savefig(f'PPI_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()




def plot_ERP_diff(stretch=True):

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    cond_sel = ['FR_CV_1', 'MECA', 'CO2']

    t_start_ERP = ERP_time_vec[0]
    t_stop_ERP = ERP_time_vec[1]

    cluster_stats_type = 'manual_perm'

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=True)
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=False)

    sujet_group = ['allsujet', 'rep', 'non_rep']

    if stretch:
        time_vec = np.arange(stretch_point_TF)
    else:
        time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)
        mask_time_vec_zoomin = (time_vec >= -0.5) & (time_vec < 0.5) 

    conditions_diff = {'allsujet' : ['MECA', 'CO2'], 'rep' : ['MECA', 'CO2'], 'non_rep' : ['MECA', 'CO2'], 'repnorep' : ['FR_CV_1', 'MECA', 'CO2']}
    odor_list_diff = {'allsujet' : ['+', '-'], 'rep' : ['+', '-'], 'non_rep' : ['+', '-'], 'repnorep' : ['o', '+', '-']}

    ######## IDENTIFY MIN MAX ########

    print('IDENTIFY MIN MAX')

    #### identify min max for allsujet

    absmax_group = {}

    _absmax = np.array([])

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(cond_sel):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_stretch = xr_data.loc[:, cond, odor, nchan, :].median('sujet').values
                sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                data_stretch_up, data_stretch_down = data_stretch + sem, data_stretch - sem
                _absmax = np.concatenate([_absmax, np.abs(data_stretch_up), np.abs(data_stretch_down)])

    absmax_group['allsujet'] = _absmax.max()

    #### identify min max for rep/norep

    _absmax = np.array([])

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(cond_sel):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                for group in ['rep', 'non_rep']:

                    if group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])

                    data_stretch_up, data_stretch_down = data_stretch + sem, data_stretch - sem
                    _absmax = np.concatenate([_absmax, np.abs(data_stretch_up), np.abs(data_stretch_down)])

    absmax_group['repnorep'] = _absmax.max()

    ######## SUMMARY NCHAN INTRA ########

    #group = sujet_group[0]
    for group in sujet_group:

        print(f'plot intra {group}')

        if group == 'allsujet':
            n_sujet = xr_data.loc[:, cond, odor, nchan, :].shape[0]
        if group == 'rep':
            n_sujet = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0]
        elif group == 'non_rep':
            n_sujet = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0]

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_diff[group]))

            fig.set_figheight(10)
            fig.set_figwidth(7)

            if stretch:
                plt.suptitle(f'stretch {nchan} {group} {n_sujet} intra')
            else:
                plt.suptitle(f'nostretch {nchan} {group} {n_sujet} intra')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(conditions_diff[group]):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    ax = axs[odor_i, cond_i]

                    # pval_pre = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['pre_test_pval'].values[0]
                    # pval_post = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['p_unc'].values[0]
                    # if pval_pre <= 0.05 and pval_post <= 0.05: 
                    #     pval_title = pval_stars(pval_post)
                    # else:
                    #     pval_title = 'ns'

                    # ax.set_title(f"{cond} {pval_title}")

                    ax.set_title(f"{cond}")

                    if cond_i == 0:
                        ax.set_ylabel(f"{odor}")

                    if group == 'allsujet':
                        ax.set_ylim(-absmax_group['allsujet'], absmax_group['allsujet'])
                    else:
                        ax.set_ylim(-absmax_group['repnorep'], absmax_group['repnorep'])

                    ax.plot(time_vec, data_stretch, color='r')
                    ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                    ax.plot(time_vec, baseline, label='FR_CV_1', color='b')

                    ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                    clusters = cluster_stats_intra.loc[group, nchan, odor, cond, :]

                    if group == 'allsujet':
                        ax.fill_between(time_vec, -absmax_group['allsujet'], absmax_group['allsujet'], where=clusters.astype('int'), alpha=0.3, color='r')
                    else:
                        ax.fill_between(time_vec, -absmax_group['repnorep'], absmax_group['repnorep'], where=clusters.astype('int'), alpha=0.3, color='r')

                    # else:
                    #     clusters = cluster_stats_intra['intra'][group][nchan][odor][cond]['cluster']
                    #     cluster_p_values = cluster_stats_intra['intra'][group][nchan][odor][cond]['pval']
                    #     #c = clusters[0]
                    #     for i_c, c in enumerate(clusters):
                    #         c = c[0]
                    #         if cluster_p_values[i_c] <= 0.05:
                    #             h = ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color="r", alpha=0.3)
                    #         # else:
                    #         #     ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color='k', alpha=0.3)

                    #     # ax.legend((h,), ("cluster p-value < 0.05",))

                    ax.invert_yaxis()

                    if stretch:
                        if group == 'allsujet':
                            ax.vlines(stretch_point_TF/2, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
                        else:
                            ax.vlines(stretch_point_TF/2, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  
                    else:
                        if group == 'allsujet':
                            ax.vlines(0, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
                        else:
                            ax.vlines(0, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra'))

            if stretch:

                fig.savefig(f'stretch_{nchan}_intra_{group}.jpeg', dpi=150)

                if group == 'allsujet':
                    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'allsujet'))
                    fig.savefig(f'stretch_{nchan}_intra_{group}.jpeg', dpi=150)

                elif group == 'rep' or group == 'non_rep':
                    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'rep and norep'))
                    fig.savefig(f'stretch_{nchan}_intra_{group}.jpeg', dpi=150)

                fig.clf()
                plt.close('all')
                gc.collect()

            else:

                fig.savefig(f'nostretch_{nchan}_intra_{group}.jpeg', dpi=150)

                if group == 'allsujet':
                    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'allsujet'))
                    fig.savefig(f'nostretch_{nchan}_intra_{group}.jpeg', dpi=150)

                elif group == 'rep' or group == 'non_rep':
                    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'rep and norep'))
                    fig.savefig(f'nostretch_{nchan}_intra_{group}.jpeg', dpi=150)

                fig.clf()
                plt.close('all')
                gc.collect()
    
    

    ######## SUMMARY NCHAN INTER ########

    #group = sujet_group[-1]
    for group in sujet_group:

        print(f'plot inter {group}')

        if group == 'allsujet':
            n_sujet = xr_data.loc[:, cond, odor, nchan, :].shape[0]
        if group == 'rep':
            n_sujet = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0]
        elif group == 'non_rep':
            n_sujet = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0]

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list_diff[group]), ncols=len(cond_sel))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            plt.suptitle(f'{nchan} {group} {n_sujet} inter')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(cond_sel):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list_diff[group]):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, cond, 'o', nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[:, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].median('sujet').values
                        sem = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].median('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    ax = axs[odor_i, cond_i]

                    # pval_pre = df_stats_interintra['inter'].query(f"nchan == '{nchan}' and group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['pre_test_pval'].values[0]
                    # pval_post = df_stats_interintra['inter'].query(f"nchan == '{nchan}' and group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['p_unc'].values[0]
                    # if pval_pre <= 0.05 and pval_post <= 0.05: 
                    #     pval_title = pval_stars(pval_post)
                    # else:
                    #     pval_title = 'ns'

                    # ax.set_title(f"{cond} {pval_title}")

                    ax.set_title(f"{cond}")

                    if cond_i == 0:
                        ax.set_ylabel(f"{odor}")

                    if group == 'allsujet':
                        ax.set_ylim(-absmax_group['allsujet'], absmax_group['allsujet'])
                    else:
                        ax.set_ylim(-absmax_group['repnorep'], absmax_group['repnorep'])

                    ax.plot(time_vec, data_stretch, color='r')
                    ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                    ax.plot(time_vec, baseline, label='o', color='b')
                    ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                    if cluster_stats_type == 'manual_perm':

                        clusters = cluster_stats_inter.loc[group, nchan, odor, cond].values

                        if group == 'allsujet':
                            ax.fill_between(time_vec, -absmax_group['allsujet'], absmax_group['allsujet'], where=clusters.astype('int'), alpha=0.3, color='r')
                        else:
                            ax.fill_between(time_vec, -absmax_group['repnorep'], absmax_group['repnorep'], where=clusters.astype('int'), alpha=0.3, color='r')

                    else:

                        clusters = cluster_stats_inter[group][nchan][cond][odor]['cluster']
                        cluster_p_values = cluster_stats_inter[group][nchan][cond][odor]['pval']
                        #c = clusters[0]
                        for i_c, c in enumerate(clusters):
                            c = c[0]
                            if cluster_p_values[i_c] <= 0.05:
                                h = ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color="r", alpha=0.3)
                            # else:
                            #     ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color='k', alpha=0.3)

                        # ax.legend((h,), ("cluster p-value < 0.05",))

                    ax.invert_yaxis()

                    if stretch:
                        if group == 'allsujet':
                            ax.vlines(stretch_point_TF/2, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
                        else:
                            ax.vlines(stretch_point_TF/2, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  
                    else:
                        if group == 'allsujet':
                            ax.vlines(0, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
                        else:
                            ax.vlines(0, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'inter'))
            fig.savefig(f'{nchan}_inter_{group}.jpeg', dpi=150)

            if group == 'allsujet':
                os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'inter', 'allsujet'))
                fig.savefig(f'{nchan}_inter_{group}.jpeg', dpi=150)

            if group == 'rep' or group == 'non_rep':
                os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'inter', 'rep and norep'))
                fig.savefig(f'{nchan}_inter_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()
            

    ######## SUMMARY NCHAN REPNOREP ########

    print(f'plot repnorep')

    n_sujet = f"rep:{xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0]}/norep:{xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0]}" 

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel))

        fig.set_figheight(10)
        fig.set_figwidth(7)

        if stretch:
            plt.suptitle(f'stretch {nchan} {group} {n_sujet} intra')
        else:
            plt.suptitle(f'nostretch {nchan} {group} {n_sujet} intra')

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(cond_sel):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].median('sujet').values
                sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
                baseline = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].median('sujet').values
                sem_baseline = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])
                # if cond != 'VS':
                #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                ax = axs[odor_i, cond_i]

                # pval_pre = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['pre_test_pval'].values[0]
                # pval_post = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['p_unc'].values[0]
                # if pval_pre <= 0.05 and pval_post <= 0.05: 
                #     pval_title = pval_stars(pval_post)
                # else:
                #     pval_title = 'ns'

                # ax.set_title(f"{cond} {pval_title}")

                ax.set_title(f"{cond}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                ax.set_ylim(-absmax_group['repnorep'], absmax_group['repnorep'])

                ax.plot(time_vec, data_stretch, color='r')
                ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                ax.plot(time_vec, baseline, label='norep', color='b')

                ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

                clusters = cluster_stats_rep_norep.loc[nchan, odor, cond, :]

                ax.fill_between(time_vec, -absmax_group['repnorep'], absmax_group['repnorep'], where=clusters.astype('int'), alpha=0.3, color='r')

                ax.invert_yaxis()

                if stretch:
                    ax.vlines(stretch_point_TF/2, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  
                else:
                    ax.vlines(0, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  


        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra'))

        if stretch:

            fig.savefig(f'stretch_{nchan}_intra_{group}.jpeg', dpi=150)

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'repnorep'))
            fig.savefig(f'stretch_{nchan}_intra_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

        else:

            fig.savefig(f'nostretch_{nchan}_intra_{group}.jpeg', dpi=150)

            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'repnorep'))
            fig.savefig(f'nostretch_{nchan}_intra_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()
    

    return












    ######## SUMMARY NCHAN INTER ZOOMIN ########

    print('PLOT SUMMARY ERP')

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list_diff), ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            plt.suptitle(f'{nchan} {group} {n_sujet} inter')

            #cond_i, cond = 1, 'MECA'
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list_diff):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, cond, 'o', nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[:, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, 'o', nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    ax = axs[odor_i, cond_i]

                    # pval_pre = df_stats_interintra['inter'].query(f"nchan == '{nchan}' and group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['pre_test_pval'].values[0]
                    # pval_post = df_stats_interintra['inter'].query(f"nchan == '{nchan}' and group == '{group}' and cond == '{cond}'").query(f"A == '{odor}' or B == '{odor}'")['p_unc'].values[0]
                    # if pval_pre <= 0.05 and pval_post <= 0.05: 
                    #     pval_title = pval_stars(pval_post)
                    # else:
                    #     pval_title = 'ns'

                    # ax.set_title(f"{cond} {pval_title}")

                    ax.set_title(f"{cond}")

                    if cond_i == 0:
                        ax.set_ylabel(f"{odor}")

                    if group == 'allsujet':
                        ax.set_ylim(-absmax_group['allsujet'], absmax_group['allsujet'])
                    else:
                        ax.set_ylim(-absmax_group['repnorep'], absmax_group['repnorep'])

                    ax.plot(time_vec[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin], color='r')
                    ax.fill_between(time_vec[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin]+sem[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin]-sem[mask_time_vec_zoomin], alpha=0.25, color='m')

                    ax.plot(time_vec[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin], label='o', color='b')
                    ax.fill_between(time_vec[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin]+sem_baseline[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin]-sem_baseline[mask_time_vec_zoomin], alpha=0.25, color='c')

                    if cluster_stats_type == 'manual_perm':
                        clusters = cluster_stats['inter'][group][nchan][cond][odor]

                        if group == 'allsujet':
                            ax.fill_between(time_vec[mask_time_vec_zoomin], -absmax_group['allsujet'], absmax_group['allsujet'], where=clusters[mask_time_vec_zoomin].astype('int'), alpha=0.3, color='r')
                        else:
                            ax.fill_between(time_vec[mask_time_vec_zoomin], -absmax_group['repnorep'], absmax_group['repnorep'], where=clusters[mask_time_vec_zoomin].astype('int'), alpha=0.3, color='r')

                    else:

                        clusters = cluster_stats['inter'][group][nchan][cond][odor]['cluster']
                        cluster_p_values = cluster_stats['inter'][group][nchan][cond][odor]['pval']
                        #c = clusters[0]
                        for i_c, c in enumerate(clusters):
                            c = c[0]
                            if cluster_p_values[i_c] <= 0.05:
                                h = ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color="r", alpha=0.3)
                            # else:
                            #     ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color='k', alpha=0.3)

                        # ax.legend((h,), ("cluster p-value < 0.05",))

                    ax.invert_yaxis()

                    if group == 'allsujet':
                        ax.vlines(0, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
                    else:
                        ax.vlines(0, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  

            fig.tight_layout()
            plt.legend()

            # plt.show()
            
            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'inter', 'zoomin'))
            fig.savefig(f'{nchan}_inter_{group}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()

    ######## TOPOPLOT DIFF ########

    print('TOPOPLOT DIFF')

    time_diff = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    time_diff_step = 0.5
    dict_diff = {'group' : sujet_group, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_diff}
    data_diff = np.zeros((len(sujet_group), len(conditions_diff), len(odor_list), len(chan_list_eeg), len(time_diff)))

    xr_data_diff_mean = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_point = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())

    #group = sujet_group[0]
    for group in sujet_group:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].mean('sujet').values
                        baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        
                    diff = baseline - data_stretch 

                    for time_start in time_diff:

                        time_stop = time_start + time_diff_step

                        mask_sel_time_diff_mean = (time_vec >= time_start) & (time_vec <= time_stop)
                        mask_sel_time_diff_point = (time_vec >= time_start-0.01) & (time_vec <= time_start+0.01)

                        xr_data_diff_mean.loc[group, cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_mean])

                        xr_data_diff_point.loc[group, cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_point])


    #### select params
    # xr_data_diff = xr_data_diff_mean
    xr_data_diff = xr_data_diff_point

    classing_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    #### scales
    # scale_min = xr_data_diff.loc['allsujet', :, :, :, :].min().values
    # scale_max = xr_data_diff.loc['allsujet', :, :, :, :].max().values

    max_abs = np.array([np.abs(xr_data_diff.loc['allsujet', :, :, :, :].min().values), np.abs(xr_data_diff.loc['allsujet', :, :, :, :].max().values)]).max()
    scale_min, scale_max = -max_abs, max_abs

    #### plot
    # group = sujet_group[0]
    for group in sujet_group:

        for time_start_i, time_start in enumerate(time_diff):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_diff))

            if group == 'allsujet':
                plt.suptitle(f'{group} {time_start} diff (s{xr_data.shape[0]})')
            if group == 'rep':
                plt.suptitle(f'{group} {time_start} diff (s{len(sujet_best_list_rev)})')
            if group == 'non_rep':
                plt.suptitle(f'{group} {time_start} diff (s{len(sujet_no_respond_rev)})')

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #c, cond = 0, 'FR_CV_1'
            for c, cond in enumerate(conditions_diff):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(cond, fontweight='bold', rotation=0)
                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    data_plot = xr_data_diff.loc[group, cond, odor, :, time_start].values

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'diff_baseline_{group}_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()

    #### for cond
    for c, cond in enumerate(conditions_diff):

        for time_start_i, time_start in enumerate(time_diff):

            #### scales
            scale_min = xr_data_diff.loc[group, cond, :, :, :].values.min()
            scale_max = xr_data_diff.loc[group, cond, :, :, :].values.max()

            #### plot
            fig, axs = plt.subplots(nrows=len(odor_list), ncols=2)
            plt.suptitle(f'{time_start} diff')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for c, sujet_type in enumerate(['respond', 'no_respond']):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        if sujet_type == 'respond':
                            ax.set_title(f'{cond} {sujet_type} (s{len(sujet_best_list_rev)})', fontweight='bold', rotation=0)
                        if sujet_type == 'no_respond':
                            ax.set_title(f'{cond} {sujet_type} (s{len(sujet_no_respond_rev)})', fontweight='bold', rotation=0)

                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    if sujet_type == 'respond':
                        data_plot = xr_data_diff.loc['rep', cond, odor, :, time_start].values
                    if sujet_type == 'no_respond':
                        data_plot = xr_data_diff.loc['non_rep', cond, odor, :, time_start].values

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'{cond}_diff_baseline_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()

    #### across time
    for group in sujet_group:

        for c, cond in enumerate(conditions_diff):

            #### scales
            # scale_min = xr_data_diff.loc[group, cond, :, :, :].values.min()
            # scale_max = xr_data_diff.loc[group, cond, :, :, :].values.max()

            max_abs = np.array([np.abs(xr_data_diff.loc[group, cond, :, :, :].values.min()), np.abs(xr_data_diff.loc[group, cond, :, :, :].values.max())]).max()
            scale_min, scale_max = -max_abs, max_abs

            #### plot
            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(time_diff))
            plt.suptitle(f'{group} {cond} diff')
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for c, time_start in enumerate(time_diff):

                #r, odor_i = 0, odor_list[0]
                for r, odor in enumerate(odor_list):

                    #### plot
                    ax = axs[r, c]

                    if r == 0:
                        ax.set_title(f'{time_start}', fontweight='bold', rotation=0)

                    if c == 0:
                        ax.set_ylabel(f'{odor}')
                    
                    data_plot = xr_data_diff.loc[group, cond, odor, :, time_start].values

                    mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
            fig.savefig(f'time_{cond}_{group}_diff_baseline_topo.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()


    # if stretch == False:

    #     ######## SUMMARY NCHAN INTRA ZOOMIN ########

    #     #group = sujet_group[0]
    #     for group in sujet_group:

    #         if group == 'allsujet':
    #             n_sujet = xr_data.loc[:, cond, odor, nchan, :].shape[0]
    #         if group == 'rep':
    #             n_sujet = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0]
    #         elif group == 'non_rep':
    #             n_sujet = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0]

    #         #nchan_i, nchan = 0, chan_list_eeg[0]
    #         for nchan_i, nchan in enumerate(chan_list_eeg):

    #             fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions_diff))

    #             fig.set_figheight(10)
    #             fig.set_figwidth(10)

    #             plt.suptitle(f'{nchan} {group} {n_sujet} intra')

    #             #cond_i, cond = 1, 'MECA'
    #             for cond_i, cond in enumerate(conditions_diff):

    #                 #odor_i, odor = 0, odor_list[0]
    #                 for odor_i, odor in enumerate(odor_list):

    #                     if group == 'allsujet':
    #                         data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
    #                         sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
    #                         baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
    #                         sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, nchan, :].shape[0])
    #                         # if cond != 'VS':
    #                         #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
    #                         #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
    #                     elif group == 'rep':
    #                         data_stretch = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
    #                         sem = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].shape[0])
    #                         baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
    #                         sem_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].shape[0])
    #                         # if cond != 'VS':
    #                         #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
    #                         #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
    #                     elif group == 'non_rep':
    #                         data_stretch = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].mean('sujet').values
    #                         sem = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].shape[0])
    #                         baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
    #                         sem_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].shape[0])
    #                         # if cond != 'VS':
    #                         #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
    #                         #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

    #                     ax = axs[odor_i, cond_i]

    #                     # pval_pre = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['pre_test_pval'].values[0]
    #                     # pval_post = df_stats_interintra['intra'].query(f"nchan == '{nchan}' and group == '{group}' and odor == '{odor}'").query(f"A == '{cond}' or B == '{cond}'")['p_unc'].values[0]
    #                     # if pval_pre <= 0.05 and pval_post <= 0.05: 
    #                     #     pval_title = pval_stars(pval_post)
    #                     # else:
    #                     #     pval_title = 'ns'

    #                     # ax.set_title(f"{cond} {pval_title}")

    #                     ax.set_title(f"{cond}")

    #                     if cond_i == 0:
    #                         ax.set_ylabel(f"{odor}")

    #                     if group == 'allsujet':
    #                         ax.set_ylim(-absmax_group['allsujet'], absmax_group['allsujet'])
    #                     else:
    #                         ax.set_ylim(-absmax_group['repnorep'], absmax_group['repnorep'])

    #                     ax.plot(time_vec[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin], color='r')
    #                     ax.fill_between(time_vec[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin]+sem[mask_time_vec_zoomin], data_stretch[mask_time_vec_zoomin]-sem[mask_time_vec_zoomin], alpha=0.25, color='m')

    #                     ax.plot(time_vec[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin], label='FR_CV_1', color='b')
    #                     ax.fill_between(time_vec[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin]+sem_baseline[mask_time_vec_zoomin], baseline[mask_time_vec_zoomin]-sem_baseline[mask_time_vec_zoomin], alpha=0.25, color='c')

    #                     if cluster_stats_type == 'manual_perm':
    #                         clusters = cluster_stats['intra'][group][nchan][odor][cond]
    #                         if group == 'allsujet':
    #                             ax.fill_between(time_vec[mask_time_vec_zoomin], -absmax_group['allsujet'], absmax_group['allsujet'], where=clusters[mask_time_vec_zoomin].astype('int'), alpha=0.3, color='r')
    #                         else:
    #                             ax.fill_between(time_vec[mask_time_vec_zoomin], -absmax_group['repnorep'], absmax_group['repnorep'], where=clusters[mask_time_vec_zoomin].astype('int'), alpha=0.3, color='r')

    #                     else:
    #                         clusters = cluster_stats['intra'][group][nchan][odor][cond]['cluster']
    #                         cluster_p_values = cluster_stats['intra'][group][nchan][odor][cond]['pval']
    #                         #c = clusters[0]
    #                         for i_c, c in enumerate(clusters):
    #                             c = c[0]
    #                             if cluster_p_values[i_c] <= 0.05:
    #                                 h = ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color="r", alpha=0.3)
    #                             # else:
    #                             #     ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color='k', alpha=0.3)

    #                         # ax.legend((h,), ("cluster p-value < 0.05",))

    #                     ax.invert_yaxis()

    #                     if group == 'allsujet':
    #                         ax.vlines(0, ymin=-absmax_group['allsujet'], ymax=absmax_group['allsujet'], colors='g')  
    #                     else:
    #                         ax.vlines(0, ymin=-absmax_group['repnorep'], ymax=absmax_group['repnorep'], colors='g')  

    #             fig.tight_layout()
    #             plt.legend()

    #             # plt.show()

    #             #### save
    #             os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra', 'zoomin'))
    #             fig.savefig(f'{nchan}_intra_{group}.jpeg', dpi=150)

    #             fig.clf()
    #             plt.close('all')
    #             gc.collect()







def plot_ERP_rep_norep(xr_data, cluster_stats_rep_norep, cluster_stats_type):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

    ######## SUMMARY NCHAN ########

    print('PLOT SUMMARY ERP')

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions[2]
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                data_stretch = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                n_sujet_rep = xr_data.loc[sujet_best_list, cond, odor, nchan, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

                data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                n_sujet_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :]['sujet'].shape[0]

                scales_val['min'].append(data_stretch.min())
                scales_val['max'].append(data_stretch.max())

        scales_val['min'] = np.array(scales_val['min']).min()
        scales_val['max'] = np.array(scales_val['max']).max()

        plt.suptitle(f'{nchan} rep:{n_sujet_rep} no_rep:{n_sujet_norep}')

        #cond_i, cond = 1, 'MECA'
        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_stretch_rep = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                sem_rep = xr_data.loc[sujet_best_list, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list, cond, odor, nchan, :].shape[0])

                data_stretch_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                sem_norep = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, cond, odor, nchan, :].shape[0])

                ax.set_title(f"{cond}")

                ax.set_ylim(allplot_erp_ylim[0], allplot_erp_ylim[1])

                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                ax.plot(time_vec, data_stretch_rep, label='rep', color='r')
                ax.fill_between(time_vec, data_stretch_rep+sem_rep, data_stretch_rep-sem_rep, alpha=0.25, color='m')

                ax.plot(time_vec, data_stretch_norep, label='no_rep', color='b')
                ax.fill_between(time_vec, data_stretch_norep+sem_norep, data_stretch_norep-sem_norep, alpha=0.25, color='c')

                if cluster_stats_type == 'manual_perm':

                    clusters = cluster_stats_rep_norep[nchan][odor][cond]
                    ax.fill_between(time_vec, allplot_erp_ylim[0], allplot_erp_ylim[1], where=clusters.astype('int'), alpha=0.3, color='r')

                else:

                    clusters = cluster_stats_rep_norep[nchan][odor][cond]['cluster']
                    cluster_p_values = cluster_stats_rep_norep[nchan][odor][cond]['pval']
                    #c = clusters[0]
                    for i_c, c in enumerate(clusters):
                        c = c[0]
                        if cluster_p_values[i_c] <= 0.05:
                            h = ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color="r", alpha=0.3)
                        # else:
                        #     ax.axvspan(time_vec[c.start], time_vec[c.stop - 1], color='k', alpha=0.3)

                # ax.legend((h,), ("cluster p-value < 0.05",))
                
                ax.invert_yaxis()

                ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

        fig.tight_layout()
        plt.legend()

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'inter'))
        fig.savefig(f'{nchan}_rep_norep.jpeg', dpi=150)
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'intra'))
        fig.savefig(f'{nchan}_rep_norep.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()


    ######## TOPOPLOT DIFF ########

    print('TOPOPLOT DIFF')

    time_diff = [-2, -1.5, -1, -0.5, 0, 0.5, 1]
    time_diff_step = 0.5
    dict_diff = {'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg, 'time' : time_diff}
    data_diff = np.zeros((len(conditions), len(odor_list), len(chan_list_eeg), len(time_diff)))

    dict_diff_alltime = {'metric' : ['minmax', 'sum'], 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_diff_alltime = np.zeros((len(['minmax', 'sum']), len(conditions), len(odor_list), len(chan_list_eeg)))

    xr_data_diff_mean = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_point = xr.DataArray(data=data_diff, dims=dict_diff.keys(), coords=dict_diff.values())
    xr_data_diff_alltime = xr.DataArray(data=data_diff_alltime, dims=dict_diff_alltime.keys(), coords=dict_diff_alltime.values())

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):
                    
                data_stretch_rep = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                
                data_stretch_norep = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                diff = data_stretch_norep.mean(axis=0) - data_stretch_rep.mean(axis=0)
                
                xr_data_diff_alltime.loc['sum', cond, odor, nchan] = np.sum(diff)
                xr_data_diff_alltime.loc['minmax', cond, odor, nchan] = diff.max() - diff.min()

                for time_start in time_diff:

                    time_stop = time_start + time_diff_step

                    mask_sel_time_diff_mean = (time_vec >= time_start) & (time_vec <= time_stop)
                    mask_sel_time_diff_point = (time_vec >= time_start-0.01) & (time_vec <= time_start+0.01)

                    xr_data_diff_mean.loc[cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_mean])

                    xr_data_diff_point.loc[cond, odor, nchan, time_start] = np.median(diff[mask_sel_time_diff_point])


    #### select params
    # xr_data_diff = xr_data_diff_mean
    xr_data_diff = xr_data_diff_point

    classing_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

    #### scales
    scale_min = xr_data_diff.min().values
    scale_max = xr_data_diff.max().values

    #### plot
    for time_start_i, time_start in enumerate(time_diff):

        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))

        plt.suptitle(f'{time_start} diff rep_norep')

        fig.set_figheight(10)
        fig.set_figwidth(10)

        #c, cond = 0, 'FR_CV_1'
        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(cond, fontweight='bold', rotation=0)
                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff.loc[cond, odor, :, time_start].values*-1

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'diff_repnorep_{classing_letter[time_start_i]}{time_start}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### across time
    for c, cond in enumerate(conditions):

        #### scales
        scale_min = xr_data_diff.loc[cond, :, :, :].values.min()
        scale_max = xr_data_diff.loc[cond, :, :, :].values.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(time_diff))
        plt.suptitle(f'{cond} diff')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, time_start in enumerate(time_diff):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{time_start}', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff.loc[cond, odor, :, time_start].values

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        plt.suptitle(cond)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'time_repnorep_{cond}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    #### all time

    #### scales
    for metric_alltime in ['minmax', 'sum']:

        scale_min = xr_data_diff_alltime.loc[metric_alltime].values.min()
        scale_max = xr_data_diff_alltime.loc[metric_alltime].values.max()

        #### plot
        fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(conditions))
        plt.suptitle(f'alltime diff {metric_alltime}')
        fig.set_figheight(10)
        fig.set_figwidth(10)

        for c, cond in enumerate(conditions):

            #r, odor_i = 0, odor_list[0]
            for r, odor in enumerate(odor_list):

                #### plot
                ax = axs[r, c]

                if r == 0:
                    ax.set_title(f'{cond}', fontweight='bold', rotation=0)

                if c == 0:
                    ax.set_ylabel(f'{odor}')
                
                data_plot = xr_data_diff_alltime.loc[metric_alltime, cond, odor, :].values

                mne.viz.plot_topomap(data_plot, info, axes=ax, vlim=(scale_min, scale_max), show=False)

        # plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_diff'))
        fig.savefig(f'alltime_repnorep_{metric_alltime}_topo.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()


















########################################
######## ERP RESPONSE PROFILE ########
########################################




def plot_ERP_response_profile(xr_data, xr_data_sem):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    # t_start_PPI = PPI_time_vec[0]
    # t_stop_PPI = PPI_time_vec[1]

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    mask_sujet_rep = []
    for sujet in sujet_list:
        if sujet in sujet_best_list:
            mask_sujet_rep.append(True)
        else:
            mask_sujet_rep.append(False)
    mask_sujet_rep = np.array(mask_sujet_rep)   

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

    mask_frontal = []
    for nchan in chan_list_eeg:
        if nchan in ['Fp1', 'Fz', 'Fp2']:
            mask_frontal.append(True)
        else:
            mask_frontal.append(False)
    mask_frontal = np.array(mask_frontal)

    mask_central = []
    for nchan in chan_list_eeg:
        if nchan in ['C3', 'Cz', 'C4']:
            mask_central.append(True)
        else:
            mask_central.append(False)
    mask_central = np.array(mask_central)

    mask_occipital = []
    for nchan in chan_list_eeg:
        if nchan in ['O1', 'Oz', 'O2']:
            mask_occipital.append(True)
        else:
            mask_occipital.append(False)
    mask_occipital = np.array(mask_occipital)

    mask_temporal = []
    for nchan in chan_list_eeg:
        if nchan in ['T7', 'TP9', 'TP10', 'T8']:
            mask_temporal.append(True)
        else:
            mask_temporal.append(False)
    mask_temporal = np.array(mask_temporal)

    ######## SUJET ########

    dict_time = {'metric' : ['time', 'amp'], 'sujet' : sujet_list, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_time = np.zeros((2, len(sujet_list), len(conditions_diff), len(odor_list), len(chan_list_eeg))) 

    xr_erp_profile = xr.DataArray(data_time, coords=dict_time.values(), dims=dict_time.keys())

    print('SUJET')

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        print(sujet)

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 0, 'CO2'
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values
                    sem = xr_data_sem.loc[sujet, cond, odor, nchan, :].values
                    baseline = xr_data.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                    sem_baseline = xr_data_sem.loc[sujet, 'FR_CV_1', odor, nchan, :].values

                    # if ((data_stretch + sem) < (baseline - sem_baseline)).sum() != 0:

                    ERP_time_i = np.argmax(np.abs(data_stretch - baseline))

                    xr_erp_profile.loc['time', sujet, cond, odor, nchan] = time_vec[ERP_time_i]
                    xr_erp_profile.loc['amp', sujet, cond, odor, nchan] = (data_stretch - baseline)[ERP_time_i]

                    if debug:

                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        plt.plot(time_vec, data_stretch, label=cond, color='r')
                        plt.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        plt.plot(time_vec, baseline, label='VS', color='b')
                        plt.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        plt.gca().invert_yaxis()

                        plt.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                        plt.show()

                        plt.plot(time_vec, data_stretch - baseline, label=cond, color='r')
                        plt.show()

                        plt.plot(time_vec, np.abs(data_stretch - baseline), label=cond, color='r')
                        plt.show()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile.loc['time', :, cond, odor, nchan].values.min())
                scales_val_time['max'].append(xr_erp_profile.loc['time', :, cond, odor, nchan].values.max())

                scales_val_amp['min'].append(xr_erp_profile.loc['amp', :, cond, odor, nchan].values.min())
                scales_val_amp['max'].append(xr_erp_profile.loc['amp', :, cond, odor, nchan].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_time = xr_erp_profile.loc['time', :, cond, odor, nchan].values
                data_amp = xr_erp_profile.loc['amp', :, cond, odor, nchan].values
                
                ax.scatter(data_time[mask_sujet_rep], data_amp[mask_sujet_rep], label='rep')
                ax.scatter(data_time[~mask_sujet_rep], data_amp[~mask_sujet_rep], label='no_rep')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.legend()
        plt.suptitle(f"{nchan}")

        # plt.show()

        fig.savefig(f'{nchan}_allsujet.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()

    if debug:

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            for sujet in sujet_list:

                fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

                #cond_i, cond = 0, 'CO2'
                for cond_i, cond in enumerate(conditions_diff):

                    #odor_i, odor = 2, odor_list[2]
                    for odor_i, odor in enumerate(odor_list):

                        ax = axs[odor_i, cond_i]

                        if odor_i == 0:
                            ax.set_title(cond)
                        if cond_i == 0:
                            ax.set_ylabel(odor)

                        data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values
                        sem = xr_data_sem.loc[sujet, cond, odor, nchan, :].values
                        baseline = xr_data.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                        sem_baseline = xr_data_sem.loc[sujet, 'FR_CV_1', odor, nchan, :].values
                        
                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        ax.plot(time_vec, data_stretch, label=cond, color='r')
                        ax.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        ax.plot(time_vec, baseline, label='VS', color='b')
                        ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        ax.invert_yaxis()

                        ax.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                plt.suptitle(f"{sujet}")
                plt.show()

    ######## GROUP ########

    dict_time = {'metric' : ['time', 'amp'], 'group' : sujet_group, 'cond' : conditions_diff, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    data_time = np.zeros((2, len(sujet_group), len(conditions_diff), len(odor_list), len(chan_list_eeg))) 

    xr_erp_profile_group = xr.DataArray(data_time, coords=dict_time.values(), dims=dict_time.keys())

    print('GROUP')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #group = sujet_group[0]
    for group in sujet_group:

        print(group)

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            #cond_i, cond = 0, 'CO2'
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        data_stretch = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[:, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[:, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[:, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[:, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'rep':
                        data_stretch = xr_data.loc[sujet_best_list, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_best_list, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_best_list, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_best_list, cond, odor, nchan, 'down', :].mean('sujet').values
                    elif group == 'non_rep':
                        data_stretch = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].mean('sujet').values
                        sem = xr_data.loc[sujet_no_respond, cond, odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, cond, odor, nchan, :].shape[0])
                        baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                        sem_baseline = xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].std('sujet').values / np.sqrt(xr_data.loc[sujet_no_respond, 'FR_CV_1', odor, nchan, :].shape[0])
                        # if cond != 'VS':
                        #     data_surr_up = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'up', :].mean('sujet').values
                        #     data_surr_down = xr_surr.loc[sujet_no_respond, cond, odor, nchan, 'down', :].mean('sujet').values

                    # if ((data_stretch + sem) < (baseline - sem_baseline)).sum() != 0:

                    ERP_time_i = np.argmax(np.abs(data_stretch - baseline))

                    xr_erp_profile_group.loc['time', group, cond, odor, nchan] = time_vec[ERP_time_i]
                    xr_erp_profile_group.loc['amp', group, cond, odor, nchan] = (data_stretch - baseline)[ERP_time_i]

                    if debug:

                        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

                        plt.plot(time_vec, data_stretch, label=cond, color='r')
                        plt.fill_between(time_vec, data_stretch+sem, data_stretch-sem, alpha=0.25, color='m')

                        plt.plot(time_vec, baseline, label='VS', color='b')
                        plt.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')
                        
                        plt.gca().invert_yaxis()

                        plt.vlines(0, ymin=allplot_erp_ylim[0], ymax=allplot_erp_ylim[1], colors='g')  

                        plt.show()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    #nchan_i, nchan = 0, chan_list_eeg[0]
    for nchan_i, nchan in enumerate(chan_list_eeg):

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        for group in sujet_group:

            #cond_i, cond = 2, conditions_diff[2]
            for cond_i, cond in enumerate(conditions_diff):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    scales_val_time['min'].append(xr_erp_profile_group.loc['time', :, cond, odor, nchan].values.min())
                    scales_val_time['max'].append(xr_erp_profile_group.loc['time', :, cond, odor, nchan].values.max())

                    scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values.min())
                    scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)

                data_time = xr_erp_profile_group.loc['time', :, cond, odor, nchan].values
                data_amp = xr_erp_profile_group.loc['amp', :, cond, odor, nchan].values
                
                ax.scatter(xr_erp_profile_group.loc['time', 'allsujet', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'allsujet', cond, odor, nchan].values, label='allsujet')
                ax.scatter(xr_erp_profile_group.loc['time', 'rep', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'rep', cond, odor, nchan].values, label='rep')
                ax.scatter(xr_erp_profile_group.loc['time', 'non_rep', cond, odor, nchan].values, xr_erp_profile_group.loc['amp', 'non_rep', cond, odor, nchan].values, label='non_rep')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.legend()
        plt.suptitle(f"{nchan}")

        # plt.show()

        fig.savefig(f'{nchan}_allgroup.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()       

    #### plot allgroup allchan
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    scales_val_time = {'min' : [], 'max' : []}
    scales_val_amp = {'min' : [], 'max' : []}

    #cond_i, cond = 2, conditions_diff[2]
    for cond_i, cond in enumerate(conditions_diff):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            scales_val_time['min'].append(xr_erp_profile_group.loc['time', :, cond, odor, :].values.min())
            scales_val_time['max'].append(xr_erp_profile_group.loc['time', :, cond, odor, :].values.max())

            scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', :, cond, odor, :].values.min())
            scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', :, cond, odor, :].values.max())

    min_time = np.array(scales_val_time['min']).min()
    max_time = np.array(scales_val_time['max']).max()

    min_amp = np.array(scales_val_amp['min']).min()
    max_amp = np.array(scales_val_amp['max']).max()

    #cond_i, cond = 0, 'CO2'
    for cond_i, cond in enumerate(conditions_diff):

        #odor_i, odor = 2, odor_list[2]
        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i, cond_i]

            if odor_i == 0:
                ax.set_title(cond)
            if cond_i == 0:
                ax.set_ylabel(odor)
            
            ax.scatter(xr_erp_profile_group.loc['time', 'allsujet', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'allsujet', cond, odor, :].values, label='allsujet')
            ax.scatter(xr_erp_profile_group.loc['time', 'rep', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'rep', cond, odor, :].values, label='rep')
            ax.scatter(xr_erp_profile_group.loc['time', 'non_rep', cond, odor, :].values, xr_erp_profile_group.loc['amp', 'non_rep', cond, odor, :].values, label='non_rep')

            ax.set_ylim(min_amp, max_amp)
            ax.set_xlim(min_time, max_time)

            ax.invert_yaxis()

            ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
            ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

    plt.suptitle(f"{group}")
    plt.legend()

    # plt.show()

    fig.savefig(f'{nchan}_allgroup.jpeg', dpi=150)
    fig.clf()
    plt.close('all')
    gc.collect()       

    #### plot allchan
    for group in sujet_group:

        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.min())
                scales_val_time['max'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.max())

                scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.min())
                scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)
                
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_frontal], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_frontal], label='frontal')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_central], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_central], label='central')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_occipital], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_occipital], label='occipital')
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values[mask_temporal], xr_erp_profile_group.loc['amp', group, cond, odor, :].values[mask_temporal], label='temporal')

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.suptitle(f"{group}")
        plt.legend()

        # plt.show()

        fig.savefig(f'allchan_topo_{group}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()   



        fig, axs = plt.subplots(ncols=len(conditions_diff), nrows=len(odor_list))

        fig.set_figheight(10)
        fig.set_figwidth(10)

        scales_val_time = {'min' : [], 'max' : []}
        scales_val_amp = {'min' : [], 'max' : []}

        #cond_i, cond = 2, conditions_diff[2]
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                scales_val_time['min'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.min())
                scales_val_time['max'].append(xr_erp_profile_group.loc['time', group, cond, odor, :].values.max())

                scales_val_amp['min'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.min())
                scales_val_amp['max'].append(xr_erp_profile_group.loc['amp', group, cond, odor, :].values.max())

        min_time = np.array(scales_val_time['min']).min()
        max_time = np.array(scales_val_time['max']).max()

        min_amp = np.array(scales_val_amp['min']).min()
        max_amp = np.array(scales_val_amp['max']).max()

        #cond_i, cond = 0, 'CO2'
        for cond_i, cond in enumerate(conditions_diff):

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if odor_i == 0:
                    ax.set_title(cond)
                if cond_i == 0:
                    ax.set_ylabel(odor)
                
                ax.scatter(xr_erp_profile_group.loc['time', group, cond, odor, :].values, xr_erp_profile_group.loc['amp', group, cond, odor, :].values)

                ax.set_ylim(min_amp, max_amp)
                ax.set_xlim(min_time, max_time)

                ax.invert_yaxis()

                ax.vlines(0, ymin=min_amp, ymax=max_amp, colors='k')
                ax.hlines(0, xmin=min_time, xmax=max_time, colors='g')

        plt.suptitle(f"{group}")

        # plt.show()

        fig.savefig(f'allchan_allchan_{group}.jpeg', dpi=150)
        fig.clf()
        plt.close('all')
        gc.collect()    














########################################
######## ERP RESPONSE STATS ########
########################################



def plot_erp_response_stats(xr_data):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    # t_start_PPI = PPI_time_vec[0]
    # t_stop_PPI = PPI_time_vec[1]

    t_start_PPI = ERP_time_vec[0]
    t_stop_PPI = ERP_time_vec[1]

    PPI_lm_start = PPI_lm_time[0]
    PPI_lm_stop = PPI_lm_time[1] 

    sujet_group = ['allsujet', 'rep', 'non_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
    time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
    mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

    conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

    sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])

    #### generate df
    # time_vec_mask = xr_data['time'].values[(time_vec >= -1) & (time_vec <= 0)]
    time_vec_mask = xr_data['time'].values[(time_vec >= -2) & (time_vec <= 2)]

    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].sum('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)

    df_min = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
    df_max = xr_data.loc[:, :, :, :, time_vec_mask].max('time').to_dataframe(name='val').reset_index(drop=False)

    df_minmax = df_min.copy()
    df_minmax['val'] = np.abs(df_min['val'].values) + np.abs(df_max['val'].values)

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

    predictor = 'cond'
    outcome = 'val'

    for group in sujet_group:

        for nchan in chan_list_eeg:

            fig, axs = plt.subplots(ncols=len(odor_list))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                ax.set_ylabel(odor)

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}'")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")

                ax = auto_stats(df_stats, predictor, outcome, ax=ax, subject='sujet', design='within', mode='box', transform=False, verbose=False)
                
            plt.suptitle(f"{odor_list} {nchan} {group}")
            plt.tight_layout()

            # plt.show()

            fig.savefig(f'stats_inter_{nchan}_{group}.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()    

    predictor = 'odor'
    outcome = 'val'

    for group in sujet_group:

        for nchan in chan_list_eeg:

            fig, axs = plt.subplots(ncols=len(conditions))

            fig.set_figheight(10)
            fig.set_figwidth(10)

            #cond_i, cond = 2, conditions[2]
            for cond_i, cond in enumerate(conditions):

                ax = axs[cond_i]

                ax.set_ylabel(odor)

                if group == 'allsujet':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}'")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}'")
                if group == 'rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                if group == 'non_rep':
                    # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                    df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")

                ax = auto_stats(df_stats, predictor, outcome, ax=ax, subject='sujet', design='within', mode='box', transform=False, verbose=False)
                
            plt.suptitle(f"{conditions} {nchan} {group}")
            plt.tight_layout()

            # plt.show()

            fig.savefig(f'stats_intra_{nchan}_{group}.jpeg', dpi=150)
            fig.clf()
            plt.close('all')
            gc.collect()  




def get_df_stats(xr_data):

    if os.path.exists(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff', 'df_stats_all_intra.xlsx')):

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff'))

        print('ALREADY COMPUTED', flush=True)

        df_stats_all_intra = pd.read_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter = pd.read_excel('df_stats_all_inter.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter}

    else:

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        # t_start_PPI = PPI_time_vec[0]
        # t_stop_PPI = PPI_time_vec[1]

        t_start_PPI = ERP_time_vec[0]
        t_stop_PPI = ERP_time_vec[1]

        sujet_group = ['allsujet', 'rep', 'non_rep']

        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)
        mask_time_PPI = (time_vec > -2.5) & (time_vec < 0)

        conditions_diff = ['MECA', 'CO2', 'FR_CV_2']

        #### generate df
        # time_vec_mask = xr_data['time'].values[(time_vec >= -1) & (time_vec <= 0)]
        time_vec_mask = xr_data['time'].values[(time_vec >= -2) & (time_vec <= 2)]

        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].sum('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        # df_sig = xr_data.loc[:, :, :, :, time_vec_mask].median('time').to_dataframe(name='val').reset_index(drop=False)

        df_min = xr_data.loc[:, :, :, :, time_vec_mask].min('time').to_dataframe(name='val').reset_index(drop=False)
        df_max = xr_data.loc[:, :, :, :, time_vec_mask].max('time').to_dataframe(name='val').reset_index(drop=False)

        df_minmax = df_min.copy()
        df_minmax['val'] = np.abs(df_min['val'].values) + np.abs(df_max['val'].values)

        #### plot
        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'ERP_response_profile'))

        #### df generation
        predictor = 'odor'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            for nchan_i, nchan in enumerate(chan_list_eeg):

                #cond_i, cond = 2, conditions[2]
                for cond_i, cond in enumerate(conditions):

                    if group == 'allsujet':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}'")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}'")
                    if group == 'rep':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_best_list_rev.tolist()}")
                    if group == 'non_rep':
                        # df_stats = df_sig.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                        df_stats = df_minmax.query(f"cond == '{cond}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
        
                    if group_i + nchan_i + cond_i == 0:
                        df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'nchan', np.array([nchan]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'cond', np.array([cond]*df_stats_all.shape[0]))

                    else:
                        _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'nchan', np.array([nchan]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'cond', np.array([cond]*_df_stats_all.shape[0]))
                        df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['inter'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_inter = df_stats_all.copy()

        predictor = 'cond'
        outcome = 'val'

        for group_i, group in enumerate(sujet_group):

            for nchan_i, nchan in enumerate(chan_list_eeg):

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    if group == 'allsujet':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}'")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}'")
                    if group == 'rep':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list.tolist()}")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_best_list_rev.tolist()}")
                    if group == 'non_rep':
                        # df_stats = df_sig.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
                        df_stats = df_minmax.query(f"odor == '{odor}' and nchan == '{nchan}' and sujet in {sujet_no_respond_rev.tolist()}")
        
                    if group_i + nchan_i + odor_i == 0:
                        df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        df_stats_all.insert(0, 'group', np.array([group]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'nchan', np.array([nchan]*df_stats_all.shape[0]))
                        df_stats_all.insert(0, 'odor', np.array([odor]*df_stats_all.shape[0]))

                    else:
                        _df_stats_all = get_auto_stats_df(df_stats, predictor, outcome, subject='sujet', design='within', transform=False, verbose=False)[['pre_test', 'pre_test_pval', 'Contrast', 'A', 'B', 'p_unc']]
                        _df_stats_all.insert(0, 'group', np.array([group]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'nchan', np.array([nchan]*_df_stats_all.shape[0]))
                        _df_stats_all.insert(0, 'odor', np.array([odor]*_df_stats_all.shape[0]))
                        df_stats_all = pd.concat([df_stats_all, _df_stats_all], axis=0)

        df_stats_all['comp_type'] = np.array(['intra'] * df_stats_all.shape[0])
        df_stats_all = df_stats_all.reset_index(drop=True)

        df_stats_all_intra = df_stats_all.copy()

        df_stats_all_intra = df_stats_all_intra.query(f"A == 'FR_CV_1' or B == 'FR_CV_1'")
        df_stats_all_inter = df_stats_all_inter.query(f"A == 'o' or B == 'o'")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff'))

        df_stats_all_intra.to_excel('df_stats_all_intra.xlsx')
        df_stats_all_inter.to_excel('df_stats_all_inter.xlsx')

        df_stats_all_intra.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'FR_CV_1' or B == 'FR_CV_1'").to_excel('df_stats_all_intra_signi.xlsx')
        df_stats_all_inter.query(f"pre_test_pval <= 0.05 and p_unc <= 0.05").query(f"A == 'o' or B == 'o'").to_excel('df_stats_all_inter_signi.xlsx')

        df_stats_all = {'intra' : df_stats_all_intra, 'inter' : df_stats_all_inter}

    return df_stats_all









def get_cluster_stats(xr_data):

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', 'cluster_stats.pkl')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('ALREADY COMPUTED', flush=True)

        with open('cluster_stats.pkl', 'rb') as fp:
            cluster_stats = pickle.load(fp)

        with open('cluster_stats_rep_norep.pkl', 'rb') as fp:
            cluster_stats_rep_norep = pickle.load(fp)


    else:

        ch_types = ['eeg'] * len(chan_list_eeg)
        info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
        info.set_montage('standard_1020')

        mat_adjacency = mne.channels.find_ch_adjacency(info, 'eeg')[0]

        conditions_diff = ['MECA', 'CO2', 'FR_CV_2']
        odor_diff = ['+', '-']
        sujet_group = ['allsujet', 'rep', 'non_rep']

        cluster_stats = {}

        cluster_stats['intra'] = {}

        #group = sujet_group[0]
        for group in sujet_group:

            cluster_stats['intra'][group] = {}

            #nchan = chan_list_eeg[0]
            for nchan in chan_list_eeg:

                print(group, nchan)

                cluster_stats['intra'][group][nchan] = {}

                #odor_i, odor = 2, odor_list[2]
                for odor_i, odor in enumerate(odor_list):

                    cluster_stats['intra'][group][nchan][odor] = {}

                    #cond = conditions_diff[0]
                    for cond in conditions_diff:

                        cluster_stats['intra'][group][nchan][odor][cond] = {}

                        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values

                        if debug:

                            plt.plot(data_baseline.mean(axis=0))
                            plt.plot(data_baseline.mean(axis=0))
                            plt.show()

                        # n_conditions = 2
                        # n_observations = data_cond.shape[0]
                        # pval = 0.05  # arbitrary
                        # dfn = n_conditions - 1  # degrees of freedom numerator
                        # dfd = n_observations - 2  # degrees of freedom denominator
                        # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                        # thresh = int(np.round(thresh))
                        
                        # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                        #     [data_baseline, data_cond],
                        #     n_permutations=1000,
                        #     threshold=None,
                        #     tail=1,
                        #     n_jobs=4,
                        #     out_type="mask",
                        #     verbose='CRITICAL'
                        # )

                        data_diff = data_baseline-data_cond

                        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                        cluster_stats['intra'][group][nchan][odor][cond]['cluster'] = clusters
                        cluster_stats['intra'][group][nchan][odor][cond]['pval'] = cluster_p_values

        cluster_stats['inter'] = {}

        for group in sujet_group:

            cluster_stats['inter'][group] = {}

            for nchan in chan_list_eeg:

                print(group, nchan)

                cluster_stats['inter'][group][nchan] = {}

                #cond_i, cond = 2, conditions[2]
                for cond_i, cond in enumerate(conditions):

                    cluster_stats['inter'][group][nchan][cond] = {}

                    for odor in odor_diff:

                        cluster_stats['inter'][group][nchan][cond][odor] = {}

                        data_baseline = xr_data.loc[:, cond, 'o', nchan, :].values
                        data_cond = xr_data.loc[:, cond, odor, nchan, :].values

                        # n_conditions = 2
                        # n_observations = data_cond.shape[0]
                        # pval = 0.05  # arbitrary
                        # dfn = n_conditions - 1  # degrees of freedom numerator
                        # dfd = n_observations - 2  # degrees of freedom denominator
                        # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                        # thresh = int(np.round(thresh))
                        
                        # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                        #     [data_baseline, data_cond],
                        #     n_permutations=1000,
                        #     threshold=thresh,
                        #     tail=1,
                        #     n_jobs=4,
                        #     out_type="mask",
                        #     verbose='CRITICAL'
                        # )

                        data_diff = data_baseline-data_cond

                        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                        cluster_stats['inter'][group][nchan][cond][odor]['cluster'] = clusters
                        cluster_stats['inter'][group][nchan][cond][odor]['pval'] = cluster_p_values

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        
        with open('cluster_stats.pkl', 'wb') as fp:
            pickle.dump(cluster_stats, fp)

        cluster_stats_rep_norep = {}
        sujet_no_respond_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_no_respond])
        sujet_best_list_rev = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])

        for nchan in chan_list_eeg:

            print(nchan)

            cluster_stats_rep_norep[nchan] = {}

            #odor_i, odor = 2, odor_list[2]
            for odor_i, odor in enumerate(odor_list):

                cluster_stats_rep_norep[nchan][odor] = {}

                for cond in conditions:

                    cluster_stats_rep_norep[nchan][odor][cond] = {}

                    data_baseline = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].values
                    data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].values

                    # n_conditions = 2
                    # n_observations = data_cond.shape[0]
                    # pval = 0.05  # arbitrary
                    # dfn = n_conditions - 1  # degrees of freedom numerator
                    # dfd = n_observations - 2  # degrees of freedom denominator
                    # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                    # thresh = int(np.round(thresh))
                    
                    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                    #     [data_baseline, data_cond],
                    #     n_permutations=1000,
                    #     threshold=thresh,
                    #     tail=1,
                    #     n_jobs=4,
                    #     out_type="mask",
                    #     verbose='CRITICAL'
                    # )

                    n_obs_min = np.array([data_baseline.shape[0], data_cond.shape[0]]).min()

                    baseline_sel = np.random.choice(range(data_baseline.shape[0]), size=n_obs_min)
                    cond_sel = np.random.choice(range(data_cond.shape[0]), size=n_obs_min)

                    data_diff = data_baseline[baseline_sel,:] - data_cond[cond_sel,:]

                    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                            data_diff,
                            n_permutations=1000,
                            threshold=None,
                            stat_fun=None,
                            # adjacency=mat_adjacency,
                            tail=0,
                            n_jobs=4,
                            out_type="mask",
                            verbose='CRITICAL'
                        )

                    cluster_stats_rep_norep[nchan][odor][cond]['cluster'] = clusters
                    cluster_stats_rep_norep[nchan][odor][cond]['pval'] = cluster_p_values

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        
        with open('cluster_stats_rep_norep.pkl', 'wb') as fp:
            pickle.dump(cluster_stats_rep_norep, fp)

    return cluster_stats, cluster_stats_rep_norep













def get_df_mdp():

    os.chdir(os.path.join(path_data, 'psychometric'))
    df_mdp = pd.read_excel('OLFADYS_mdp.xlsx')
    df_mdp = df_mdp.query(f"sujet in {sujet_list_rev.tolist()}")

    ### sq
    for sujet in df_mdp['sujet'].unique():

        for session in df_mdp['session'].unique():

            for cond in df_mdp['cond'].unique():

                res = df_mdp.query(f"sujet == '{sujet}' and session == '{session}' and cond == '{cond}' and question in ['QS_1', 'QS_2', 'QS_3', 'QS_4', 'QS_5']").groupby(['sujet']).sum()['value'].values[0]

                df_mdp = pd.concat([df_mdp, pd.DataFrame({'sujet' : [sujet], 'session' : [session], 'cond' : [cond], 'question' : ['SQ'], 'value' : [res]}) ])

    ### A1+SQ
    for sujet in df_mdp['sujet'].unique():

        for session in df_mdp['session'].unique():

            for cond in df_mdp['cond'].unique():

                res = df_mdp.query(f"sujet == '{sujet}' and session == '{session}' and cond == '{cond}' and question in ['SQ', 'A1']").groupby(['sujet']).sum()['value'].values[0]

                df_mdp = pd.concat([df_mdp, pd.DataFrame({'sujet' : [sujet], 'session' : [session], 'cond' : [cond], 'question' : ['A1+SQ'], 'value' : [res]}) ])

    ### A2
    for sujet in df_mdp['sujet'].unique():

        for session in df_mdp['session'].unique():

            for cond in df_mdp['cond'].unique():

                res = df_mdp.query(f"sujet == '{sujet}' and session == '{session}' and cond == '{cond}' and question in ['A2_1', 'A2_2', 'A2_3', 'A2_4', 'A2_5']").groupby(['sujet']).sum()['value'].values[0]

                df_mdp = pd.concat([df_mdp, pd.DataFrame({'sujet' : [sujet], 'session' : [session], 'cond' : [cond], 'question' : ['A2'], 'value' : [res]}) ])

    ### A1+A2
    for sujet in df_mdp['sujet'].unique():

        for session in df_mdp['session'].unique():

            for cond in df_mdp['cond'].unique():

                res = df_mdp.query(f"sujet == '{sujet}' and session == '{session}' and cond == '{cond}' and question in ['A1', 'A2']").groupby(['sujet']).sum()['value'].values[0]

                df_mdp = pd.concat([df_mdp, pd.DataFrame({'sujet' : [sujet], 'session' : [session], 'cond' : [cond], 'question' : ['A1+A2'], 'value' : [res]}) ])

    ### selecct best
    mask_best = []
    for sujet in df_mdp['sujet'].values:
        if sujet in sujet_best_list:
            mask_best.append('REP')
        else:
            mask_best.append('NOREP')

    df_mdp['select_best'] = mask_best

    return df_mdp




def get_df_ERP_metric_allsujet(xr_data, xr_cluster_based_perm):

    #### load mdp
    df_mdp = get_df_mdp()

    df_ERP_metrics_allsujet = pd.DataFrame({'sujet' : [], 'odor' : [], 'phase' : [], 'chan' : [], 'amplitude' : [], 'time' : []})
    time_vec = xr_data['time'].values

    for sujet in sujet_list:

        for odor in odor_list:

            for chan in chan_list_eeg:

                for phase in ['inspi', 'expi']:

                    if phase == 'inspi':
                        time_vec_mask_phase = time_vec >= 0

                    elif phase == 'expi':
                        time_vec_mask_phase = time_vec < 0

                    time_vec_phase = time_vec[time_vec_mask_phase]
                    ERP_mean_phase = xr_data.loc[sujet, 'CO2', odor, chan, :][time_vec_mask_phase].values
                    mask_cluster = xr_cluster_based_perm.loc[sujet, odor, chan, :][time_vec_mask_phase].values

                    mask_cluster[0], mask_cluster[-1] = 0, 0 # to ensure diff detection

                    if np.abs(np.diff(mask_cluster)).sum() != 0: 

                        start_stop_chunk = np.where(np.diff(mask_cluster))[0]

                        max_chunk_signi = []
                        max_chunk_time = []
                        for start_i in np.arange(0, start_stop_chunk.size, 2):

                            _argmax = np.argmax(np.abs(ERP_mean_phase)[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                            max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                            max_chunk_signi.append(ERP_mean_phase[_argmax])

                        cluster_selected = np.argmax(np.abs(max_chunk_signi))
                        max_rep = max_chunk_signi[cluster_selected]
                        time_max_rep = time_vec_phase[max_chunk_time[cluster_selected]]

                    else:

                        continue

                    _df = pd.DataFrame({'sujet' : [sujet], 'odor' : [odor], 'phase' : [phase], 'chan' : [chan], 'amplitude' : [max_rep], 'time' : [time_max_rep]})
                    df_ERP_metrics_allsujet = pd.concat([df_ERP_metrics_allsujet, _df])
    
    #### generate df with A2 ratings
    df_ERP_metric_A2_ratings = df_ERP_metrics_allsujet.query(f"odor == '+'")

    A2_ratings = []
    rep_state = []
    for row_i in range(df_ERP_metric_A2_ratings.shape[0]):
        sujet = df_ERP_metrics_allsujet.iloc[row_i,:]['sujet']
        sujet_rev = f"{sujet[2:]}{sujet[:2]}"
        _diff_A2_p = df_mdp.query(f"sujet == '{sujet_rev}' and cond == 'CO2' and session == '+' and question == 'A2'")['value'].values[0]
        _diff_A2_o = df_mdp.query(f"sujet == '{sujet_rev}' and cond == 'CO2' and session == 'o' and question == 'A2'")['value'].values[0]
        A2_ratings.append(_diff_A2_p - _diff_A2_o)
        if sujet in sujet_best_list_rev:
            rep_state.append(True)
        else:
            rep_state.append(False)

    df_ERP_metric_A2_ratings['rep_state'] = rep_state
    df_ERP_metric_A2_ratings['A2_diff'] = A2_ratings

    return df_ERP_metrics_allsujet, df_ERP_metric_A2_ratings


def plot_ERP_metrics_A2_lm(df_ERP_metric_A2_ratings):

    rep_state_color = {True : 'green', False : 'red'}

    min, max = df_ERP_metric_A2_ratings['amplitude'].min(), df_ERP_metric_A2_ratings['amplitude'].max()

    for chan in chan_list_eeg:

        fig, axs = plt.subplots(ncols=2)

        for phase_i, phase in enumerate(['inspi', 'expi']):

            ax = axs[phase_i]

            _df = df_ERP_metric_A2_ratings.query(f"chan == '{chan}' and phase == '{phase}'")

            if _df.size == 0:
                continue

            for _amplitude, _A2_diff, _rep_state in zip(_df['amplitude'].values, _df['A2_diff'].values, _df['rep_state'].values):

                ax.scatter(x=_A2_diff, y=_amplitude, color=rep_state_color[_rep_state])

            ax.set_xlabel('A2')
            ax.set_ylabel('amplitude')
            ax.set_xlim(-10, 10)
            ax.set_ylim(min, max)
            ax.set_title(phase)

        plt.suptitle(chan)
        plt.tight_layout()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'lm_A2_ampl'))
        fig.savefig(f"{chan}_lm_A2ampl.png")





def plot_ERP_metrics_response(df_ERP_metrics_allsujet):

    df_ERP_metrics_allsujet.query(f"phase == 'inspi'")['amplitude'].mean()

    time_vec = np.arange(ERP_time_vec[0], ERP_time_vec[1], 1/srate)
    angles_vec = np.linspace(0, 2*np.pi, num=time_vec.size)

    rep_sujet = []
    for _sujet in df_ERP_metrics_allsujet['sujet']:
        if _sujet in sujet_best_list_rev:
            rep_sujet.append(True)
        else:
            rep_sujet.append(False)
    df_ERP_metrics_allsujet['rep_sujet'] = rep_sujet

    color_sujet_rep = {True : 'tab:green', False : 'tab:red'}

    ######## PLOT RESPONSE ########

    for chan in chan_list_eeg:

        min, max = df_ERP_metrics_allsujet['amplitude'].min(), df_ERP_metrics_allsujet['amplitude'].max()

        fig, axs = plt.subplots(ncols=3, subplot_kw={'projection': 'polar'}, figsize=(10,8))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]

            _df = df_ERP_metrics_allsujet.query(f"chan == '{chan}' and odor == '{odor}'")

            time_responses_i = [np.where(time_vec == time_val)[0][0] for time_val in _df['time']]
            angle_responses = angles_vec[time_responses_i]

            for _angle, _time, _rep_status in zip(angle_responses, _df['amplitude'], _df['rep_sujet']):
                ax.scatter(_angle, _time, color=color_sujet_rep[_rep_status], s=10)

            ax.set_xticks(np.linspace(0, 2 * np.pi, num=4, endpoint=False))
            ax.set_xticklabels(np.round(np.linspace(-2.5,2.5, num=4, endpoint=False), 2))
            ax.set_yticks(np.round(np.linspace(min,max, num=3, endpoint=True), 2))
            ax.set_rlim([min,max])
            ax.set_title(f"{odor}")

        plt.suptitle(f'{chan} CO2')
        plt.tight_layout()

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'time', 'channel_wise_allsujet'))
        plt.savefig(f"{chan}_CO2.png")



################################################
######## GENERATE PPI EVALUATION ########
################################################



def generate_ppi_evaluation(xr_data):

    t_start_PPI = PPI_time_vec[0]
    t_stop_PPI = PPI_time_vec[1]

    #### generate parameters
    dict = {'sujet' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'indice' : [], 'PPI' : []}

    chan_list_blind_evaluation = ['Cz', 'Fz']

    for sujet in sujet_list:

        for cond_i, cond in enumerate(conditions):

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                for nchan_i, nchan in enumerate(chan_list_blind_evaluation):

                    dict['sujet'].append(sujet)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['indice'].append(0)
                    dict['PPI'].append(0)

    df_PPI_blind_evaluation = pd.DataFrame(dict)
    df_PPI_blind_evaluation = df_PPI_blind_evaluation.sample(frac=1).reset_index(drop=True)
    df_PPI_blind_evaluation['indice'] = np.arange(df_PPI_blind_evaluation.shape[0])+1

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_PPI_blind_evaluation.to_excel('df_PPI_blind_evaluation.xlsx')

    ######## SUMMARY NCHAN ########

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))

    for indice in range(df_PPI_blind_evaluation.shape[0]):

        print_advancement(indice, df_PPI_blind_evaluation.shape[0], [25, 50, 75])

        fig, ax = plt.subplots()

        plt.suptitle(indice+1)

        fig.set_figheight(10)
        fig.set_figwidth(10)

        sujet, cond, odor, nchan = df_PPI_blind_evaluation.iloc[indice].values[0], df_PPI_blind_evaluation.iloc[indice].values[1], df_PPI_blind_evaluation.iloc[indice].values[2], df_PPI_blind_evaluation.iloc[indice].values[3]

        data_stretch = xr_data.loc[sujet, cond, odor, nchan, :].values

        ax.set_ylim(-3, 3)

        stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
        time_vec = np.linspace(t_start_PPI, t_stop_PPI, stretch_point_PPI)

        ax.plot(time_vec, data_stretch)
        # ax.plot(time_vec, data_stretch.std(axis=0), color='k', linestyle='--')
        # ax.plot(time_vec, -data_stretch.std(axis=0), color='k', linestyle='--')

        ax.invert_yaxis()

        ax.vlines(0, ymin=-3, ymax=3, colors='g')  
        ax.hlines(0, xmin=PPI_time_vec[0], xmax=PPI_time_vec[-1], colors='g') 

        # plt.show()

        #### save
        fig.savefig(f'{indice+1}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()













########################################
######## RESPI ERP ANALYSIS ########
########################################



def get_data_itl():

    t_start_PPI = mean_respi_ERP_time_vec[0]
    t_stop_PPI = mean_respi_ERP_time_vec[1]

    #### load data ITL
    data_respi_ITL = {}

    os.chdir(os.path.join(path_data, 'data_itl_leo'))

    sujet_list_itl = np.unique(np.array([file[:4] for file in os.listdir()]))

    #sujet_itl = sujet_list_itl[0]
    for sujet_itl in sujet_list_itl:

        data_respi_ITL[sujet_itl] = {}

        for cond in ['VS', 'CO2', 'ITL']:

            respi_file_name = [file for file in os.listdir() if file.find(sujet_itl) != -1 and file.find(f'{cond}.edf') != -1][0]
            inspi_file_name = [file for file in os.listdir() if file.find(sujet_itl) != -1 and file.find(f'{cond}.Markers') != -1][0]
            
            respi = mne.io.read_raw_edf(respi_file_name).get_data()[-1,:]
            f = open(inspi_file_name, "r")
            inspi_starts = [int(line.split(',')[2][1:]) for line_i, line in enumerate(f.read().split('\n')) if len(line.split(',')) == 5 and line.split(',')[0] == 'Response']
            inspi_starts = np.array(inspi_starts)

            if debug:
                fig, ax = plt.subplots()
                plt.plot(respi[:10000])
                plt.vlines(inspi_starts[inspi_starts<10000], ymin=respi.min(), ymax=respi.max(), color='r')
                export_fig(f'respi.png', fig)

            #### chunk
            stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)

            clean_cycles_i = [start_i for start_i, start_time in enumerate(inspi_starts) if int(start_time + t_start_PPI*srate) > 0]
            inspi_starts = inspi_starts[clean_cycles_i]
            clean_cycles_i = [start_i for start_i, start_time in enumerate(inspi_starts) if int(start_time + t_stop_PPI*srate) < respi.shape[0]]
            inspi_starts = inspi_starts[clean_cycles_i]

            respi_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

            x_mean, x_std = respi.mean(), respi.std()

            for start_i, start_time in enumerate(inspi_starts):

                t_start = int(start_time + t_start_PPI*srate)
                t_stop = int(start_time + t_stop_PPI*srate)

                # respi_stretch[start_i, :] = (respi[t_start: t_stop] - x_mean) / x_std
                respi_stretch[start_i, :] = respi[t_start: t_stop] - respi[t_start: t_stop].mean()

            data_respi_ITL[sujet_itl][cond] = respi_stretch

            if debug:

                time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                fig, ax = plt.subplots()
                plt.plot(time_vec, respi_stretch.mean(axis=0)*-1)
                plt.vlines(0, ymin=(respi_stretch.mean(axis=0)*-1).min(), ymax=(respi_stretch.mean(axis=0)*-1).max(), color='r')
                export_fig(f'respi.png', fig)

    #### normalize by VS
    val = np.array([])

    for sujet_itl in sujet_list_itl:

        val = np.concatenate((val, data_respi_ITL[sujet_itl]['VS'].reshape(-1)), axis=0)

    VS_mean, VS_std = val.mean(), val.std()

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
    dict = {'sujet' : sujet_list_itl, 'cond' : ['VS', 'CO2', 'ITL'], 'times' : time_vec}
    data = np.zeros((sujet_list_itl.shape[0], 3, time_vec.shape[0]))
    xr_itl = xr.DataArray(data=data, dims=dict.keys(), coords=dict.values())

    for sujet_itl in sujet_list_itl:

        for cond in ['VS', 'CO2', 'ITL']:

            _zscore = (data_respi_ITL[sujet_itl][cond] - VS_mean) / VS_std
            xr_itl.loc[sujet_itl, cond, :] = np.median(_zscore, axis=0)

    return xr_itl





def get_data_erp_respi():

    t_start_PPI = -4
    t_stop_PPI = 4

    #### extract zscore params
    zscore_prms = {'mean' : {}, 'std' : {}}

     #odor = odor_list[1]
    for odor in odor_list:

        zscore_prms['mean'][odor] = {}
        zscore_prms['std'][odor] = {}

    #sujet_i, sujet = 0, sujet_list[0]
    for sujet_i, sujet in enumerate(sujet_list):

        print_advancement(sujet_i, len(sujet_list), [25, 50, 75])

        #odor = odor_list[0]
        for odor in odor_list:

            respi_concat = []

            #cond = 'FR_CV_1'
            for cond in conditions:

                #### load
                respi_concat = np.concatenate((respi_concat, load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]), axis=0)

            if debug:

                plt.plot(respi_concat)
                plt.show()

            zscore_prms['mean'][odor][sujet] = np.array(respi_concat).mean()
            zscore_prms['std'][odor][sujet] = np.array(respi_concat).std()

    #### extract data
    data_respi = {}

    #sujet_i, sujet = 26, sujet_list[26]
    for sujet_i, sujet in enumerate(sujet_list):

        print_advancement(sujet_i, len(sujet_list), [25, 50, 75])

        data_respi[sujet] = {}

        respfeatures = load_respfeatures(sujet)

        #cond = 'MECA'
        for cond in conditions:

            data_respi[sujet][cond] = {}

            #odor = odor_list[1]
            for odor in odor_list:

                data_respi[sujet][cond][odor] = {}

                #### load
                respi = load_data_sujet(sujet, cond, odor)[chan_list.index('PRESS'),:]
                respi = (respi - zscore_prms['mean'][odor][sujet]) / zscore_prms['std'][odor][sujet]

                if debug:

                    plt.plot(respi)
                    plt.show()

                respfeatures_i = respfeatures[cond][odor]
                inspi_starts = respfeatures_i.query(f"select == 1")['inspi_index'].values

                #### chunk
                stretch_point_PPI = int(np.abs(t_start_PPI)*srate + t_stop_PPI*srate)
                respi_stretch = np.zeros((inspi_starts.shape[0], int(stretch_point_PPI)))

                # x_mean, x_std = respi.mean(), respi.std()

                for start_i, start_time in enumerate(inspi_starts):

                    t_start = int(start_time + t_start_PPI*srate)
                    t_stop = int(start_time + t_stop_PPI*srate)

                    # respi_stretch[start_i, :] = (respi[t_start: t_stop] - x_mean) / x_std
                    # respi_stretch[start_i, :] = (respi[t_start: t_stop] - zscore_prms['mean'][sujet]) / zscore_prms['std'][sujet]
                    # respi_stretch[start_i, :] = respi[t_start: t_stop] - x_mean
                    respi_stretch[start_i, :] = respi[t_start: t_stop]

                if debug:

                    plt.plot(respi)
                    plt.vlines(inspi_starts, ymax=respi.max(), ymin=respi.min(), color='k')
                    plt.show()

                    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

                    for inspi_i, _ in enumerate(inspi_starts):

                        plt.plot(time_vec, respi_stretch[inspi_i, :], alpha=0.3)

                    plt.vlines(0, ymax=respi_stretch.max(), ymin=respi_stretch.min(), color='k')
                    # plt.hlines([-3, 3], xmax=t_start_PPI, xmin=t_stop_PPI, color='k')
                    plt.plot(time_vec, respi_stretch.mean(axis=0), color='r')
                    plt.title(f'{cond} {odor} : {respi_stretch.shape[0]}')
                    plt.show()

                data_respi[sujet][cond][odor] = respi_stretch

    # for sujet_i, sujet in enumerate(sujet_list):

    #     #### normalize by VS
    #     val = np.array([])

    #     for odor_i, odor in enumerate(odor_list):

    #         val = np.concatenate((val, data_respi[sujet]['FR_CV_1'][odor].reshape(-1)), axis=0)
    #         val = np.concatenate((val, data_respi[sujet]['FR_CV_2'][odor].reshape(-1)), axis=0)

    #     VS_mean, VS_std = val.mean(), val.std()

    #     for cond in conditions:

    #         for odor_i, odor in enumerate(odor_list):

    #             data_respi[sujet][cond][odor] = (data_respi[sujet][cond][odor] - VS_mean) / VS_std

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)
    dict = {'sujet' : sujet_list, 'odor' : odor_list, 'cond' : conditions, 'times' : time_vec}
    data = np.zeros((len(sujet_list), len(odor_list), len(conditions), time_vec.shape[0]))
    xr_respi = xr.DataArray(data=data, dims=dict.keys(), coords=dict.values())

    for sujet in sujet_list:

        for odor in odor_list:

            for cond in conditions:

                xr_respi.loc[sujet, odor, cond, :] = np.mean(data_respi[sujet][cond][odor], axis=0)

    return xr_respi







def plot_mean_respi():

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'ERP', f'allsujet_ERP_respi.nc')):

        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

        print('xr_respi ALREADY COMPUTED', flush=True)

        xr_respi = xr.open_dataarray(f'allsujet_ERP_respi.nc')
        xr_itl = xr.open_dataarray(f'allsujet_ERP_respi_itl.nc')

    else:

        xr_respi = get_data_erp_respi()
        xr_itl = get_data_itl()

        #### save data
        os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
        xr_respi.to_netcdf('allsujet_ERP_respi.nc')
        xr_itl.to_netcdf('allsujet_ERP_respi_itl.nc')

    t_start_PPI = -4
    t_stop_PPI = 4

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    max = np.array([(xr_itl.mean('sujet').values*-1).max(), (xr_respi.mean('sujet').values).max()]).max()
    min = np.array([(xr_itl.mean('sujet').values*-1).min(), (xr_respi.mean('sujet').min().values).min()]).min()

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))

    time_vec = np.arange(t_start_PPI, t_stop_PPI, 1/srate)

    fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

    for odor_i, odor in enumerate(odor_list):

        ax = axs[odor_i]

        ax.set_title(f"{odor}")

        ax.plot(time_vec, xr_itl.mean('sujet').loc['VS',:].values*-1, label='VS_itl', color='g', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['CO2',:].values*-1, label='CO2_itl', color='r', linestyle=':', dashes=(5, 10))
        ax.plot(time_vec, xr_itl.mean('sujet').loc['ITL',:].values*-1, label='ITL_itl', color='b', linestyle=':', dashes=(5, 10))

        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'VS_1', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'FR_CV_2', :].mean('sujet'), label=f'VS_2', color='g')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
        ax.plot(time_vec, xr_respi.loc[:, odor, 'MECA', :].mean('sujet'), label=f'MECA', color='b')

        ax.vlines(0, ymin=min, ymax=max, color='k')

    plt.legend()
    
    plt.suptitle(f"comparison ITL")

    # plt.show()

    plt.savefig(f"allsujet_ERP_comparison_ITL.png")
    plt.close('all')


    #### plot sujet respi
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'respi'))
    sujet_group = ['allsujet', 'rep', 'no_rep']

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(conditions), figsize=(15,10))

        for cond_i, cond in enumerate(conditions):

            ax = axs[cond_i]

            ax.set_title(f"{cond}")

            ax.plot(time_vec, xr_data.loc[:, 'o', cond, :].mean('sujet'), label=f'o', color='b')
            ax.plot(time_vec, xr_data.loc[:, '-', cond, :].mean('sujet'), label=f'-', color='r')
            ax.plot(time_vec, xr_data.loc[:, '+', cond, :].mean('sujet'), label=f'+', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_COND_mean_{group}.png")
        
        plt.close('all')

    for group in sujet_group:

        if group == 'allsujet':
            xr_data = xr_respi
        if group == 'rep':
            xr_data = xr_respi.loc[sujet_best_list]
        if group == 'no_rep':
            xr_data = xr_respi.loc[sujet_no_respond]

        max = (xr_data.mean('sujet').values).max()
        min = (xr_data.mean('sujet').values).min()

        fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]

            ax.set_title(f"{odor}")

            ax.plot(time_vec, xr_data.loc[:, odor, 'FR_CV_1', :].mean('sujet'), label=f'FR_CV_1', color='c')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'FR_CV_2', :].mean('sujet'), label=f'FR_CV_2', color='b')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'CO2', :].mean('sujet'), label=f'CO2', color='r')
            ax.plot(time_vec, xr_data.loc[:,  odor, 'MECA', :].mean('sujet'), label=f'MECA', color='g')

            ax.set_ylim(min, max)

            ax.vlines(0, ymin=min, ymax=max, color='k')

        plt.legend()
        
        plt.suptitle(f"{group} ERP, n:{xr_data['sujet'].shape[0]}")

        # plt.show()

        plt.savefig(f"ERP_ODOR_mean_{group}.png")
        
        plt.close('all')

    if debug:

        cond = 'FR_CV_1'
        odor = 'o'

        for sujet in sujet_list:

            plt.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=str(sujet.data), alpha=0.25)

        plt.plot(time_vec, xr_data.loc[:, odor, cond, :].mean('sujet'), label=str(sujet.data), alpha=1, color='r')
        plt.vlines(0, ymin=xr_data.loc[:, odor, cond, :].min(), ymax=xr_data.loc[:, odor, cond, :].max(), color='k')
        plt.title(f'allsujet {cond} {odor}: {sujet_list.shape[0]}')
        # plt.legend()
        plt.show()

        for sujet in sujet_list:

            fig, axs = plt.subplots(ncols=len(odor_list))

            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                for cond in conditions:

                    ax.plot(time_vec, xr_data.loc[sujet, odor, cond, :], label=cond)

            plt.legend()
            plt.suptitle(sujet)
            plt.show()

        plt.plot(xr_data.loc[:, 'o', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '+', cond, :].mean('sujet'))
        plt.plot(xr_data.loc[:, '-', cond, :].mean('sujet'))
















def plot_ERP_mean_subject_wise(stretch=False):

    print('ERP PLOT', flush=True)

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    t_start_ERP = ERP_time_vec[0]
    t_stop_ERP = ERP_time_vec[1]

    if stretch:
        xr_data, xr_data_sem = compute_ERP_stretch()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=True)
    else:
        xr_data, xr_data_sem = compute_ERP()
        cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=False)

    sujet_group = ['allsujet', 'rep', 'non_rep']

    if stretch:
        time_vec = np.arange(stretch_point_TF)
    else:
        time_vec = np.arange(t_start_ERP, t_stop_ERP, 1/srate)

    cond_sel_list = ['FR_CV_1', 'CO2']

    ######## INTRA ########

    #group = sujet_group[0]
    for group in sujet_group:

        ######## IDENTIFY MIN MAX ########

        print(f'PLOT SUMMARY ERP {group}')

        #### identify min max for allsujet

        min, max = [], []

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            if group == 'allsujet':
                min.append(xr_data.loc[:, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[:, cond_sel_list, :, nchan, :].max().values) 
            elif group == 'rep':
                min.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].max().values) 
            elif group == 'non_rep':
                min.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].max().values) 

        min, max = np.array(min).min(), np.array(max).max()

        min_mean, max_mean = [], []

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            if group == 'allsujet':
                min_mean.append(xr_data.loc[:, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[:, cond_sel_list, :, nchan, :].mean('sujet').max().values) 
            elif group == 'rep':
                min_mean.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].mean('sujet').max().values) 
            elif group == 'non_rep':
                min_mean.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].mean('sujet').max().values) 

        min_mean, max_mean = np.array(min_mean).min(), np.array(max_mean).max()

        ######## SUMMARY NCHAN INTRA ########

        if debug:
            nchan = 'FC1' 
            cond = 'CO2'
            odor = 'o'

            fig, ax = plt.subplots()

            if group == 'allsujet':
                _sujet_list = sujet_list
            elif group == 'rep':
                _sujet_list = sujet_best_list_rev
            elif group == 'non_rep':
                _sujet_list = sujet_no_respond_rev

            for sujet in _sujet_list:

                ax.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :], label=sujet)

            ax.invert_yaxis()
            plt.legend()
            plt.show()

            for sujet in _sujet_list:

                plt.plot(time_vec, xr_data.loc['12BD', cond, odor, nchan, :], label=sujet)
                plt.title(sujet)
                plt.show()

        rep_state_colors = {True : 'g', False : 'r'}

        #nchan_i, nchan = chan_list_eeg.index('FC1'), 'FC1'  
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel_list)+1)

            fig.set_figheight(10)
            fig.set_figwidth(10)

            plt.suptitle(f'{group} {nchan} intra')

            #cond_i, cond = 1, 'CO2'
            for cond_i, cond in enumerate(cond_sel_list):

                #odor_i, odor = 1, odor_list[1]
                for odor_i, odor in enumerate(odor_list):

                    ax = axs[odor_i, cond_i]

                    if group == 'allsujet':
                        data_mean = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'rep':
                        data_mean = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'non_rep':
                        data_mean = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].mean('sujet').values

                    ax.plot(time_vec, data_mean, color='k')

                    if odor_i == 0 :
                        ax.set_title(f"{cond}")

                    if cond_i == 0:
                        ax.set_ylabel(f"{odor}")

                    if group == 'allsujet':
                        _sujet_list = sujet_list
                    elif group == 'rep':
                        _sujet_list = sujet_best_list_rev
                    elif group == 'non_rep':
                        _sujet_list = sujet_no_respond_rev

                    for sujet in _sujet_list:

                        if sujet in sujet_best_list_rev:
                            rep_state = True
                        else:
                            rep_state = False 

                        ax.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :], color=rep_state_colors[rep_state], alpha=0.2)

                    if stretch:
                        ax.vlines(stretch_point_TF/2, ymin=min, ymax=max, colors='k')
                    else:
                        ax.vlines(0, ymin=min, ymax=max, colors='k')  

                    ax.set_ylim(min, max)

                    ax.invert_yaxis()

            cond_i, cond = 2, 'BOTH'

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='b', label='VS')

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'CO2', odor, nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'CO2', odor, nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'CO2', odor, nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='r', label='CO2')

                if odor_i == 0 :
                    ax.set_title(f"{cond}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                if stretch:
                    ax.vlines(stretch_point_TF/2, ymin=min_mean, ymax=max_mean, colors='k')
                else:
                    ax.vlines(0, ymin=min_mean, ymax=max_mean, colors='k')  

                ax.set_ylim(min_mean, max_mean)

                clusters = cluster_stats_intra.loc[group, nchan, odor, 'CO2',:].values
                ax.fill_between(time_vec, min_mean, max_mean, where=clusters.astype('int'), alpha=0.3, color='r')

                ax.invert_yaxis()

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff_subject_wise', 'intra', 'mean'))

            if stretch:
                fig.savefig(f'stretch_{group}_{nchan}_intra.jpeg', dpi=150)
            else:
                fig.savefig(f'nostretch_{group}_{nchan}_intra.jpeg', dpi=150)

            plt.close('all')


    ######## INTER ########

    #group = sujet_group[0]
    for group in sujet_group:

        ######## IDENTIFY MIN MAX ########

        print(f'PLOT SUMMARY ERP {group}')

        #### identify min max for allsujet

        min, max = [], []

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            if group == 'allsujet':
                min.append(xr_data.loc[:, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[:, cond_sel_list, :, nchan, :].max().values) 
            elif group == 'rep':
                min.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].max().values) 
            elif group == 'non_rep':
                min.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].min().values)
                max.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].max().values) 

        min, max = np.array(min).min(), np.array(max).max()

        min_mean, max_mean = [], []

        #nchan_i, nchan = 0, chan_list_eeg[0]
        for nchan_i, nchan in enumerate(chan_list_eeg):

            if group == 'allsujet':
                min_mean.append(xr_data.loc[:, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[:, cond_sel_list, :, nchan, :].mean('sujet').max().values) 
            elif group == 'rep':
                min_mean.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[sujet_best_list_rev, cond_sel_list, :, nchan, :].mean('sujet').max().values) 
            elif group == 'non_rep':
                min_mean.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].mean('sujet').min().values)
                max_mean.append(xr_data.loc[sujet_no_respond_rev, cond_sel_list, :, nchan, :].mean('sujet').max().values) 

        min_mean, max_mean = np.array(min_mean).min(), np.array(max_mean).max()

        ######## SUMMARY NCHAN INTRA ########

        if debug:
            nchan = 'FC1' 
            cond = 'CO2'
            odor = 'o'

            fig, ax = plt.subplots()

            if group == 'allsujet':
                _sujet_list = sujet_list
            elif group == 'rep':
                _sujet_list = sujet_best_list_rev
            elif group == 'non_rep':
                _sujet_list = sujet_no_respond_rev

            for sujet in _sujet_list:

                ax.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :], label=sujet)

            ax.invert_yaxis()
            plt.legend()
            plt.show()

            for sujet in _sujet_list:

                plt.plot(time_vec, xr_data.loc['12BD', cond, odor, nchan, :], label=sujet)
                plt.title(sujet)
                plt.show()

        rep_state_colors = {True : 'g', False : 'r'}

        #nchan_i, nchan = chan_list_eeg.index('FC1'), 'FC1'  
        for nchan_i, nchan in enumerate(chan_list_eeg):

            fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_sel_list)+2)

            fig.set_figheight(10)
            fig.set_figwidth(10)

            plt.suptitle(f'{group} {nchan} inter')

            #cond_i, cond = 1, 'CO2'
            for cond_i, cond in enumerate(cond_sel_list):

                #odor_i, odor = 1, odor_list[1]
                for odor_i, odor in enumerate(odor_list):

                    ax = axs[odor_i, cond_i*2]

                    if group == 'allsujet':
                        data_mean = xr_data.loc[:, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'rep':
                        data_mean = xr_data.loc[sujet_best_list_rev, cond, odor, nchan, :].mean('sujet').values
                    elif group == 'non_rep':
                        data_mean = xr_data.loc[sujet_no_respond_rev, cond, odor, nchan, :].mean('sujet').values

                    ax.plot(time_vec, data_mean, color='k')

                    if odor_i == 0 :
                        ax.set_title(f"{cond}")

                    if cond_i == 0:
                        ax.set_ylabel(f"{odor}")

                    if group == 'allsujet':
                        _sujet_list = sujet_list
                    elif group == 'rep':
                        _sujet_list = sujet_best_list_rev
                    elif group == 'non_rep':
                        _sujet_list = sujet_no_respond_rev

                    for sujet in _sujet_list:

                        if sujet in sujet_best_list_rev:
                            rep_state = True
                        else:
                            rep_state = False 

                        ax.plot(time_vec, xr_data.loc[sujet, cond, odor, nchan, :], color=rep_state_colors[rep_state], alpha=0.2)

                    if stretch:
                        ax.vlines(stretch_point_TF/2, ymin=min, ymax=max, colors='k')
                    else:
                        ax.vlines(0, ymin=min, ymax=max, colors='k')  

                    ax.set_ylim(min, max)

                    ax.invert_yaxis()

            cond_i, cond = 1, 'FR_CV_1_mean'

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'FR_CV_1', 'o', nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', 'o', nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', 'o', nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='b', label='o')

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='r', label=odor)

                if odor_i == 0 :
                    ax.set_title(f"{cond}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                if stretch:
                    ax.vlines(stretch_point_TF/2, ymin=min_mean, ymax=max_mean, colors='k')
                else:
                    ax.vlines(0, ymin=min_mean, ymax=max_mean, colors='k')  

                ax.set_ylim(min_mean, max_mean)

                if odor != 'o':
                    clusters = cluster_stats_inter.loc[group, nchan, odor, 'FR_CV_1',:].values
                    ax.fill_between(time_vec, min_mean, max_mean, where=clusters.astype('int'), alpha=0.3, color='r')

                ax.invert_yaxis()

            plt.legend()

            cond_i, cond = 3, 'CO2_mean'

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, cond_i]

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'CO2', 'o', nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'CO2', 'o', nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'CO2', 'o', nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='b', label='o')

                if group == 'allsujet':
                    data_mean = xr_data.loc[:, 'CO2', odor, nchan, :].mean('sujet').values
                elif group == 'rep':
                    data_mean = xr_data.loc[sujet_best_list_rev, 'CO2', odor, nchan, :].mean('sujet').values
                elif group == 'non_rep':
                    data_mean = xr_data.loc[sujet_no_respond_rev, 'CO2', odor, nchan, :].mean('sujet').values
                ax.plot(time_vec, data_mean, color='r', label=odor)

                if odor_i == 0 :
                    ax.set_title(f"{cond}")

                if cond_i == 0:
                    ax.set_ylabel(f"{odor}")

                if stretch:
                    ax.vlines(stretch_point_TF/2, ymin=min_mean, ymax=max_mean, colors='k')
                else:
                    ax.vlines(0, ymin=min_mean, ymax=max_mean, colors='k')  

                ax.set_ylim(min_mean, max_mean)

                if odor != 'o':
                    clusters = cluster_stats_inter.loc[group, nchan, odor, 'CO2',:].values
                    ax.fill_between(time_vec, min_mean, max_mean, where=clusters.astype('int'), alpha=0.3, color='r')

                ax.invert_yaxis()

            fig.tight_layout()
            plt.legend()

            # plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'summary_diff_subject_wise', 'inter', 'mean'))

            if stretch:
                fig.savefig(f'stretch_{group}_{nchan}_inter.jpeg', dpi=150)
            else:
                fig.savefig(f'nostretch_{group}_{nchan}_inter.jpeg', dpi=150)

            plt.close('all')













################################
######## SLOPE REG ########
################################

def plot_slope_versus_discomfort(xr_lm_data, conditions):

    os.chdir(os.path.join(path_data, 'psychometric'))

    df_mdp = pd.read_excel('OLFADYS_mdp.xlsx')
    df_q = pd.read_excel('OLFADYS_questionnaire.xlsx')

    val = np.zeros((df_q['sujet'].unique().shape[0]*df_q['session'].unique().shape[0]*df_q['cond'].unique().shape[0], 4), dtype='object')
    i = 0
    for sujet_i, sujet in enumerate(df_q['sujet'].unique()):
        for session_i, session in enumerate(df_q['session'].unique()):
            for cond_i, cond in enumerate(df_q['cond'].unique()):
                df_i = df_q.query(f"sujet == '{sujet}' & session == '{session}' & cond == '{cond}'")
                val_p = 300 - df_i[df_i['question'].isin([1, 4, 5])]['val'].sum()
                val_n = df_i[df_i['question'].isin([2, 3, 6])]['val'].sum()
                val_stai = ((val_n + val_p)/600)*100
                val[i, :] = np.array([sujet, session, cond, val_stai])
                i += 1
    df_stai = pd.DataFrame(val, columns=['sujet', 'session', 'cond', 'val'])
    df_stai['val'] = df_stai['val'].astype(np.float64) 

    df_relax = df_q.query(f"question == 7").drop(['question', 'raw_val', 'scale'], axis=1)

    question_A2 = ['A2_1', 'A2_2', 'A2_3', 'A2_4', 'A2_5']
    df_A2 = df_mdp.query(f"question in {question_A2}").groupby(['sujet', 'session', 'cond']).mean().reset_index()

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))

    xr_respi = xr.open_dataarray(f'allsujet_ERP_respi.nc')

    os.chdir(os.path.join(path_data, 'respi_detection'))
    df_CO2 = pd.read_excel('OLFADYS_alldata_mean.xlsx').query(f"cond == 'CO2'")[['sujet', 'cond', 'odor', 'PetCO2', 'VAS_S', 'VAS_A']]
    df_CO2 = df_CO2.reset_index(drop=True)
    
    for row_i in range(df_CO2.shape[0]):
        if df_CO2.iloc[row_i]['odor'] == 'p':
            df_CO2['odor'][row_i] = '+'
        if df_CO2.iloc[row_i]['odor'] == 'n':
            df_CO2['odor'][row_i] = '-'

    os.chdir(os.path.join(path_data, 'respi_detection'))
    df_respi_paris = pd.read_excel('OLFADYS_alldata_mean.xlsx').query(f"sujet in {sujet_list.tolist()}").reset_index(drop=True)

    for row_i in range(df_respi_paris.shape[0]):
        if df_respi_paris.iloc[row_i]['odor'] == 'p':
            df_respi_paris['odor'][row_i] = '+'
        if df_respi_paris.iloc[row_i]['odor'] == 'n':
            df_respi_paris['odor'][row_i] = '-'

    #### A2
    reg_slope_dis = {'sujet' : [], 'odor' : [], 'cond' : [], 'nchan' : [], 'slope' : [], 'A2' : [], 'STAI' : [], 'relax' : [], 'rep' : [], 
                     'amplitude' : [], 'amp_mean_diff' : [], 'PCO2' : [], 'VAS_S' : [], 'VAS_A' : [], 'VT' : [], 'Ve' : []}

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        respfeatures = load_respfeatures(sujet)

        for odor in odor_list:

            for cond in conditions:

                if cond in ['FR_CV_1', 'FR_CV_2']:
                    continue

                for nchan in chan_list_eeg:

                    slope = float(xr_lm_data.loc[sujet, cond, odor, nchan, 'slope'].data)

                    sujet_sel = f"{sujet[2:]}{sujet[:2]}"
                    A2 = float(df_A2.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['value'].values)
                    relax = float(df_relax.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['val'].values)
                    STAI = float(df_stai.query(f"sujet == '{sujet_sel}' and session == '{odor}' and cond == '{cond}'")['val'].values)

                    if sujet in sujet_best_list:
                        rep = True
                    else:
                        rep = False

                    amplitude = respfeatures[cond][odor].query(f"select == 1")['total_amplitude'].mean()
                    amp_mean_diff = xr_respi.loc[sujet, odor, cond, :].values.max() - xr_respi.loc[sujet, odor, cond, :].values.min()

                    if cond == 'CO2':
                        PCO2 = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['PetCO2'].values[0]
                        VAS_A = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['VAS_A'].values[0]
                        VAS_S = df_CO2.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == 'CO2'")['VAS_S'].values[0]
                    else:
                        PCO2 = 0
                        VAS_A = 0
                        VAS_S = 0

                    VT = df_respi_paris.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == '{cond}'")['VT'].values[0]
                    Ve = df_respi_paris.query(f"sujet == '{sujet}' and odor == '{odor}' and cond == '{cond}'")['Ve'].values[0]

                    reg_slope_dis['sujet'].append(sujet)
                    reg_slope_dis['odor'].append(odor)
                    reg_slope_dis['cond'].append(cond)
                    reg_slope_dis['nchan'].append(nchan)
                    reg_slope_dis['slope'].append(slope)
                    reg_slope_dis['A2'].append(A2)
                    reg_slope_dis['relax'].append(relax)
                    reg_slope_dis['STAI'].append(STAI)
                    reg_slope_dis['rep'].append(rep)
                    reg_slope_dis['amplitude'].append(amplitude)
                    reg_slope_dis['amp_mean_diff'].append(amp_mean_diff)
                    reg_slope_dis['PCO2'].append(PCO2)
                    reg_slope_dis['VAS_A'].append(VAS_A)
                    reg_slope_dis['VAS_S'].append(VAS_S)
                    reg_slope_dis['VT'].append(VT)
                    reg_slope_dis['Ve'].append(Ve)

    df_reg = pd.DataFrame(reg_slope_dis)

    #### plot
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'reg'))

    #metric = 'Ve'
    for metric in ['A2', 'STAI', 'relax', 'amplitude', 'amp_mean_diff', 'PCO2', 'VAS_S', 'VAS_A', 'VT', 'Ve']:

        print(metric)

        for nchan in chan_list_eeg:

            df_plot = df_reg.query(f"nchan == '{nchan}'")

            g = sns.FacetGrid(df_plot, row='odor', col="cond", hue='rep')
            g.map(sns.scatterplot, "slope", metric)
            plt.gca().invert_xaxis()
            plt.suptitle(f'{nchan}')
            plt.legend()
            # plt.show()

            plt.savefig(f'{metric}_{nchan}.png')
            plt.close('all')

    










################################
######## PPI PROPORTION ########
################################


def plot_PPI_proportion(xr_PPI_count):

    cond_temp = ['VS', 'CO2', 'MECA']

    group_list = ['allsujet', 'rep', 'no_rep']

    sujet_best_list = np.array([f"{sujet[2:]}{sujet[:2]}" for sujet in sujet_best_list])
    sujet_no_respond = np.array([sujet for sujet in sujet_list if sujet not in sujet_best_list])

    #### MCN, TS

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'blind_evaluation'))
    df_blind_eva = pd.read_excel('df_PPI_blind_evaluation.xlsx')

    examinateur_list = ['MCN', 'TS']

    xr_dict = {'examinateur' : examinateur_list, 'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'nchan' : chan_list_eeg}
    _xr_PPI_count = xr.DataArray(data=np.zeros((len(examinateur_list), len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg))), dims=xr_dict.keys(), coords=xr_dict.values())

    for examinateur in examinateur_list:

        #sujet = sujet_list[0]
        for sujet in sujet_list:

            if sujet in ['28NT']:
                continue

            #cond = 'CO2'
            for cond in conditions:

                if cond in ['FR_CV_1', 'FR_CV_2']:
                    continue

                #odor = odor_list[0]
                for odor in odor_list:

                    #nchan_i, nchan = 0, chan_list_eeg[0]
                    for nchan_i, nchan in enumerate(chan_list_eeg):

                        if nchan in ['Cz', 'Fz']:

                            _eva = df_blind_eva.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and nchan == '{nchan}'")[examinateur].values[0]
                            _xr_PPI_count.loc[examinateur, sujet, cond, odor, nchan] = _eva

    df_PPI_count = _xr_PPI_count.to_dataframe(name='PPI').reset_index()

    dict = {'examinateur' : [], 'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for examinateur in examinateur_list:

        for group in group_list:

            #cond = 'VS'
            for cond in cond_temp:

                for odor in odor_list:

                    for nchan in chan_list_eeg:

                        if group == 'allsujet':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        if group == 'no_rep':
                            df_plot = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                        
                        if df_plot['PPI'].sum() == 0:
                            prop = 0
                        else:
                            prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                        dict['examinateur'].append(examinateur)
                        dict['group'].append(group)
                        dict['cond'].append(cond)
                        dict['odor'].append(odor)
                        dict['nchan'].append(nchan)
                        dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"examinateur == '{examinateur}' and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"examinateur == '{examinateur}' and sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for examinateur in examinateur_list:

        for nchan in ['Cz', 'Fz']:

            df_plot = df_PPI_plot.query(f"examinateur == '{examinateur}' and nchan == '{nchan}'")
            sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
            plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
            plt.ylim(0,100)
            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{nchan}_{examinateur}.png")
            plt.close('all')

    #### JG

    df_PPI_count = xr_PPI_count.to_dataframe(name='PPI').reset_index()
    df_PPI_count = df_PPI_count.query(f"examinateur == 'JG'")

    dict = {'group' : [], 'cond' : [], 'odor' : [], 'nchan' : [], 'proportion' : []}

    for group in group_list:

        #cond = 'VS'
        for cond in conditions:

            for odor in odor_list:

                for nchan in chan_list_eeg:

                    if group == 'allsujet':
                        df_plot = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    if group == 'no_rep':
                        df_plot = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")
                    
                    if df_plot['PPI'].sum() == 0:
                        prop = 0
                    else:
                        prop = np.round(df_plot['PPI'].sum() / df_plot['sujet'].shape[0], 5)*100

                    dict['group'].append(group)
                    dict['cond'].append(cond)
                    dict['odor'].append(odor)
                    dict['nchan'].append(nchan)
                    dict['proportion'].append(prop)
            
    df_PPI_plot = pd.DataFrame(dict)

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'proportion'))

    n_sujet_all = df_PPI_count.query(f"cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_rep = df_PPI_count.query(f"sujet in {sujet_best_list.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]
    n_sujet_no_rep = df_PPI_count.query(f"sujet in {sujet_no_respond.tolist()} and cond == '{cond}' and nchan == '{nchan}' and odor == '{odor}'")['sujet'].shape[0]

    for nchan in chan_list_eeg:

        df_plot = df_PPI_plot.query(f"nchan == '{nchan}'")
        sns.catplot(data=df_plot, x="odor", y="proportion", hue='group', kind="point", col='cond')
        plt.suptitle(f"{nchan} / all:{n_sujet_all},rep:{n_sujet_rep},no_rep:{n_sujet_no_rep}")
        plt.ylim(0,100)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{nchan}_JG.png")
        plt.close('all')




    


################################
######## TIMING ERP ########
################################


def timing_ERP_IE_SUM_export_df(xr_data):

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data['time'].data

    RRP_time_ampl = {'group' : [], 'odor' : [], 'phase' : [], 'time' : [], 'amplitude' : []}

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        timing_data = np.zeros((len(odor_list), len(['expi', 'inspi']), len(chan_list_eeg), len(['value', 'time'])))

        for odor_i, odor in enumerate(['o', '+', '-']):

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            cond = 'CO2'

            print('intra', subgroup_type, odor, cond)

            if subgroup_type == 'allsujet':
                data_cond = xr_data.loc[:, cond, odor, :, :].values
                
            elif subgroup_type == 'rep':
                data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                mask_signi = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                for respi_phase_i, respi_phase in enumerate(['expi', 'inspi']):

                    if respi_phase == 'inspi':
                        mask_signi_phase = np.concatenate(( np.zeros(int(time_vec.size/2)).astype('bool'), mask_signi[int(time_vec.size/2):] ))

                    elif respi_phase == 'expi':
                        mask_signi_phase = np.concatenate(( mask_signi[:int(time_vec.size/2)], np.zeros(int(time_vec.size/2)).astype('bool') ))

                    if mask_signi_phase.sum() == 0:
                        continue     

                    else:
                        mask_signi_phase[0] = False
                        mask_signi_phase[-1] = False   

                    if np.diff(mask_signi_phase).sum() > 2: 

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        max_chunk_signi = []
                        max_chunk_time = []
                        for start_i in np.arange(0, start_stop_chunk.size, 2):

                            _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                            max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                            max_chunk_signi.append(data_cond_chan.mean(axis=0)[_argmax])

                        max_rep = max_chunk_signi[np.argmax(np.abs(max_chunk_signi))]
                        time_max_rep = time_vec[max_chunk_time[np.where(max_chunk_signi == max_rep)[0][0]]]

                    else:

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[0]:start_stop_chunk[1]])
                        max_rep = data_cond_chan.mean(axis=0)[start_stop_chunk[0] + _argmax]
                        time_max_rep = time_vec[start_stop_chunk[0] +_argmax]

                    timing_data[odor_i, respi_phase_i, chan_i, 0] = max_rep
                    timing_data[odor_i, respi_phase_i, chan_i, 1] = time_max_rep

        xr_coords = {'odor' : odor_list, 'phase' : ['expi', 'inspi'], 'chan' : chan_list_eeg, 'type' : ['value', 'time']}
        xr_timing = xr.DataArray(data=timing_data, coords=xr_coords)       

        #### export df
        for odor_i, odor in enumerate(odor_list):

            for phase in ['expi', 'inspi']:

                angles_vec = np.linspace(0, 2*np.pi, num=time_vec.size)
                mask_sel = (xr_timing.loc[odor, phase, :, 'time'] != 0).data
                time_responses_filter = xr_timing.loc[odor, phase, :, 'time'].data[mask_sel]
                ampl_responses_filter = xr_timing.loc[odor, phase, :, 'value'].data[mask_sel]

                time_responses_i = [np.where(time_vec == time_val)[0][0] for time_val in time_responses_filter]
                angle_responses = angles_vec[time_responses_i]

                _phase_mean = np.angle(np.mean(np.exp(1j*angle_responses)))
                
                mean_time = np.round(time_vec[[angle_i for angle_i, angle_val in enumerate(angles_vec) if np.mod(_phase_mean, 2 * np.pi) < angle_val][0]], 3)

                RRP_time_ampl['group'].append(subgroup_type)
                RRP_time_ampl['odor'].append(odor)
                RRP_time_ampl['phase'].append(phase)
                RRP_time_ampl['time'].append(mean_time)
                RRP_time_ampl['amplitude'].append(np.mean(ampl_responses_filter))

    df_export = pd.DataFrame(RRP_time_ampl)
    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'time'))
    df_export.to_excel('df_intra_allgroup.xlsx')


def timing_ERP_IE_SUM_plot(xr_data):   

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = xr_data['time'].data

    ######## PLOT RESPONSE FOR ALL GROUPS ########

    #subgroup_type = 'allsujet'
    for subgroup_type in ['allsujet', 'rep', 'no_rep']:

        ######## INTRA ########
        #### load stats
        timing_data = np.zeros((len(odor_list), len(['expi', 'inspi']), len(chan_list_eeg), len(['value', 'time'])))

        for odor_i, odor in enumerate(['o', '+', '-']):

            if subgroup_type == 'allsujet':
                data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'rep':
                data_baseline = xr_data.loc[sujet_best_list_rev, 'FR_CV_1', odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_baseline = xr_data.loc[sujet_no_respond_rev, 'FR_CV_1', odor, :, :].values

            cond = 'CO2'

            print('intra', subgroup_type, odor, cond)

            if subgroup_type == 'allsujet':
                data_cond = xr_data.loc[:, cond, odor, :, :].values
                
            elif subgroup_type == 'rep':
                data_cond = xr_data.loc[sujet_best_list_rev, cond, odor, :, :].values

            elif subgroup_type == 'no_rep':
                data_cond = xr_data.loc[sujet_no_respond_rev, cond, odor, :, :].values

            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                mask_signi = get_permutation_cluster_1d(data_baseline_chan, data_cond_chan, ERP_n_surrogate)

                for respi_phase_i, respi_phase in enumerate(['expi', 'inspi']):

                    if respi_phase == 'inspi':
                        mask_signi_phase = np.concatenate(( np.zeros(int(time_vec.size/2)).astype('bool'), mask_signi[int(time_vec.size/2):] ))

                    elif respi_phase == 'expi':
                        mask_signi_phase = np.concatenate(( mask_signi[:int(time_vec.size/2)], np.zeros(int(time_vec.size/2)).astype('bool') ))

                    if mask_signi_phase.sum() == 0:
                        continue     

                    else:
                        mask_signi_phase[0] = False
                        mask_signi_phase[-1] = False   

                    if np.diff(mask_signi_phase).sum() > 2: 

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        max_chunk_signi = []
                        max_chunk_time = []
                        for start_i in np.arange(0, start_stop_chunk.size, 2):

                            _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[start_i]:start_stop_chunk[start_i+1]])
                            max_chunk_time.append(start_stop_chunk[start_i] +_argmax)
                            max_chunk_signi.append(data_cond_chan.mean(axis=0)[_argmax])

                        max_rep = max_chunk_signi[np.argmax(np.abs(max_chunk_signi))]
                        time_max_rep = time_vec[max_chunk_time[np.where(max_chunk_signi == max_rep)[0][0]]]

                    else:

                        start_stop_chunk = np.where(np.diff(mask_signi_phase))[0]

                        _argmax = np.argmax(np.abs(data_cond_chan.mean(axis=0))[start_stop_chunk[0]:start_stop_chunk[1]])
                        max_rep = data_cond_chan.mean(axis=0)[start_stop_chunk[0] + _argmax]
                        time_max_rep = time_vec[start_stop_chunk[0] +_argmax]

                    timing_data[odor_i, respi_phase_i, chan_i, 0] = max_rep
                    timing_data[odor_i, respi_phase_i, chan_i, 1] = time_max_rep

        xr_coords = {'odor' : odor_list, 'phase' : ['expi', 'inspi'], 'chan' : chan_list_eeg, 'type' : ['value', 'time']}
        xr_timing = xr.DataArray(data=timing_data, coords=xr_coords)     

        ### plot
        min, max = xr_timing.loc[:,:,:,'value'].data.min(), xr_timing.loc[:,:,:,'value'].data.max()
        vlim = np.max([np.abs(min), np.abs(max)])
        if subgroup_type == 'rep':
            vlim = 0.22
        color_phase = {'expi' : 'tab:green', 'inspi' : 'tab:orange'}

        fig, axs = plt.subplots(ncols=3, subplot_kw={'projection': 'polar'}, figsize=(10,8))

        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]
            mean_vec = {}

            for phase in ['expi', 'inspi']:

                angles_vec = np.linspace(0, 2*np.pi, num=time_vec.size)
                mask_sel = (xr_timing.loc[odor, phase, :, 'time'] != 0).data
                time_responses_filter = xr_timing.loc[odor, phase, :, 'time'].data[mask_sel]
                ampl_responses_filter = xr_timing.loc[odor, phase, :, 'value'].data[mask_sel]

                time_responses_i = [np.where(time_vec == time_val)[0][0] for time_val in time_responses_filter]
                angle_responses = angles_vec[time_responses_i]

                _phase_mean = np.angle(np.mean(np.exp(1j*angle_responses)))
                
                mean_vec[phase] = np.round(time_vec[[angle_i for angle_i, angle_val in enumerate(angles_vec) if np.mod(_phase_mean, 2 * np.pi) < angle_val][0]], 2)

                ax.scatter(angle_responses, ampl_responses_filter, color=color_phase[phase], s=10)

                ax.plot(angles_vec, np.zeros((time_vec.size)), color='k')

                ax.plot([_phase_mean, _phase_mean], [0, np.mean(ampl_responses_filter)], color=color_phase[phase])

            ax.set_xticks(np.linspace(0, 2 * np.pi, num=4, endpoint=False))
            ax.set_xticklabels(np.round(np.linspace(-2.5,2.5, num=4, endpoint=False), 2))
            ax.set_yticks(np.round(np.linspace(-vlim,vlim, num=3, endpoint=True), 2))
            ax.set_rlim([-vlim,vlim])
            ax.set_title(f"{odor} \n inspi : {mean_vec['inspi']}, expi : {mean_vec['expi']}")

        plt.suptitle(subgroup_type)
        plt.tight_layout()

        # plt.show()

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'time'))
        plt.savefig(f"{subgroup_type}_CO2.png")

    




########################
######## MINMAX ########
########################



def compute_topoplot_stats_allsujet_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    ######## INTRA ########
    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        data_baseline = xr_data.loc[:, 'FR_CV_1', odor, :, :].values

        for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

            print('intra', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1)

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                if pval < 0.05:
                    
                    mask_signi[chan_i] = True

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    plt.tight_layout()

    plt.suptitle(f'minmax ALLSUJET INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_intra_allsujet.jpeg")
    
    # plt.show()

    ######## INTER ########
    #### scale
    min = np.array([])
    max = np.array([])

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

    for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        data_baseline = xr_data.loc[:, cond, 'o', :, :].values

        for odor_i, odor in enumerate(['+', '-']):

            print('inter', odor, cond)

            data_cond = xr_data.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                if pval < 0.05:
                        
                    mask_signi[chan_i] = True
            
            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    plt.tight_layout()

    plt.suptitle(f'minmax ALLSUJET INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}')

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_inter_allsujet.jpeg")

    # plt.show()







def compute_topoplot_stats_repnorep_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #sujet_sel = 'rep'
    for sujet_sel in ['rep', 'no_rep']:

        #### select data
        if sujet_sel == 'rep':
            xr_data_sel = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        elif sujet_sel == 'no_rep':
            xr_data_sel = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        ######## INTRA ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for odor_i, odor in enumerate(['o', '+', '-']):

        #         data_baseline = xr_data_vlim.loc[:, 'FR_CV_1', odor, :, :].values

        #         for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

        #             min = np.append(min, data_topo.min())
        #             max = np.append(max, data_topo.max())

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

                min = np.append(min, data_topo.min())
                max = np.append(max, data_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

        for odor_i, odor in enumerate(['o', '+', '-']):

            data_baseline = xr_data_sel.loc[:, 'FR_CV_1', odor, :, :].values

            for cond_i, cond in enumerate(['MECA', 'CO2', 'FR_CV_2']):

                print('intra', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                    data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                    pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                    if pval < 0.05:
                        
                        mask_signi[chan_i] = True

                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

        plt.tight_layout()

        plt.suptitle(f"minmax {sujet_sel} INTRA (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"minmax_intra_{sujet_sel}.jpeg")
        
        # plt.show()

        ######## INTER ########
        #### scale
        min = np.array([])
        max = np.array([])

        # for sujet_sel_vlim in ['rep', 'no_rep']:

        #     if sujet_sel_vlim == 'rep':
        #         xr_data_vlim = xr_data.loc[sujet_best_list_rev, :, :, :, :]
        #     elif sujet_sel_vlim == 'no_rep':
        #         xr_data_vlim = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

        #     for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

        #         data_baseline = xr_data_vlim.loc[:, cond, 'o', :, :].values

        #         for odor_i, odor in enumerate(['+', '-']):

        #             data_cond = xr_data_vlim.loc[:, cond, odor, :, :].values

        #             data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
        #             data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

        #             data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

        #             min = np.append(min, data_topo.min())
        #             max = np.append(max, data_topo.max())
            
        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

                min = np.append(min, data_topo.min())
                max = np.append(max, data_topo.max())

        min = min.min()
        max = max.max()
        vlim = np.array([max, min]).max()

        #### plot
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,15))

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_sel.loc[:, cond, 'o', :, :].values

            for odor_i, odor in enumerate(['+', '-']):

                print('inter', odor, cond)

                data_cond = xr_data_sel.loc[:, cond, odor, :, :].values

                mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

                data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
                data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

                data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0) 

                #chan_i, chan = 0, chan_list_eeg[0]
                for chan_i, chan in enumerate(chan_list_eeg):

                    data_baseline_chan = data_baseline[:, chan_i, :]
                    data_cond_chan = data_cond[:, chan_i, :] 

                    data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                    data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                    pval = pg.ttest(data_baseline_red, data_cond_red, paired=True, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]

                    if pval < 0.05:
                            
                        mask_signi[chan_i] = True
                
                ax = axs[odor_i, cond_i]

                ax.set_title(f"{cond} {odor}")

                mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                                mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

        plt.tight_layout()

        plt.suptitle(f"minmax {sujet_sel} INTER (cond-baseline) {np.round(-vlim,2)}:{np.round(vlim,2)}")

        os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
        fig.savefig(f"minmax_inter_{sujet_sel}.jpeg")

        # plt.show()



def compute_topoplot_stats_repnorep_diff_minmax(xr_data):

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg)
    info = mne.create_info(chan_list_eeg, ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### sel data
    xr_data_rep = xr_data.loc[sujet_best_list_rev, :, :, :, :]
    xr_data_norep = xr_data.loc[sujet_no_respond_rev, :, :, :, :]

    #### scale
    min = np.array([])
    max = np.array([])

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            min = np.append(min, data_topo.min())
            max = np.append(max, data_topo.max())

    min = min.min()
    max = max.max()
    vlim = np.array([max, min]).max()

    #### plot
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15,15))

    for odor_i, odor in enumerate(['o', '+', '-']):

        for cond_i, cond in enumerate(['FR_CV_1', 'MECA', 'CO2', 'FR_CV_2']):

            print('intra', odor, cond)

            data_baseline = xr_data_rep.loc[:, cond, odor, :, :].values
            data_cond = xr_data_norep.loc[:, cond, odor, :, :].values

            mask_signi = np.zeros(len(chan_list_eeg)).astype('bool')

            data_baseline_red = data_baseline.max(axis=2) - data_baseline.min(axis=2) 
            data_cond_red = data_cond.max(axis=2) - data_cond.min(axis=2)

            data_topo = data_cond_red.mean(axis=0) - data_baseline_red.mean(axis=0)

            #chan_i, chan = 0, chan_list_eeg[0]
            for chan_i, chan in enumerate(chan_list_eeg):

                data_baseline_chan = data_baseline[:, chan_i, :]
                data_cond_chan = data_cond[:, chan_i, :] 

                data_baseline_red = data_baseline_chan.max(axis=1) - data_baseline_chan.min(axis=1) 
                data_cond_red = data_cond_chan.max(axis=1) - data_cond_chan.min(axis=1) 

                pval = pg.ttest(data_baseline_red, data_cond_red, paired=False, alternative='two-sided', correction=None, confidence=0.95)['p-val'].values[0]


                if pval < 0.05:
                    
                    mask_signi[chan_i] = True

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond} {odor}")

            mne.viz.plot_topomap(data=data_topo, axes=ax, show=False, names=chan_list_eeg, pos=info,
                            mask=mask_signi, mask_params=mask_params, vlim=(-vlim, vlim))

    # plt.tight_layout()

    plt.suptitle(f"minmax norep - rep {np.round(-vlim,2)}:{np.round(vlim,2)}")

    os.chdir(os.path.join(path_results, 'allplot', 'ERP', 'topoplot_summary_stats'))
    fig.savefig(f"minmax_repnorep.jpeg")
    
    # plt.show()






################################
######## EXPORT DF ########
################################


def export_df_ERP(stretch=False):

    if stretch:
        extract_time = {'expi' : [0, int(stretch_point_TF/2)], 'inspi' : [int(stretch_point_TF/2), stretch_point_TF]}
        xr_data, xr_data_sem = compute_ERP_stretch()
        xr_data = xr_data.rename({'phase' : 'phase_vec'})
    else:
        extract_time = {'inspi' : [-1, 0], 'expi' : [0, 1]}
        xr_data, xr_data_sem = compute_ERP()

    chan_list_MI = ['C3', 'Cz', 'C4', 'FC1', 'FC2']
    cond_sel = ['FR_CV_1', 'CO2']
    phase_list = ['inspi', 'expi']

    df_ERP = pd.DataFrame()
    for phase in phase_list:
        if stretch:
            mask_time = xr_data['phase_vec'].values[(xr_data['phase_vec'].values < extract_time[phase][1]) & (xr_data['phase_vec'].values >= extract_time[phase][0])]
            _df_ERP = xr_data.loc[:,cond_sel,:,chan_list_MI,mask_time].sum('phase_vec').to_dataframe(name='val').reset_index(drop=False)
        else:
            mask_time = xr_data['time'].values[(xr_data['time'].values < extract_time[phase][1]) & (xr_data['time'].values >= extract_time[phase][0])]
            _df_ERP = xr_data.loc[:,cond_sel,:,chan_list_MI,mask_time].sum('time').to_dataframe(name='val').reset_index(drop=False)
        _df_ERP = pd.pivot_table(_df_ERP, values='val', columns=['nchan'], index=['sujet', 'cond', 'odor']).reset_index(drop=False)
        _df_ERP['phase'] =  [phase]*_df_ERP.shape[0]
        df_ERP = pd.concat([df_ERP, _df_ERP])

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
    if stretch:
        df_ERP.to_excel('stretch_df_ERP_allsujet.xlsx')
    else:
        df_ERP.to_excel('nostretch_df_ERP_allsujet.xlsx')










