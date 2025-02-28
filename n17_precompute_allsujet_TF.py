
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False








def compute_TF_allsujet():

    #tf_mode = 'TF'
    for tf_mode in ['TF', 'ITPC']:

        if tf_mode == 'ITPC':
            continue

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

                #sujet_group_i, sujet_group = 0, sujet_group[0] 
                for sujet_group_i, sujet_group in enumerate(group_sujet):

                    if sujet_group == 'allsujet':  
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median, axis=0)
                    if sujet_group == 'rep':
                        sel = np.array([np.where(sujet_list == sujet)[0][0] for sujet in sujet_best_list_rev])
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median[sel,:], axis=0)
                    if sujet_group == 'no_rep':  
                        sel = np.array([np.where(sujet_list == sujet)[0][0] for sujet in sujet_no_respond_rev])
                        data_xr[nchan,sujet_group_i,:,:,:,:] = np.median(tf_median[sel,:], axis=0)

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





    
    

def generate_df_TF_allsujet():

    print('COMPUTE', flush=True)

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'TF', f'df_allsujet_TF.xlsx')):
        print('ALREADY COMPUTED', flush=True)
        return

    else:

        cond_sel = ['FR_CV_1', 'CO2']
        phase_vec = np.arange(stretch_point_TF)

        #sujet_i, sujet = 0, sujet_list[0]
        # for sujet_i, sujet in enumerate(sujet_list):
        def get_df_TF_sujet(sujet):

            df_TF_sujet = pd.DataFrame()

            #cond_i, cond = 0, conditions[0]
            for cond_i, cond in enumerate(cond_sel):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    os.chdir(os.path.join(path_precompute, sujet, 'TF'))
                    _tf = np.median(np.load(f'{sujet}_tf_conv_{cond}_{odor}.npy')[:,:,:,:], axis=1)

                    for band, freq in freq_band_dict['wb'].items():

                        frex_mask = (frex >= freq[0]) & (frex < freq[1])

                        for resp_phase in ['inspi', 'expi']:

                            if resp_phase == 'inspi':
                                phase_mask = phase_vec < stretch_point_TF/2

                            if resp_phase == 'expi':
                                phase_mask = phase_vec >= stretch_point_TF/2

                            for chan_i, chan in enumerate(chan_list_short):

                                _Pxx = np.median(_tf[chan_i,:,:][frex_mask,:][:,phase_mask])

                                _df_chunk = pd.DataFrame({'sujet' : [sujet], 'chan' : [chan], 'cond' : [cond], 'odor' : [odor], 'band' : [band], 'phase' : [resp_phase], 'Pxx' : [_Pxx]})

                                df_TF_sujet = pd.concat([df_TF_sujet, _df_chunk], axis=0)

            return df_TF_sujet

        #get_df_TF_sujet(sujet)
        alldf = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_df_TF_sujet)(sujet) for sujet in sujet_list)
        
        df_TF_allsujet = pd.DataFrame()

        for sujet_i, sujet in enumerate(sujet_list):

            df_TF_allsujet = pd.concat([df_TF_allsujet, alldf[sujet_i]])

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        df_TF_allsujet.to_excel('df_allsujet_TF.xlsx')










########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':
        
    compute_TF_allsujet()
    # execute_function_in_slurm_bash_mem_choice('n16_precompute_allsujet_TF', 'compute_TF_allsujet', [], '20G')

    generate_df_TF_allsujet()


        







