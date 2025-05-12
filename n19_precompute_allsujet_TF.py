
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



    
    

def generate_df_TF_allsujet():

    print('COMPUTE', flush=True)

    if os.path.exists(os.path.join(path_precompute, 'allsujet', 'TF', f'df_allsujet_TF.xlsx')):
        print('ALREADY COMPUTED', flush=True)
        return

    else:

        phase_vec = np.arange(stretch_point_TF)

        #sujet_i, sujet = 0, sujet_list[0]
        # for sujet_i, sujet in enumerate(sujet_list):
        def get_df_TF_sujet(sujet):

            print(sujet)

            df_TF_sujet = pd.DataFrame()

            os.chdir(os.path.join(path_precompute, 'allsujet', 'TF', 'conv'))
            _tf = np.load(f'{sujet}_tf_conv_allcond.npy')

            #cond_i, cond = 0, conditions[0]
            for cond_i, cond in enumerate(conditions):

                #odor_i, odor = 0, odor_list[0]
                for odor_i, odor in enumerate(odor_list):

                    for band, freq in freq_band_dict_lmm.items():

                        frex_mask = (frex >= freq[0]) & (frex < freq[1])

                        for resp_phase in ['inspi', 'expi']:

                            if resp_phase == 'inspi':
                                phase_mask = phase_vec < stretch_point_TF/2

                            if resp_phase == 'expi':
                                phase_mask = phase_vec >= stretch_point_TF/2

                            for chan_i, chan in enumerate(chan_list_eeg):

                                _Pxx = np.median(_tf[cond_i, odor_i, chan_i,:,:][frex_mask,:][:,phase_mask])

                                _df_chunk = pd.DataFrame({'sujet' : [sujet], 'chan' : [chan], 'cond' : [cond], 'odor' : [odor], 'band' : [band], 'phase' : [resp_phase], 'Pxx' : [_Pxx], 'rep' : [sujet in sujet_best_list_rev]})

                                df_TF_sujet = pd.concat([df_TF_sujet, _df_chunk], axis=0)

            return df_TF_sujet

        #get_df_TF_sujet(sujet)
        alldf = joblib.Parallel(n_jobs = n_core, prefer = 'processes')(joblib.delayed(get_df_TF_sujet)(sujet) for sujet in sujet_list)
        
        df_TF_allsujet = pd.DataFrame()

        for sujet_i, sujet in enumerate(sujet_list):

            df_TF_allsujet = pd.concat([df_TF_allsujet, alldf[sujet_i]])

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        df_TF_allsujet.to_excel('df_allsujet_TF.xlsx')

        # os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        # df_TF_allsujet = pd.read_excel('df_allsujet_TF.xlsx')

        df_Pxx_region = pd.DataFrame()

        for sujet_i, sujet in enumerate(sujet_list):

            print(sujet)
                
            for cond_i, cond in enumerate(conditions):

                for odor_i, odor in enumerate(odor_list):

                    for band_i, band in enumerate(freq_band_dict_lmm):
                
                        for phase_i, phase in enumerate(['inspi', 'expi']): 
                            
                            for region_i, region in enumerate(chan_list_lobes_lmm.items()):

                                _val = np.median(df_TF_allsujet.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and band == '{band}' and phase == '{phase}' and chan in {region[-1]}")['Pxx'].values)
                                _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'odor' : [odor], 'band' : [band], 'phase' : [phase], 'region' : [region[0]], 'Pxx' : [_val], 'rep' : [sujet in sujet_best_list_rev]})

                                df_Pxx_region = pd.concat([df_Pxx_region, _df])

        os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
        df_Pxx_region.to_excel('df_allsujet_TF_region.xlsx')






########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':
        

    generate_df_TF_allsujet()


        







