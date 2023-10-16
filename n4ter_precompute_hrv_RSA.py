

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import physio
import pickle

from n0_config_params import *
from n0bis_config_analysis_functions import *

debug = False





################################
######## RSA COMPUTE ########
################################

#sujet = sujet_list[0]
def precompute_RSA_sujet(sujet):

    #### identify if already computed for all
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

    if os.path.exists(f'{sujet}_RSA.pkl'):
        print('ALREADY COMPUTED', flush=True)
        return

    #### open params
    respfeatures_allcond = load_respfeatures(sujet)

    RSA_allcond = {}

    #odor = odor_list[0]
    for odor in odor_list:

        RSA_allcond[odor] = {}

        #cond = conditions[2]
        for cond in conditions:

            print(f'#### {sujet} RSA {cond} {odor} ####', flush=True)

            #### load
            data = load_data_sujet(sujet, cond, odor)
            raw_resp = data[chan_list.index('PRESS'),:] 
            raw_ecg = data[chan_list.index('ECG'),:]

            respfeatures = respfeatures_allcond[cond][odor]

            if debug:
                
                plt.plot(zscore(raw_resp))
                plt.plot(zscore(raw_ecg))
                plt.show()            

            # resp, resp_cycles = physio.compute_respiration(raw_resp, srate, parameter_preset='human_airflow')
            ecg, ecg_peaks = physio.compute_ecg(raw_ecg, srate, parameter_preset='human_ecg')

            resp_cycles = respfeatures

            points_per_cycle = 500

            rsa_cycles, cyclic_cardiac_rate = physio.compute_rsa(resp_cycles, ecg_peaks, srate=srate, two_segment=True, points_per_cycle=points_per_cycle)

            if debug :
                
                inspi_expi_ratio = 0.5

                one_cycle = np.arange(points_per_cycle) / points_per_cycle

                fig, ax = plt.subplots()
                ax.plot(one_cycle, cyclic_cardiac_rate.T, color='k', alpha=.3)
                ax.plot(one_cycle, np.mean(cyclic_cardiac_rate, axis=0), color='darkorange', lw=3)
                ax.axvspan(0, inspi_expi_ratio, color='g', alpha=0.3)
                ax.axvspan(inspi_expi_ratio, 1, color='r', alpha=0.3)
                ax.set_xlabel('One respiratory cycle')
                ax.set_ylabel('Heart rate [bpm]')
                ax.set_xlim(0, 1)
                ax.text(0.2, 60, 'inhalation', ha='center', color='g')
                ax.text(0.85, 60, 'exhalation', ha='center', color='r')
                ax.set_title('All RSA cycle streched to resp cycles')

                plt.show()

            RSA_allcond[odor][cond] = cyclic_cardiac_rate

    #### save & transert
    print('SAVE', flush=True)
    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

    with open(f'{sujet}_RSA.pkl', 'wb') as f:
        pickle.dump(RSA_allcond, f)







################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    #sujet = sujet_list[1]
    for sujet in sujet_list:
    
        precompute_RSA_sujet(sujet)
        execute_function_in_slurm_bash_mem_choice('n4ter_precompute_hrv_RSA', 'precompute_RSA_sujet', [sujet], '10G')



