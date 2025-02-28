

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


debug = False

#sujet = sujet_list[1]
def plot_RSA(sujet):

    print(f'{sujet}')

    os.chdir(os.path.join(path_precompute, sujet, 'HRV'))
                
    with open(f'{sujet}_RSA.pkl', 'rb') as f:
        RSA_allcond = pickle.load(f)

    os.chdir(os.path.join(path_results, sujet, 'RSA'))

    #### INTRA

    cond_plot = ['MECA', 'CO2', 'FR_CV_2']

    fig, axs = plt.subplots(nrows=len(odor_list), ncols=len(cond_plot))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    scales_val = {'min' : [], 'max' : []}

    #cond_i, cond = 2, cond_plot[2]
    for cond_i, cond in enumerate(conditions):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            val = RSA_allcond[odor][cond]

            scales_val['min'].append(val.mean(axis=0).min())
            scales_val['max'].append(val.mean(axis=0).max())

    scales_val['min'] = np.array(scales_val['min']).min()
    scales_val['max'] = np.array(scales_val['max']).max()

    plt.suptitle(f'{sujet} intra')

    #cond_i, cond = 1, 'MECA'
    for cond_i, cond in enumerate(cond_plot):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            data_cond = RSA_allcond[odor][cond].mean(axis=0)
            sem = RSA_allcond[odor][cond].std(axis=0) / np.sqrt(RSA_allcond[odor][cond].shape[0])
            baseline = RSA_allcond[odor]['FR_CV_1'].mean(axis=0)
            sem_baseline = RSA_allcond[odor]['FR_CV_1'].std(axis=0) / np.sqrt(RSA_allcond[odor]['FR_CV_1'].shape[0])    

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond}")

            if cond_i == 0:
                ax.set_ylabel(f"{odor}")

            ax.set_ylim(scales_val['min'], scales_val['max'])

            time_vec = np.arange(baseline.shape[0])

            ax.plot(time_vec, data_cond, color='r')
            ax.fill_between(time_vec, data_cond+sem, data_cond-sem, alpha=0.25, color='m')

            ax.plot(time_vec, baseline, label='FR_CV_1', color='b')
            ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

            ax.vlines(time_vec.size/2, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

    fig.tight_layout()
    plt.legend()

    # plt.show()

    #### save
    fig.savefig(f'intra_RSA.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()

    #### INTER

    odor_plot = ['+', '-']

    fig, axs = plt.subplots(nrows=len(odor_plot), ncols=len(conditions))

    fig.set_figheight(10)
    fig.set_figwidth(10)

    scales_val = {'min' : [], 'max' : []}

    #cond_i, cond = 2, cond_plot[2]
    for cond_i, cond in enumerate(conditions):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            val = RSA_allcond[odor][cond]

            scales_val['min'].append(val.mean(axis=0).min())
            scales_val['max'].append(val.mean(axis=0).max())

    scales_val['min'] = np.array(scales_val['min']).min()
    scales_val['max'] = np.array(scales_val['max']).max()

    plt.suptitle(f'{sujet} inter')

    #cond_i, cond = 1, 'MECA'
    for cond_i, cond in enumerate(conditions):

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_plot):

            data_cond = RSA_allcond[odor][cond].mean(axis=0)
            sem = RSA_allcond[odor][cond].std(axis=0) / np.sqrt(RSA_allcond[odor][cond].shape[0])
            baseline = RSA_allcond['o'][cond].mean(axis=0)
            sem_baseline = RSA_allcond['o'][cond].std(axis=0) / np.sqrt(RSA_allcond['o'][cond].shape[0])    

            ax = axs[odor_i, cond_i]

            ax.set_title(f"{cond}")

            if cond_i == 0:
                ax.set_ylabel(f"{odor}")

            ax.set_ylim(scales_val['min'], scales_val['max'])

            time_vec = np.arange(baseline.shape[0])

            ax.plot(time_vec, data_cond, color='r')
            ax.fill_between(time_vec, data_cond+sem, data_cond-sem, alpha=0.25, color='m')

            ax.plot(time_vec, baseline, label='o', color='b')
            ax.fill_between(time_vec, baseline+sem_baseline, baseline-sem_baseline, alpha=0.25, color='c')

            ax.vlines(time_vec.size/2, ymin=scales_val['min'], ymax=scales_val['max'], colors='g')  

    fig.tight_layout()
    plt.legend()

    # plt.show()

    #### save
    fig.savefig(f'inter_RSA.jpeg', dpi=150)

    fig.clf()
    plt.close('all')
    gc.collect()

    print('done')



################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    for sujet in sujet_list:
    
        plot_RSA(sujet)




