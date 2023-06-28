

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import seaborn as sns

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False


################################
######## COMPUTE ########
################################

def get_figure_allsujet():

    #### load data
    #n_classes = '4classes'
    for n_classes in ['2classes', '4classes']:

        #sujet = sujet_list[0]
        for sujet in sujet_list:
            
            if sujet in ['31HJ', '25DF']:
                continue

            os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

            if sujet == sujet_list[0]:

                xr_hrv_tracker_allsujet = xr.open_dataarray(f'{n_classes}_no_ref_{sujet}_hrv_tracker_alltestsize.nc')
                xr_hrv_tracker_score_allsujet = xr.open_dataarray(f'{n_classes}_no_ref_{sujet}_hrv_tracker_score_alltestsize.nc')

            else:

                xr_hrv_tracker_allsujet_i = xr.open_dataarray(f'{n_classes}_no_ref_{sujet}_hrv_tracker_alltestsize.nc')
                xr_hrv_tracker_score_allsujet_i = xr.open_dataarray(f'{n_classes}_no_ref_{sujet}_hrv_tracker_score_alltestsize.nc')

                xr_hrv_tracker_allsujet = xr.concat([xr_hrv_tracker_allsujet, xr_hrv_tracker_allsujet_i], dim='sujet')
                xr_hrv_tracker_score_allsujet = xr.concat([xr_hrv_tracker_score_allsujet, xr_hrv_tracker_score_allsujet_i], dim='sujet')
        
        xr_hrv_tracker_allsujet = xr_hrv_tracker_allsujet.astype('int')
        n_sujet = xr_hrv_tracker_allsujet['sujet'].shape[0]

        #### count
        classes = np.unique(xr_hrv_tracker_allsujet.values)
        classes_percentage = {}
        #classe = classes[0]
        for classe in classes:
            classes_percentage[classe] = (((xr_hrv_tracker_allsujet.loc[:,:,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet).drop('type')
        
        #### figure
        os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

        #train_size = xr_hrv_tracker_allsujet['train_percentage'].values[0]
        for train_size in xr_hrv_tracker_allsujet['train_percentage'].values:

            fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                for classe in classes:
                    ax.plot(classes_percentage[classe].loc[train_size, odor, :].values, label=classe, linestyle='-')
        
                ax.plot(xr_hrv_tracker_allsujet.loc[:, train_size, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
                ax.plot(xr_hrv_tracker_allsujet.loc[:, train_size, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
                ax.set_ylim(0, classes.max())
                ax.set_title(f'{odor}')
                ax.legend()

            plt.suptitle(f'train size : {train_size}')

            # fig.show()

            fig.savefig(f'test.png')

            fig.savefig(f'allsujet_no_ref_hrv_tracker_trainsize_{train_size}.png')

        #### plot score
        xr_hrv_tracker_score_allsujet.rename('value')
        df_score = xr_hrv_tracker_score_allsujet.to_dataframe(name='value').reset_index()

        sns.pointplot(data=df_score, x='train_percentage', y='value', hue='odor')
        plt.savefig('allsujet_no_ref_detection_perf_allsize.png')





    #### load data with ref
    #sujet = sujet_list[1]
    for sujet in sujet_list:

        if sujet == '31HJ':
            continue

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

        if sujet == sujet_list[0]:

            xr_hrv_tracker_allsujet = xr.open_dataarray(f'o_ref_{sujet}_hrv_tracker.nc')
            xr_hrv_tracker_score_allsujet = xr.open_dataarray(f'o_ref_{sujet}_hrv_tracker_score.nc')

        else:

            xr_hrv_tracker_allsujet_i = xr.open_dataarray(f'o_ref_{sujet}_hrv_tracker.nc')
            xr_hrv_tracker_score_allsujet_i = xr.open_dataarray(f'o_ref_{sujet}_hrv_tracker_score.nc')

            xr_hrv_tracker_allsujet = xr.concat([xr_hrv_tracker_allsujet, xr_hrv_tracker_allsujet_i], dim='sujet')
            xr_hrv_tracker_score_allsujet = xr.concat([xr_hrv_tracker_score_allsujet, xr_hrv_tracker_score_allsujet_i], dim='sujet')

    #### figure
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

    train_size = 0.8

    odor_list_test = [odor for odor in odor_list if odor != 'o']

    fig, axs = plt.subplots(ncols=len(odor_list_test), figsize=(15,10))

    #odor_i = odor_list[0]
    for odor_i, odor in enumerate(odor_list_test):

        ax = axs[odor_i]

        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'prediction', :].mean(axis=0).data, color='y', label='prediction', linestyle='--')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].mean(axis=0).data + xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].mean(axis=0).data - xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
        ax.set_ylim(0, 4)
        ax.set_title(f'{odor}')
        ax.legend()

    plt.suptitle(f'train size : {train_size}')

    # fig.show()

    fig.savefig(f'allsujet_o_ref_hrv_tracker_trainsize_{train_size}.png')

    #### figure short list subjects
    allsujet_xr = xr_hrv_tracker_allsujet.sujet.values
    sujet_best_list_included = [f"{sujet[-2:]}{sujet[:2]}" for sujet in sujet_best_list]
    sujet_best_list_included = [sujet for sujet in sujet_best_list_included if sujet in allsujet_xr]

    sujet_best_list_not_included = [sujet for sujet in allsujet_xr if sujet not in sujet_best_list_included]

    #### plot
    fig, axs = plt.subplots(ncols=len(odor_list_test), nrows=2, figsize=(15,10))

    for target_i, target in enumerate(['o_respond', 'o_no_respond']):

        if target == 'o_respond':
            xr_hrv_tracker_allsujet_filter = xr_hrv_tracker_allsujet.loc[sujet_best_list_included, :, :, :]
        if target == 'o_no_respond':
            xr_hrv_tracker_allsujet_filter = xr_hrv_tracker_allsujet.loc[sujet_best_list_not_included, :, :, :]

        #odor_i = odor_list[0]
        for odor_i, odor in enumerate(odor_list_test):

            ax = axs[target_i, odor_i]

            ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'prediction', :].mean(axis=0).data, color='y', label='prediction', linestyle='--')
            ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
            ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
            ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].mean(axis=0).data + xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
            ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].mean(axis=0).data - xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
            ax.set_ylim(0, 4)
            ax.set_title(f'{target}, {odor}')
            ax.legend()

    plt.suptitle(f'train size : {train_size}')

    # fig.show()

    fig.savefig(f'allsujet_o_ref_odor_repond_hrv_tracker_trainsize_{train_size}.png')









################################
######## EXECUTE ########
################################


if __name__ == '__main__':


    get_figure_allsujet()




