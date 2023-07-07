

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

def get_figure_allsujet(mode_plot_prediction):

    classes_colors = ['b', 'g', 'c', 'm', 'y']

    #n_classes = '2classes'
    for n_classes in ['2classes', '4classes']:

        ################################
        ######## TRAIN MODIF ########
        ################################

        #### load data
        #sujet = sujet_list[0]
        for sujet in sujet_list:
            
            if sujet in ['31HJ']:
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
        
        n_sujet = xr_hrv_tracker_allsujet['sujet'].shape[0]

        #### count
        classes = np.unique(xr_hrv_tracker_allsujet.values).astype('int')
        #classe = classes[0]
        for classe_i, classe in enumerate(classes):
            if classe_i == 0:
                xr_classes = (((xr_hrv_tracker_allsujet.loc[:,:,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet).drop('type').expand_dims({'classe' : [classe]})
            else:
                xr_classes = xr.concat([xr_classes, (((xr_hrv_tracker_allsujet.loc[:,:,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet).drop('type').expand_dims({'classe' : [classe]})], dim='classe')

        xr_classes_max_prob = xr_classes.argmax(axis=0)

        #### figure
        os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

        #train_size = xr_hrv_tracker_allsujet['train_percentage'].values[0]
        for train_size in xr_hrv_tracker_allsujet['train_percentage'].values:

            fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

            #odor_i, odor = 0, odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i]

                if mode_plot_prediction == 'max_proba':

                    ax.plot(xr_classes_max_prob.loc[train_size, odor, :].values, color='m', linestyle='-', label='max_proba')

                if mode_plot_prediction == 'full_proba':

                    for classe_i, classe in enumerate(classes):
                        ax.plot(xr_classes.loc[classe, train_size, odor, :].values, color=classes_colors[classe_i], label=classe, linestyle='-')

                ax.plot(xr_hrv_tracker_allsujet.loc[:, train_size, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
                ax.plot(xr_hrv_tracker_allsujet.loc[:, train_size, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
                ax.set_ylim(0, classes.max()+0.5)
                ax.set_title(f'{odor}')
                ax.legend()

            plt.suptitle(f'train size : {train_size}')

            # fig.show()

            fig.savefig(f'{n_classes}_trainsize_{train_size}_allsujet_no_ref_hrv_tracker.png')

            plt.close()

        #### plot score
        xr_hrv_tracker_score_allsujet.rename('value')
        df_score = xr_hrv_tracker_score_allsujet.to_dataframe(name='value').reset_index()

        plt.figure()
        sns.pointplot(data=df_score, x='train_percentage', y='value', hue='odor')
        # plt.show()
        plt.savefig(f'{n_classes}_allsujet_no_ref_detection_perf_allsize.png')



        ################################
        ######## WITH REF ########
        ################################

        #### load data with ref
        #sujet = sujet_list[1]
        for sujet in sujet_list:

            if sujet in ['31HJ']:
                continue

            os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

            if sujet == sujet_list[0]:

                xr_hrv_tracker_allsujet = xr.open_dataarray(f'{n_classes}_o_ref_{sujet}_hrv_tracker.nc')
                xr_hrv_tracker_score_allsujet = xr.open_dataarray(f'{n_classes}_o_ref_{sujet}_hrv_tracker_score.nc')

            else:

                xr_hrv_tracker_allsujet_i = xr.open_dataarray(f'{n_classes}_o_ref_{sujet}_hrv_tracker.nc')
                xr_hrv_tracker_score_allsujet_i = xr.open_dataarray(f'{n_classes}_o_ref_{sujet}_hrv_tracker_score.nc')

                xr_hrv_tracker_allsujet = xr.concat([xr_hrv_tracker_allsujet, xr_hrv_tracker_allsujet_i], dim='sujet')
                xr_hrv_tracker_score_allsujet = xr.concat([xr_hrv_tracker_score_allsujet, xr_hrv_tracker_score_allsujet_i], dim='sujet')

        n_sujet = xr_hrv_tracker_allsujet['sujet'].shape[0]

        #### count
        classes = np.unique(xr_hrv_tracker_allsujet.values).astype('int')
        #classe = classes[0]
        for classe_i, classe in enumerate(classes):
            if classe_i == 0:
                xr_classes = (((xr_hrv_tracker_allsujet.loc[:,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet).drop('type').expand_dims({'classe' : [classe]})
            else:
                xr_classes = xr.concat([xr_classes, (((xr_hrv_tracker_allsujet.loc[:,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet).drop('type').expand_dims({'classe' : [classe]})], dim='classe')

        xr_classes_max_prob = xr_classes.argmax(axis=0)

        #### figure
        os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

        train_size = 0.8

        fig, axs = plt.subplots(ncols=len(odor_list), figsize=(15,10))

        #odor_i, odor = 0, odor_list[0]
        for odor_i, odor in enumerate(odor_list):

            ax = axs[odor_i]

            if mode_plot_prediction == 'max_proba':

                ax.plot(xr_classes_max_prob.loc[odor, :].values, color='m', linestyle='-', label='max_proba')

            if mode_plot_prediction == 'full_proba':

                for classe_i, classe in enumerate(classes):
                    ax.plot(xr_classes.loc[classe, odor, :].values, color=classes_colors[classe_i], label=classe, linestyle='-')

            ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
            ax.plot(xr_hrv_tracker_allsujet.loc[:, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
            ax.set_ylim(0, classes.max()+0.5)
            ax.set_title(f'{odor}')
            ax.legend()

        plt.suptitle(f'train size : {train_size}')

        # fig.show()

        fig.savefig(f'{n_classes}_allsujet_o_ref_hrv_tracker_trainsize_{train_size}.png')

        #### figure short list subjects
        allsujet_xr = xr_hrv_tracker_allsujet.sujet.values
        sujet_best_list_included = [f"{sujet[-2:]}{sujet[:2]}" for sujet in sujet_best_list]
        sujet_best_list_included = [sujet for sujet in sujet_best_list_included if sujet in allsujet_xr]

        sujet_best_list_not_included = [sujet for sujet in allsujet_xr if sujet not in sujet_best_list_included]

        #### plot
        fig, axs = plt.subplots(ncols=2, nrows=len(odor_list), figsize=(15,10))
        
        #target_i, target = 0, 'o_respond'
        for target_i, target in enumerate(['o_respond', 'o_no_respond']):

            if target == 'o_respond':
                sujet_selection = sujet_best_list_included
            if target == 'o_no_respond':
                sujet_selection = sujet_best_list_not_included

            n_sujet_sel = len(sujet_selection)

            for classe_i, classe in enumerate(classes):
                if classe_i == 0:
                    xr_classes = (((xr_hrv_tracker_allsujet.loc[sujet_selection,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet_sel).drop('type').expand_dims({'classe' : [classe]})
                else:
                    xr_classes = xr.concat([xr_classes, (((xr_hrv_tracker_allsujet.loc[sujet_selection,:,'prediction',:] == classe) * 1).sum(dim='sujet') / n_sujet_sel).drop('type').expand_dims({'classe' : [classe]})], dim='classe')

            xr_classes_max_prob = xr_classes.argmax(axis=0)

            xr_hrv_tracker_allsujet_filter = xr_hrv_tracker_allsujet.loc[sujet_selection, :, :, :]

            #odor_i = odor_list[0]
            for odor_i, odor in enumerate(odor_list):

                ax = axs[odor_i, target_i]

                if mode_plot_prediction == 'max_proba':

                    ax.plot(xr_classes_max_prob.loc[odor, :].values, color='m', linestyle='-', label='max_proba')

                if mode_plot_prediction == 'full_proba':

                    for classe_i, classe in enumerate(classes):
                        ax.plot(xr_classes.loc[classe, odor, :].values, color=classes_colors[classe_i], label=classe, linestyle='-')
        
                ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'label', :].mean(axis=0).data, color='k', label='real', linestyle='--')
                ax.plot(xr_hrv_tracker_allsujet_filter.loc[:, odor, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
                ax.set_ylim(0, classes.max()+0.5)

                if odor_i == 0:
                    ax.set_title(f'{target}')
                if target_i == 0:
                    ax.set_ylabel(f'{odor}')
                ax.legend()

        plt.suptitle(f'train size : {train_size}')

        # fig.show()

        fig.savefig(f'allsujet_o_ref_odor_repond_hrv_tracker_trainsize_{train_size}.png')









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    mode_plot_prediction = 'max_proba'
    # mode_plot_prediction = 'full_proba'

    get_figure_allsujet(mode_plot_prediction)




