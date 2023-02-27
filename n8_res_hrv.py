

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from n0_config_params import *
from n0bis_config_analysis_functions import *


debug = False


########################
######## LOAD ########
########################

def open_xr_data():

    #### load data
    #sujet = sujet_list[1]
    for sujet in sujet_list:

        if sujet == '31HJ':
            continue

        os.chdir(os.path.join(path_precompute, sujet, 'HRV'))

        if sujet == sujet_list[0]:

            xr_hrv_tracker_allsujet = xr.open_dataarray(f'{sujet}_hrv_tracker.nc')
            xr_hrv_tracker_score_allsujet = xr.open_dataarray(f'{sujet}_hrv_tracker_score.nc')

        else:

            xr_hrv_tracker_allsujet_i = xr.open_dataarray(f'{sujet}_hrv_tracker.nc')
            xr_hrv_tracker_score_allsujet_i = xr.open_dataarray(f'{sujet}_hrv_tracker_score.nc')

            xr_hrv_tracker_allsujet = xr.concat([xr_hrv_tracker_allsujet, xr_hrv_tracker_allsujet_i], dim='sujet')
            xr_hrv_tracker_score_allsujet = xr.concat([xr_hrv_tracker_score_allsujet, xr_hrv_tracker_score_allsujet_i], dim='sujet')

    #### figure
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))

    #odor_i = odor_list[0]
    for odor_i in odor_list:

        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor_i, 'prediction', :].mean(axis=0).data, color='y', label='prediction', linestyle='--')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor_i, 'label', :].mean(axis=0).data, color='k', label='real')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor_i, 'trig_odor', :].mean(axis=0).data, color='r', label='odor_trig')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor_i, 'trig_odor', :].mean(axis=0).data + xr_hrv_tracker_allsujet.loc[:, 'o', 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
        ax.plot(xr_hrv_tracker_allsujet.loc[:, odor_i, 'trig_odor', :].mean(axis=0).data - xr_hrv_tracker_allsujet.loc[:, 'o', 'trig_odor', :].std(axis=0).data, color='r', linestyle='--')
        ax.set_ylim(0, 4)
        ax.set_title(f'odor : {odor_i}')
        ax.legend()
        # fig.show()
        plt.close()

        fig.savefig(f'hrv_tracker_{odor_i}.png')






