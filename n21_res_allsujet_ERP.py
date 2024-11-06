
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
import cv2

from n0_config_params import *
from n0bis_config_analysis_functions import *
from n0ter_stats import *
from n21bis_res_allsujet_ERP import *

from mne.stats import permutation_cluster_test
from mne.stats import permutation_cluster_1samp_test

debug = False







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    ######## MAIN WORKFLOW ########

    xr_data, xr_data_sem = compute_ERP()
    xr_data_stretch, xr_data_sem_stretch = compute_ERP_stretch()
    cluster_stats_type = 'manual_perm'
    cluster_stats_intra, cluster_stats_inter, cluster_stats_rep_norep = get_cluster_stats_manual_prem(stretch=False)
    cluster_stats_intra_stretch, cluster_stats_inter_stretch, cluster_stats_rep_norep_stretch = get_cluster_stats_manual_prem(stretch=True)

    plot_ERP_diff(stretch=False) # used for analysis
    plot_ERP_mean_subject_wise(stretch=False)

    plot_ERP_diff(stretch=True) # used for analysis
    plot_ERP_mean_subject_wise(stretch=True)







    ######## LOAD DATA ########

    # cond_erp = ['VS', 'MECA', 'CO2']

    print(f'#### compute allsujet ####', flush=True)

    xr_data, xr_data_sem = compute_ERP()
    df_stats_interintra = get_df_stats(xr_data)
    xr_lm_data, xr_lm_pred, xr_lm_pred_coeff = compute_lm_on_ERP(xr_data)


    ######## LINEAR REGRESSION PLOT ########

    xr_PPI_count = get_PPI_count(xr_data)
    
    # shuffle_way = 'inter_cond'
    # shuffle_way = 'intra_cond'
    shuffle_way = 'linear_based'
    xr_surr = compute_surr_ERP(xr_data, shuffle_way)
    
    print(f'#### plot allsujet ####', flush=True)

    plot_ERP(xr_data, xr_lm_data, xr_lm_pred, xr_lm_pred_coeff, xr_surr, xr_PPI_count)

    ######## IDENTIFY WHEN AND WHERE ERP OCCURE ########

    plot_ERP_response_profile(xr_data, xr_data_sem)

    ######## ERP ANALYSIS ########

    cluster_stats_type = 'manual_perm'
    # cluster_stats, cluster_stats_rep_norep = get_cluster_stats(xr_data)
    cluster_stats, cluster_stats_rep_norep = get_cluster_stats_manual_prem(xr_data)
    xr_cluster_based_perm = get_cluster_stats_manual_prem_subject_wise()

    df_ERP_metrics_allsujet, df_ERP_metric_A2_ratings = get_df_ERP_metric_allsujet(xr_data, xr_cluster_based_perm)

    plot_ERP_metrics_response(df_ERP_metrics_allsujet)
    plot_ERP_metrics_A2_lm(df_ERP_metric_A2_ratings)

    plot_ERP_diff(xr_data, cluster_stats, cluster_stats_type) # used for analysis
    plot_ERP_mean_subject_wise(xr_data, cluster_stats)

    plot_ERP_rep_norep(xr_data, cluster_stats_rep_norep, cluster_stats_type)

    plot_erp_response_stats(xr_data)

    ######## ERP TIME ########

    timing_ERP_IE_SUM_plot(xr_data)
    timing_ERP_IE_SUM_export_df(xr_data)

    ######## PPI ########

    #### plot PPI across subject
    plot_PPI_proportion(xr_PPI_count)

    #### manual evaluation
    generate_ppi_evaluation(xr_data)

    ######## MEAN RESPI ########

    plot_mean_respi()

    ######## REG ########

    print(f'#### plot discomfort / slope ####', flush=True)
    plot_slope_versus_discomfort(xr_data, xr_lm_data)

    ######## TOPOPLOTS AND STATS ########

    compute_topoplot_stats_allsujet_minmax(xr_data)
    compute_topoplot_stats_repnorep_minmax(xr_data)
    compute_topoplot_stats_repnorep_diff_minmax(xr_data)

    for perm_type in ['mne', 'inhouse']:
        compute_topoplot_stats_allsujet_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_perm(xr_data, perm_type)
        compute_topoplot_stats_repnorep_diff_perm(xr_data, perm_type)




