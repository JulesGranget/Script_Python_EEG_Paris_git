

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *
from n21bis_res_allsujet_ERP import *

debug = False







################################
######## EXECUTE ########
################################

if __name__ == '__main__':
    
    ######## MAIN WORKFLOW ########

    # plot_ERP_diff(stretch=False) # used for analysis
    # plot_ERP_mean_subject_wise(stretch=False)

    plot_ERP_diff(stretch=True) # used for analysis
    plot_ERP_mean_subject_wise(stretch=True)

    export_df_ERP(stretch=False)
    export_df_ERP(stretch=True)

    explore_sd_cleaning_preocmpute()





