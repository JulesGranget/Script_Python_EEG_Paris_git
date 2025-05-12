
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import gc
import xarray as xr
import seaborn as sns
import pickle
import cv2

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

from sklearn.svm import SVR  # Support Vector Regressor for regression
from sklearn.svm import SVC  # Support Vector Classifier for classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


debug = False




################################
######## GENERATE DF ########
################################

def get_df_PRED(stretch=False):

    if stretch:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'PRED', 'stretch_df_PRED.xlsx')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'PRED'))
            df_PRED = pd.read_excel('stretch_df_PRED.xlsx')
            return df_PRED
        
    else:

        if os.path.exists(os.path.join(path_precompute, 'allsujet', 'PRED', 'nostretch_df_PRED.xlsx')):

            os.chdir(os.path.join(path_precompute, 'allsujet', 'PRED'))
            df_PRED = pd.read_excel('nostretch_df_PRED.xlsx')
            return df_PRED
    
    chan_list_MI = ['C3', 'Cz', 'C4', 'FC1', 'FC2']

    os.chdir(os.path.join(path_precompute, 'allsujet', 'FC'))
    if stretch:
        df_MI = pd.read_excel('stretch_df_MI_allsujet.xlsx')
    else:
        df_MI = pd.read_excel('nostretch_df_MI_allsujet.xlsx')

    os.chdir(os.path.join(path_precompute, 'allsujet', 'ERP'))
    if stretch:
        df_ERP = pd.read_excel('stretch_df_ERP_allsujet.xlsx')
    else:
        df_ERP = pd.read_excel('nostretch_df_ERP_allsujet.xlsx')

    df_PRED = df_MI.copy()
    df_add = pd.DataFrame()
    
    for row_i in range(df_MI.shape[0]):

        _sujet, _cond, _odor, _phase = df_PRED.iloc[row_i,:]['sujet'], df_PRED.iloc[row_i,:]['cond'], df_PRED.iloc[row_i,:]['odor'], df_PRED.iloc[row_i,:]['phase']
        _row_add = df_ERP.query(f"sujet == '{_sujet}' and cond == '{_cond}' and odor == '{_odor}' and phase == '{_phase}'")[chan_list_MI]

        if _sujet in sujet_best_list_rev:
            rep_stat = True
        else:
            rep_stat = False

        _row_add['rep_stat'] = rep_stat
        df_add = pd.concat([df_add, _row_add], axis=0)

    df_PRED = pd.concat([df_PRED, df_add], axis=1)

    os.chdir(os.path.join(path_precompute, 'allsujet', 'PRED'))
    if stretch:
        df_PRED.to_excel('stretch_df_PRED.xlsx')
    else:
        df_PRED.to_excel('nostretch_df_PRED.xlsx')

    return df_PRED


def get_odor_df():

    os.chdir(os.path.join(path_data, 'psychometric'))
    df_odor_profile = pd.read_excel('OLFADYS_odor_profiles.xlsx')
    df_odor_choice = pd.read_excel('OLFADYS_odor_choice.xlsx')

    question_list = df_odor_profile['question'].unique()

    sujet_sel_mask = []

    for row_i in range(df_odor_profile.shape[0]):

        if df_odor_profile['sujet'].iloc[row_i] in sujet_best_list:

            sujet_sel_mask.append('YES')

        else:

            sujet_sel_mask.append('NO')

    df_odor_profile['select_best'] = sujet_sel_mask

    odor_choice = {}

    for sujet in df_odor_choice['sujet'].unique():

        odor_choice[sujet] = df_odor_choice.query(f"sujet == '{sujet}'")['odor'].values.tolist()

    for sujet_i, sujet in enumerate(sujet_list_rev):

        if sujet_i == 0:

            df_odor_profile_filtered = df_odor_profile.query(f"sujet == '{sujet}' and odor in {odor_choice[sujet]}")

        else:

            df_odor_profile_filtered = pd.concat((df_odor_profile_filtered, df_odor_profile.query(f"sujet == '{sujet}' and odor in {odor_choice[sujet]}")), axis=0)

    for row_i in range(df_odor_profile_filtered.shape[0]):

        sujet_i = df_odor_profile_filtered['sujet'].iloc[row_i]

        if df_odor_profile_filtered['odor'].iloc[row_i] == odor_choice[sujet_i][0]:

            odor_quality = '+'

        elif df_odor_profile_filtered['odor'].iloc[row_i] == odor_choice[sujet_i][1]:

            odor_quality = '-'

        df_odor_profile_filtered['odor'].iloc[row_i] = odor_quality

    df_odor_profile_filtered = df_odor_profile_filtered.query(f"odor == '+'")

    return question_list, df_odor_profile_filtered



def get_df_PRED_TF():

    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
    df_TF_allsujet = pd.read_excel('df_allsujet_TF.xlsx')

    cond_sel = ['FR_CV_1', 'CO2']
    data_PRED_TF = {}
    data_PRED_TF['sujet'] = []
    data_PRED_TF['rep_state'] = []

    for sujet in sujet_list:

        for chan in chan_list_short:

            for cond in cond_sel:

                for odor in odor_list:

                    for band, freq in freq_band_dict['wb'].items():

                        for phase in ['inspi', 'expi']:

                            data_PRED_TF[f"{chan}_{cond}_{odor}_{band}_{phase}"] = []

    for sujet in sujet_list:

        data_PRED_TF['sujet'].append(sujet)
    
        if sujet in sujet_best_list_rev:

            data_PRED_TF['rep_state'].append(True)

        else:

            data_PRED_TF['rep_state'].append(False)

        for chan in chan_list_short:

            for cond in cond_sel:

                for odor in odor_list:

                    for band, freq in freq_band_dict['wb'].items():

                        for phase in ['inspi', 'expi']:

                            _Pxx = df_TF_allsujet.query(f"sujet == '{sujet}' and chan == '{chan}' and cond == '{cond}' and odor == '{odor}' and band == '{band}' and phase == '{phase}'")['Pxx'].values[0]
                            
                            data_PRED_TF[f"{chan}_{cond}_{odor}_{band}_{phase}"].append(_Pxx)

    df_PRED_TF = pd.DataFrame(data_PRED_TF)
    feature_list_RS = [feature for feature in list(data_PRED_TF.keys()) if feature.find('FR_CV_1') != -1]
    feature_list_whole = list(data_PRED_TF.keys())

    return df_PRED_TF, feature_list_RS, feature_list_whole




def search_pred():


    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import sklearn

    #### load data Cxy
    os.chdir(os.path.join(path_precompute, 'allsujet', 'PSD_Coh'))
    df_Cxy = pd.read_excel(f"Cxy_allsujet.xlsx").drop(columns=['Unnamed: 0'])

    array_Cxy = np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_eeg)))

    sujet_idx = {s: i for i, s in enumerate(sujet_list)}
    cond_idx = {c: i for i, c in enumerate(conditions)}
    odor_idx = {o: i for i, o in enumerate(odor_list)}
    chan_idx = {ch: i for i, ch in enumerate(chan_list_eeg)}

    for _, row in df_Cxy.iterrows():
        i = sujet_idx[row['sujet']]
        j = cond_idx[row['cond']]
        k = odor_idx[row['odor']]
        l = chan_idx[row['chan']]
        array_Cxy[i, j, k, l] = row['Cxy']

    xr_Cxy = xr.DataArray(data=array_Cxy, dims=['sujet', 'cond', 'odor', 'chan'], coords={'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'chan' : chan_list_eeg})

    data_region_Cxy = np.zeros((len(sujet_list), len(conditions), len(odor_list), len(chan_list_lobes_lmm.keys())))

    for sujet_i, sujet in enumerate(sujet_list):
            
        for cond_i, cond in enumerate(conditions):

            for odor_i, odor in enumerate(odor_list):
                
                for region_i, region in enumerate(chan_list_lobes_lmm.items()):

                    data_region_Cxy[sujet_i, cond_i, odor_i, region_i] = xr_Cxy.loc[sujet, cond, odor, region[-1]].median('chan').values

    xr_Cxy_region = xr.DataArray(data=data_region_Cxy, dims=['sujet', 'cond', 'odor', 'region'], coords={'sujet' : sujet_list, 'cond' : conditions, 'odor' : odor_list, 'region' : list(chan_list_lobes_lmm.keys())})
    df_Cxy_region = xr_Cxy_region.to_dataframe(name='Cxy').reset_index(drop=False) 

    #### load Pxx
    phase_list = ['inspi', 'expi']

    os.chdir(os.path.join(path_precompute, 'allsujet', 'TF'))
    df_Pxx = pd.read_excel(f"df_allsujet_TF.xlsx").drop(columns=['Unnamed: 0'])

    df_Pxx_region = pd.DataFrame()

    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)
            
        for cond_i, cond in enumerate(conditions):

            for odor_i, odor in enumerate(odor_list):

                for band_i, band in enumerate(freq_band_dict_lmm):
            
                    for phase_i, phase in enumerate(phase_list): 
                        
                        for region_i, region in enumerate(chan_list_lobes_lmm.items()):

                            _val = np.median(df_Pxx.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and band == '{band}' and phase == '{phase}' and chan in {region[-1]}")['Pxx'].values)
                            _df = pd.DataFrame({'sujet' : [sujet], 'cond' : [cond], 'odor' : [odor], 'band' : [band], 'phase' : [phase], 'region' : [region[0]], 'Pxx' : [_val]})

                            df_Pxx_region = pd.concat([df_Pxx_region, _df])

    #### load HRV
    os.chdir(os.path.join(path_results, 'allplot', 'HRV'))
    df_hrv = pd.read_excel(f"allsujet_df_hrv_physio.xlsx")

    #### load respi
    os.chdir(os.path.join(path_results, 'allplot', 'RESPI'))
    df_respi = pd.read_excel(f"allsujet_df_respi.xlsx").drop(columns=['Unnamed: 0'])

    #### load stai
    os.chdir(os.path.join(path_data, 'psychometric'))
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
    df_stai = df_stai.query(f"sujet in {sujet_list_rev.tolist()}")

    #### load odor_profile
    os.chdir(os.path.join(path_data, 'psychometric'))
    df_odor_profile = pd.read_excel('OLFADYS_odor_profiles.xlsx')
    df_odor_choice = pd.read_excel('OLFADYS_odor_choice.xlsx')

    question_list = df_odor_profile['question'].unique()

    sujet_sel_mask = []

    for row_i in range(df_odor_profile.shape[0]):

        if df_odor_profile['sujet'].iloc[row_i] in sujet_best_list:

            sujet_sel_mask.append('YES')

        else:

            sujet_sel_mask.append('NO')

    df_odor_profile['select_best'] = sujet_sel_mask

    odor_choice = {}

    for sujet in df_odor_choice['sujet'].unique():

        odor_choice[sujet] = df_odor_choice.query(f"sujet == '{sujet}'")['odor'].values.tolist()

    for sujet_i, sujet in enumerate(sujet_list_rev):

        if sujet_i == 0:

            df_odor_profile_filtered = df_odor_profile.query(f"sujet == '{sujet}' and odor in {odor_choice[sujet]}")

        else:

            df_odor_profile_filtered = pd.concat((df_odor_profile_filtered, df_odor_profile.query(f"sujet == '{sujet}' and odor in {odor_choice[sujet]}")), axis=0)

    for row_i in range(df_odor_profile_filtered.shape[0]):

        sujet_i = df_odor_profile_filtered['sujet'].iloc[row_i]

        if df_odor_profile_filtered['odor'].iloc[row_i] == odor_choice[sujet_i][0]:

            odor_quality = '+'

        elif df_odor_profile_filtered['odor'].iloc[row_i] == odor_choice[sujet_i][1]:

            odor_quality = '-'

        df_odor_profile_filtered['odor'].iloc[row_i] = odor_quality

    df_odor = df_odor_profile_filtered[['sujet', 'question', 'odor', 'value']]

    #### construct data_set
    cond_sel = ['FR_CV_1']
    odor_sel = ['o', '+']

        #### Cxy
    df_Cxy_clf = pd.DataFrame()
    df_Cxy_clf_region = pd.DataFrame()

    for sujet in sujet_list:

        _df_Cxy_clf = pd.DataFrame()
        _df_Cxy_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for region in chan_list_lobes_lmm:

                    _metric = f"{cond}_{odor}_{region}_Cxy"
                    _val = df_Cxy_region.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and region == '{region}'")['Cxy'].values[0]

                    _df_Cxy_clf[_metric] = [_val]

        df_Cxy_clf_region = pd.concat([df_Cxy_clf_region, _df_Cxy_clf], axis=0)

    for sujet in sujet_list:

        _df_Cxy_clf = pd.DataFrame()
        _df_Cxy_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for chan in chan_list_eeg:

                    _metric = f"{cond}_{odor}_{chan}_Cxy"
                    _val = df_Cxy.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and chan == '{chan}'")['Cxy'].values[0]

                    _df_Cxy_clf[_metric] = [_val]

        df_Cxy_clf = pd.concat([df_Cxy_clf, _df_Cxy_clf], axis=0)

        #### Pxx
    df_Pxx_clf = pd.DataFrame()
    df_Pxx_clf_region = pd.DataFrame()

    for sujet in sujet_list:

        print(sujet)

        _df_Pxx_clf = pd.DataFrame()
        _df_Pxx_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for region in chan_list_lobes_lmm:

                    for band in list(freq_band_dict_lmm.keys()):

                        for phase in phase_list:

                            _metric = f"{cond}_{odor}_{region}_{band}_{phase}_Pxx"
                            _val = df_Pxx_region.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and region == '{region}' and band == '{band}' and phase == '{phase}'")['Pxx'].values[0]

                            _df_Pxx_clf[_metric] = [_val]

        df_Pxx_clf_region = pd.concat([df_Pxx_clf_region, _df_Pxx_clf], axis=0)

    for sujet in sujet_list:

        print(sujet)

        _df_Pxx_clf = pd.DataFrame()
        _df_Pxx_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for chan in chan_list_eeg:

                    for band in list(freq_band_dict_lmm.keys()):

                        for phase in phase_list:

                            _metric = f"{cond}_{odor}_{chan}_{band}_{phase}_Pxx"
                            _val = df_Pxx.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}' and chan == '{chan}' and band == '{band}' and phase == '{phase}'")['Pxx'].values[0]

                            _df_Pxx_clf[_metric] = [_val]

        df_Pxx_clf = pd.concat([df_Pxx_clf, _df_Pxx_clf], axis=0)

        #### HRV
    metric_HRV_list = ['HRV_Mean', 'HRV_SD', 'HRV_Median', 'HRV_Mad', 'HRV_CV', 'HRV_MCV', 'HRV_Asymmetry', 'HRV_RMSSD', 'HRV_RespHRV']

    df_hrv_clf = pd.DataFrame()

    for sujet in sujet_list:

        _df_hrv_clf = pd.DataFrame()
        _df_hrv_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for metric_HRV in metric_HRV_list:

                    _metric = f"{cond}_{odor}_{metric_HRV}"
                    _val = df_hrv.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}'")[metric_HRV].values[0]

                    _df_hrv_clf[_metric] = [_val]

        df_hrv_clf = pd.concat([df_hrv_clf, _df_hrv_clf], axis=0)

    metric_HRV_list_nathalie = ['HRV_RespHRV']

    df_hrv_clf_nathalie = pd.DataFrame()

    for sujet in sujet_list:

        _df_hrv_clf = pd.DataFrame()
        _df_hrv_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for metric_HRV in metric_HRV_list_nathalie:

                    _metric = f"{cond}_{odor}_{metric_HRV}"
                    _val = df_hrv.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}'")[metric_HRV].values[0]

                    _df_hrv_clf[_metric] = [_val]

        df_hrv_clf_nathalie = pd.concat([df_hrv_clf_nathalie, _df_hrv_clf], axis=0)
    
        #### RESPI
    metric_respi_list = ['inspi_duration', 'expi_duration', 'cycle_freq', 'inspi_volume', 'expi_volume', 'total_amplitude',
                          'inspi_amplitude', 'expi_amplitude']
    
    df_respi_clf = pd.DataFrame()

    for sujet in sujet_list:

        _df_respi_clf = pd.DataFrame()
        _df_respi_clf['sujet'] = [sujet]

        for cond in cond_sel:
        
            for odor in odor_sel:

                for metric_respi in metric_respi_list:

                    _metric = f"{cond}_{odor}_{metric_respi}"
                    _val = df_respi.query(f"sujet == '{sujet}' and cond == '{cond}' and odor == '{odor}'")[metric_respi].values[0]

                    _df_respi_clf[_metric] = [_val]

        df_respi_clf = pd.concat([df_respi_clf, _df_respi_clf], axis=0)
    
        #### STAI
    df_stai_clf = pd.DataFrame()

    for sujet in sujet_list:

        _df_stai_clf = pd.DataFrame()
        _df_stai_clf['sujet'] = [sujet]

        sujet_rev = f"{sujet[2:]}{sujet[:2]}"

        for cond in cond_sel:
        
            for odor in odor_sel:

                _metric = f"{cond}_{odor}_stai"
                _val = df_stai.query(f"sujet == '{sujet_rev}' and cond == '{cond}' and session == '{odor}'")['val'].values[0]

                _df_stai_clf[_metric] = [_val]

        df_stai_clf = pd.concat([df_stai_clf, _df_stai_clf], axis=0)
    
        #### ODOR
    odor_question_list = ['appreciation', 'eveil', 'familiarite', 'intensite', 'evocation']

    df_odor_clf = pd.DataFrame()

    for sujet in sujet_list:

        _df_odor_clf = pd.DataFrame()
        _df_odor_clf['sujet'] = [sujet]

        sujet_rev = f"{sujet[2:]}{sujet[:2]}"

        for metric_odor in odor_question_list:

            for odor_protocol in ['+', '-']:

                _val = df_odor.query(f"sujet == '{sujet_rev}' and question == '{metric_odor}' and odor == '{odor_protocol}'")['value'].values[0]

                _df_odor_clf[f"{metric_odor}_{odor_protocol}"] = [_val]

        df_odor_clf = pd.concat([df_odor_clf, _df_odor_clf], axis=0)

        #### tot
    df_clf = pd.concat([df_Cxy_clf, df_Pxx_clf.iloc[:,1:], df_hrv_clf.iloc[:,1:], df_respi_clf.iloc[:,1:], df_stai_clf.iloc[:,1:], df_odor_clf.iloc[:,1:]], axis=1)
    df_clf = pd.concat([df_Cxy_clf_region, df_Pxx_clf_region.iloc[:,1:], df_hrv_clf.iloc[:,1:], df_respi_clf.iloc[:,1:], df_stai_clf.iloc[:,1:], df_odor_clf.iloc[:,1:]], axis=1)

    df_clf = pd.concat([df_Pxx_clf_region, df_hrv_clf_nathalie.iloc[:,1:], df_respi_clf.iloc[:,1:], df_stai_clf.iloc[:,1:], df_odor_clf.iloc[:,1:]], axis=1)

    #### pred alldata
    X_raw_col_name = df_clf.iloc[:,1:].columns
    X_raw = df_clf.iloc[:,1:].values
    y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

    scaler = StandardScaler()
    scaler.fit(X_raw)
    X = scaler.transform(X_raw)

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, random_state=42)
    clf.fit(X, y)

    scores = cross_val_score(clf, X, y, cv=4, scoring='accuracy')

    best_score = 0
    best_tree = None

    for est in clf.estimators_:
        y_pred = est.predict(X)
        score = accuracy_score(y, y_pred)
        if score > best_score:
            best_score = score
            best_tree = est

    fig, ax = plt.subplots(figsize=(20, 10))
    sklearn.tree.plot_tree(best_tree, 
            feature_names=X_raw_col_name, 
            class_names=['0', '1'],
            filled=True, rounded=True,
            max_depth=10, fontsize=10)
    plt.show()

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(X_raw_col_name)[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    #### Cxy

    chan_list_sel = ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2']
    X = (xr_Cxy.loc[:, 'FR_CV_1', '+', :] - xr_Cxy.loc[:, 'FR_CV_1', 'o', :]).values
    y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    #### Pxx

    chan_list_sel = ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2']
    band = 'gamma'
    phase = 'inspi'

    X = (xr_Pxx.loc[:, band, phase, 'FR_CV_1', '+', :] - xr_Pxx.loc[:, band, phase, 'FR_CV_1', 'o', :]).values
    y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    scaler = StandardScaler()

    df_res_clf = pd.DataFrame({'metric' : [], 'model' : [], 'band' : [], 'phase' : [], 'accuracy' : [], 'accuracy_cv' : []})

    for metric in ['VS+_-_VSo', 'VSo', 'CO2o_-_VSo', 'VS+_VSo']:

        for band in freq_band_dict_lmm:

            for phase in phase_list_lmm:

                for model in ['SVM', 'logreg', 'rf']:

                    if metric == 'VS+_-_VSo':
                        X_raw = (xr_Pxx.loc[:, band, phase, 'FR_CV_1', '+', :] - xr_Pxx.loc[:, band, phase, 'FR_CV_1', 'o', :]).values
                    elif metric == 'VSo':
                        X_raw = xr_Pxx.loc[:, band, phase, 'FR_CV_1', 'o', :].values
                    elif metric == 'CO2o_-_VSo':
                        X_raw = (xr_Pxx.loc[:, band, phase, 'CO2', 'o', :] - xr_Pxx.loc[:, band, phase, 'FR_CV_1', 'o', :]).values
                    elif metric == 'VS+_VSo':
                        X_raw = np.hstack((xr_Pxx.loc[:, band, phase, 'FR_CV_1', '+', :].values, xr_Pxx.loc[:, band, phase, 'FR_CV_1', 'o', :].values))

                    scaler.fit(X_raw)

                    X = scaler.transform(X_raw)
    
                    y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

                    if model == 'SVM':
                        clf = svm.SVC()
                    elif model == 'rf':
                        clf = RandomForestClassifier()
                    else:
                        clf = LogisticRegression()

                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)

                    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

                    _df_res_clf = pd.DataFrame({'metric' : [metric], 'model' : [model], 'band' : [band], 'phase' : [phase], 'accuracy' : [accuracy_score(y_test, y_pred)], 'accuracy_cv' : [scores.mean()]})
                    df_res_clf = pd.concat([df_res_clf, _df_res_clf], axis=0)

    model = 'SVM'
    df_res_clf.query(f"model == '{model}'")

    model = 'SVM'
    g = sns.FacetGrid(df_res_clf.query(f"model == '{model}'"), col="band")
    g.map_dataframe(sns.scatterplot, x="phase", y="accuracy_cv", hue="metric")
    g.add_legend()
    plt.suptitle(model)
    plt.ylim(0,1)
    plt.show()

    model = 'logreg'
    g = sns.FacetGrid(df_res_clf.query(f"model == '{model}'"), col="band")
    g.map_dataframe(sns.scatterplot, x="phase", y="accuracy_cv", hue="metric")
    g.add_legend()
    plt.suptitle(model)
    plt.ylim(0,1)
    plt.show()

    model = 'rf'
    g = sns.FacetGrid(df_res_clf.query(f"model == '{model}'"), col="band")
    g.map_dataframe(sns.scatterplot, x="phase", y="accuracy_cv", hue="metric")
    g.add_legend()
    plt.suptitle(model)
    plt.ylim(0,1)
    plt.show()

    df_res_clf = pd.DataFrame({'metric' : [], 'model' : [], 'band' : [], 'accuracy' : [], 'accuracy_cv' : []})

    for metric in ['VS+_-_VSo_IE', 'VSo_IE']:

        for band in freq_band_dict_lmm:

            for model in ['SVM', 'logreg', 'rf']:

                if metric == 'VS+_-_VSo_IE':
                    X_raw = np.hstack(((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', :] - xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', :]).values, 
                                        (xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', :] - xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', :]).values))
                elif metric == 'VS+_VSo_IE':
                    X_raw = np.hstack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', :].values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', :].values))

                scaler.fit(X_raw)

                X = scaler.transform(X_raw)

                y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

                if model == 'SVM':
                    clf = svm.SVC()
                elif model == 'rf':
                    clf = RandomForestClassifier()
                else:
                    clf = LogisticRegression()

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

                _df_res_clf = pd.DataFrame({'metric' : [metric], 'model' : [model], 'band' : [band], 'accuracy' : [accuracy_score(y_test, y_pred)], 'accuracy_cv' : [scores.mean()]})
                df_res_clf = pd.concat([df_res_clf, _df_res_clf], axis=0)


    g = sns.FacetGrid(df_res_clf, col="band")
    g.map_dataframe(sns.scatterplot, x="model", y="accuracy_cv", hue="metric")
    g.add_legend()
    plt.ylim(0,1)
    plt.show()

    chan_list_sel_clf = ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2']

    df_res_clf = pd.DataFrame({'metric' : [], 'model' : [], 'band' : [], 'accuracy' : [], 'accuracy_cv' : []})

    for metric in ['VS+_-_VSo_IE', 'VSo_IE', 'VS+_VSo_IE']:

        for band in freq_band_dict_lmm:

            for model in ['SVM', 'logreg', 'rf']:

                if metric == 'VS+_-_VSo_IE':
                    X_raw = np.stack(((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values, 
                                        (xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values), axis=1)
                elif metric == 'VS+_VSo_IE':
                    X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, 
                                        xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)
                elif metric == 'VSo_IE':
                    X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)

                scaler.fit(X_raw)

                X = scaler.transform(X_raw)

                y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

                if model == 'SVM':
                    clf = svm.SVC()
                elif model == 'rf':
                    clf = RandomForestClassifier()
                else:
                    clf = LogisticRegression()

                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                scores = cross_val_score(clf, X, y, cv=4, scoring='accuracy')

                _df_res_clf = pd.DataFrame({'metric' : [metric], 'model' : [model], 'band' : [band], 'accuracy' : [accuracy_score(y_test, y_pred)], 'accuracy_cv' : [scores.mean()]})
                df_res_clf = pd.concat([df_res_clf, _df_res_clf], axis=0)


    g = sns.FacetGrid(df_res_clf, col="band")
    g.map_dataframe(sns.scatterplot, x="model", y="accuracy_cv", hue="metric")
    g.add_legend()
    plt.ylim(0,1)
    plt.show()

    chan_list_sel_clf = ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2']

    df_res_clf = pd.DataFrame({'metric' : [], 'band' : [], 'accuracy' : [], 'accuracy_cv' : []})

    for metric in ['VS+_-_VSo_IE', 'VSo_IE', 'VS+_VSo_IE']:

        for band in freq_band_dict_lmm:

            if metric == 'VS+_-_VSo_IE':
                X_raw = np.stack(((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values, 
                                    (xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values), axis=1)
            elif metric == 'VS+_VSo_IE':
                X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, 
                                    xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)
            elif metric == 'VSo_IE':
                X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)

            pipeline = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.SVC())

            param_grid = {
                'svc__C': [0.1, 1, 10],
                'svc__kernel': ['linear', 'rbf'],
                'svc__gamma': ['scale', 'auto']
            }

            grid = sklearn.model_selection.GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                refit='accuracy',
                cv=4,
                return_train_score=True,
            )

            y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

            grid.fit(X_raw, y)

            scaler.fit(X_raw)

            X = scaler.transform(X_raw)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

            clf = svm.SVC(C=grid.best_params_['svc__C'], kernel=grid.best_params_['svc__kernel'], gamma=grid.best_params_['svc__gamma'])

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            scores = cross_val_score(clf, X, y, cv=4, scoring='accuracy')

            _df_res_clf = pd.DataFrame({'metric' : [metric], 'band' : [band], 'accuracy' : [accuracy_score(y_test, y_pred)], 'accuracy_cv' : [scores.mean()]})
            df_res_clf = pd.concat([df_res_clf, _df_res_clf], axis=0)

    sns.scatterplot(df_res_clf, x="band", y="accuracy_cv", hue="metric")
    plt.ylim(0,1)
    plt.show()

    chan_list_sel_clf = ['Fp1', 'Fz', 'F3', 'F7', 'F4', 'F8', 'Fp2']

    df_res_clf = pd.DataFrame({'metric' : [], 'band' : [], 'accuracy' : [], 'accuracy_cv' : []})

    for metric in ['VS+_-_VSo_IE', 'VSo_IE', 'VS+_VSo_IE']:

        if metric == 'VS+_-_VSo_IE':
            X_raw = np.stack(((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values, 
                                (xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf] - xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf]).median('chan').values), axis=1)
        elif metric == 'VS+_VSo_IE':
            X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, 
                                xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', '+', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)
        elif metric == 'VSo_IE':
            X_raw = np.stack((xr_Pxx.loc[:, band, 'inspi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values, xr_Pxx.loc[:, band, 'expi', 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values), axis=1)

        X_raw = xr_Pxx.loc[:, :, :, 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').values.reshape(sujet_list.size, -1)
        X_raw_col_name = xr_Pxx.loc[:, :, :, 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').stack(feature=('band', 'phase'))['feature'].values
        X_raw_col_name = [f"{b}_{p}" for b, p in X_raw_col_name]
        X_raw = xr_Pxx.loc[:, :, :, 'FR_CV_1', 'o', chan_list_sel_clf].median('chan').stack(feature=('band', 'phase')).values

        y = [1 if sujet in sujet_best_list_rev else 0 for sujet in sujet_list]

        scaler.fit(X_raw)

        X = scaler.transform(X_raw)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        clf.fit(X, y)

        from sklearn.metrics import accuracy_score

        best_score = 0
        best_tree = None

        for est in clf.estimators_:
            y_pred = est.predict(X)  # or X_train if splitting
            score = accuracy_score(y, y_pred)
            if score > best_score:
                best_score = score
                best_tree = est


        fig, ax = plt.subplots(figsize=(20, 10))
        sklearn.tree.plot_tree(best_tree, 
                feature_names=X_raw_col_name, 
                class_names=['0', '1'],
                filled=True, rounded=True,
                max_depth=10, fontsize=10)

        plt.show()

        importances = clf.feature_importances_  # array of shape (n_features,)
        indices = np.argsort(importances)[::-1]

        # Optional: plot
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), np.array(X_raw_col_name)[indices], rotation=90)
        plt.tight_layout()
        plt.show()

        

        y_pred = clf.predict(X_test)

        scores = cross_val_score(clf, X, y, cv=4, scoring='accuracy')

        _df_res_clf = pd.DataFrame({'metric' : [metric], 'band' : [band], 'accuracy' : [accuracy_score(y_test, y_pred)], 'accuracy_cv' : [scores.mean()]})
        df_res_clf = pd.concat([df_res_clf, _df_res_clf], axis=0)

    sns.scatterplot(df_res_clf, x="band", y="accuracy_cv", hue="metric")
    plt.ylim(0,1)
    plt.show()







################################
######## EXECUTE ########
################################

if __name__ == '__main__':

    ######## MAIN WORKFLOW ########
    search_pred()





    ######## LOAD ########
    stretch=False
    stretch=True

    df_PRED = get_df_PRED(stretch=stretch)
    df_PRED_TF, feature_list_RS, feature_list_whole = get_df_PRED_TF()
    
    question_list, df_odor_profile = get_odor_df()
    df_odor_profile = pd.pivot_table(df_odor_profile, values='value', index='sujet', columns='question').reset_index(drop=False)
    order_i = np.argsort(np.array([int(f"{sujet[2:]}") for sujet in df_odor_profile['sujet'].values]))
    df_odor_profile = df_odor_profile.iloc[order_i,:]

    ######## FEATURES SELECTION ########
    features_list = ['C3-C4', 'C3-Cz', 'C3-FC1', 'C3-FC2', 'C4-FC1', 'C4-FC2', 'Cz-C4', 'Cz-FC1', 'Cz-FC2', 'FC1-FC2', 'C3', 'Cz', 'C4', 'FC1', 'FC2']
    features_list = ['C3', 'Cz', 'C4', 'FC1', 'FC2']
    question_list = ['appreciation', 'eveil', 'evocation', 'familiarite', 'intensite']

    features_list_whole = []

    for phase in ['inspi', 'expi']:

        for odor in ['o', '+']:

            features_list_whole.extend([f"{col_name}_{odor}_{phase}" for col_name in features_list])

    features_list_whole.extend(question_list)

    feature_list_TF_dict = {}
    feature_list_RS_TF_dict = {}
    feature_list_RS_odor_TF_dict = {}

    for band, freq in freq_band_dict['wb'].items():

        feature_list_TF_dict[band] = [feature for feature in feature_list_whole if feature.find(band) != -1]
        feature_list_RS_TF_dict[band] = [feature for feature in feature_list_whole if feature.find(band) != -1 and feature.find('FR_CV_1') != -1 and feature.find('o') != -1]
        feature_list_RS_odor_TF_dict[band] = [feature for feature in feature_list_whole if feature.find(band) != -1 and feature.find('FR_CV_1') != -1 and feature.find('-') == -1]

    ######## DF ADJUST ########
    df_PRED_RS = {}
    df_PRED_RS_odor = {}
    for phase in ['inspi', 'expi']:
        df_PRED_RS[phase] = df_PRED.query(f"cond == 'FR_CV_1' and odor == 'o' and phase == '{phase}'")
        data_PRED_RS_odor = df_PRED.query(f"cond == 'FR_CV_1' and odor == '+' and phase == '{phase}'")[features_list].values - df_PRED.query(f"cond == 'FR_CV_1' and odor == 'o' and phase == '{phase}'")[features_list].values
        df_PRED_RS_odor[phase] = pd.DataFrame(data=data_PRED_RS_odor, columns=[features_list])
        df_PRED_RS_odor[phase]['rep_state'] = df_PRED_RS[phase]['rep_state'].values

    df_PRED_whole_data = pd.DataFrame()
    for phase in ['inspi', 'expi']:
        for odor in ['o', '+']:
            _df_PRED_data = df_PRED.query(f"cond == 'FR_CV_1' and odor == '{odor}' and phase == '{phase}'")[features_list].values
            _features_list = [f"{col_name}_{odor}_{phase}" for col_name in features_list]
            _df_PRED_add = pd.DataFrame(data=_df_PRED_data, columns=_features_list)
            df_PRED_whole_data = pd.concat([df_PRED_whole_data, _df_PRED_add], axis=1)
    df_PRED_whole_data['rep_state'] = df_PRED_RS[phase]['rep_state'].values

    df_PRED_whole_data = pd.concat([df_PRED_whole_data, df_odor_profile[question_list]], axis=1)

    ######## SELECT DF PRED ########
    df_classification = df_PRED_RS['inspi']
    df_classification = df_PRED_RS_odor['inspi']
    df_classification = df_PRED_RS['expi']
    df_classification = df_PRED_RS_odor['expi']

    df_classification = df_PRED_whole_data
    df_classification = df_PRED_TF

    X = df_classification[features_list] # Features
    X = df_classification[features_list_whole] # Features
    X = df_classification[feature_list_RS] # Features
    X = df_classification[feature_list_whole] # Features
    X = df_classification[feature_list_TF_dict['theta']] # Features
    X = df_classification[feature_list_RS_TF_dict['l_gamma']] # Features
    X = df_classification[feature_list_RS_odor_TF_dict['h_gamma']] # Features

    y = df_classification['rep_state'] # Target

    df_res_all_model = pd.DataFrame()

    ######## SPLIT ########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ######## SVM ########
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 0.01]}
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
    grid.fit(X_train_scaled, y_train)

    svc = SVC(kernel='rbf', random_state=42, C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], class_weight='balanced')

    svc.fit(X_train_scaled, y_train)

    y_pred = svc.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["False", "True"], zero_division=0))

    df_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=["False", "True"], zero_division=0, output_dict=True))
    _df_res = pd.DataFrame({'model' : ['SVM', 'SVM'], 'cat' : ['True', 'False'], 'f1_score' : [df_report.loc['f1-score', 'True'], df_report.loc['f1-score', 'False']], 'accuracy' : [df_report.loc['precision', 'accuracy'], df_report.loc['precision', 'accuracy']]})
    df_res_all_model = pd.concat([df_res_all_model, _df_res])
    
    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        df_report.to_excel(f"stretch_SVM.xlsx")
    else:
        df_report.to_excel(f"nostretch_SVM.xlsx")

    # Print predicted vs actual values
    print("Predicted:", y_pred)
    print("Actual:", y_test.values)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[0]):
            c = conf_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.title(f"Confusion Matrix:SVM stretch:{stretch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        plt.savefig(f"stretch_SVM.png")
    else:
        plt.savefig(f"nostretch_SVM.png")

    plt.show()

    ######## RANDOM FOREST ########

    # Initialize and train the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    df_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=["False", "True"], zero_division=0, output_dict=True))

    _df_res = pd.DataFrame({'model' : ['RF', 'RF'], 'cat' : ['True', 'False'], 'f1_score' : [df_report.loc['f1-score', 'True'], df_report.loc['f1-score', 'False']], 'accuracy' : [df_report.loc['precision', 'accuracy'], df_report.loc['precision', 'accuracy']]})
    df_res_all_model = pd.concat([df_res_all_model, _df_res])
    
    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        df_report.to_excel(f"stretch_RF.xlsx")
    else:
        df_report.to_excel(f"nostretch_RF.xlsx")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[0]):
            c = conf_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.title(f"Confusion Matrix:RF stretch:{stretch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        plt.savefig(f"stretch_RF.png")
    else:
        plt.savefig(f"nostretch_RF.png")

    plt.show()

    # ROC Curve and AUC Score
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Feature Importance
    importances = rf.feature_importances_
    feature_names = X.columns

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

    # Plot Feature Importance
    order_i = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[order_i], y=feature_names[order_i], palette="viridis")
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    # Print feature importance
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importance:")
    print(importance_df)

    ######## LINEAR ########

    # Initialize Logistic Regression model
    logreg = LogisticRegression()

    # Train the model
    logreg.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = logreg.predict(X_test_scaled)
    y_pred_proba = logreg.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    df_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=["False", "True"], zero_division=0, output_dict=True))

    _df_res = pd.DataFrame({'model' : ['LOGREG', 'LOGREG'], 'cat' : ['True', 'False'], 'f1_score' : [df_report.loc['f1-score', 'True'], df_report.loc['f1-score', 'False']], 'accuracy' : [df_report.loc['precision', 'accuracy'], df_report.loc['precision', 'accuracy']]})
    df_res_all_model = pd.concat([df_res_all_model, _df_res])
    
    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        df_report.to_excel(f"stretch_LOGREG.xlsx")
    else:
        df_report.to_excel(f"nostretch_LOGREG.xlsx")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[0]):
            c = conf_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.title(f"Confusion Matrix:LOGREG stretch:{stretch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.chdir(os.path.join(path_results, 'allplot', 'PRED'))
    if stretch:
        plt.savefig(f"stretch_LOGREG.png")
    else:
        plt.savefig(f"nostretch_LOGREG.png")

    plt.show()

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # Cross-Validation
    # cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
    # print("Cross-Validation Scores:", cv_scores)
    # print("Mean Cross-Validation Accuracy:", np.mean(cv_scores))

    #### COMPARE ALL MODELS
    sns.catplot(data=df_res_all_model, kind='bar', x='model', y='f1_score', hue='cat')
    plt.ylim(0,1)
    plt.show()

    sns.catplot(data=df_res_all_model, kind='bar', x='model', y='accuracy', hue='cat')
    plt.ylim(0,1)
    plt.show()