
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





################################
######## EXECUTE ########
################################

if __name__ == '__main__':

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