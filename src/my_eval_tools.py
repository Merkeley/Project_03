'''
    File: my_eval_tools.py

    Description: This file contains several functions that can be used to 
        in the evaluation of machine learning models.

    Author: MBoals

    Date Created: 2/10/20

    Date Modified:

    Update Log:

    Functions Defined:

'''
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

def calc_roc_data(model, X_dat, y_dat, n_step) :
    threshold_list = np.linspace(0, 1, n_step, endpoint=True)
    roc_data = [[0,0,0]]

    for i, threshold in enumerate(threshold_list) :
        y_predict = (model.predict_proba(X_dat)[:, 1] >= threshold)
        confusion_m = confusion_matrix(y_dat, y_predict)
        fpr = confusion_m[1][0] / (confusion_m[1][0]+confusion_m[1][1])
        tpr = confusion_m[0][0] / (confusion_m[0][0]+confusion_m[0][1])
        roc_data.append([threshold, fpr, tpr])

    roc_data.append([1,1,1])

    return roc_data


def calc_hybrid_roc_data(model1, model2, X_dat, y_dat, n_step) :
    threshold_list = np.linspace(0, 1, n_step, endpoint=True)
    roc_data = [[0,0,0]]

    proba_array = hybrid_predict_proba(model1, model2, X_dat) 

    for i, threshold in enumerate(threshold_list) :
        y_predict = (proba_array[:, 1] >= threshold)
        confusion_m = confusion_matrix(y_dat, y_predict)
        fpr = confusion_m[1][0] / (confusion_m[1][0]+confusion_m[1][1])
        tpr = confusion_m[0][0] / (confusion_m[0][0]+confusion_m[0][1])
        roc_data.append([threshold, fpr, tpr])

    roc_data.append([1,1,1])

    return roc_data

def hybrid_predict_proba(model1, model2, X_dat) :
    proba_array = np.empty([X_dat.shape[0], 2])

    for i, index in enumerate(X_dat.index) :
        sample = np.array(X_dat.iloc[i, 1:]).reshape(1,-1)

        if X_dat.iloc[i,0] == 1 : # Month to month contract
            sample_proba = model1.predict_proba(sample)
        else:
            sample_proba = model2.predict_proba(sample)

        proba_array[i] = sample_proba

    return proba_array


def calc_pr_sweep(predictor, in_data, actual, n_steps) :

    results = predict_sweep(predictor, in_data, actual, n_steps)

    precision = []
    recall = []
    thresh_list = []

    for trial in results :
        threshold = trial[0]
        tn = trial[1]
        fp = trial[2]
        fn = trial[3]
        tp = trial[4]

        if tp+fp > 0 :
            precision.append(tp / (tp+fp))
        else :
            precision.append(1)

        if tp+fn > 0 :
            recall.append(tp/(tp+fn))
        else :
            recall.append(1)

        thresh_list.append(threshold)

    return precision, recall, thresh_list


def predict_sweep(predictor, in_data, target, n_steps) :
    # Make sure we have a valid number of steps
    if n_steps == 0 :
        return None

    # Set the initial conditions for the sweep
    threshold = 0;
    threshold_step = 1 / n_steps

    result_list = []
    for i in range(n_steps+1) :
        y_predict = [1 if predictor(in_data)[x][1] >= threshold else 0 for x in range(in_data.shape[0])]
        tn, fp, fn, tp = confusion_matrix(target, y_predict).ravel()
        result_list.append([threshold, tn, fp, fn, tp])
        threshold += threshold_step

    return result_list


def hybrid_predict(model1, model2, X_dat) :
    y_hat = np.empty(X_dat.shape[0])

    for i, index in enumerate(X_dat.index) :
        sample = np.array(X_dat.iloc[i, 1:]).reshape(1,-1)

        if X_dat.iloc[i,0] == 1 : # Month to month contract
            predict = model1.predict(sample)
        else:
            predict = model2.predict(sample)

        y_hat[i] = predict

    return y_hat
