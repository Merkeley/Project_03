'''
    File: telecom_dat_clean.py

    Description: This file contains a single function that is used to clean the
        data for Metis Data Science Bootcamp Project 3

    Author: MBoals

    Date Created: 1/27/20

    Date Modified: 

    Update Log:

    Functions Defined:

'''
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_dat(in_frame) :
    # Convert all the charges from trings to floats
    total_charges = in_frame['TotalCharges']
    f_charges = []
    for charges in total_charges :
        try:
            f_charges.append(float(charges))
        except:
            f_charges.append(0.0)

    in_frame['TotalCharges'] = f_charges

    in_frame = in_frame.replace('Yes', 1)
    in_frame = in_frame.replace('No', 0)

    in_frame = in_frame.replace('No internet service', 0)
    in_frame = in_frame.replace('No phone service', 0)

    contract_dummies = pd.get_dummies(in_frame['Contract'])
    internet_dummies = pd.get_dummies(in_frame['InternetService'])
    gender_dummies = pd.get_dummies(in_frame['gender'])

    in_frame[contract_dummies.columns] = contract_dummies
    in_frame[internet_dummies.columns] = internet_dummies
    in_frame[gender_dummies.columns] = gender_dummies

    in_frame.drop('Two year', axis=1, inplace=True)
    in_frame.drop(0, axis=1, inplace=True)
    in_frame.drop('Male', axis=1, inplace=True)

    in_frame.drop(['Contract', 'gender', 'InternetService'], axis=1, inplace=True)

    std = StandardScaler()
    cols_to_scale = ['MonthlyCharges', 'TotalCharges']
    in_frame[cols_to_scale] = std.fit_transform(in_frame[cols_to_scale])

    return(in_frame)
