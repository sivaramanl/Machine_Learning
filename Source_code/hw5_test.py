# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:33:07 2018

@author: Sivaraman Lakshmipathy
"""

import pandas as pd
import numpy as np
import pickle
import sys

def read_from_file(fileName):
    print("\nReading input from file")
    data = pd.read_csv(fileName, low_memory=False)
    return data

def getFeaturesTarget(data_fin, target_col):
    #Separating the target variable and features
    data_fin_copy = data_fin.copy()
    y = data_fin_copy[target_col]
    data_fin_copy.drop([target_col], axis = 1, inplace = True)
    X = data_fin_copy
    return X,y

def main():
    fileName = "test.csv"
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    model_fileName = "predict_smoking_model.sav"
    #Step 1 - Import data
    data = read_from_file(fileName)
    
    #Target column = Smoking
    target_col = "Smoking"
    X,y = getFeaturesTarget(data, target_col)
    
    #Dimension reduction
    dimension_reduction_model = pickle.load(open("predict_smoking_feature_reduction_model.sav", 'rb'))
    X_reduced = dimension_reduction_model.transform(X)
    
    trained_model = pickle.load(open(model_fileName, 'rb'))
    y_pred = trained_model.predict(X_reduced)
    print("Accuracy", np.mean(y==y_pred))
    
if __name__ == '__main__':
    main()