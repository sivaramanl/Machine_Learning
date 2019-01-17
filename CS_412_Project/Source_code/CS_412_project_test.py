# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:33:07 2018

@author: Sivaraman Lakshmipathy
"""

import pandas as pd
import numpy as np
import pickle
import sys

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import model_selection, metrics

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
    print("\nPerforming dimensionality reduction using the saved feature reduction model.")
    dimension_reduction_model = pickle.load(open("predict_smoking_feature_reduction_model.sav", 'rb'))
    X_reduced = dimension_reduction_model.transform(X)
    
    print("\nLoading saved model.")
    trained_model = pickle.load(open(model_fileName, 'rb'))
    y_pred = trained_model.predict(X_reduced)
    print("Accuracy", np.mean(y==y_pred))
    
    print("\nCalculating the precision, recall and F1 scores for the classifier.")
    preds = model_selection.cross_val_predict(trained_model, X_reduced, y, cv=10)
    accScore = metrics.accuracy_score(y,preds)
    labels = ["former smoker", "never smoked", "tried smoking", "current smoker"]
    precision = metrics.precision_score(y,preds,average=None,labels=labels)
    recall = metrics.recall_score(y,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
    print("\nOverall Acurracy: ",accScore,"\n")
    for i in range(len(labels)):
        print("Precision of %s class: %f" %(labels[i],precision[i]))
        print("Recall of %s class: %f" %(labels[i],recall[i]))
        print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")
    
if __name__ == '__main__':
    main()