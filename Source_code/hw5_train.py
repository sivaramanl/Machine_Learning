# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:29:05 2018

@author: Sivaraman Lakshmipathy
"""

#CS 412 - Intro to ML
#Homework Assignment 5
#Task 2 - Smoking type classification

# Import libraries
import pandas as pd
import numpy as np
import sys

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn import model_selection, metrics

def read_from_file(fileName):
    print("\nReading input from file")
    data = pd.read_csv(fileName, low_memory=False)
    return data

def separate_data(data):
    #Differentiate between numeric and categorical columns
    print("\nDifferentiating between numeric and categorical columns.")
    cols = data.columns
    num_cols_index = data._get_numeric_data().columns

    categ_cols = list(set(cols) - set(num_cols_index))
    print(len(categ_cols), " categorical columns identified.")

    num_cols = list(set(num_cols_index))
    print(len(num_cols), " numerical columns identified.")
    
    return num_cols, categ_cols

def removeEntriesWithoutTarget(data, target_col):
    print("\nRemoving entries without target variable.")
    if len(data[target_col].isnull() > 0):
        data_fin = data[data[target_col].isnull() == False]
    #print(data_fin.shape)
    return data_fin

def fillNAEntries(data):
    print("\nFilling entries with NA values with the mode values of their corresponding columns.")
    data = data.fillna(data.mode().iloc[0])
    return data

def oneHotEncode(data_fin, categ_cols):
    for entry in categ_cols:
        pp = pd.get_dummies(data_fin[entry], dummy_na = False, prefix = entry)
        data_fin = pd.concat([data_fin,pp],axis=1)
        data_fin.drop([entry],axis=1, inplace=True)
    print("Current shape of data", data_fin.shape)
    return data_fin
    
def getFeaturesTarget(data_fin, target_col):
    #Separating the target variable and features
    data_fin_copy = data_fin.copy()
    y = data_fin_copy[target_col]
    data_fin_copy.drop([target_col], axis = 1, inplace = True)
    X = data_fin_copy
    return X,y

def trainTestSplit(X, y, target_col, tripleSplit):
    if tripleSplit:
        X1, X_test, y1, y_test = train_test_split(X, y, test_size = 0.15, random_state=4)

        X_test2 = X_test.copy()
        y_test2 = y_test.copy()
        #persist test to csv
        df = pd.DataFrame(X_test2)
        df[target_col] = y_test2
        df.to_csv("test.csv", index=False)
        
        X1_2 = X1.copy()
        y1_2 = y1.copy()
        df = pd.DataFrame(X1_2)
        df[target_col] = y1_2
        df.to_csv("train.csv", index=False)

        X_train, X_dev, y_train, y_dev = train_test_split(X1, y1, test_size = 0.15, random_state=1)
        return X_train, X_dev, y_train, y_dev
    else:
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.15, random_state=4)
        return X_train, X_dev, y_train, y_dev

def decisionTreeClassifier(X_train, y_train, X_dev, y_dev):
    print("\nTraining decision tree classifier")
    tree_clf = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=10)
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_dev)
    accuracy = np.mean(y_dev == y_pred)
    print("Accuracy", accuracy)
    return tree_clf, accuracy

def logisticRegressionClassifier(X_train, y_train, X_dev, y_dev, solver='sag', multi_class_val=None):
    print("\nTraining logistic regression classifier")
    if multi_class_val == None:
        logreg = LogisticRegression(solver=solver)
    else:
        logreg = LogisticRegression(solver=solver, multi_class=multi_class_val)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_dev)
    accuracy = np.mean(y_dev == y_pred)
    print("Accuracy", accuracy)
    return logreg, accuracy

def svmClassifier(X_train, y_train, X_dev, y_dev, kernel_svm):
    print("\nTraining SVM classifier with", kernel_svm, "classifier")
    svclassifier = SVC(kernel=kernel_svm)
    svclassifier.fit(X_train, y_train) 
    y_pred = svclassifier.predict(X_dev)
    accuracy = np.mean(y_dev == y_pred)
    print("Accuracy", accuracy)
    return svclassifier, accuracy

def getReducedModel(X, y):
    print("\nReducing dimensionality by identifying top features using RandomForestClassifier")
    print("Original number of features:", X.shape[1])
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X_reduced = model.transform(X)
    print("Reduced number of features:", X_reduced.shape[1])
    return model, X_reduced

def performPCA(X_train, X_dev):
    print("\nPerforming PCA")
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train2 = pca.transform(X_train)
    X_dev2 = pca.transform(X_dev)
    return pca, X_train2, X_dev2

def gradientBoostingClassifier(X_train, y_train, X_dev, y_dev):
    print("\nPerforming Gradient Boosting.")
    gb = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.25, max_depth = 5, random_state = 0)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_dev)
    accuracy = np.mean(y_dev == y_pred)
    print("Accuracy", accuracy)
    return gb, accuracy

def performGridSearch(baseClassifier, X_train, y_train, X_dev, y_dev):
    print("\nPerforming Grid Search for Gradient boosting")
    parameters = {
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "max_depth":[3,5,8],
            "n_estimators":[10]
            }

    clf = GridSearchCV(baseClassifier, parameters, cv=10, n_jobs=-1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    accuracy = np.mean(y_dev == y_pred)
    print("Accuracy", accuracy)
    print("Best params", clf.best_params_)
    return clf, accuracy

def main():
    input_file = "responses.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    #Step 1 - Import data
    data = read_from_file(input_file)
    #Copying the original data for safety
    data_orig = data.copy()

    #Target column = Smoking
    target_col = "Smoking"
    
    num_cols, categ_cols = separate_data(data)
    
    data_fin = removeEntriesWithoutTarget(data, target_col)
    if target_col in categ_cols:
        categ_cols.remove(target_col)
        
    data_fin = fillNAEntries(data_fin)
    
    #Create a backup of preprocessed data
    data_fin_backup = data_fin.copy()
    
    data_fin = data_fin_backup.copy()
    
    print("\nOriginal shape of data", data_fin.shape)
    print("Performing one hot encoding of categorical data.")
    data_fin = oneHotEncode(data_fin, categ_cols)
    
    X,y = getFeaturesTarget(data_fin, target_col)
    
    print("\nSplitting data into train/dev/test and persisting test data in test.csv")
    X_train, X_dev, y_train, y_dev = trainTestSplit(X, y, target_col, tripleSplit = True)
    
    models = []
    models_name = []
    accuracies = []
    
    model, accuracy = decisionTreeClassifier(X_train, y_train, X_dev, y_dev)
    models.append(model)
    models_name.append("DecisionTreeClassifier")
    accuracies.append(accuracy)
    
    model, accuracy = logisticRegressionClassifier(X_train, y_train, X_dev, y_dev)
    models.append(model)
    accuracies.append(accuracy)
    
    model, accuracy = svmClassifier(X_train, y_train, X_dev, y_dev, "rbf")
    models.append(model)
    models_name.append("SVM Classifier")
    accuracies.append(accuracy)
    
    reduction_model, X = getReducedModel(X, y)
    fileName = "predict_smoking_feature_reduction_model.sav"
    pickle.dump(reduction_model, open(fileName, 'wb'))
    X_red = X.copy()
    
    X_train, X_dev, y_train, y_dev = trainTestSplit(X, y, target_col, tripleSplit = False)
    
    model, accuracy = logisticRegressionClassifier(X_train, y_train, X_dev, y_dev, "sag", "multinomial")
    models.append(model)
    models_name.append("Logistic Regression Classifier (reduced features)")
    accuracies.append(accuracy)
    
    model, accuracy = svmClassifier(X_train, y_train, X_dev, y_dev, "rbf")
    models.append(model)
    models_name.append("SVM Classifier (reduced features)")
    accuracies.append(accuracy)
    
    pca_model, X_train, X_dev = performPCA(X_train, X_dev)
    fileName = "predict_smoking_pca_model.sav"
    pickle.dump(pca_model, open(fileName, 'wb'))
    
    model, accuracy = logisticRegressionClassifier(X_train, y_train, X_dev, y_dev, "sag", "multinomial")
    models.append(model)
    models_name.append("Logistic Regression Classifier (PCA)")
    accuracies.append(accuracy)
    
    model, accuracy = svmClassifier(X_train, y_train, X_dev, y_dev, "rbf")
    models.append(model)
    models_name.append("SVM Classifier (PCA)")
    accuracies.append(accuracy)
    
    model, accuracy = gradientBoostingClassifier(X_train, y_train, X_dev, y_dev)
    models.append(model)
    models_name.append("Gradient Boosting")
    accuracies.append(accuracy)
    
    model, accuracy = performGridSearch(model, X_train, y_train, X_dev, y_dev)
    models.append(model)
    models_name.append("Grid search for Gradient Boosting.")
    accuracies.append(accuracy)
    
    print("\nResults:")
    print("Highest accuracy is achieved using Logistic regression classifier with multinomial classification on reduced feature set.")
    print("Hence, persisting the model to predict on test values.")
    
    print("Calculating the precision, recall and F1 scores for the Logistic regression classifier.")
    preds = model_selection.cross_val_predict(models[3], X_red, y, cv=10)
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
    
    fileName = "predict_smoking_model.sav"
    pickle.dump(models[3], open(fileName, 'wb'))

if __name__ == '__main__':
    main()