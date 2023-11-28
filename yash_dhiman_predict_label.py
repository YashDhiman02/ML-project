
import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


class prediction():

reader=pd.read_csv("training_data.csv") # my data is stored in a variable called reader
target=pd.read_csv("train_data_classlabels.csv") # classlabels of my data are stored in target variable 
target = target.iloc[:,0]
test=pd.read_csv("testing_data.csv")
test = test.drop(['Time','Amount'],axis=1)
reader = reader.drop(['Time','Amount'],axis=1)

print(test.isnull().sum())

clf=RandomForestClassifier()

pipeline = Pipeline([('scaler', MinMaxScaler()),
                    ('feature_selection', SelectKBest(chi2, k=18)),
                    ('clf',RandomForestClassifier(class_weight='balanced',criterion='log_loss', n_estimators=50, max_depth=50, max_features='log2')),
                ])


pipeline.fit(reader,target) 

X_test = test
predictions = pipeline.predict(X_test)
predictions=pd.DataFrame({"Class":predictions})
predictions.to_csv('test_Predictions.csv',index=False)