# general 
import numpy as np
import pandas as pd
#%pip install sktime

import pickle

# for data pre-processing
from sklearn.model_selection import train_test_split
from sktime.classification.kernel_based import RocketClassifier

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report

#set RSEED 
RSEED = 42

#loading our preprocessed datasets
X = np.load('/home/kirafriedrichs/X_sklearnsplit.npy')
y = np.load('/home/kirafriedrichs/y_sklearnsplit.npy')

#train-test-split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RSEED)

print('Import of datasets was successful.')
print('Shape of train dataset:')
print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

clf = RocketClassifier(use_multivariate="yes", num_kernels=5000, random_state=42, n_jobs=-1) 
print('Starting training. Rocket is ready for takeoff..')
clf.fit(X_train, y_train) 
print('Training was successful.')
print('Running the prediction of the test set...')
y_pred = clf.predict(X_test)

# accuracy
print('The accuracy score is:')
print(accuracy_score(y_test, y_pred))
#print('Classification Report:')
#print(classification_report(y_test, y_pred))
# confusion matrix
#print('Confusion matrix:')
#print(confusion_matrix(y_test, y_pred))
#multilabel confusion matrix
#multilabel_confusion_matrix(y_test, y_pred)

# save the model to disk
model_name = 'rocket_sklearnsplit.pkl'
pickle.dump(clf, open(model_name, 'wb'))

#to load model use:
#pickled_model = pickle.load(open(model_name, 'rb'))


