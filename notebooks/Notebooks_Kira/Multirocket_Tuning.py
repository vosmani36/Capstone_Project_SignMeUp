# import packages
print('Loading all the packages...')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.rocket import MiniRocket
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report
import pickle


#loading our preprocessed datasets
print('Loading the preprocessed data...')
X_train = np.load('/home/kirafriedrichs/X_train_fewlips.npy')
y_train = np.load('/home/kirafriedrichs/y_train_fewlips.npy')

X_test = np.load('/home/kirafriedrichs/X_test_fewlips.npy')
y_test = np.load('/home/kirafriedrichs/y_test_fewlips.npy')
print('Import of datasets was successful.')

print('Shape of train dataset:')
print(X_train.shape, y_train.shape)

print('Shape of test dataset:')
print(X_test.shape, y_test.shape)


#Set up Grid for RandomSearch
print('Setting up parameters to be random searched...')
param_grid = {
    'multirocket__num_kernels': [1000, 5000, 10000],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__solver': ['sag', 'saga']
}


#apply MultiROCKET and Logistic Regression with RandomSearch
print('Apply MultiRocket and Logistic Regression with RandomSearch...')
print('define Multirocket transformer...')
multirocket = MiniRocketMultivariate(random_state=42, n_jobs=-1)
print('Define Logistic Regression Classifier...')
clf = LogisticRegression(random_state=42, n_jobs=-1)                               #max_iter=100 by default, maybe set higher
print('Setting up Pipeline for Random Search...')
pipeline = Pipeline([('multirocket', multirocket), ('clf', clf)])
print('Finding best parameters for multirocket transformer and logistic regression classifier and set up model...')
model = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', random_state=42, error_score='raise')

#fit model with best params
print('Fit the model...')
model.fit(X_train, y_train)

#print best parameters
print('Best estimator:', model.best_estimator_)
print('Best score:', model.best_score_)
print('Best parameters:', model.best_params_)

#print test accuracy
print('Predicting y_test with X_test and calculating the accuracy...')
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)


#print classification report
print(classification_report(y_test, y_pred))


# save the model to disk
model_name = 'multirocket_randomsearch.pkl'
pickle.dump(clf, open(model_name, 'wb'))






