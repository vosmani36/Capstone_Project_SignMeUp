# import packages
print('Loading all the packages...')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import Rocket
from sktime.transformations.panel.rocket import MiniRocket
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report
import pickle


#loading our preprocessed datasets
print('Loading the preprocessed data...')
X_train = np.load('/home/kirafriedrichs/X_train.npy')
y_train = np.load('/home/kirafriedrichs/y_train.npy')

X_test = np.load('/home/kirafriedrichs/X_test.npy')
y_test = np.load('/home/kirafriedrichs/y_test.npy')
print('Import of datasets was successful.')

print('Shape of train dataset:')
print(X_train.shape, y_train.shape)

print('Shape of test dataset:')
print(X_test.shape, y_test.shape)


#Set up GridSearch
print('Starting GridSearch...')
param_grid = {
    'rocket__num_kernels': [1000, 5000, 10000],
    'clf__C': [0.01, 0.1, 1, 10]
}


#apply MiniROCKET and logistic regression with GridSearch
print('Apply Rocket and Logistic Regression with GridSearch...')
rocket = Rocket(random_state=42)
clf = LogisticRegression(solver='sag', random_state=42)
pipeline = Pipeline([('rocket', rocket), ('clf', clf)])
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


#print best parameters
print('Best parameters:', grid.best_params_)


#print test accuracy
print('predicting y_test with X_test and calculating the accuracy.')
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)


#print classification report
print(classification_report(y_test, y_pred))



'''
# Define the MiniROCKET model
from sktime.transformations.panel.rocket import MiniRocket
model = MiniRocket(num_kernels=5000, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Extract features
X_train_features = model.transform(X_train)

# Train the classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C = 0.1, random_state=42)
clf.fit(X_train_features, y_train)

# Evaluate the model
X_test_features = model.transform(X_test)
y_pred = clf.predict(X_test_features)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)
'''


# save the model to disk
model_name = 'minirocket_gridsearch.pkl'
pickle.dump(clf, open(model_name, 'wb'))






