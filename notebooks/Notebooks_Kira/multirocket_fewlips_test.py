# Multirocket test

# import packages
print('Loading all the packages...')
import numpy as np
from sktime.classification.kernel_based import RocketClassifier
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



#Train classifier
print('Training classifier...')
multirocket = RocketClassifier(rocket_transform='multirocket', random_state=42, num_kernels=5000, n_jobs=-1)

#fit model with best params
print('Fit the multirocket...')
multirocket.fit(X_train, y_train)


#print test accuracy
print('Predicting y_test with X_test and calculating the accuracy...')
y_pred = multirocket.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)


#print classification report
print(classification_report(y_test, y_pred))


# save the model to disk
model_name = 'multirocket_5000.pkl'
pickle.dump(multirocket, open(model_name, 'wb'))