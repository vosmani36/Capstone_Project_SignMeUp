# load packages 
#%pip install tensorflow
# general 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# for model building and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import TensorBoard

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report

# set random seed
RSEED = 42

#loading our preprocessed datasets
X = np.load('/home/kirafriedrichs/X_sklearnsplit_z.npy')
y = np.load('/home/kirafriedrichs/y_sklearnsplit_z.npy')

#train-test-split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RSEED)

print('Import of datasets was successful.')
#print('Shape of train dataset:')
#print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

#print('Encoding target labels...')
#y_train = pd.get_dummies(y_train)
#print('Shape of train dataset:')
#print(y_train.shape)

model = tf.keras.models.load_model('./LSTM_model_sklearnsplit_z.h5')

# show summary of model
model.summary()

# prediction on test data
print('Running predictions on test set...')
y_pred = model.predict(X_test)
y_pred = (pd.DataFrame(y_pred)).idxmax(axis=1)

# accuracy score
print(f'Accuracy score on test dataset is: {round(accuracy_score(y_test, y_pred),3)}.')
print(classification_report(y_test, y_pred))

# assume y_test and y_pred are the ground truth and predicted labels, respectively
report = classification_report(y_test, y_pred, output_dict=True)

# extract the F1 scores for each class
f1_scores = {}
for class_label, metrics in report.items():
    if class_label != 'accuracy' and class_label != 'macro avg' and class_label != 'weighted avg':
        f1_scores[class_label] = metrics['f1-score']

# sort the F1 scores in ascending order and select the top 20 classes (aka the 20 worst classes)
worst_classes = sorted(f1_scores, key=f1_scores.get, reverse=False)[:20]
print(f'Worst 20 sings are: {worst_classes}.')

best_classes = sorted(f1_scores, key=f1_scores.get, reverse=True)[:20]
print(f'Best 20 sings are: {best_classes}.')

good_classes = sorted(f1_scores, key=f1_scores.get, reverse=False)[20:250]
print(f'Good 230 sings are: {good_classes}.')

print(f'The list with the worst signs counts {len(worst_classes)} signs, the list with the good signs counts {len(good_classes)} signs.')

#save model
#print(f'Saving model as {model_name}.h5')
#model.save(model_name+'.h5')
list_common = list(set(worst_classes).intersection(good_classes))
print(f'To be sure: worst and good sign lists have in common: {list_common}.')


print('Done.')