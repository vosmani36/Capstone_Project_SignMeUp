
#%pip install tensorflow
# general 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# for model building and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.callbacks import TensorBoard

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#defining directories
INPUT = '../data/'
OUTPUT = '../data/'

#loading datasets
X = np.load(f'{INPUT}default_data_HandsLips6/X_test_h6.npy')
y = np.load(f'{INPUT}default_data_HandsLips6/y_test_h6.npy')

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print('Import of datasets was successful.')
#print('Shape of train dataset:')
#print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

#print('Encoding target labels...')
#y_train = pd.get_dummies(y_train)
#print('Shape of train dataset:')
#print(y_train.shape)

# set random seed
RSEED = 42

#loading model
model = tf.keras.models.load_model('./LSTM_Model6_HandsLips/LSTM_model_6.h5')

# show summary of model
model.summary()

# prediction on test data
print('Running predictions on test set...')
#measure runtime for predictions
count_pred = X_test.shape[0]
t = time.time()

y_pred = model.predict(X_test)
y_pred = (pd.DataFrame(y_pred)).idxmax(axis=1)

elapsed_time = time.time()-t
print(f'Elapsed time for prediction of whole test set is {elapsed_time} seconds. Average time for one prediction is {elapsed_time/count_pred}seconds.')

# accuracy score
print(f'Accuracy score on test dataset is: {round(accuracy_score(y_test, y_pred),3)}.')
print(classification_report(y_test, y_pred))

#save classification report for detection of best/worst classes
report = classification_report(y_test, y_pred, output_dict=True)

# extract the F1 scores for each class
f1_scores = {}
for class_label, metrics in report.items():
    if class_label != 'accuracy' and class_label != 'macro avg' and class_label != 'weighted avg':
        f1_scores[class_label] = metrics['f1-score']

# sort the F1 scores in descending order and select the top 20 classes
top_classes = sorted(f1_scores, key=f1_scores.get, reverse=True)[:20]
print(f'Best 20 sings are: {top_classes}.')

# sort the F1 scores in descending order and select the worst 20 classes
worst_classes = sorted(f1_scores, key=f1_scores.get, reverse=False)[:20]
print(f'Worst 20 sings are: {worst_classes}.')

#save model
#print(f'Saving model as {model_name}.h5')
#model.save(model_name+'.h5')
print('Done.')
