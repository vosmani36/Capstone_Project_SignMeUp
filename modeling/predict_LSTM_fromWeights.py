
# this notebook loads model weights from training of a LSTM neural network and runs prediction on the test set
# it will give out accuracy score, classification report and lists of best and worst predicted signs

#%pip install tensorflow
# general 
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import time

# for model building and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report

INPUT = '../data/'
OUTPUT = '../data/'
# name for saving the trained model
model_name = 'LSTM_model'

# Define dictionary to store results
training_history = {}


#loading our preprocessed datasets
X = np.load(f'{INPUT}default_data_HandsLips_Z_split/X_z.npy')
y = np.load(f'{INPUT}default_data_HandsLips_Z_split/y_z.npy')

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print('Import of datasets was successful.')
print('Shape of train dataset:')
print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

#encoding target variables by defining dummies
print('Encoding target labels...')
y_train = pd.get_dummies(y_train)
print('Shape of train dataset:')
print(y_train.shape)

# set random seed
RSEED = 42

# setup sequential model
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_train.shape[1], X_train.shape[2]))) # input_shape = shape of x_train without first dimension
model.add(LSTM(64, return_sequences=True, activation='relu')) 
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax')) # 250 as output, due to 250 signs

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    name='Adam')

# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# categorical_crossentropy must be used for multiclass classification model! 

# show summary of model
model.summary()
#training_history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=1, callbacks=[cp_callback])

# The model weights (that are considered the best) are loaded into the model.
model.load_weights('./LSTM_model9_HandsLips_Z/best_model_weights.h5')
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

# assume y_test and y_pred are the ground truth and predicted labels, respectively
report = classification_report(y_test, y_pred, output_dict=True)

# extract the F1 scores for each class
f1_scores = {}
for class_label, metrics in report.items():
    if class_label != 'accuracy' and class_label != 'macro avg' and class_label != 'weighted avg':
        f1_scores[class_label] = metrics['f1-score']

# sort the F1 scores in descending order and select the top 20 classes
top_classes = sorted(f1_scores, key=f1_scores.get, reverse=True)[:20]
print(f'Best 20 sings sorted are: {top_classes}.')

# sort the F1 scores in descending order and select the worst 20 classes
worst_classes = sorted(f1_scores, key=f1_scores.get, reverse=False)[:20]
print(f'Worst 20 sings are: {worst_classes}.')

zero_classes = []
for class_id, f1_score in f1_scores.items():
    if f1_score == 0:
        zero_classes.append(class_id)
print(f'Signs with F1 score of 0: {zero_classes}')

#save model
print(f'Saving model as {model_name}.h5')
model.save(model_name+'.h5')

print('Done.')
