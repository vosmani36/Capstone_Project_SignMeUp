
#%pip install tensorflow
# general 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for model building and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Define dictionary to store results
training_history = {}

# Plotting function for Accuracy
def plot_metric(history):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Categorical Accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.savefig(f'plot_metric_{model_name}.png')

# Plotting function for loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='categorical_crossentropy')
    plt.plot(history.history['val_loss'], label='val_categorical_crossentropy')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid(True)
    plt.savefig(f'plot_loss_{model_name}.png')


#loading our preprocessed datasets
print('Loading X and y data...')
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
print('Import of datasets was successful.')
print('Shape of train dataset:')
print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

print('Encoding target labels...')
y_train = pd.get_dummies(y_train)
print('Shape of train dataset:')
print(y_train.shape)

# set random seed
RSEED = 24

# name for saving the trained model
model_name = 'LSTM_model_2_1'

# setup sequential model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))) # input_shape = shape of x_train without first dimension
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(250, activation='softmax')) # 250 as output, due to 250 signs?

# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# categorical_crossentropy must be used for multiclass classification model! 

# show summary of model
model.summary()
training_history = model.fit(X_train, y_train, validation_split=0.2 ,epochs=1, verbose=1)

#Generating and saving plots
print('Generating and saving metric/loss plots.')
plot_metric(training_history)
plot_loss(training_history)

# prediction on test data
print('Running predictions on test set...')
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# accuracy score
print(f'Accuracy score on test dataset is: {round(accuracy_score(y_test, y_pred),3)}.')

#save model
print(f'Saving model in models/{model_name}')
model.save('models/'+model_name)

print('Done.')
