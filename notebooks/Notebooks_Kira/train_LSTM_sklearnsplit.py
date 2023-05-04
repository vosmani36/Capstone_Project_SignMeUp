
#%pip install tensorflow
# general 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# for model building and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# for model evaluation
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Define dictionary to store results
training_history = {}

# for checkpoint callbacks
checkpoint_filepath = '/home/kirafriedrichs/LSTM_logback_230'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, 
    verbose=1, 
    save_weights_only=True,
    # Model weights are saved at the end of every epoch, if it's the best seen so far
    monitor='val_categorical_accuracy',
    mode='max',
    save_best_only=True)



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

# set random seed
RSEED = 42

#loading our preprocessed datasets
X = np.load('/home/kirafriedrichs/X_230_z.npy')
y = np.load('/home/kirafriedrichs/y_230_z.npy')

#creating new label map with reduced sign number
unique_values = sorted(list(set(y)))  # get a sorted list of unique values
new_label_map = {i: unique_values[i] for i in range(len(unique_values))}

#mapping the new label map on y dataset
y = np.array([list(new_label_map.keys())[list(new_label_map.values()).index(val)] for val in y])

#train-test-split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RSEED)

print('Import of datasets was successful.')
print('Shape of train dataset:')
print(X_train.shape, y_train.shape)
print('Shape of test dataset:')
print(X_test.shape, y_test.shape)

print('Encoding target labels...')
y_train = pd.get_dummies(y_train)
print('Shape of train dataset:')
print(y_train.shape)

# name for saving the trained model
model_name = 'LSTM_model_sklearnsplit_z230'

# setup sequential model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))) # input_shape = shape of x_train without first dimension
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(230, activation='softmax')) # 250 as output, due to 250 signs?

# compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# categorical_crossentropy must be used for multiclass classification model! 

# show summary of model
model.summary()
training_history = model.fit(X_train, y_train, validation_split=0.2 ,epochs=100, verbose=1, callbacks=[cp_callback])

#Generating and saving plots
print('Generating and saving metric/loss plots.')
plot_metric(training_history)
plot_loss(training_history)

# The model weights (that are considered the best) are loaded into the
# model.
model.load_weights(checkpoint_filepath)

# prediction on test data
print('Running predictions on test set...')
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# accuracy score
print(f'Accuracy score on test dataset is: {round(accuracy_score(y_test, y_pred),3)}.')

#save model
print(f'Saving model as {model_name}.h5')
model.save(model_name+'.h5')

print('Done.')
