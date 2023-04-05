# tutorial

## About
Here you find our notebooks, data and models based on the Youtube tutorial "Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model" by Nicholas Renotte (https://www.youtube.com/watch?v=doDUihpj6ro). 

<br>

## Usage

<br>

### __Generating your own data__

You can generate your own data using the notebook `Example_Data_Generation.ipynb`. Just run all, then a window with your webcam feed will pop up. 

In the top left corner you can see a red text indicating the current word / sign you should present and the current number of sequence / video. 

Whenever, the green text "STARTING COLLECTION" pops up, you have 1 second? time to perform the your sign. With this notebook you can collect data for the signs / words "hello", "thanks" and "iloveyou" (in that order) and 30 videos / sequences are collected for each sign / word. 

Feel free, to change and play around as you like! :D

<br>

### __Training your own model__

You can train your own model using the notebook `Modeling.ipynb`. 

You can just run all using some prepared test data. If you generated your own data using the `Example_Data_Generation.ipynb` notebook, change the code from

`DATA_PATH = os.path.join('MP_Data_test')` 

to 

`DATA_PATH = os.path.join('MP_Data')`
.

After training, the model will be saved in this directory under the name 'first_model_whoop_whoop.h5' ;) it can be loaded again for later works or into other notebooks. 

<br>

### __Doing real-time predictions with your webcam__

You can perform real-time predictions on your webcam feed using the notebook `Real-Time-Prediction.ipynb`. 

On the top left you see a history of the last 5 detected words (a word is only added to the history, if it differs from the preceding word). 

Beneath the history, real-time barplots can be seen, that represent the prediction confidence for a corresponding word. 
