import numpy as np
import tensorflow as tf

#MODE = 'kaggle'
MODE = 'tutorial'

# load model
if MODE == 'kaggle': 
    MODEL = tf.keras.models.load_model('models/LSTM_model_20signs_7.h5')
elif MODE == 'tutorial': 
    MODEL = tf.keras.models.load_model('models/action_model')

#------------------------------
# PRE-PROCESSING CONFIGURATION
#------------------------------

#limit dataset for quick test
QUICK_TEST = True
QUICK_LIMIT = 500

#Define length of sequences for padding or cutting; 22 is the median length of all sequences
LENGTH = 22

#define min or max length of sequences; sequences too long/too short will be dropped
#max value of 92 was defined by calculating the interquartile range
MIN_LENGTH = 10
MAX_LENGTH = 92

#final data will be flattened, if false data will be 3 dimensional
FLATTEN = False

#define initialization of numpy array 
ARRAY = False #(True=Zeros, False=empty values)

#Define padding mode 
#1 = padding at start&end; 2 = padding at end; 3 = no padding, 4 = copy first/lastframe, 5 = copy last frame)
#Note: Mode 3 will give you an error due to different lengths, working on that
PADDING = 2
CONSTANT_VALUE = 0 #only required for mode 1 and 2; enter tf.constant(float('nan')) for NaN

#define if z coordinate will be dropped
DROP_Z = True

#mirror, flips x coordinate for data augmentation
MIRROR = True

#define if csv file should be filtered
CSV_FILTER  = False
#define how many participants for test set
TEST_COUNT = 5 #5 participants account for ca 23% of dataset
#generate test or train dataset (True = Train dataset; False = Test dataset)
#TRAIN = True #only works if CSV_FILTER is activated
TRAIN = True

#filter for specific signs
SIGN_FILTER = True
sign_list = [0,1,5,8]

#define filenames for x and y:
feature_data = 'X' #x data
feature_labels = 'y' #y data

#use for test dataset
#feature_data = 'X_test_h6' #x data
#feature_labels = 'y_test_h6' #y data


RANDOM_STATE = 42

#Defining Landmarks
#index ranges for each landmark type
#dont change these landmarks
FACE = list(range(0, 468))
LEFT_HAND = list(range(468, 489))
POSE = list(range(489, 522))
POSE_UPPER = list(range(489, 510))
RIGHT_HAND = list(range(522, 543))
LIPS = [61, 185, 40, 39, 37,  0, 267, 269, 270, 409,
                 291,146, 91,181, 84, 17, 314, 405, 321, 375, 
                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
                 95, 88, 178, 87, 14,317, 402, 318, 324, 308]
lipsUpperOuter= [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter= [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner= [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner= [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
#defining landmarks that will be merged
averaging_sets = []

#generating list with all landmarks selected for preprocessing
#change landmarks you want to use here:
point_landmarks_right = RIGHT_HAND + lipsUpperInner + lipsLowerInner
point_landmarks_left = LEFT_HAND + lipsUpperInner + lipsLowerInner

#calculating sum of total landmarks used
LANDMARKS = len(point_landmarks_right) + len(averaging_sets)
print(f'Total count of used landmarks: {LANDMARKS}')

#defining input shape for model
if DROP_Z:
    INPUT_SHAPE = (LENGTH,LANDMARKS*2)
else:
    INPUT_SHAPE = (LENGTH,LANDMARKS*3)
print(INPUT_SHAPE)


#------------------------------
# GAME MECHANICS
#------------------------------

COUNTDOWN = 0

if MODE == 'kaggle': 
    LABEL_MAP = {'brown': 0,  'callonphone': 1,  'cow': 2,  'cry': 3,  'dad': 4,  'fireman': 5,  'frog': 6,  'gum': 7,  'icecream': 8,  'minemy': 9,  'nose': 10,  'owl': 11,  'please': 12,  'radio': 13,  'shhh': 14,  'shirt': 15,  'tomorrow': 16,  'uncle': 17,  'water': 18,  'who': 19}
    THRESHOLD = 0.5 # confidence metrics (only render prediction results, if confidence is above threshold)

elif MODE == 'tutorial': 
    LABEL_MAP = {'Hello!': 0, 'Thanks!': 1, 'I love you!': 2}
    THRESHOLD = 0.3 # confidence metrics (only render prediction results, if confidence is above threshold)


SELECTED_SIGNS = list(LABEL_MAP.keys())
SELECTED_LABELS = [LABEL_MAP[x] for x in SELECTED_SIGNS]

#------------------------------
# VISUALIZATION
#------------------------------
TRANSITION_FRAMES = 10


#------------------------------
# TUTORIAL MODE
#------------------------------

if MODE == 'tutorial': 
    LENGTH = 30