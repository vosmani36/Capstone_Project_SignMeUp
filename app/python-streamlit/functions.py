import numpy as np
import pandas as pd
import cv2 # for camera feed
import mediapipe as mp # for accessing and reading from webcam
import tensorflow as tf

# developer modules
from params import  MODE, LENGTH, DROP_Z, averaging_sets, point_landmarks_left, point_landmarks_right, FLATTEN, INPUT_SHAPE, RIGHT_HAND, LEFT_HAND, PADDING, CONSTANT_VALUE, THRESHOLD

# Initiate mediapipe model and utils
mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities


# ------------------------------
# Mediapipe
# ------------------------------

# function to extract coordinates (+visibility) of all landmarks --> keypoints
# and concatenates everything into a flattened list 
if MODE == 'kaggle': 
    def extract_keypoints(mph_results): 
        pose = np.array([[r.x, r.y] for r in mph_results.pose_landmarks.landmark]).flatten() if mph_results.pose_landmarks else np.zeros(33*2) # x, y, z and extra value visibility
        face = np.array([[r.x, r.y] for r in mph_results.face_landmarks.landmark]).flatten() if mph_results.face_landmarks else np.zeros(468*2)
        lh = np.array([[r.x, r.y] for r in mph_results.left_hand_landmarks.landmark]).flatten() if mph_results.left_hand_landmarks else np.zeros(21*2)
        rh = np.array([[r.x, r.y] for r in mph_results.right_hand_landmarks.landmark]).flatten() if mph_results.right_hand_landmarks else np.zeros(21*2)
        return np.concatenate([face, lh, pose, rh])
        # a flattened list with list of all pose, face, lh, rh landmark x, y, z, (+visibility) coordinates
elif MODE == 'tutorial': 
    # function to extract coordinates (+visibility) of all landmarks --> keypoints
    # and concatenates everything into a flattened list 
    def extract_keypoints(results): 
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) # x, y, z and extra value visibility
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([face, lh, pose, rh])
        # a flattened list with list of all pose, face, lh, rh landmark x, y, z, (+visibility) coordinates

# ------------------------------
# Visualization
# ------------------------------

# function to draw landmarks points and connecting lines on top of an image, e.g. on top of your camera feed
def draw_styled_landmarks(image, results): 
    # draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(224,208,64), thickness=1, circle_radius=1))
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(224,208,64), thickness=2, circle_radius=2)) 
    # draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(224,208,64), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(235,206,135), thickness=2, circle_radius=2)) 
    # draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(224,208,64), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(128,128,240), thickness=2, circle_radius=2))
 
# function to visualize predicted word probabilities with a dynamic real-time bar chart
def prob_viz(pred, SELECTED_SIGNS, input_frame): 
    output_frame = input_frame.copy() 
    bar_zero = 15
    
    for num, prob in enumerate(pred): 
        cv2.rectangle(output_frame, 
                      pt1=(bar_zero, 65+num*50), 
                      pt2=(bar_zero+int(prob*100*5), 95+num*50), 
                      color=(200, 200, 200), thickness=-1)
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.putText(img=output_frame, 
                    text=SELECTED_SIGNS[num], 
                    org=(bar_zero, 90+num*50), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=(50, 50, 50), 
                    thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
    return output_frame


# ------------------------------
# Pre-processing
# ------------------------------

# helper function for pre-processing
def tf_nan_mean(x, axis=0):
    #calculates the mean of a TensorFlow tensor x along a specified axis while ignoring any NaN values in the tensor.
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis)

# helper function for pre-processing
def right_hand_percentage(x):
    #calculates percentage of right hand usage
    right = tf.gather(x, RIGHT_HAND, axis=1)
    left = tf.gather(x, LEFT_HAND, axis=1)
    right_count = tf.reduce_sum(tf.where(tf.math.is_nan(right), tf.zeros_like(right), tf.ones_like(right)))
    left_count = tf.reduce_sum(tf.where(tf.math.is_nan(left), tf.zeros_like(left), tf.ones_like(left)))
    return right_count / (left_count+right_count)

#generating preprocessing layer that will be added to final model
class FeatureGen(tf.keras.layers.Layer):
    #defines custom tensorflow layer 
    def __init__(self):
        #initializes layer
        super(FeatureGen, self).__init__()
    
    def call(self, x_in, MIRROR=False):
        #drop z coordinates if required
        if DROP_Z:
            x_in = x_in[:, :, 0:2]
        if MIRROR:
            #flipping x coordinates
            x_in = np.array(x_in)
            x_in[:, :, 0] = (x_in[:, :, 0]-1)*(-1)
            x_in = tf.convert_to_tensor(x_in)

        #generates list with mean values for landmarks that will be merged
        x_list = [tf.expand_dims(tf_nan_mean(x_in[:, av_set[0]:av_set[0]+av_set[1], :], axis=1), axis=1) for av_set in averaging_sets]
        
        #extracts specific columns from input x_in defined by landmarks
        handedness = right_hand_percentage(x_in)
        if handedness > 0.5:
            x_list.append(tf.gather(x_in, point_landmarks_right, axis=1))
        else: 
            x_list.append(tf.gather(x_in, point_landmarks_left, axis=1))

        #concatenates the two tensors from above along axis 1/columns
        x = tf.concat(x_list, 1)

        #padding to desired length of sequence (defined by LENGTH)
        #get current number of rows
        x_padded = x
        current_rows = tf.shape(x_padded)[0]
        #if current number of rows is greater than desired number of rows, truncate excess rows
        if current_rows > LENGTH:
            x_padded = x_padded[:LENGTH, :, :]

        #if current number of rows is less than desired number of rows, add padding
        elif current_rows < LENGTH:
            #calculate amount of padding needed
            pad_rows = LENGTH - current_rows

            if PADDING ==4: #copy first/last frame
                if pad_rows %2 == 0: #if pad_rows is even
                    padding_front = tf.repeat(x_padded[0:1, :], pad_rows//2, axis=0)
                    padding_back = tf.repeat(x_padded[-1:, :], pad_rows//2, axis=0)
                else: #if pad_rows is odd
                    padding_front = tf.repeat(x_padded[0:1, :], (pad_rows//2)+1, axis=0)
                    padding_back = tf.repeat(x_padded[-1:, :], pad_rows//2, axis=0)
                x_padded = tf.concat([padding_front, x_padded, padding_back], axis=0)
            elif PADDING == 5: #copy last frame
                padding_back = tf.repeat(x_padded[-1:, :], pad_rows, axis=0)
                x_padded = tf.concat([x_padded, padding_back], axis=0)
            else:
                if PADDING ==1: #padding at start and end
                    if pad_rows %2 == 0: #if pad_rows is even
                        paddings = [[pad_rows//2, pad_rows//2], [0, 0], [0, 0]]
                    else: #if pad_rows is odd
                        paddings = [[pad_rows//2+1, pad_rows//2], [0, 0], [0, 0]]
                elif PADDING ==2: #padding only at the end of sequence
                    paddings = [[0, pad_rows], [0, 0], [0, 0]]
                elif PADDING ==3: #no padding
                    paddings = [[0, 0], [0, 0], [0, 0]]
                x_padded = tf.pad(x_padded, paddings, mode='CONSTANT', constant_values=CONSTANT_VALUE)

        x = x_padded
        current_rows = tf.shape(x)[0]

        #interpolate single missing values
        x = pd.DataFrame(np.array(x).flatten()).interpolate(method='linear', limit=2, limit_direction='both')
        #fill missing values with zeros
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        
        #reshape data to 2D or 3D array
        if FLATTEN:
            x = tf.reshape(x, (1, current_rows*INPUT_SHAPE[1]))
        else:
            x = tf.reshape(x, (1, current_rows, INPUT_SHAPE[1]))

        return x

#define converter using generated layer
feature_converter = FeatureGen()


# ------------------------------
# Real-time prediction 
# ------------------------------

def real_time_prediction(results, sequence, predictions, THRESHOLD, LENGTH, MODEL, SELECTED_LABELS, TRANSITION_FRAMES, SELECTED_SIGNS): 
    sign = ''
    prob = 0

    # Extract key points into a sequence
    keypoints = extract_keypoints(results) # extract keypoints x, y, z for face, left_hand, pose, right_hand from mediapipe holistic predictions, keypoints.shape e.g. (543, 3)
    sequence.append(keypoints) # keep appending keypoints (frames) to a sequence, np.array(sequence).shape e.g. (22, 543, 3)
    sequence = sequence[-LENGTH:] # takes last e.g. 22 frames of the sequence

    # Predict upon full sequence
    if len(sequence) == LENGTH: 
        # pre-processing
        if MODE == 'kaggle': 
            model_input = feature_converter(np.array(sequence))
            #print(f'OMG! Frenzy Franzi is converting your mediapipe input! See how the shape is changing from {np.array(sequence).shape} to {model_input.shape}! SO AWESOME!!!')
        elif MODE == 'tutorial': 
            model_input = np.expand_dims(sequence, axis=0)
        
        # prediction
        pred = MODEL.predict(model_input)[0] # MODEL.fit() expects something in shape (num_sequences, 30, 1662), e.g. (1, 30, 1662) for a single sequence                    
        print(len(pred))
        pred = pred[SELECTED_LABELS] # selects only a subset of signs, as defined in SELECTED_LABELS
        predictions.append(np.argmax(pred)) # appends all predictions

        # 3. Visualization logic
        # makes sure the last x frames had the same prediction (more stable transition from one sign to another) 
        if np.unique(predictions[-TRANSITION_FRAMES:])[0]==np.argmax(pred): 
            # if the confidence of the most confident prediction is above threshold
            if pred[np.argmax(pred)] > THRESHOLD: 
                sign = SELECTED_SIGNS[np.argmax(pred)]
                prob = pred[np.argmax(pred)]
                prob = np.round(float(prob), 2)
            else: 
                sign = ' '
                prob = 0
                
    return sign, prob
                


# ------------------------------
# Streamlit
# ------------------------------

