import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np

import tempfile
import time

# developer modules
from functions import draw_styled_landmarks, real_time_prediction
from params import LENGTH, SELECTED_SIGNS, TRANSITION_FRAMES, SELECTED_LABELS, MODEL, THRESHOLD


# ------------------------------
# Basic App Scaffolding
# ------------------------------

# Title
st.title('SignMeUp')

# Markdown styling
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create Sidebar
st.sidebar.title('SignMeUp Sidebar')
st.sidebar.subheader('Parameter')

# Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['Video Recognition', 'About', 'Contact']
)


# ------------------------------
# About Page
# ------------------------------

if app_mode == 'About':
    st.markdown('''
                ## About \n
                In this application we are using **MediaPipe** landmark prediction for recognizing American Sign Language. **StreamLit** is used to create the Web Graphical User Interface (GUI) \n
                
                - [Github](https://github.com/vosmani36/Capstone_Project_SignMeUp/tree/main/notebooks) \n
    ''')

    ## Add Sidebar and Window style
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,
        unsafe_allow_html=True,
    ) 


# ------------------------------
# Video Recognition Page
# ------------------------------

elif app_mode == 'Video Recognition':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Start')

    ## Get Video
    stframe = st.empty()
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    if use_webcam:
        video = cv.VideoCapture(0)
    else:
        video = cv.VideoCapture('https://cdn.dribbble.com/users/17914/screenshots/4902225/video-placeholder.png')

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(video.get(cv.CAP_PROP_FPS))

    ## Recording
    fps = 0
    sign_recognized = ' '
    prob_recognized = 0
    i = 0

    kpil, kpil2, kpil3 = st.columns(3)

    with kpil:
        st.markdown('**Frame Rate**')
        kpil_text = st.markdown('0')

    with kpil2:
        st.markdown('**Sign**')
        kpil2_text = st.markdown('0')

    with kpil3:
        st.markdown('**Probability**')
        kpil3_text = st.markdown('0')

    st.markdown('<hr/>', unsafe_allow_html=True)


    ## Live Video Mediapipe Holistic
    
    # New detection variables 
    sequence = [] # to collect all 22 frames for prediction
    sentence = [] # history of all predictions (predicted words)
    predictions = []

    # Real-time prediction
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

            prevTime = 0

            while video.isOpened():
                i +=1
                ret, frame = video.read()
                if not ret:
                    continue
                
                # Make MediaPipe detections 
                results = holistic.process(frame) 

                # Draw detected landmarks
                draw_styled_landmarks(frame, results)
                
                # Real-time prediction
                sign_recognized, prob_recognized = real_time_prediction(results, sequence, predictions, THRESHOLD, LENGTH, MODEL, SELECTED_LABELS, TRANSITION_FRAMES, SELECTED_SIGNS)
                
                # FPS Counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime

                # Dashboard
                kpil_text.write(f"<h1 style='text-align: center; color:(52, 75, 102);'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpil2_text.write(f"<h1 style='text-align: center; color:(52, 75, 102);'>{sign_recognized}</h1>", unsafe_allow_html=True)
                kpil3_text.write(f"<h1 style='text-align: center; color:(52, 75, 102);'>{prob_recognized}</h1>",
                                 unsafe_allow_html=True)

                frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
                stframe.image(frame,channels='BGR', use_column_width=True)