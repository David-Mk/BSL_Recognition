import os
# Supress TF console warnings and informative messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model

from frame_collection import detection, extract_keypoints, landmarks, actions

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Detection models
mp_holistic = mp.solutions.holistic

# Possible fix??
# Load trained model
# model.load_weights('recognition.h5')
loaded_model = load_model('recognition.h5')

# Detection variables
# sequence = []
# sentence = []
# treshold = 0.4

# Camera capture. In case of errors, try swap number inside (camera index) or change them with frame_collection.py
capture = cv2.VideoCapture(0)

def main():

    sequence = []
    sentence = []
    threshold = 0.4
    # Load Mediapipe detection model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:

        while capture.isOpened():

            # Adjusting camera feed
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

            # Make detection
            image, result = detection(frame, hol)
            print(result)
            
            # Add landrmarks
            landmarks(image, result)

            # Make predictions
            keypoints = extract_keypoints(result)
            sequence.append(keypoints) #better then insert
            
            # Number of sequences should be the same as sequences_length in frame_collection.py
            sequence = sequence[-20:]
            
            if len(sequence) == 20:
                res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
            
                # Visualization
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:

                            sentence.append(actions[np.argmax(res)])
                        else:

                            sentence.append(actions[np.argmax(res)])

                # Get last 5 values of predictions        
                if len(sentence) > 5:

                    sentence = sentence[-5:]

            # Draw rectangle and text on camera feed
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
            # Show the screen    
            cv2.imshow('Camera feed', image)

            # Break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Remove camera after break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
