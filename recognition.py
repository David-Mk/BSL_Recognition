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
sequence = []
sentence = []
treshold = 0.4

# Camera capture. In case of errors, try swap number inside (camera index) or change them with frame_collection.py
capture = cv2.VideoCapture(0)

def main():

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
            sequence.append(keypoints)
            sequence = sequence[:30]
            
            if len(sequence) == 30:
                res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
            
            cv2.imshow('Camera feed', image)


            # Break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
