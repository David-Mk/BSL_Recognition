import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keras.models import Sequential


from frame_collection import detection, extract_keypoints, landmarks 

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Detection models
mp_holistic = mp.solutions.holistic


# Possible fix??
model = Sequential()
model.load_weights('recognition.h5')



sequence, sentence = []
treshold = 0.4

# Camera capture
capture = cv2.VideoCapture(0)


def main():

    # Load Mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:

        while capture.isOpened():

            # Adjusting camera feed
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)

            # Make detection
            image, result = detection(frame, hol)
            print(result)
            
            # Adding landrmarks
            landmarks(image, result)

            keypoints = extract_keypoints(result)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(res)
            
            cv2.imshow('Camera feed', frame)


            # Break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
