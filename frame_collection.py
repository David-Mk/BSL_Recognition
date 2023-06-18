import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Detection models
mp_holistic = mp.solutions.holistic

# Data exporting path
DATA_PATH = os.path.join('bsl_data')

# Frame models (each model is word, letter or number from BSL dictionary) 
actions = np.array(['apartament', 'car', 'home'])

# Number of videos and videos length. For all recordings, numbers shall be the same
sequences_count = 30
sequences_length = 20

for a in actions:
    for s in range(sequences_count):
        try:
            os.makedirs(os.path.join(DATA_PATH, a, str(s)))
        except:
            pass


# Detection and color handling
def detection(image, model):

    # Color conversion is required, because Mediapipe and OpenCV work with different color input values
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

# Landmarks models, segmentation and customization
def landmarks(image, result):

    # Face landmarks
    mp_draw.draw_landmarks(
        image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_draw.DrawingSpec(color=(80, 110, 10),
                            thickness=1, circle_radius=1),
        mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Body landmarks
    mp_draw.draw_landmarks(
        image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(245, 117, 66),
                            thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # Right hand landmarks
    mp_draw.draw_landmarks(
        image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(80, 22, 10),
                            thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Left hand landmarks
    mp_draw.draw_landmarks(
        image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(80, 22, 10),
                            thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))


# Add coordinates and points to arrays
def extract_keypoints(result):

    # OPTION WITHOUT LOGICAL CONTROL (nor reliable, because when fuction fails to extract landamrks, recording is throwing an exception)
    #face = []
    #
    #face = np.array([[r.x, r.y, r.z, r.visibility]])
    #
    #for r in result.face_landmarks.landmark:
    #
    #    data = np.array([r.x, r.y, r.z])
    #    face.append(data)
    #
    #body = []
    #
    #for r in result.pose_landmarks.landmark:
    #
    #    data = np.array([r.x, r.y, r.z, r.visibility])
    #    body.append(data)
    #
    #lhand = []
    #
    #for r in result.left_hand_landmarks.landmark:
    #
    #    data = np.array([r.x, r.y, r.z])
    #    lhand.append(data)
    #
    #rhand = []
    #
    #for r in result.right_hand_landmarks.landmark:
    #
    #    data = np.array([r.x, r.y, r.z])
    #    rhand.append(data)
    
    # Add points to arrays and handling undetected ot unregognized parts
    body = np.array([[r.x, r.y, r.z, r.visibility] 
                    for r in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(132)
    
    face = np.array([[r.x, r.y, r.z] 
                    for r in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(1404)
    
    rhand = np.array([[r.x, r.y, r.z] 
                    for r in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(63)
    
    lhand = np.array([[r.x, r.y, r.z] 
                    for r in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(63)
    
    return np.concatenate([body, face, rhand, lhand])

# Camera capture. In case of errors, try swap number inside (camera index) or change them with recognition.py
capture = cv2.VideoCapture(1)

def main():
    
    # Load Mediapipe detection model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:

        # while capture.isOpened():
            
        # Range of captured frames and added landmarks to them
        for a in actions:
            
            for s in range(sequences_count):
            
                for frame_num in range(sequences_length):

                    # Adjusting camera feed
                    ret, frame = capture.read()
                    frame = cv2.flip(frame, 1)

                    # Make detection
                    image, result = detection(frame, hol)

                    # Add landrmarks
                    landmarks(image, result)
                    
                    # Collection process information
                    if frame_num == 0:
                       
                        cv2.putText(image, 'COLLECTION PROCESS', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'COLLECTING FRAMES FOR: {} VIDEO NUMBER: {}'.format(a, s), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        # Wait between frames collection process
                        cv2.waitKey(500)
                    else: 
                        
                        cv2.putText(image, 'COLLECTING FRAMES FOR: {} VIDEO NUMBER: {}'.format(a, s), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Export points for detections
                    keypoints = extract_keypoints(result)
                    npy_path = os.path.join(DATA_PATH, a, str(s), str(frame_num))
                    np.save(npy_path, keypoints)

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
