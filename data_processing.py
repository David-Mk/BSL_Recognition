import os
# Supress TF console warnings and informative messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import train_test_split as tts
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

from frame_collection import actions, sequences_count, sequences_length, DATA_PATH

def main():

    # Label categories
    label_names = {label:num for num, label in enumerate(actions)}

    print('Labels shape:\n', label_names)
    sequences, labels = [], []

    # Data labeling for LSTM model
    for action in actions:

        for sequence in range(sequences_count):
        
            window = []

            for frame_num in range(sequences_length):

                result = np.load(os.path.join(DATA_PATH, action, str(sequence), '{}.npy'.format(frame_num)))
                window.append(result)
                
            sequences.append(window)
            labels.append(label_names[action])
               
    print('Sequences shape:\n', np.array(sequences).shape)

    # Data preparation
    X = np.array(sequences)
    print('X array shape:\n', X.shape)

    y = to_categorical(labels).astype(int)
    print('y array shape:\n', y.shape)

    # Train-test-split. Test size is 5% from whole data 
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.05)
    print('X_train array shape:\n', X_train.shape)
    print('X_test array shape:\n', X_test.shape)
    print('y_train array shape:\n', y_train.shape)
    print('y_test array shape:\n', y_test.shape)
    
    return X_train, X_test, y_train, y_test, result

if __name__ == "__main__":

    main()
