from sklearn.model_selection import train_test_split as tts
import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.keras.utils import to_categorical
import numpy as np
import os

# For some reason, import bellow doesn't work. I used workaround above to import TF and Keras
#from tensorflow.keras.utils import to_categorical

from frame_collection import actions, sequences_count, sequences_length, DATA_PATH

def main():

    label_names = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []

    # Data labeling for LSTM model
    for action in actions:

        # WARNING!!! sequences_count? no_sequences was before 
        for sequence in range(sequences_count):
        
            window = []

            for frame_num in range(sequences_length):

                result = np.load(os.path.join(DATA_PATH, action, str(sequence), '{}.npy'.format(frame_num)))
                window.append(result)
            sequences.append(window)
            labels.append(label_names[action])
    
    # Data preparation
    X = np.array(sequences)
    print(X.shape)

if __name__ == "__main__":

    main()

#y = to_categorical(labels).astype(int)
#print(y.shape)
#
## Train-test-split fitting
#X_train, X_test, y_train, y_test = tts(X, y, test_size=0.05)
