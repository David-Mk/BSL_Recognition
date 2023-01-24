import os
# Supress TF warnings and informative messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard

from frame_collection import actions

from data_processing import X_train, X_test, y_train, y_test

# Input directory
logs_dir = os.path.join('Logs')
tb_checker = TensorBoard(log_dir = logs_dir)

def main():
    
    # Neural model is Long short-term memory (LSTM). It's used for better performance of Sequential model training
    # Defining the model as Sequential (The Sequential model is a linear stack of layers.)
    model = Sequential()

    # Adding layers. Density is customizable. Activation functions are defined, because of complexity of data. ReLu is used, because of it's simplicity during training
    # Since TF with LSTM is used, first 2 layers have return sequences and last one is set on False

    # ADD! Set dimensions, according to the shape of the training data
    model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape = (20, 1662)))
    model.add(LSTM(128, return_sequences = True, activation = 'relu'))
    model.add(LSTM(64, return_sequences = False, activation = 'relu'))

    # For image classification are used Dense Layers
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))

    # This layers specify the output. Activation is softmax for finilizing layer and multi-class classification prediction
    model.add(Dense(actions.shape[0], activation = 'softmax'))

    # Add Adam optimizer and compute loss function 
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='categorical_accuracy')

    # Fit model and set training process
    model.fit(X_train, y_train, epochs=3000, callback=[tb_checker])

if __name__ == "__main__":

    main()