import os
# Supress TF console warnings and informative messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix as mcm, accuracy_score as accuracy

from frame_collection import actions
from data_processing import main as data_main

# Training logs directory
logs_dir = os.path.join('Logs')
tb_checker = TensorBoard(log_dir = logs_dir)

def main():
    
    X_train, y_train, result = data_main()

    # Neural model is Long short-term memory (LSTM). It's used for better performance of Sequential model training
    # Defining the model as Sequential (The Sequential model is a linear stack of layers)
    model = Sequential()

    # Add layers. Density is customizable. ReLu is used, because of it's simplicity during training
    # Since TF with LSTM is used, first 2 layers have return sequences set on True and last one on False
    # ADD! Set dimensions, according to the shape of the training data
    model.add(LSTM(64, return_sequences = True, activation = 'relu', input_shape = (20, 1662)))
    model.add(LSTM(128, return_sequences = True, activation = 'relu'))
    model.add(LSTM(64, return_sequences = False, activation = 'relu'))

    # For image classification are used Dense Layers
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))

    # This layer specify the output. Activation is softmax for finilizing layer and multi-class classification prediction
    model.add(Dense(actions.shape[0], activation = 'softmax'))

    # Add Adam optimizer and compute loss function 
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='categorical_accuracy')

    # Fit model and set training process 
    model.fit(X_train, y_train, epochs=300, callbacks=[tb_checker])

    # Summary and evaluation of model training process 
    print(model.summary())

    # yhat = model.predict(X_train)
    # yhat = np.argmax(yhat, axis = 1).tolist()
    # 
    # ytrue = np.argmax(y_test, axis = 1).tolist()
    # 
    # mcm(ytrue, yhat)
    #
    # print(mcm)
    # print(accuracy(ytrue, yhat))
    
    # Make predictions
    # print(actions[np.argmax(result[1])])
    # print(actions[np.argmax(y_test[1])])
    
    # Save and delete model
    model.save('recognition.h5')
    del model
    
if __name__ == "__main__":

    main()