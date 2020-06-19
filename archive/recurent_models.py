from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, SimpleRNN, Dropout, Dense

from simple_data_making import word_list, max_len, x_train_full, x_test_full, y_train, y_test
from func import metrics, show


def simple_rnn():
    model = Sequential()
    model.add(Embedding(word_list, 32, input_length=max_len))
    model.add(SimpleRNN(10))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=metrics)
    show(model, x_train_full, x_test_full, y_train, y_test)
