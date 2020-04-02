#%%

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
import matplotlib.pyplot as plt
import numpy


def toVec(dataset, dim=200):
    result = numpy.zeros((len(dataset), dim))
    for i, val in enumerate(dataset):
        result[i, val] = 1
    return result


datalen = 25000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=datalen)
x_train, y_train = x_train.copy(), y_train.copy()

x_train = pad_sequences(x_train, 300, padding="post")
x_test = pad_sequences(x_test, 300, padding="post")

x_train = toVec(x_train, datalen)
x_test = toVec(x_test, datalen)

print(x_train[0])

model = Sequential()
model.add(Dense(200, activation="relu", input_shape=(25000,)))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

acc = list()
val_acc = list()

for i in range(6):
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=0)
    acc += history.history['accuracy']
    val_acc += history.history['val_accuracy']
    print(history.history["accuracy"])
    print(history.history["val_accuracy"])
    scores = historyTest = model.evaluate(x_test, y_test, verbose=0)
    print("################################")
    print("Test:", scores)
    plt.plot(history.history['accuracy'], label='Learning')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


