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

x_train = pad_sequences(x_train, 1000, padding="post")
x_test = pad_sequences(x_test, 1000, padding="post")

x_train = toVec(x_train, datalen)
x_test = toVec(x_test, datalen)


model = Sequential()
model.add(Dense(200, activation="relu", input_shape=(datalen,)))
model.add(Dense(50, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])

acc = list()
val_acc = list()

for i in range(6):
    history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
    acc += history.history['acc']
    val_acc += history.history['val_acc']
    print("accuracy: ", history.history["acc"])
    print("val-accuracy: ", history.history["val_acc"])
    print("loss: ", history.history["loss"])
    scores = historyTest = model.evaluate(x_test, y_test, verbose=0)
    print("################################")
    print("Test:", scores)
    plt.plot(acc, label='Learning')
    plt.plot(val_acc, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()