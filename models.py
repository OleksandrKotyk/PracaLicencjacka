import os.path
from copy import deepcopy
from os import path
from random import randint
from time import time

from pandas import DataFrame
from tensorflow.keras.backend import clear_session
from tensorflow.python.keras.layers import Dense, Embedding, Flatten, SimpleRNN, Dropout, LSTM, Bidirectional
from tensorflow.python.keras.models import Sequential
from termcolor import cprint

from data_making import make_data, data_preparing, applying
from func import metrics, show


def human_time(time_val):
    return time_val


def make_tables(simple_s, name):
    new = {i: [] for i in ["name", "loss", "Precision", "Recall", "F1 score", "Accuracy", "Time", "Epoch"]}
    for i in simple_s:
        s = simple_s[i][0][0]
        f1_score = 2 * s[1] * s[2] / (s[1] + s[2]) if s[1] + s[2] != 0 else 0
        new["name"] += [i]
        new["loss"] += [round(s[0], 2)]
        new["Precision"] += [round(s[1], 2)]
        new["Recall"] += [round(s[2], 2)]
        new["F1 score"] += [round(f1_score, 2)]
        new["Accuracy"] += [round(s[3], 2)]
        new["Time"] += [human_time(simple_s[i][1])]
        new["Epoch"] += [simple_s[i][0][1]]

    df = DataFrame(new, columns=["name", "loss", "Precision", "Recall", "F1 score", "Accuracy", "Time", "Epoch"])

    df.to_excel("xlses/" + name + '.xls')


def run_models(from_main_fun, epoch_s, adding="", optimizer="Adagrad"):
    scores = {}

    def run_one_model(model, x_train, x_test, epc, name="model"):
        name += adding
        cprint(name, "green")
        start_time = time()
        sc_epc = show(model, x_train, x_test, y_train, y_test, plt=True, eps=epc, plt_title=name)
        scores[name] = [sc_epc, (time() - start_time)]
        clear_session()

    def find_best_for_sequential(x_train, x_test):
        print(end="\n\n")
        max_score = 0
        best = [0, 0]
        for k in range(10):
            model = Sequential()
            num = randint(100, 250)
            num2 = randint(20, num - 20)
            model.add(Dense(num, activation="relu", input_dim=dict_len))
            model.add(Dense(num2, activation="relu"))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
            for _ in range(2):
                model.fit(x_train, y_train, epochs=5, batch_size=200, validation_split=0.1, verbose=0)
                srs = model.evaluate(x_test, y_test, verbose=0)
                if max_score < srs[3]:
                    max_score = srs[3]
                    best = [num, num2]
            clear_session()
        print("Best first layer for sequential:", best[0])
        print("Best second layer for sequential:", best[1])
        return best

    def sequential(best):
        model = Sequential()
        model.add(Dense(best[0], activation="relu", input_dim=dict_len))
        model.add(Dense(best[1], activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    def embedding():
        model = Sequential()
        model.add(Embedding(dict_len, 32, input_length=pad_len))
        model.add(Flatten())
        model.add(Dense(30, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    def simple_rnn():
        model = Sequential()
        model.add(Embedding(dict_len, 32, input_length=pad_len))
        model.add(SimpleRNN(10))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    def double_lstm():
        model = Sequential()
        model.add(Embedding(dict_len, 32, input_length=pad_len))
        model.add(LSTM(10, return_sequences=True))
        model.add(LSTM(10))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    def bidirectional_lstm():
        model = Sequential()
        model.add(Embedding(dict_len, 32, input_length=pad_len))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    def lstm():
        model = Sequential()
        model.add(Embedding(dict_len, 32, input_length=pad_len))
        model.add(LSTM(10))
        model.add(Dropout(0.6))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model

    x_train_pre, x_test_pre, x_train_vec, x_test_vec, x_train_vec_td_idf, x_test_vec_td_idf, y_test, y_train, dict_len, pad_len = from_main_fun

    bst = find_best_for_sequential(x_train_vec, x_test_vec)

    run_one_model(sequential(bst), x_train_vec_td_idf, x_test_vec_td_idf, epc=epoch_s[0], name="Sequential td-idf")
    run_one_model(sequential(bst), x_train_vec, x_test_vec, epc=epoch_s[0], name="Sequential")
    run_one_model(embedding(), x_train_pre, x_test_pre, epc=epoch_s[1], name="Embedding")
    run_one_model(simple_rnn(), x_train_pre, x_test_pre, epc=epoch_s[2], name="Simple RNN")
    run_one_model(lstm(), x_train_pre, x_test_pre, epc=epoch_s[5], name="Simple LSTM")
    run_one_model(double_lstm(), x_train_pre, x_test_pre, epc=epoch_s[3], name="Double LSTM")
    run_one_model(bidirectional_lstm(), x_train_pre, x_test_pre, epc=epoch_s[4], name="Bidirectional LSTM")
    cprint("End of modeling", "cyan")
    print(end="\n\n")
    return scores


if not path.isdir('xlses'):
    os.mkdir("xlses")

if not path.isdir('plots'):
    os.mkdir("plots")

epochs = [40, 40, 40, 40, 40, 40]
for number_of_d in [1000, 3000, 7000, 11000, 15000]:
    main_data = make_data(data_len=number_of_d)
    cprint(str(number_of_d) + ' documents', 'yellow')
    main_data["review"], terms = applying(main_data["review"], rem_stop_words=True)
    cprint('---------------------------------------\n', 'blue')
    for quantity_of_w in [1000, 5000, 10000, 15000, 20000]:

        if len(terms) >= quantity_of_w < 10000 and number_of_d == 7000:
            cprint(str(quantity_of_w) + " terms", "cyan")
            data = data_preparing(deepcopy(main_data), terms, num_of_wds=quantity_of_w)
            scr = run_models(data, adding=" SGD " + str(number_of_d) + "d " + str(quantity_of_w) + "t", epoch_s=epochs,
                             optimizer="sgd")
            make_tables(scr, "SGD " + str(number_of_d) + "d_" + str(quantity_of_w) + "t")

        if len(terms) >= quantity_of_w:
            cprint(str(quantity_of_w) + " terms", "cyan")
            data = data_preparing(deepcopy(main_data), terms, num_of_wds=quantity_of_w)
            scr = run_models(data, adding=" " + str(number_of_d) + "d " + str(quantity_of_w) + "t", epoch_s=epochs)
            make_tables(scr, " " + str(number_of_d) + "d_" + str(quantity_of_w) + "t")
