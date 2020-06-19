from math import floor

from nltk import FreqDist
from numpy.ma import log, zeros
from pandas import read_csv
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from termcolor import cprint
from time import time

from func import remove_html, remove_spec_char, tokenize, remove_stop_words, make_stem, replace_triple_more, \
    remove_single_char, make_enum, to_vec


def to_vec_td_idf(dataset, dim):
    words_in_data = FreqDist([j for i in dataset for j in set(i)])
    result = zeros((len(dataset), dim))
    for i, val in enumerate(dataset):
        word_map = FreqDist(val)
        for wd in word_map:
            td = word_map[wd] / len(val)
            idf = log(len(dataset) / words_in_data[wd])
            # result[i, wd] = 1
            result[i, wd] = td * idf
    return result


def applying(rev, rem_stop_words=True):
    rev = rev.apply(remove_html)
    rev = rev.apply(lambda x: x.lower())
    rev = rev.apply(remove_spec_char)
    rev = rev.apply(tokenize)
    if rem_stop_words:
        rev = rev.apply(remove_stop_words)
    # rev = rev.apply(make_lem)
    rev = rev.apply(make_stem)
    if rem_stop_words:
        rev = rev.apply(remove_stop_words)
    rev = rev.apply(replace_triple_more)
    rev = rev.apply(remove_single_char)

    return rev, FreqDist([j for i in rev.iloc for j in i])


def information_gain(main_data, word_list):
    pos_rev = main_data.loc[main_data["sentiment"] == "positive"]["review"]
    neg_rev = main_data.loc[main_data["sentiment"] == "negative"]["review"]
    pos_word_occur = FreqDist([j for i in pos_rev.iloc for j in i])
    neg_word_occur = FreqDist([j for i in neg_rev.iloc for j in i])

    for i in word_list:
        if i not in pos_word_occur:
            pos_word_occur[i] = 0
        if i not in neg_word_occur:
            neg_word_occur[i] = 0

    ig = {}

    for i in word_list:
        f = ((pos_word_occur[i] + neg_word_occur[i]) / len(main_data) *
             entropy([pos_word_occur[i], neg_word_occur[i]], base=2))
        s = ((len(main_data) - pos_word_occur[i] - neg_word_occur[i]) / len(main_data) *
             entropy([len(pos_rev) - pos_word_occur[i], len(neg_rev) - neg_word_occur[i]], base=2))
        ig[i] = entropy([len(neg_rev), len(pos_rev)], base=2) - (f + s)

    sorted_ig = sorted(ig.items(), key=lambda x: x[1])
    cprint('The best IG words', 'green')
    for i in sorted_ig[-10:]:
        print(i)

    return sorted_ig


def make_data(data_len=None):
    data = read_csv("IMDB_Dataset.csv", encoding="utf-8")
    # data = read_csv("/content/drive/My Drive/IMDB_Dataset.csv", encoding="utf-8")
    data_len = int(input("Data len:")) if not data_len else data_len
    data = data.loc[data["sentiment"] == "positive"].iloc[:floor(data_len / 2)].append(
        data.loc[data["sentiment"] == "negative"].iloc[:floor(data_len / 2)])
    data = data.sample(data_len)

    return data


def main_fun(data, rem_stop_words, is_ig=True, num_of_wds=None, pad_len=None):
    start = time()
    data["review"], words = applying(data["review"], rem_stop_words)

    cprint("Quantity of the words: {}".format(len(words)), "blue")

    if is_ig:
        ig = information_gain(data, words)
        num_of_wds = int(input("Number of words:")) if not num_of_wds else num_of_wds
        enum_words = {j[0]: i + 1 for i, j in enumerate(ig[-num_of_wds:])}
    else:
        print(len(words))
        enum_words = {j: i + 1 for i, j in enumerate(words)}

    data["review"] = data["review"].apply(make_enum, args=[enum_words])

    max_len = max([len(i) for i in data["review"]])
    cprint("Words reducing", "green")
    print("Number of empty reviews:", len([None for i in data["review"] if len(i) == 0]))
    print("Max length:", max_len)

    sentiments = data["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    x_train, x_test, y_train, y_test = train_test_split(data["review"], sentiments, test_size=0.2)

    pad_len = max_len if not pad_len else pad_len
    x_train_pre = pad_sequences(x_train, pad_len, padding="pre")
    x_test_pre = pad_sequences(x_test, pad_len, padding="pre")

    x_train_vec = to_vec(x_train, len(enum_words) + 1)
    x_test_vec = to_vec(x_test, len(enum_words) + 1)

    x_train_vec_td_idf = to_vec_td_idf(x_train, len(enum_words) + 1)
    x_test_vec_td_idf = to_vec_td_idf(x_test, len(enum_words) + 1)

    cprint("Time of cleaning: {}". format(time() - start), "green")
    cprint("End of data cleaning", "red")
    return x_train_pre, x_test_pre, x_train_vec, x_test_vec, x_train_vec_td_idf, x_test_vec_td_idf, y_test, y_train, len(
        enum_words), pad_len
