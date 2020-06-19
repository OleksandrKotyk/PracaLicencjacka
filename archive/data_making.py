from copy import deepcopy

from nltk import FreqDist, Counter
from numpy.ma import array
from pandas import read_csv
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from func import remove_html, remove_spec_char, tokenize, remove_stop_words, make_stem, replace_triple_more, \
    remove_single_char, make_enum, make_sent_num, to_vec, td_idf_to_vec


def make_ig():
    pos_count = 0
    neg_count = 0
    pos_word_occur = {i: 0 for i in word_list}
    neg_word_occur = {i: 0 for i in word_list}
    pos_word_unique_occur = {i: 0 for i in word_list}
    neg_word_unique_occur = {i: 0 for i in word_list}

    for i in range(len(reviews)):
        if data["sentiment"].iloc[i] == "positive":
            pos_count += 1
            word_map = Counter(array(reviews.iloc[i]))
            for j in word_map:
                pos_word_occur[j] += word_map[j]
                pos_word_unique_occur[j] += 1
        else:
            neg_count += 1
            word_map = Counter(array(reviews.iloc[i]))
            for j in word_map:
                neg_word_occur[j] += word_map[j]
                neg_word_unique_occur[j] += 1

    print("Num of pos reviews:", pos_count)
    print("Num of neg reviews:", neg_count)

    ig = {}
    for i in word_list:
        f = ((pos_word_occur[i] + neg_word_occur[i]) / len(data) *
             entropy([pos_word_occur[i], neg_word_occur[i]], base=2))
        s = ((len(data) - pos_word_occur[i] - neg_word_occur[i]) / len(data) *
             entropy([pos_count - pos_word_occur[i], neg_count - neg_word_occur[i]], base=2))
        ig[i] = entropy([neg_count, pos_count], base=2) - (f + s)

    sorted_ig = sorted(ig.items(), key=lambda x: x[1])
    for i in sorted_ig[-10:]:
        print(i)

    return sorted_ig, pos_word_unique_occur, neg_word_unique_occur


data_len = int(input("Data len:"))

data = read_csv("../IMDB_Dataset.csv", encoding="utf-8")
data = data.sample(data_len)

reviews = data["review"]
reviews = reviews.apply(remove_html)
reviews = reviews.apply(lambda x: x.lower())
reviews = reviews.apply(remove_spec_char)
reviews = reviews.apply(tokenize)
# reviews = reviews.apply(remove_stop_words)
# reviews = reviews.apply(make_lem)
reviews = reviews.apply(make_stem)
reviews = reviews.apply(replace_triple_more)
reviews = reviews.apply(remove_single_char)
full_reviews = deepcopy(reviews)
reviews = reviews.apply(remove_stop_words)

word_list = FreqDist([j for i in reviews.iloc for j in i])
print("Num of words after cleaning:", len(word_list))

sorted_IG, pos_word_unique, neg_word_unique = make_ig()

word_counted = {}
word_dict = FreqDist([j for i in full_reviews.iloc for j in i])
word_dict_len_full = len(word_dict) + 1
for i, j in enumerate(word_dict):
    word_counted[j] = i + 1
enum_full_rev = full_reviews.apply(make_enum, args=[word_counted])


word_dict_len = int(input())
word_counted = {}
for i, j in enumerate(sorted_IG[-word_dict_len:]):
    word_counted[j[0]] = i + 1
word_dict_len += 1
enum_reviews = reviews.apply(make_enum, args=[word_counted])

sm, max_len, max_full_len = 0, 0, 0
for i in enum_reviews:
    if len(i) == 0:
        sm += 1
    if len(i) > max_len:
        max_len = len(i)

for i in enum_full_rev:
    if len(i) > max_full_len:
        max_full_len = len(i)

print("Empty:", sm, "Largest:", max_len)
print("Max full len:", max_full_len)

sent = data["sentiment"].apply(make_sent_num)
x_train, x_test, y_train, y_test = train_test_split(enum_reviews, sent, test_size=0.2)
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(enum_full_rev, sent, test_size=0.2)

x_train_pre = pad_sequences(x_train, max_len, padding="pre")
x_test_pre = pad_sequences(x_test, max_len, padding="pre")

# neg_word_unique = {word_counted[i[0]]: i[1] for i in neg_word_unique.items() if i[0] in word_counted}
# pos_word_unique = {word_counted[i[0]]: i[1] for i in pos_word_unique.items() if i[0] in word_counted}
# x_train_vec = td_idf_to_vec(x_train, word_dict_len, pos_word_unique, neg_word_unique)
# x_test_vec = td_idf_to_vec(x_test, word_dict_len, pos_word_unique, neg_word_unique)


x_train_vec = to_vec(x_train, word_dict_len)
x_test_vec = to_vec(x_test, word_dict_len)

x_train_full = pad_sequences(x_train_full, max_full_len, padding="pre")
x_test_full = pad_sequences(x_test_full, max_full_len, padding="pre")
print(x_test_full[0])
