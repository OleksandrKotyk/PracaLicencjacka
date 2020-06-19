from math import log

from matplotlib.pyplot import plot, xlabel, ylabel, legend, show as plot_show, title as plot_title
from nltk import RegexpTokenizer, PorterStemmer, WordNetLemmatizer, Counter, download
from nltk.corpus import stopwords
from numpy.ma import zeros
from regex import sub
from tensorflow.keras.metrics import Precision, Recall

download("stopwords")
download("wordnet")

stopWords = stopwords.words("english")
tokenizer = RegexpTokenizer(r"[a-z]+")
post_stemmer = PorterStemmer()
lem = WordNetLemmatizer()
metrics = [Precision(), Recall(), 'accuracy']


def remove_html(text):
    text = sub(r"<.+?>", " ", text)
    return text


def remove_spec_char(text):
    ret = ""
    for k in text:
        if k.isalpha():
            ret += k
        else:
            ret += " "
    return ret


def remove_stop_words(text_array):
    text_array = [j for j in text_array if j not in stopWords]
    return text_array


def tokenize(text):
    text_array = tokenizer.tokenize(text)
    return text_array


def make_stem(text_array):
    text_array = [post_stemmer.stem(j) for j in text_array]
    return text_array


def remove_single_char(text_array):
    ret = []
    for j in text_array:
        if len(j) > 1:
            ret.append(j)
    return ret


def make_lem(text_array):
    ret = []
    for i in text_array:
        ret.append(lem.lemmatize(i))
    return ret


def replace_triple_more(text_array):
    ret = []
    for i in text_array:
        ret.append(sub(r'(.)\1\1+', r'\1', i))
    return ret


def make_enum(text_array, enum_words):
    ret = []
    for i in text_array:
        if enum_words.get(i) is not None:
            ret.append(enum_words[i])
    return ret


def make_sent_num(text):
    if text == "positive":
        return 1
    else:
        return 0


def to_vec(dataset, dim):
    result = zeros((len(dataset), dim))
    for i, val in enumerate(dataset):
        result[i, val] = 1
    return result


def td_idf_to_vec(dataset, dim, neg_word_occ, pos_word_occ):
    result = zeros((len(dataset), dim))
    for i, val in enumerate(dataset):
        word_map = Counter(val)
        for wd in val:
            print(word_map)
            result[i][wd] = log(len(dataset) / (neg_word_occ[wd] + pos_word_occ[wd]), 10) * word_map[wd] / len(val)
    return result


def show(model, x_tr, x_te, y_tr, y_te, verb=0, plt=True, bs=200, eps=None, plt_title=None, num=1):
    # print(model.summary())
    acc = list()
    val_acc = list()
    epochs = 10 if not eps else eps
    max_scores = [-1, -1, -1, -1]
    best_epoch = None
    for _ in range(epochs):
        history = model.fit(x_tr, y_tr, epochs=1, batch_size=bs, validation_split=0.1, verbose=verb)
        acc += history.history['accuracy']
        val_acc += history.history['val_accuracy']
        scores = model.evaluate(x_te, y_te, verbose=0)
        if _ % num == 0:
            f1_score = 2 * scores[1] * scores[2] / (scores[1] + scores[2]) if scores[1] + scores[2] != 0 else 0
            print("Accuracy: {}    F1 score: {}".format(scores[3], f1_score))
        if acc[-1] > 95 or abs(acc[-1] - val_acc[-1] > 7):
            print(acc[-1])
            print(abs(acc[-1] - val_acc[-1] > 7))
            break
        if max_scores[3] < scores[3]:
            max_scores = scores
            best_epoch = _ + 1

    scores = model.evaluate(x_te, y_te, verbose=0)
    f1_score = 2 * scores[1] * scores[2] / (scores[1] + scores[2]) if scores[1] + scores[2] != 0 else 0
    print("Accuracy: {}    F1 score: {}".format(scores[3], f1_score))
    if plt:
        plot_title(plt_title)
        plot(acc, label='Learning')
        plot(val_acc, label='Validation')
        xlabel('Epoch')
        ylabel('Accuracy')
        legend()
        plot_show()

    return scores, epochs, max_scores, best_epoch


def td_idf(text_array, neg_word_occ, pos_word_occ):
    for i in text_array:
        word_map = Counter(i)
        for wd in word_map:
            word_map[wd] = log(len(text_array) / (neg_word_occ[wd] + pos_word_occ[wd]), 10) * word_map[wd] / len(i)
        print(word_map)
        break

# print("val-accuracy: ", history.history["val_accuracy"])
# print("loss: ", history.history["loss"])
# print("################################")
# print("Loss:", scores[0])
# print("Precision:", scores[1])
# print("Recall:", scores[2])
# print("F1 score:", (2 * scores[1] * scores[2]) / (scores[1] + scores[2]))

# scores.append(2 * scores[1] * scores[2]) / (scores[1] + scores[2])
# print("################################")
# print("Loss:", scores[0], end=" ")
# print("Precision:", scores[1])
# print("Recall:", scores[2], end=" ")
# print("Accuracy:", scores[3], end=" ")
# print("F1 score:", (2 * scores[1] * scores[2]) / (scores[1] + scores[2]))
# print("Sum of epochs:", epochs)
