from math import log

from matplotlib.pyplot import plot, xlabel, ylabel, legend, title as plot_title, grid, xticks, \
    yticks, figure, savefig, close
from nltk import RegexpTokenizer, PorterStemmer, WordNetLemmatizer, Counter, download
from nltk.corpus import stopwords
from numpy.ma import zeros, arange
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


def show(model, x_tr, x_te, y_tr, y_te, verb=0, plt=True, bs=200, eps=None, plt_title=None, num=None):
    acc = list()
    val_acc = list()
    test_acc = list()
    loss = list()
    epochs = 10 if not eps else eps
    best_scores = [-1, -1, -1, -1]
    best_epoch = None
    loss_now = 1000000
    go = True
    for _ in range(epochs):
        history = model.fit(x_tr, y_tr, epochs=1, batch_size=bs, validation_split=0.05, verbose=verb)
        acc += history.history['accuracy']
        val_acc += history.history['val_accuracy']
        scores = model.evaluate(x_te, y_te, verbose=0)
        test_acc += [scores[3]]
        loss += history.history['val_loss']
        if num is not None and _ % num == 0:
            f1_score = 2 * scores[1] * scores[2] / (scores[1] + scores[2]) if scores[1] + scores[2] != 0 else 0
            print("Accuracy: {}    F1 score: {}    Loss: {}".format(scores[3], f1_score, scores[0]))
        if loss_now > history.history['val_loss'][0] and go:
            loss_now = history.history['val_loss'][0]
            if best_scores[3] < scores[3]:
                best_scores = scores
                best_epoch = _ + 1
        else:
            go = False

    if plt:
        figure(figsize=(8, 5))
        grid(True)
        xticks(arange(1, epochs + 1, 1))
        yticks(arange(0, 1.05, 0.05))
        plot_title(plt_title)
        plot(arange(1, epochs + 1, 1), acc, label='Learning')
        plot(arange(1, epochs + 1, 1), val_acc, label='Validation')
        plot(arange(1, epochs + 1, 1), test_acc, label='Test')
        plot(arange(1, epochs + 1, 1), loss, label='Validation loss')
        xlabel('Epoch')
        ylabel('Accuracy')
        legend()
        # plot_show()
        savefig("plots/" + plt_title + ".png")
        close("all")
    del model

    return best_scores, best_epoch


def td_idf(text_array, neg_word_occ, pos_word_occ):
    for i in text_array:
        word_map = Counter(i)
        for wd in word_map:
            word_map[wd] = log(len(text_array) / (neg_word_occ[wd] + pos_word_occ[wd]), 10) * word_map[wd] / len(i)
        print(word_map)
        break
