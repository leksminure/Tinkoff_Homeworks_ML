import numpy as np
from nltk.stem.snowball import SnowballStemmer
import pickle
from langdetect import detect

languages_dict = {'da': "danish",
                  'de': "german",
                  'en': "english",
                  'es': "spanish",
                  'fi': "finnish",
                  'fr': "french",
                  'hu': "hungarian",
                  'it': "italian",
                  'nl': "dutch",
                  'no': "norwegian",
                  'pt': "portuguese",
                  'ro': "romanian",
                  'ru': "russian",
                  'sv': "swedish"}


class Document:
    def __init__(self,
                 displayed_title,
                 displayed_lyric,
                 hidden_title,
                 hidden_lyric,
                 language,
                 tfidf_dict_title,
                 tfidf_dict_lyric):
        self.displayed_title = displayed_title
        self.displayed_lyric = displayed_lyric
        self.hidden_title = hidden_title
        self.hidden_lyric = hidden_lyric
        self.language = language
        self.tfidf_dict_title = tfidf_dict_title
        self.tfidf_dict_lyric = tfidf_dict_lyric

    def format(self, query):
        return [self.displayed_title, self.displayed_lyric[:150] + ' ...']


def intersected_list(lists):
    for element in lists:  # проверка на пустоту хотя бы одного списка
        if not element:
            return []
    n = len(lists)
    counters = [0 for i in range(n)]  # список индексов: по i индексу списка - индекс на котором находится указатель в i списке
    result = []
    max_len = sum(len(list) for list in lists)
    while len(result) < 1000 and sum(counters) < max_len - n + 1:
        array_along_axis = [arr[counters[id]] for id, arr in enumerate(lists)]
        if array_along_axis == [array_along_axis[0]] * n:
            result.append(array_along_axis[0])
            index_of_list_with_max_len = np.argmax(list(map(len, lists)))
            if counters[index_of_list_with_max_len] != len(lists[index_of_list_with_max_len])-1:
                counters[index_of_list_with_max_len] += 1
            else:
                index_of_list_with_min_len = np.argmin(list(map(len, lists)))
                counters[index_of_list_with_min_len] += 1
        else:
            index_of_min = np.argmin(np.array(array_along_axis))
            counters[index_of_min] += 1
            if counters[index_of_min] == len(lists[index_of_min]):
                return result
    return result


index = {}


def build_index():
    with open('documents.pkl', 'rb') as file:
        global docs
        docs = pickle.load(file)

    for idx, document in enumerate(docs):
        for word in set((document.hidden_title + ' ' + document.hidden_lyric).split()):
            if word not in index:
                index[word] = []
            index[word].append(idx)
    for word in index:
        index[word].sort()


def score(query, document):
    keywords = query.lower().split()
    score = 0
    for word in keywords:
        if word in document.tfidf_dict_title:
            score += 2.5 * document.tfidf_dict_title[word]
        if word in document.tfidf_dict_lyric:
            score += document.tfidf_dict_lyric[word]
    return score


def retrieve(query):
    if not query:
        return []
    query = query.lower()
    query_just_alpha = ' '.join([word for word in query if word.isalpha()])
    querys_language = detect(query_just_alpha) if len(query_just_alpha) > 0 else ''  # ломается если нет буквенного текста
    if querys_language in languages_dict:
        stemmer = SnowballStemmer(language=languages_dict[querys_language])
        query = ' '.join([stemmer.stem(word) for word in query.split()])
    index_lists = [index[word] if word in index else [] for word in query.split()]
    candidates = [docs[i] for i in intersected_list(index_lists)]
    return candidates[:50]
