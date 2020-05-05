import random
import json
import os
import numpy as np
from numpy import linalg as la
from functools import reduce

from clean_text import clean_text

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'model/')

with open(DATA_DIR + 'model.json', 'r') as file:
    dict_words = json.load(file)
dict_words["UNK"] = [0] * 128


def metric_for_vec(sample):
    np_sample = [[np.array(dict_words[word] if word in dict_words else dict_words["UNK"]) for word in vec] for vec in
                 sample]

    metric_sample = [reduce(np.add, np_vec) for np_vec in np_sample]

    return metric_sample


def cosine_vec(vec1, vec2):
    return np.dot(vec1, vec2) / (la.norm(vec1) * la.norm(vec2)) if la.norm(vec1) * la.norm(vec2) else 0.0


def cosine_matrix(metric_sample1, metric_sample2):
    sample = metric_sample1 + metric_sample2
    metric_matrix = np.eye(len(sample))
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            metric_matrix[i, j] = metric_matrix[j, i] = cosine_vec(sample[i], sample[j])

    return metric_matrix


def knn_distance(sample1, sample2, metric_matrix, k):
    score = 0

    for item in range(len(sample1)):
        line_item = list(zip(sample1 + sample2, tuple(metric_matrix[item])))
        line_sample = line_item[:len(sample1)]
        list_knn = sorted(line_item, key=lambda x: -x[1])[:k + 1]
        score += len(set(line_sample).intersection(list_knn)) - 1

    for item in range(len(sample2)):
        line_item = list(zip(sample1 + sample2, tuple(metric_matrix[len(sample1) + item])))
        line_sample = line_item[len(sample1) + 1:]
        list_knn = sorted(line_item, key=lambda x: -x[1])[:k + 1]
        score += len(set(line_sample).intersection(list_knn)) - 1

    return score / (2 * k * len(sample1))


def choose_sample(text, n_word, n_vec):
    list_random = random.sample(range(len(text) - n_word), n_vec)
    list_sample = [tuple(text[pos:n_word + pos]) for pos in list_random]

    return list_sample


def semantic_knn(text1: str, text2: str, n_iter=50, n_word=15, n_vec=20, k=8, std=0.15) -> float:
    """
    Computes semantic distance between two texts.
    In each iteration generates two samples where 1st sample from text1, 2nd sample from text2
    Compute knn_score (uses knn_text()).
    Calculate mean(knn_scores).
    Compute final score using normal distribution (0.5, std).

    :param text1: first text
    :param text2: second text
    :param n_iter: number of iterations
    :param n_word: number of words in vector
    :param n_vec: number of vectors in sample
    :param k: number of nearest neighbors
    :param std: standard deviation for normal distribution

    :return: semantic distance between two texts
    """
    clean_text1 = clean_text(text1)
    clean_text2 = clean_text(text2)
    knn_scores = []
    for _ in range(n_iter):
        sample1 = choose_sample(clean_text1, n_word, n_vec)
        sample2 = choose_sample(clean_text2, n_word, n_vec)
        metric_sample1 = metric_for_vec(sample1)
        metric_sample2 = metric_for_vec(sample2)
        knn_scores.append(knn_distance(sample1, sample2, cosine_matrix(metric_sample1, metric_sample2), k))

    mean_u = np.mean(knn_scores)
    print(mean_u)

    mean = 0.5
    formula = lambda x: np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

    return formula(mean_u) * 100


if __name__ == '__main__':
    with open('texts/text1.txt', 'r') as file:
        text1 = file.read()
    with open('texts/text6.txt', 'r') as file:
        text2 = file.read()
    print(semantic_knn(text1, text2))
