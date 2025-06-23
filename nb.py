#SYSTEM IMPORTS
import numpy as np
import pandas as pd
from typing import List, Type, Dict, Any
import re
import os
import sys
import pickle

_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

#TYPE DECLARED
NaiveBayesType: Type = Type["NaiveBayes"]

class NaiveBayes:
    def __init__(self: NaiveBayesType) -> None:
        self.vocab: List[str] = []
        self.index: Dict[str, int] = {}
        self.class_counts: List[int] = []
        self.priors: List[float] = []
        self.likelihoods: Any = None
        self.num_classes: int = 0
        self.vocab_size: int = 0


    def word_vector(self: NaiveBayesType, tokens: List[str]) -> List[int]:
        #counting BoW approach: converts a list of tokens into a word vector based on frequency
        vector = [0] * len(self.vocab)
        for token in tokens:
            if token in self.index:
                vector[self.index[token]] += 1
        return vector

    def train(self: NaiveBayesType, X_train: List[List[str]], y_train: List[str]) -> None:
        #calculates the priors and likelihoods 
        self.vocab = list(set(word for sentence in X_train for word in sentence))
        self.vocab_size = len(self.vocab)
        self.index = {word: i for i, word in enumerate(self.vocab)}

        self.num_classes = len(set(y_train)) #just 2 for fake/real
        self.class_counts = [0] * self.num_classes
        self.priors = [0] * self.num_classes
        self.likelihoods = np.zeros((self.num_classes, self.vocab_size))

        for label in y_train:
            self.class_counts[label] += 1

        total = len(y_train)
        self.priors = [count/total for count in self.class_counts]

        for i, sentence in enumerate(X_train):
            label = y_train[i]
            word_vector = self.word_vector(sentence)
            for j, count in enumerate(word_vector):
                self.likelihoods[label, j] += count

        self.class_counts = np.array(self.class_counts)

        #smooth distribution to avoid 0 probs
        self.likelihoods += 1
        self.likelihoods /= (self.class_counts[:, None] + self.vocab_size)


    #returns the best class using Naive Bayes
    def predict_naive_bayes(self: NaiveBayesType, words: List[int], priors: List[float], likelihoods: np.ndarray) -> int:
        best_prob = -np.inf
        best_class = -1

        for y in range(len(priors)):
            log_post = np.log(priors[y])

            for w, word_count in enumerate(words):
                if word_count > 0:
                    log_post += np.log(likelihoods[y, w])

            if log_post > best_prob:
                best_prob = log_post
                best_class = y

        return best_class

    #predicts the labels for the articles
    def predict(self: NaiveBayesType, X_test: List[List[str]]) -> List[int]:
        y_pred = []
        for sentence in X_test:
            word_vector = self.word_vector(sentence)
            best_class = self.predict_naive_bayes(word_vector, self.priors, self.likelihoods)
            y_pred.append(best_class)
        return y_pred

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


