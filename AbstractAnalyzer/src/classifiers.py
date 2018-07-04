from gensim.models.keyedvectors import KeyedVectors
from itertools import chain
from enum import Enum
from numpy import average


class Classification(Enum):
    abstract = 'abstract'
    concrete = 'concrete'


class KnnClassifier:
    def __init__(self, k: int, positive_set: set, negative_set: set, corpus_path: str):
        """
        :param k: number of nearest neighbours to look at
        :param positive_set: Vocabulary to use as positively ranked
        :param negative_set: Vocabulary to use as negatively ranked
        :param corpus_path: path to the corpus to load word vectors from
        """
        self.k = k
        self.positive_set = positive_set
        self.negative_set = negative_set
        self.word_vectors_model = KeyedVectors.load_word2vec_format(corpus_path, binary=True)

    def classify_word(self, word: str):
        """
        :param word: Word to classify
        :return: NotImplemented error
        """
        raise NotImplementedError

    def _get_k_nearest_neighbours(self, distances_set, key=None):
        """
        :param distances_set: set of distances from the to be classified word
        :return: k nearest neighbours to word from the distances set
        """
        return sorted(distances_set, key=key)[:self.k]


class NeighboursKnnClassifier(KnnClassifier):
    def __init__(self, k: int, positive_set: set, negative_set: set, corpus_path: str):
        """
        :param k: number of nearest neighbours to look at
        :param positive_set: Vocabulary to use as positively ranked
        :param negative_set: Vocabulary to use as negatively ranked
        :param corpus_path: path to the corpus to load word vectors from
        """
        super().__init__(k, positive_set, negative_set, corpus_path)

    def classify_word(self, word: str):
        """
        :param word: word to classify
        :return: Classification
        """

        k_nearest_neighbours = self.__find_k_nearest_neighbours(word)

        positive_neighbours_count = sum([(word in self.positive_set) for word in k_nearest_neighbours])

        if positive_neighbours_count > len(k_nearest_neighbours) / 2:
            return Classification.abstract
        else:
            return Classification.concrete

    def __find_k_nearest_neighbours(self, word: str) -> list:
        """
        :param word: word to find neighbours of
        :return: k nearest neighbors of word
        """
        distances_from_positive_set = zip(self.positive_set,
                                          self.word_vectors_model.distances(word, self.positive_set))
        distances_from_negative_set = zip(self.negative_set,
                                          self.word_vectors_model.distances(word, self.negative_set))

        distance_from_all_set = chain(distances_from_positive_set, distances_from_negative_set)

        return [word for word, distance in self._get_k_nearest_neighbours(distance_from_all_set,
                                                                          key=lambda word_distance: word_distance[1])]


class SurroundingKnnClassifier(KnnClassifier):
    def __init__(self, k: int, positive_set: set, negative_set: set, corpus_path: str):
        """
        :param k: number of nearest neighbours to look at
        :param positive_set: Vocabulary to use as positively ranked
        :param negative_set: Vocabulary to use as negatively ranked
        :param corpus_path: path to the corpus to load word vectors from
        """
        super().__init__(k, positive_set, negative_set, corpus_path)

    def classify_word(self, word: str) -> Classification:
        """
        :param word: word to classify
        :return: classification of the word
        """

        k_nearest_positive_neighbours, k_nearest_negative_neighbours = self.__find_k_surrounding_sets(word)

        distance_from_positive_neighbours = average(
            self.word_vectors_model.distances(word, k_nearest_positive_neighbours))

        distance_from_negative_neighbours = average(
            self.word_vectors_model.distances(word, k_nearest_negative_neighbours))

        if distance_from_positive_neighbours > distance_from_negative_neighbours:
            return Classification.abstract
        else:
            return Classification.concrete

    def __find_k_surrounding_sets(self, word: str) -> tuple:
        """
        :param word: word to find surrounding sets for
        :return: k-surrounding of word for each classification class
        """
        k_nearest_positive_neighbours = self._get_k_nearest_neighbours(
            zip(self.positive_set,
                self.word_vectors_model.distances(word,
                                                  self.positive_set)))
        k_nearest_negative_neighbours = self._get_k_nearest_neighbours(
            zip(self.negative_set,
                self.word_vectors_model.distances(word,
                                                  self.negative_set)))

        return k_nearest_positive_neighbours, k_nearest_negative_neighbours
