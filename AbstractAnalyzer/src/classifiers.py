from AbstractAnalyzer.configuration.path import PathConfigs

from gensim.models.word2vec import Word2Vec, Text8Corpus
from itertools import chain
from numpy import average


class Classification:
    abstract = 'abstract'
    concrete = 'concrete'


class KnnClassifier:
    def __init__(self, k, positive_set, negative_set):
        """
        :param k: number of nearest neighbours to look at
        :type k: int
        :param positive_set: Vocabulary to use as positively ranked
        :type positive_set: list
        :param negative_set: Vocabulary to use as negatively ranked
        :type negative_set: list
        """
        self.k = k
        self.positive_set = positive_set
        self.negative_set = negative_set
        self.word_vectors_model = Word2Vec(Text8Corpus(PathConfigs.text8_corpus_path)).wv

    def classify_word(self, word):
        """
        :param word: Word to classify
        :type word: str
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
    def __init__(self, k, positive_set, negative_set):
        """
        :param k: number of nearest neighbours to look at
        :type k: int
        :param positive_set: Vocabulary to use as positively ranked
        :type positive_set: list
        :param negative_set: Vocabulary to use as negatively ranked
        :type negative_set: list
        """
        super().__init__(k, positive_set, negative_set)

    def classify_word(self, word):
        """
        :param word: word to classify
        :type word: str
        :return: Classification
        """
        word = word.lower()

        k_nearest_neighbours = self.__find_k_nearest_neighbours(word)

        positive_neighbours_count = sum([(word in self.positive_set) for word in k_nearest_neighbours])

        if positive_neighbours_count > len(k_nearest_neighbours) / 2:
            return Classification.abstract
        else:
            return Classification.concrete

    def __find_k_nearest_neighbours(self, word):
        """
        :param word: word to find neighbours of
        :type word: str
        :return: k nearest neighbors of word
        :rtype: list
        """
        distances_from_positive_set = zip(self.positive_set,
                                          self.word_vectors_model.distances(word, self.positive_set))
        distances_from_negative_set = zip(self.negative_set,
                                          self.word_vectors_model.distances(word, self.negative_set))

        distance_from_all_set = chain(distances_from_positive_set, distances_from_negative_set)

        return [word for word, distance in self._get_k_nearest_neighbours(distance_from_all_set,
                                                                          key=lambda word_distance: word_distance[1])]


class SurroundingKnnClassifier(KnnClassifier):
    def __init__(self, k, positive_set, negative_set):
        """
        :param k: number of nearest neighbours to look at
        :type k: int
        :param positive_set: Vocabulary to use as positively ranked
        :type positive_set: list
        :param negative_set: Vocabulary to use as negatively ranked
        :type negative_set: list
        """
        super().__init__(k, positive_set, negative_set)

    def classify_word(self, word):
        """
        :param word: word to classify
        :type word: str
        :return: classification of the word
        :rtype: Classification
        """
        word = word.lower()

        k_nearest_positive_neighbours, k_nearest_negative_neighbours = self.__find_k_surrounding_sets(word)

        distance_from_positive_neighbours = average(
            self.word_vectors_model.distances(word, k_nearest_positive_neighbours))

        distance_from_negative_neighbours = average(
            self.word_vectors_model.distances(word, k_nearest_negative_neighbours))

        if distance_from_positive_neighbours > distance_from_negative_neighbours:
            return Classification.abstract
        else:
            return Classification.concrete

    def __find_k_surrounding_sets(self, word):
        """
        :param word: word to find surrounding sets for
        :type word: str
        :return: k-surrounding of word for each classification class
        :rtype: tuple
        """
        k_nearest_positive_neighbours = self._get_k_nearest_neighbours(zip(self.positive_set,
                                                                           self.word_vectors_model.distances(word,
                                                                                                             self.positive_set)))
        k_nearest_negative_neighbours = self._get_k_nearest_neighbours(zip(self.negative_set,
                                                                           self.word_vectors_model.distances(word,
                                                                                                             self.negative_set)))

        return k_nearest_positive_neighbours, k_nearest_negative_neighbours
