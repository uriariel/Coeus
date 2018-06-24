"""Abstract word classifier"""

from configuration import PathConfigs, AnalyzerConfigs
from gensim.models import word2vec
from itertools import chain
from random import sample
import csv


class DictionaryFilters(object):
    """Collection of filters for the dictionary"""

    abstract_suffixes = ['ness', 'ism', 'ion', 'ity']

    def abstract_word_filter(self, word):
        """
        :param word: word to check for abstract suffix
        :type word: str
        :return: bool
        """
        return any(map(lambda suffix: word.endswith(suffix), self.abstract_suffixes))

    def concrete_word_filter(self, word):
        """
        :param word: word to check for absence of abstract suffix
        :type word: str
        :return: bool
        """
        return all(map(lambda suffix: not word.endswith(suffix), self.abstract_suffixes))


class NeighborsKnnClassifier(object):
    def __init__(self, k, positive_set, negative_set):
        """
        :param k: number of nearest neighbors to look at
        :type k: int
        :param positive_set: Vocabulary to use as positively ranked
        :type positive_set: list
        :param negative_set: Vocabulary to use as negatively ranked
        :type negative_set: list
        """
        self.k = k
        self.positive_set = positive_set
        self.negative_set = negative_set
        self.word_vectors_model = word2vec.Word2Vec(word2vec.Text8Corpus(PathConfigs.text8_corpus)).wv

    def classify_word(self, word):
        """
        :param word: Word to classify
        :type word: str
        :return: Classification
        """
        word = word.lower()

        k_nearest_neighbors = self.__find_k_nearest_neighbours(word)

        positive_neighbors_count = sum([(word in self.positive_set) for word in k_nearest_neighbors])

        if positive_neighbors_count > len(k_nearest_neighbors) / 2:
            return 'Abstract'
        else:
            return 'Concrete'

    def __find_k_nearest_neighbours(self, word):
        """
        :param word: word to find neighbors of
        :type word: str
        :return: list
        """
        distances_from_positive_set = zip(self.positive_set,
                                          self.word_vectors_model.distances(word, self.positive_set))
        distances_from_negative_set = zip(self.negative_set,
                                          self.word_vectors_model.distances(word, self.negative_set))

        distance_from_all_set = chain(distances_from_positive_set, distances_from_negative_set)

        return [word for word, distance in sorted(distance_from_all_set,
                                                  key=lambda word_distance: word_distance[1])][:self.k]


class SurroundingKnnClassifier(object):
    def __init__(self, k, positive_set, negative_set):
        """
        :param k: number of nearest neighbors to look at
        :type k: int
        :param positive_set: Vocabulary to use as positively ranked
        :type positive_set: list
        :param negative_set: Vocabulary to use as negatively ranked
        :type negative_set: list
        """
        self.k = k
        self.positive_set = positive_set
        self.negative_set = negative_set
        self.word_vectors_model = word2vec.Word2Vec(word2vec.Text8Corpus(PathConfigs.text8_corpus)).wv

    def classify_word(self, word):
        """
        :param word: Word to classify
        :type word: str
        :return: Classification
        """
        word = word.lower()

        k_nearest_neighbors = self.__find_k_nearest_neighbours(word)

        positive_neighbors_count = sum([(word in self.positive_set) for word in k_nearest_neighbors])

        if positive_neighbors_count > len(k_nearest_neighbors) / 2:
            return 'Abstract'
        else:
            return 'Concrete'

    def __find_k_nearest_neighbours(self, word):
        """
        :param word: word to find neighbors of
        :type word: str
        :return: list
        """
        distances_from_positive_set = zip(self.positive_set,
                                          self.word_vectors_model.distances(word, self.positive_set))
        distances_from_negative_set = zip(self.negative_set,
                                          self.word_vectors_model.distances(word, self.negative_set))

        distance_from_all_set = chain(distances_from_positive_set, distances_from_negative_set)

        return sorted(distance_from_all_set, key=lambda word_distance: word_distance[1])[:self.k]


def main():
    # Read a dictionary with concreteness rating for each word
    with open(PathConfigs.filtered_dict_path, 'r') as english_word_set:
        concreteness_dict = {word.lower().strip(): concreteness for concreteness, word in csv.reader(english_word_set)}

    # Separate the dictionary into abstract and concrete word sets
    abstract_words_set = list(filter(DictionaryFilters().abstract_word_filter, concreteness_dict))
    concrete_words_set = list(filter(DictionaryFilters().concrete_word_filter, concreteness_dict))

    neigbours_knn_classifier = NeighborsKnnClassifier(AnalyzerConfigs.k,
                                                      sample(abstract_words_set, AnalyzerConfigs.word_set_size),
                                                      sample(concrete_words_set, AnalyzerConfigs.word_set_size))

    neigbours_knn_classifier.classify_word('atheism')
    classified_words = {word: neigbours_knn_classifier.classify_word(word) for word in concreteness_dict}

    # Write results to output file
    with open(PathConfigs.classified_dict_path, 'w') as classified_dict_file:
        for word, classification in classified_words.items():
            classified_dict_file.write('{}: {}\n'.format(word, classification))


if __name__ == '__main__':
    main()
