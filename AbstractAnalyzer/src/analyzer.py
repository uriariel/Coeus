"""Abstract word classifier"""
from AbstractAnalyzer.configuration.path import PathConfigs
from AbstractAnalyzer.configuration.analyzer import AnalyzerConfigs
from AbstractAnalyzer.src.classifiers import NeighboursKnnClassifier

from random import sample
import csv


class DictionaryFilters:
    """Collection of filters for the dictionary"""

    @staticmethod
    def abstract_word_filter(word):
        """
        :param word: word to check for abstract suffix
        :type word: str
        :return: bool
        """
        return any(map(lambda suffix: word.endswith(suffix), AnalyzerConfigs.abstract_suffixes))

    @staticmethod
    def concrete_word_filter(word):
        """
        :param word: word to check for absence of abstract suffix
        :type word: str
        :return: bool
        """
        return all(map(lambda suffix: not word.endswith(suffix), AnalyzerConfigs.abstract_suffixes))


def main():
    # Read a dictionary with concreteness rating for each word
    with open(PathConfigs.filtered_dict_path, 'r') as english_word_set:
        concreteness_dict = {word.lower().strip(): concreteness for concreteness, word in csv.reader(english_word_set)}

    # Separate the dictionary into abstract and concrete word sets
    abstract_words_set = list(filter(DictionaryFilters.abstract_word_filter, concreteness_dict))
    concrete_words_set = list(filter(DictionaryFilters.concrete_word_filter, concreteness_dict))

    neighbours_knn_classifier = NeighboursKnnClassifier(AnalyzerConfigs.k,
                                                        sample(abstract_words_set, AnalyzerConfigs.word_set_size),
                                                        sample(concrete_words_set, AnalyzerConfigs.word_set_size))

    neighbours_knn_classifier.classify_word('atheism')
    classified_words = {word: neighbours_knn_classifier.classify_word(word) for word in concreteness_dict}

    # Write results to output file
    with open(PathConfigs.classified_dict_path, 'w') as classified_dict_file:
        for word, classification in classified_words.items():
            classified_dict_file.write('{}: {}\n'.format(word, classification))


if __name__ == '__main__':
    main()
