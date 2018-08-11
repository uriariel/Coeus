"""Abstract word classifier"""

from AbstractAnalyzer.configuration.path import PathConfigs
from AbstractAnalyzer.configuration.analyzer import AnalyzerConfigs
from AbstractAnalyzer.src.classifiers import NeighboursKnnClassifier

from random import sample

import csv


class PreClassificationFilters:
    """Collection of filters for pre classification phase word classification"""

    def __init__(self, prefixes, suffixes):
        self.prefixes = prefixes
        self.suffixes = suffixes

    def is_abstract_word(self, word: str) -> bool:
        """
        :param word: word to check for abstraction
        :return: is the word abstract
        """
        return any(map(lambda suffix: word.endswith(suffix), self.suffixes)) or any(
            map(lambda prefix: word.startswith(prefix), self.prefixes))

    def is_concrete_word(self, word: str) -> bool:
        """
        :param word: word to check for concreteness
        :return: is the word concrete
        """
        return not self.is_abstract_word(word)


def write_results_to_output_file(classified_word_set: list, output_file_path: str):
    """
    :param classified_word_set: words with their corresponding classification
    :param output_file_path: file to write the classified words to
    """
    with open(output_file_path, 'w') as classified_dict_file:
        for word, classification in classified_word_set:
            classified_dict_file.write('{},{}\n'.format(word, classification))


def get_abstract_words_sample(all_word_set: set, sample_size: int, abstraction_filter: callable) -> set:
    """
    :param all_word_set: set to take the sample from
    :param sample_size: size of the sample to take
    :param abstraction_filter: filter for abstract words
    :return: sample of random abstract words
    """
    return set(sample(set(filter(abstraction_filter, all_word_set)), sample_size))


def get_concrete_words_sample(all_word_set: set, sample_size: int, concreteness_filter: callable) -> set:
    """
    :param all_word_set: set to take the sample from
    :param sample_size: size of the sample to take
    :param concreteness_filter: filter for concrete words
    :return: sample of random concrete words
    """
    return set(sample(set(filter(concreteness_filter, all_word_set)), sample_size))


def analyze(corpus_path: str, training_set_path: str, test_set_path: str, results_path: str,
            k: int, word_set_size: int, pre_classification_filters: PreClassificationFilters,
            abstract_training_set_path: str = '', concrete_training_set_path: str = ''):
    """
    :param corpus_path: path to corpus that should be analyzed
    :param training_set_path: path to noun set to use for selecting pre classified base
    :param test_set_path: path to word set that should be classified
    :param results_path: path to write the analyze results to
    :param k:
    :param word_set_size:
    :param pre_classification_filters: word filters for pre classification phase
    :param abstract_training_set_path:
    :param concrete_training_set_path:
    """

    with open(training_set_path, 'r') as noun_set_file:
        all_nouns_set = {word.strip() for word in noun_set_file.readlines()}

    abstract_noun_set = set()
    if abstract_training_set_path != '':
        with open(abstract_training_set_path, 'r') as abstract_noun_set_file:
            abstract_noun_set = {word.strip() for word in abstract_noun_set_file.readlines()}

    concrete_noun_set = set()
    if concrete_training_set_path != '':
        with open(concrete_training_set_path, 'r') as concrete_noun_set_file:
            concrete_noun_set = {word.strip() for word in concrete_noun_set_file.readlines()}

    k = k if k != 0 else AnalyzerConfigs.k
    word_set_size = word_set_size if word_set_size != 0 else AnalyzerConfigs.word_set_size

    neighbours_knn_classifier = NeighboursKnnClassifier(
        k,
        get_abstract_words_sample(all_nouns_set, word_set_size,
                                  pre_classification_filters.is_abstract_word) | abstract_noun_set,
        get_concrete_words_sample(all_nouns_set, word_set_size,
                                  pre_classification_filters.is_concrete_word) | concrete_noun_set,
        corpus_path)

    with open(test_set_path, 'r') as test_set_path:
        words_to_classify_set = [word.lower().strip() for word, concreteness in
                                 csv.reader(test_set_path)]

    pre_classified_words_set = [(word, neighbours_knn_classifier.classify_word(word).value) for word in
                                words_to_classify_set]

    write_results_to_output_file(pre_classified_words_set, results_path)


def analyze_hebrew():
    """Analyze hebrew corpus for abstractness"""
    analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
            training_set_path=PathConfigs.TrainingSets.hebrew_training_set_path,
            test_set_path=PathConfigs.TestingSets.hebrew_test_set_path,
            results_path=PathConfigs.Results.hebrew_classified_test_set_path,
            k=AnalyzerConfigs.k, word_set_size=AnalyzerConfigs.word_set_size,
            pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
                                                                AnalyzerConfigs.hebrew_abstract_suffixes),
            abstract_training_set_path=PathConfigs.TrainingSets.hebrew_abstract_training_set_path)


def analyze_english():
    """Analyze english corpus for abstractness"""
    analyze(corpus_path=PathConfigs.Corpuses.english_cc_corpus_path,
            training_set_path=PathConfigs.TrainingSets.english_training_set_path,
            test_set_path=PathConfigs.TestingSets.english_test_set_path,
            results_path=PathConfigs.Results.english_classified_test_set_path,
            k=AnalyzerConfigs.k, word_set_size=AnalyzerConfigs.word_set_size,
            pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.english_abstract_prefixes,
                                                                AnalyzerConfigs.english_abstract_suffixes))


def main():
    analyze_hebrew()


if __name__ == '__main__':
    main()
