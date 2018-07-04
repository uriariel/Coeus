"""Abstract word classifier"""

from AbstractAnalyzer.configuration.path import PathConfigs
from AbstractAnalyzer.configuration.analyzer import AnalyzerConfigs
from AbstractAnalyzer.src.classifiers import NeighboursKnnClassifier

from gensim.models.keyedvectors import KeyedVectors
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


def get_abstract_words_sample(all_word_set: list, sample_size: int, abstraction_filter: callable) -> list:
    """
    :param all_word_set: set to take the sample from
    :param sample_size: size of the sample to take
    :param abstraction_filter: filter for abstract words
    :return: sample of random abstract words
    """
    return sample(list(filter(abstraction_filter, all_word_set)), sample_size)


def get_concrete_words_sample(all_word_set: list, sample_size: int, concreteness_filter: callable) -> list:
    """
    :param all_word_set: set to take the sample from
    :param sample_size: size of the sample to take
    :param concreteness_filter: filter for concrete words
    :return: sample of random concrete words
    """
    return sample(list(filter(concreteness_filter, all_word_set)), sample_size)


def analyze(corpus_path: str, words_to_classify_path: str, results_path: str,
            pre_classification_filters: PreClassificationFilters):
    """
    :param corpus_path: path to corpus that should be analyzed
    :param words_to_classify_path: path to word set that should be classified
    :param results_path: path to write the analyze results to
    :param pre_classification_filters: word filters for pre classification phase
    """
    all_word_set = KeyedVectors.load_word2vec_format(corpus_path,
                                                     binary=True).wv.index2word

    neighbours_knn_classifier = NeighboursKnnClassifier(
        AnalyzerConfigs.k,
        get_abstract_words_sample(all_word_set, AnalyzerConfigs.word_set_size,
                                  pre_classification_filters.is_abstract_word),
        get_concrete_words_sample(all_word_set, AnalyzerConfigs.word_set_size,
                                  pre_classification_filters.is_concrete_word),
        corpus_path)

    with open(words_to_classify_path, 'r') as words_to_classify_path:
        words_to_classify_set = [word.lower().strip() for word, concreteness in
                                 csv.reader(words_to_classify_path)]

    pre_classified_words_set = [(word, neighbours_knn_classifier.classify_word(word).value) for word in
                                words_to_classify_set]

    write_results_to_output_file(pre_classified_words_set, results_path)


def analyze_hebrew():
    """Analyze hebrew corpus for abstractness"""
    analyze(PathConfigs.Corpuses.hebrew_cc_corpus_path, PathConfigs.DataSets.hebrew_data_set_path,
            PathConfigs.Results.hebrew_classified_data_set_path,
            PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
                                     AnalyzerConfigs.hebrew_abstract_suffixes))


def analyze_english():
    """Analyze english corpus for abstractness"""
    analyze(PathConfigs.Corpuses.english_wiki_corpus_path, PathConfigs.DataSets.english_data_set_path,
            PathConfigs.Results.english_classified_data_set_path,
            PreClassificationFilters(AnalyzerConfigs.english_abstract_prefixes,
                                     AnalyzerConfigs.english_abstract_suffixes))


def main():
    analyze_hebrew()


if __name__ == '__main__':
    main()
