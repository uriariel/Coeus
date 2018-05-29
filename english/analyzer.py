"""Abstract word classifier"""

from gensim.models import word2vec
from configuration import PathConfigs
import numpy
import csv


class DictionaryFilters(object):
    """Collection of filters for the dictionary"""

    abstract_suffixes = ['ness', 'ism']

    def abstract_word_filter(self, word):
        return any(map(lambda suffix: word.endswith(suffix), self.abstract_suffixes))

    def concrete_word_filter(self, word):
        return all(map(lambda suffix: not word.endswith(suffix), self.abstract_suffixes))


class AbstractClassifier(object):
    def __init__(self, positive_vocabulary, negative_vocabulary):
        """
        :param positive_vocabulary: Vocabulary to use as positively ranked
        :type positive_vocabulary: list
        :param negative_vocabulary: Vocabulary to use as negatively ranked
        :type negative_vocabulary: list
        """
        self.positive_vocabulary = positive_vocabulary
        self.negative_vocabulary = negative_vocabulary
        self.embeddings_model = word2vec.Word2Vec(word2vec.Text8Corpus(PathConfigs.text8_corpus))

    def classify_word(self, word):
        """
        :param word: Word to classify
        :type word: str
        :return: Classification
        """
        word = word.lower()
        distance_from_positives = numpy.average(
            self.embeddings_model.wv.distances(word, self.positive_vocabulary))
        distance_from_negatives = numpy.average(
            self.embeddings_model.wv.distances(word, self.negative_vocabulary))

        if distance_from_negatives < distance_from_positives:
            self.positive_vocabulary.append(word)
            return 'Abstract'
        else:
            self.negative_vocabulary.append(word)
            return 'Concrete'


def main():
    # Read a dictionary with concreteness rating for each word
    with open(PathConfigs.filtered_dict_path, 'r') as english_word_set:
        concreteness_dict = {word.lower().strip(): concreteness for concreteness, word in csv.reader(english_word_set)}

    # Separate the dictionary to abstract and concrete word sets
    abstract_words_set = list(filter(DictionaryFilters().abstract_word_filter, concreteness_dict))
    concrete_words_set = list(filter(DictionaryFilters().concrete_word_filter, concreteness_dict))

    abstract_classifier = AbstractClassifier(abstract_words_set[:1000], concrete_words_set[:1000])

    classified_words = {word: abstract_classifier.classify_word(word) for word in concreteness_dict}

    # Write results to output file
    with open(PathConfigs.classified_dict_path, 'w') as classified_dict_file:
        for word, classification in classified_words.items():
            classified_dict_file.write('{}: {}\n'.format(word, classification))


if __name__ == '__main__':
    main()
