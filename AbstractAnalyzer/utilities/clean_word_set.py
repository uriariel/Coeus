"""Prepare supervisioned positive and negative word sets for the classifier"""
from gensim.models.keyedvectors import KeyedVectors

from AbstractAnalyzer.configuration.path import PathConfigs
import csv


def write_filtered_set_to_file(cleaned_set: dict, output_file: str):
    with open(output_file, 'w') as cleaned_training_set_file:
        writer = csv.writer(cleaned_training_set_file)
        for word, other_data in cleaned_set.items():
            writer.writerow([word, *other_data])


def read_data_set(unfiltered_data_set_path: str) -> dict:
    with open(unfiltered_data_set_path, 'r') as unfiltered_data_set:
        concreteness_dict = {word.lower().strip(): other_data for word, *other_data in
                             csv.reader(unfiltered_data_set)}
    return concreteness_dict


def filter_data_set(unfiltered_data_set_path: str, corpus_path: str, training_set_output_path: str):
    word_set_dict = read_data_set(unfiltered_data_set_path)

    model = KeyedVectors.load_word2vec_format(corpus_path, binary=True).wv

    banned_words_list = []

    for word in word_set_dict.keys():
        try:
            model[word]
        except KeyError:
            banned_words_list.append(word)

    filtered_data_set = dict(
        filter(lambda word_concreteness: word_concreteness[0] not in banned_words_list, word_set_dict.items()))

    write_filtered_set_to_file(filtered_data_set, training_set_output_path)


def main():
    filter_data_set(PathConfigs.NounSets.hebrew_abstract_noun_set_path,
                    PathConfigs.Corpuses.hebrew_cc_corpus_path,
                    PathConfigs.NounSets.hebrew_abstract_noun_set_path)


if __name__ == '__main__':
    main()
