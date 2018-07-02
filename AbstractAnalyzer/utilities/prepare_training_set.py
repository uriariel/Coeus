"""Prepare supervisioned positive and negative word sets for the classifier"""

from gensim.models import word2vec
from AbstractAnalyzer.configuration.path import PathConfigs
import csv


def main():
    with open(PathConfigs.english_dict_path, 'r') as english_word_set:
        concreteness_dict = {word.lower().strip(): concreteness for concreteness, word in csv.reader(english_word_set)}

    model = word2vec.Word2Vec(word2vec.Text8Corpus(PathConfigs.text8_corpus_path))

    banned_words_list = []

    for word in concreteness_dict.keys():
        try:
            model.most_similar(word)
        except KeyError:
            banned_words_list.append(word)

    filtered_dict = dict(
        filter(lambda word_concreteness: word_concreteness[0] not in banned_words_list, concreteness_dict.items()))

    with open(PathConfigs.filtered_dict_path, 'w') as filtered_dict_file:
        writer = csv.writer(filtered_dict_file)
        for key, val in filtered_dict.items():
            writer.writerow([val, key])


if __name__ == '__main__':
    main()
