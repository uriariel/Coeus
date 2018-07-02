from AbstractAnalyzer.configuration.path import PathConfigs

from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec, Text8Corpus
from time import time

if __name__ == '__main__':
    start_time = time()
    # KeyedVectors.load_word2vec_format(PathConfigs.wiki_pretrained_corpus_path)
    # Word2Vec(Text8Corpus(PathConfigs.text8_corpus_path))
    end_time = time()
    print(end_time - start_time)
