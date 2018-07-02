from os.path import abspath

ASSETS_PATH = abspath('../assets/') + '/'

TEXT_8_CORPUS_PATH = ASSETS_PATH + 'text8_corpus'
WIKI_PRETRAINED_CORPUS_PATH = ASSETS_PATH + 'wiki_corpus'
ENGLISH_DICT_PATH = ASSETS_PATH + 'words_with_concreteness_sorted.csv'
FILTERED_DICT_PATH = ASSETS_PATH + 'filtered_words_with_concreteness.csv'
CLASSIFIED_WORDS_PATH = ASSETS_PATH + 'classified_words.csv'
RESULTS_DIR_PATH = '../results'


class PathConfigs(object):
    english_dict_path = ENGLISH_DICT_PATH
    text8_corpus_path = TEXT_8_CORPUS_PATH
    wiki_pretrained_corpus_path = WIKI_PRETRAINED_CORPUS_PATH
    filtered_dict_path = FILTERED_DICT_PATH
    classified_dict_path = CLASSIFIED_WORDS_PATH
    results_dir_path = RESULTS_DIR_PATH
