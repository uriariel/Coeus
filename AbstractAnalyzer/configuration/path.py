from os.path import abspath, join

ASSETS_PATH = abspath('../assets')
RESULTS_DIR_PATH = abspath('../results')

CORPUSES_PATH = join(ASSETS_PATH, 'corpuses')
DATA_SETS_PATH = join(ASSETS_PATH, 'data_sets')
RAW_FILES_PATH = join(ASSETS_PATH, 'raw_files')

ENGLISH_WIKI_CORPUS_PATH = join(CORPUSES_PATH, 'english_wiki_corpus')
HEBREW_CC_CORPUS_PATH = join(CORPUSES_PATH, 'hebrew_cc_corpus')

ENGLISH_DATA_SET_PATH = join(DATA_SETS_PATH, 'english_words_with_concreteness.csv')
HEBREW_DATA_SET_PATH = join(DATA_SETS_PATH, 'hebrew_words_with_concreteness.csv')
FILTERED_DATA_SET_PATH = join(DATA_SETS_PATH, 'filtered_words_with_concreteness.csv')

ENGLISH_DATA_SET_UNFILTERED_PATH = join(RAW_FILES_PATH, 'english_words_with_concreteness.csv')
HEBREW_DATA_SET_UNFILTERED_PATH = join(RAW_FILES_PATH, 'hebrew_words_with_concreteness.csv')

ENGLISH_CLASSIFIED_DATA_SET_PATH = join(RESULTS_DIR_PATH, 'english_classified_data_set.csv')
HEBREW_CLASSIFIED_DATA_SET_PATH = join(RESULTS_DIR_PATH, 'hebrew_classified_data_set.csv')


class PathConfigs:
    class Corpuses:
        english_wiki_corpus_path = ENGLISH_WIKI_CORPUS_PATH
        hebrew_cc_corpus_path = HEBREW_CC_CORPUS_PATH

    class DataSets:
        english_data_set_path = ENGLISH_DATA_SET_PATH
        hebrew_data_set_path = HEBREW_DATA_SET_PATH
        filtered_data_set_path = FILTERED_DATA_SET_PATH

    class RawFiles:
        english_data_set_unfiltered_path = ENGLISH_DATA_SET_UNFILTERED_PATH
        hebrew_data_set_unfiltered_path = HEBREW_DATA_SET_UNFILTERED_PATH

    class Results:
        results_dir_path = RESULTS_DIR_PATH
        english_classified_data_set_path = ENGLISH_CLASSIFIED_DATA_SET_PATH
        hebrew_classified_data_set_path = HEBREW_CLASSIFIED_DATA_SET_PATH
