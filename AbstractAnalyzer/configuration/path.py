from os.path import abspath, join

ASSETS_PATH = abspath('../assets')
RESULTS_DIR_PATH = abspath('../results')

CORPUSES_PATH = join(ASSETS_PATH, 'corpuses')
DATA_SETS_PATH = join(ASSETS_PATH, 'data_sets')
NOUN_SETS_PATH = join(ASSETS_PATH, 'noun_sets')
RAW_FILES_PATH = join(ASSETS_PATH, 'raw_files')

ENGLISH_WIKI_CORPUS_PATH = join(CORPUSES_PATH, 'english_wiki_corpus')
ENGLISH_CC_CORPUS_PATH = join(CORPUSES_PATH, 'english_cc_corpus')
HEBREW_CC_CORPUS_PATH = join(CORPUSES_PATH, 'hebrew_cc_corpus')

ENGLISH_DATA_SET_PATH = join(DATA_SETS_PATH, 'english_words_with_concreteness.csv')
HEBREW_DATA_SET_PATH = join(DATA_SETS_PATH, 'hebrew_words_with_concreteness.csv')

ENGLISH_NOUN_SET_PATH = join(NOUN_SETS_PATH, 'english_nouns_brown.csv')
HEBREW_NOUN_SET_PATH = join(NOUN_SETS_PATH, 'hebrew_nouns_mila.csv')
HEBREW_ABSTRACT_NOUN_SET_PATH = join(NOUN_SETS_PATH, 'hebrew_nouns_abstract_pattern.csv')

ENGLISH_DATA_SET_UNFILTERED_PATH = join(RAW_FILES_PATH, 'english/english_words_with_concreteness.csv')
HEBREW_DATA_SET_UNFILTERED_PATH = join(RAW_FILES_PATH, 'hebrew/hebrew_words_with_concreteness.csv')

ENGLISH_CLASSIFIED_DATA_SET_PATH = join(RESULTS_DIR_PATH, 'english/english_classified_data_set.csv')
HEBREW_CLASSIFIED_DATA_SET_PATH = join(RESULTS_DIR_PATH, 'hebrew/hebrew_classified_data_set.csv')


class PathConfigs:
    class Corpuses:
        english_wiki_corpus_path = ENGLISH_WIKI_CORPUS_PATH
        hebrew_cc_corpus_path = HEBREW_CC_CORPUS_PATH
        english_cc_corpus_path = ENGLISH_CC_CORPUS_PATH

    class DataSets:
        english_data_set_path = ENGLISH_DATA_SET_PATH
        hebrew_data_set_path = HEBREW_DATA_SET_PATH

    class NounSets:
        english_noun_set_path = ENGLISH_NOUN_SET_PATH
        hebrew_noun_set_path = HEBREW_NOUN_SET_PATH
        hebrew_abstract_noun_set_path = HEBREW_ABSTRACT_NOUN_SET_PATH

    class RawFiles:
        english_data_set_unfiltered_path = ENGLISH_DATA_SET_UNFILTERED_PATH
        hebrew_data_set_unfiltered_path = HEBREW_DATA_SET_UNFILTERED_PATH

    class Results:
        results_dir_path = RESULTS_DIR_PATH
        english_classified_data_set_path = ENGLISH_CLASSIFIED_DATA_SET_PATH
        hebrew_classified_data_set_path = HEBREW_CLASSIFIED_DATA_SET_PATH
