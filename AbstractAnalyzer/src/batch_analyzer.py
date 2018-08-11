from AbstractAnalyzer.configuration.analyzer import AnalyzerConfigs
from AbstractAnalyzer.configuration.path import PathConfigs
from AbstractAnalyzer.src.analyzer import analyze, PreClassificationFilters


def main():
    # for k_value in range(15, 27, 2):
    #     for sample_size in range(800, 1300, 50):
    #         analyze(corpus_path=PathConfigs.Corpuses.english_cc_corpus_path,
    #                 test_set_path=PathConfigs.TestingSets.english_test_set_path,
    #                 training_set_path=PathConfigs.TrainingSets.english_training_set_path,
    #                 results_path='{}_k_{}_sample_{}_foo'.format(PathConfigs.Results.english_classified_test_set_path,
    #                                                             k_value, sample_size),
    #                 pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.english_abstract_prefixes,
    #                                                                     AnalyzerConfigs.english_abstract_suffixes),
    #                 k=k_value, word_set_size=sample_size)
    #
    for k_value in range(17, 27, 2):
        for sample_size in range(500, 720, 30):
            analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
                    training_set_path=PathConfigs.TrainingSets.hebrew_training_set_path,
                    test_set_path=PathConfigs.TestingSets.hebrew_test_set_path,
                    results_path='{}_k_{}_sample_{}_doo'.format(PathConfigs.Results.hebrew_classified_test_set_path,
                                                            k_value, sample_size),
                    pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
                                                                        AnalyzerConfigs.hebrew_abstract_suffixes),
                    k=k_value, word_set_size=sample_size)

    for k_value in range(17, 27, 2):
        for sample_size in range(500, 720, 30):
            analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
                    training_set_path=PathConfigs.TrainingSets.hebrew_training_set_path,
                    test_set_path=PathConfigs.TestingSets.hebrew_test_set_path,
                    results_path='{}_k_{}_sample_{}_cheat_abstract_doo'.format(
                        PathConfigs.Results.hebrew_classified_test_set_path,
                        k_value, sample_size),
                    pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
                                                                        AnalyzerConfigs.hebrew_abstract_suffixes),
                    k=k_value, word_set_size=sample_size,
                    abstract_training_set_path=PathConfigs.TrainingSets.hebrew_abstract_training_set_path)

    for k_value in range(17, 27, 2):
        for sample_size in range(500, 720, 30):
            analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
                    training_set_path=PathConfigs.TrainingSets.hebrew_training_set_path,
                    test_set_path=PathConfigs.TestingSets.hebrew_test_set_path,
                    results_path='{}_k_{}_sample_{}_cheat_all_doo'.format(
                        PathConfigs.Results.hebrew_classified_test_set_path,
                        k_value, sample_size),
                    pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
                                                                        AnalyzerConfigs.hebrew_abstract_suffixes),
                    k=k_value, word_set_size=sample_size,
                    concrete_training_set_path=PathConfigs.TrainingSets.hebrew_concrete_training_set_path,
                    abstract_training_set_path=PathConfigs.TrainingSets.hebrew_abstract_training_set_path)

    #
    # for k_value in range(19, 27, 2):
    #     for sample_size in range(600, 1150, 50):
    #         analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
    #                 noun_set_path=PathConfigs.TrainingSets.hebrew_noun_set_path,
    #                 data_set_path=PathConfigs.TestingSets.hebrew_data_set_path,
    #                 results_path='{}_k_{}_sample_{}_cheat_2'.format(PathConfigs.Results.hebrew_classified_data_set_path,
    #                                                                 k_value, sample_size),
    #                 pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
    #                                                                     AnalyzerConfigs.hebrew_abstract_suffixes),
    #                 k=k_value, word_set_size=sample_size,
    #                 abstract_noun_set_path=PathConfigs.TrainingSets.hebrew_abstract_noun_set_path)
    #
    # for k_value in range(19, 27, 2):
    #     for sample_size in range(600, 1150, 50):
    #         analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
    #                 noun_set_path=PathConfigs.TrainingSets.hebrew_noun_set_path,
    #                 data_set_path=PathConfigs.TestingSets.hebrew_data_set_path,
    #                 results_path='{}_k_{}_sample_{}_cheat_hard_1'.format(
    #                     PathConfigs.Results.hebrew_classified_data_set_path,
    #                     k_value, sample_size),
    #                 pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
    #                                                                     AnalyzerConfigs.hebrew_abstract_suffixes),
    #                 k=k_value, word_set_size=sample_size,
    #                 abstract_noun_set_path=PathConfigs.TrainingSets.hebrew_abstract_noun_set_path,
    #                 concrete_noun_set_path=PathConfigs.TrainingSets.hebrew_concrete_noun_set_path)
    #
    # for k_value in range(19, 27, 2):
    #     for sample_size in range(600, 1150, 50):
    #         analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
    #                 noun_set_path=PathConfigs.TrainingSets.hebrew_noun_set_path,
    #                 data_set_path=PathConfigs.TestingSets.hebrew_data_set_path,
    #                 results_path='{}_k_{}_sample_{}_cheat_hard_2'.format(
    #                     PathConfigs.Results.hebrew_classified_data_set_path,
    #                     k_value, sample_size),
    #                 pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
    #                                                                     AnalyzerConfigs.hebrew_abstract_suffixes),
    #                 k=k_value, word_set_size=sample_size,
    #                 abstract_noun_set_path=PathConfigs.TrainingSets.hebrew_abstract_noun_set_path,
    #                 concrete_noun_set_path=PathConfigs.TrainingSets.hebrew_concrete_noun_set_path)
    # for k_value in range(19, 27, 2):
    #     for sample_size in range(600, 1150, 50):
    #         analyze(corpus_path=PathConfigs.Corpuses.hebrew_cc_corpus_path,
    #                 noun_set_path=PathConfigs.NounSets.hebrew_noun_set_path,
    #                 data_set_path=PathConfigs.DataSets.hebrew_data_set_path,
    #                 results_path='{}_k_{}_sample_{}'.format(PathConfigs.Results.hebrew_classified_data_set_path,
    #                                                         k_value, sample_size),
    #                 pre_classification_filters=PreClassificationFilters(AnalyzerConfigs.hebrew_abstract_prefixes,
    #                                                                     AnalyzerConfigs.hebrew_abstract_suffixes),
    #                 k=k_value, word_set_size=sample_size)


if __name__ == '__main__':
    main()
