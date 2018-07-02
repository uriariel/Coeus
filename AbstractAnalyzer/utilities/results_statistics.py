from AbstractAnalyzer.configuration.path import PathConfigs
from collections import Counter
from os import scandir


class StatisticDict(dict):
    def __getitem__(self, item):
        """
        :type item: str
        :rtype: Counter
        """
        try:
            return self.__dict__[item]
        except KeyError:
            return Counter([''])

    def __setitem__(self, key, value):
        """
        :type key: str
        :type value: str
        """
        try:
            self.__dict__[key] += Counter([value])
        except KeyError:
            self.__dict__[key] = Counter([value])

    def __iter__(self):
        """
        :rtype: iter
        """
        return self.__dict__.__iter__()

    def __repr__(self):
        """
        :rtype: str
        """
        return self.__dict__.__repr__()

    def get_items(self):
        """
        :rtype: generator
        """
        return self.__dict__.items()


def main():
    results_list = filter(lambda entry: entry.is_file(), scandir(PathConfigs.results_dir_path))
    statistics_dict = StatisticDict()
    for file in results_list:
        with open(file, 'r') as results_file:
            for result_line in results_file.readlines():
                word, classification = [token.strip() for token in result_line.split(':')]
                statistics_dict[word] = classification

    for (word, statistics) in statistics_dict.get_items():
        print('{} => {}'.format(word, ' '.join(['{}: {}'.format(classification, count) for (classification, count) in
                                               statistics.items()])))


if __name__ == '__main__':
    main()
