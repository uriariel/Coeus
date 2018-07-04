from argparse import ArgumentParser


def get_all_lines_flipped_around_comma(file_path: str) -> str:
    """
    :param file_path: file to flip lines around comma in
    :return all lines flipped around comma
    """
    flipped_comma_lines = []

    with open(file_path, 'r') as hebrew_data_set_file:
        for line in hebrew_data_set_file.readlines():
            flipped_comma_lines += '{},{}\n'.format(*line.strip().split(',')[::-1])

    return flipped_comma_lines


def write_lines_to_file(lines: str, file_path: str):
    """
    :param lines: lines to write to the file
    :param file_path: file path to write into
    """
    with open(file_path, 'w') as hebrew_data_set_file:
        hebrew_data_set_file.writelines(lines)


def arg_parse():
    """
    Parse script arguments
    :return: parsed arguments object
    """
    parser = ArgumentParser(description='Flip every line in file around it\'s comma')
    parser.add_argument('input_file_path', type=str, help='input file')
    parser.add_argument('output_file_path', type=str, help='output file')

    return parser.parse_args()


def main():
    args = arg_parse()

    write_lines_to_file(get_all_lines_flipped_around_comma(args.input_file_path), args.output_file_path)


if __name__ == '__main__':
    main()
