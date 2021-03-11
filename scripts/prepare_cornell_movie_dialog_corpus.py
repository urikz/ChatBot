import argparse
import re
import codecs
import os

MULTI_WHITESPACES_REGEX = re.compile(r'\s+')


MOVIE_CHARACTERS_FILE = 'movie_characters_metadata.txt'
MOVIE_LINES_FILE = 'movie_lines.txt'
MOVIE_CONVERSATIONS_FILE = 'movie_conversations.txt'


def get_id(s, first_letter):
    s = s.strip()
    assert s[0] == first_letter
    return int(s[1:])


def create_file_for_writing(path):
    assert not os.path.isfile(path)
    f = codecs.open(path, 'w', 'utf-8')
    return f


def parse_line(line):
    cols = []
    line2 = MULTI_WHITESPACES_REGEX.sub(' ', line).strip().split('+++$+++')
    for col in line2:
        cols.append(col.strip())
    return cols


def print_dialog(dialog, keep_history, output_file):
    if len(dialog) < 2:
        return 0
    if keep_history:
        print('\t'.join(dialog), file=output_file)
        return 1
    else:
        for i in range(1, len(dialog)):
            print(
                '%s\t%s' % (dialog[i - 1], dialog[i]),
                file=output_file,
            )
        return len(dialog) - 1


def main(args):
    assert not os.path.isfile(args.out)
    print('Will save prepared dataset to %s' % args.out)

    characters = {}
    characters_r = set()
    with codecs.open(os.path.join(args.data, MOVIE_CHARACTERS_FILE), 'r', 'utf-8', errors='ignore') as cf:
        for line in cf:
            cols = parse_line(line)
            assert len(cols) == 6
            c_id = cols[0]
            c_name = cols[3].replace(' ', '_').upper() + '_' + cols[1].replace(' ', '_').lower()
            if c_name in characters_r:
                c_name = c_name + '_' + c_id
            assert c_name not in characters_r, c_name
            characters[c_id] = c_name
            characters_r.add(c_name)


    print('Found %d characters' % len(characters))
    if args.person_vocab:
        output_file = open(args.out, 'w')
        for c_id, c_name in characters.items():
            print(c_name, file=output_file)
        output_file.close()
    else:
        lines = {}
        with codecs.open(os.path.join(args.data, MOVIE_LINES_FILE), 'r', 'utf-8', errors='ignore') as lf:
            for line in lf:
                cols = parse_line(line)
                assert len(cols) == 5
                lines[cols[0]] = (cols[1], cols[4].lower())
        print('Found %d lines' % len(lines))

        output_file = open(args.out, 'w')

        generated_samples = 0

        with codecs.open(os.path.join(args.data, MOVIE_CONVERSATIONS_FILE), 'r', 'utf-8', errors='ignore') as cf:
            for line in cf:
                cols = parse_line(line)
                assert len(cols) == 4
                c1 = cols[0]
                assert c1 in characters
                c2 = cols[1]
                assert c2 in characters
                other_character = {
                    c1: c2,
                    c2: c1,
                }
                current_lines = list(map(lambda x: x.strip("'"), cols[3].strip('[]').split(', ')))
                for current_line in current_lines:
                    assert lines[current_line][0] in other_character

                previous_person_id = None
                dialog = []
                for i in range(len(current_lines)):
                    current_person_id, current_text = lines[current_lines[i]]
                    if (
                        current_person_id == previous_person_id or
                        len(current_text) == 0
                    ):
                        generated_samples += print_dialog(
                            dialog,
                            args.keep_history,
                            output_file,
                        )
                        if len(current_text) > 0:
                            dialog = [current_text]
                            previous_person_id = current_person_id
                        else:
                            dialog = []
                            previous_person_id = None
                    else:
                        dialog.append(current_text)
                        previous_person_id = current_person_id
                generated_samples += print_dialog(
                    dialog,
                    args.keep_history,
                    output_file,
                )

        output_file.close()
        print('Generated %d samples ' % generated_samples )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument('--data', type=str, help='Path to the unzipped corpus folder')
    parser.add_argument('--out', type=str, help='Path to the preprocessed folder')
    parser.add_argument('--person-vocab', action='store_true', default=False)
    parser.add_argument('--keep-history', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
