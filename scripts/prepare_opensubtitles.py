import argparse
import codecs
import gzip
import multiprocessing
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import os
import sys
import time
import tqdm
import re
import xml.etree.ElementTree as ET

word_tokenizer = TreebankWordTokenizer()

MAX_TIME_DIFFERENCE_S = 2
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 20

# remove brackets
CLEAN_BRACKETS_REGEX = re.compile(
    '<!--.*?-->|<[^>]*>|\([^\)]*\)|\[[^\]]*\]|\{[^\}]*\}|##|~'
)
# Usually, unbalanced brackets correspond to very noisy sentences
# '#' is usually pretty bad and means lyrics of the song
BRACKETS_CHARACTERS = ['[', ']', '(', ')', '{', '}', '<', '>', '#']

MULTI_WHITESPACES_REGEX = re.compile(r'\s+')

APOSTROPHE_REPLACEMENT_REGEX = [
    (re.compile(r"(\s?)n(\s?)'(\s?)t(\s|$)"), "\\1n't\\4"),
    (re.compile(r"'(\s?)(s|re|em|im|bout|cause|ve|d|ll|ne)(\s+|$)"), " '\\2\\3"),
    # it's a common (in OpenSubtitles) spelling error to use 'il instead of 'll
    (re.compile(r"'(\s?)il(\s|$)"), " 'll\\2"),
    (re.compile(r"(\s|^)i(\s?)'(\s?)(m|mm)(\s|$)"), "\\1i 'm\\5"),
    (re.compile(r"in(\s?)'(\s|$)"), "ing\\2"),
    (re.compile(r"(\s|^)ma(\s?)'(\s?)am(\s|$)"), "\\1ma'am\\4"),
    (re.compile(r"(\s|^)c(\s?)'(\s?)mon(\s|$)"), "\\1c'mon\\4"),
    (re.compile(r"(\s|^)o(\s?)'(\s?)clock(\s|$)"), "\\1o'clock\\4"),
    (re.compile(r"(\s|^)y(\s?)'(\s?)all(\s|$)"), "\\1y'all\\4"),
]

# Some cleaning steps are taken from
# https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling/blob/master/tasks/opensubs/extract_opensubs_dialogs.py
CLEANUP_REGEX_RULES = [
    # remove speaker tag "xxx: "
    (re.compile(r'^\s*[A-z]*\s*:'), ''),
    # remove unnecessary symbols
    (re.compile(r"-{2,}"), ' '),
    # delete a space right before a period for titles
    (re.compile(r'(?<=( mr| jr| ms| dr| st|mrs)) \.'), '. '),
]

CLEANUP_REPLACE_RULES = [
    ('"', ' '),
    ("``", " "),
    ("''", " "),
    ("% %", " "),
    ("i̇", "i"),
]


def get_dir_id(filename_path):
    dirpath, filename = os.path.split(filename_path)
    _, movie_id_str = os.path.split(dirpath)
    return int(movie_id_str)


def get_list_of_files(top_path, group_by_dir=False, extension='.xml.gz'):
    result = {}
    for path, dirs, files in os.walk(top_path):
        for filename in files:
            if filename.endswith(extension):
                full_filename = os.path.realpath(os.path.join(path, filename))
                assert os.path.isfile(full_filename), 'Bad file ' + full_filename
                movie_id = (
                    get_dir_id(full_filename)
                    if group_by_dir
                    else full_filename
                )
                if movie_id not in result:
                    result[movie_id] = []
                result[movie_id].append(full_filename)
    return result


def parse_xml(filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.gz':
        with gzip.open(filepath, 'r') as f:
            return ET.parse(f)
    else:
        return ET.parse(filepath)


def normalize_whitespaces(sentence):
    return MULTI_WHITESPACES_REGEX.sub(' ', sentence).strip()


def normalize_apostrophe(sentence):
    sentence = normalize_whitespaces(sentence)
    for rule in APOSTROPHE_REPLACEMENT_REGEX:
        sentence = rule[0].sub(rule[1], sentence)
    return sentence


def clean_text(words, lowercase, max_word_length, min_word_length):
    if len(words) > 0 and words[-1] == ':':
        return None
    sentence = ' '.join(words).strip(' -')
    if lowercase:
        sentence = sentence.lower()

    sentence = CLEAN_BRACKETS_REGEX.sub('', sentence)
    if len([ch for ch in BRACKETS_CHARACTERS if ch in sentence]) > 0:
        return None

    sentence = sentence.replace('\\\'', '\'')
    if sentence.count('"') % 2 == 1:
        # There are unmatched double-quotes.
        # Usually, it means a quote got splitted into separate utterances,
        # so it's bad example of a dialog
        return None

    sentence = normalize_apostrophe(sentence)

    for (regex, replacement) in CLEANUP_REGEX_RULES:
        sentence = regex.sub(replacement, sentence)
    for (pattern, replacement) in CLEANUP_REPLACE_RULES:
        sentence = sentence.replace(pattern, replacement)

    words = word_tokenizer.tokenize(sentence)

    if (
        len(words) > 0
        and any(map(lambda k: re.search(r'\w', k) is not None, words))
        and len(words) >= min_word_length
        and len(words) <= max_word_length
    ):
        return ' '.join(words)
    else:
        return None


def parse_time_str(time_value_str):
    if not(
        time_value_str is not None
        and len(time_value_str) == 12
        and time_value_str[2] == ':'
        and time_value_str[5] == ':'
        and time_value_str[8] == ','
    ):
        return None
    try:
        return (
            int(time_value_str[0:2]) * 3600 +
            int(time_value_str[3:5]) * 60 +
            int(time_value_str[6:8])
        )
    except:
        return None


def extract_data_from_xml(
    xml_object,
    lowercase,
    max_word_length,
    min_word_length,
):
    max_time_difference = 1
    previous_end_time = -1000
    previous_sentence = None
    for sentence_node in xml_object.getroot():
        if sentence_node.tag != 's':
            continue

        words = []
        start_time, end_time = None, None

        for node in sentence_node:
            if node.tag == 'time':
                time_value = parse_time_str(node.get('value'))
                if time_value is None:
                    continue
                if node.get('id')[-1] == 'S':
                    start_time = (
                        time_value if start_time is None
                        else min(time_value, start_time)
                    )
                elif node.get('id')[-1] == 'E':
                    end_time = (
                        time_value if end_time is None
                        else max(time_value, end_time)
                    )
                else:
                    raise Exception('Unknown time-id for node: %s' % node)
            elif node.tag == 'w':
                if node.text is not None and len(node.text) > 0:
                    words.append(node.text)
            else:
                pass

        sentence = clean_text(
            words,
            lowercase,
            max_word_length,
            min_word_length,
        )
        if sentence is None:
            continue

        start_time = start_time or previous_end_time
        end_time = end_time or previous_end_time
        if (
            previous_sentence is not None
            and start_time - previous_end_time <= MAX_TIME_DIFFERENCE_S
        ):
            yield (previous_sentence + '\t' + sentence)
        previous_sentence = sentence
        previous_end_time = max(start_time, end_time)


class DataProcessor(object):
    def __init__(self, lowercase, max_word_length, min_word_length):
        self.lowercase = lowercase
        self.max_word_length = max_word_length
        self.min_word_length = min_word_length

    def __call__(self, movie_id_with_files):
        movie_id, files = movie_id_with_files
        data = set()
        files_with_error = []
        for filepath in files:
            try:
                xml_object = parse_xml(filepath)
                for conversation in extract_data_from_xml(
                    xml_object=xml_object,
                    lowercase=self.lowercase,
                    max_word_length=self.max_word_length,
                    min_word_length=self.min_word_length,
                ):
                    data.add(conversation)
            except ET.ParseError as e:
                files_with_error.append((filepath, e))
            except:
                print(
                    'Unexpected error for file %s:\n%s' % (filepath, sys.exc_info()[0]),
                    file=sys.stderr,
                )
                raise
        data_str = '\n'.join(data) + ('\n' if len(data) > 0 else '')
        return (data_str, files_with_error)


def main(args):
    np.random.seed(31415)
    assert os.path.isdir(args.data)

    output_paths = args.out.split(',')
    ratios = [1.0]
    if len(output_paths) > 1:
        ratios = list(map(float, args.ratios.split(',')))
    assert sum(ratios) == 1
    assert len(ratios) == len(output_paths)

    output_files = []
    for i, output_path in enumerate(output_paths):
        assert not os.path.isfile(args.out)
        output_files.append(codecs.open(output_path, 'w', 'utf-8'))
        print(
            'Saving the %.2f%% of the output data to %s' % (
                100.0 * ratios[i],
                output_path,
            )
        )

    start_time = time.time()
    files = get_list_of_files(args.data, group_by_dir=args.deduplicate)
    print(
        'Found %d *.xml.gz movies within %s in %d seconds' % (
            len(files),
            args.data,
            time.time() - start_time,
        )
    )

    processor = DataProcessor(
        lowercase=args.lower,
        max_word_length=args.max_length,
        min_word_length=args.min_length,
    )

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for (conversations, files_with_error) in tqdm.tqdm(
            pool.imap(processor, files.items()),
            total=len(files),
        ):
            for f, ex in files_with_error:
                tqdm.tqdm.write(
                    'Partially processed file %s due to error %s' % (f, ex)
                )
            output_file = np.random.choice(output_files, p=ratios)
            output_file.write(conversations)

    for output_file in output_files:
        output_file.close()
    print('Data has been successfully extracted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument('--data', type=str, help='Path to the dir with Open Subtitles data')
    parser.add_argument('--out', type=str, help='Path to the output files')
    parser.add_argument('--ratios', type=str, help='Probabilities for the output files')
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        default=False,
        help='Deduplicate for subtitles for the same movie',
    )
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--lower', action='store_true', default=False)
    parser.add_argument('--max-length', type=int, default=MAX_WORD_LENGTH)
    parser.add_argument('--min-length', type=int, default=MIN_WORD_LENGTH)
    args = parser.parse_args()

    if ',' in args.out:
        assert args.ratios is not None

    main(args)
