import argparse
from nltk.tokenize.treebank import TreebankWordTokenizer
import codecs
import os
import time
import tqdm

from ShaLab.data import WordVocabulary, DialogCorpus


word_tokenizer = TreebankWordTokenizer()


def filter_sentence(
    numberized_sentence,
    unk_token_id,
    max_unk_ratio,
    max_length,
):
    if max_length is not None and len(numberized_sentence) > max_length:
        return False
    if max_unk_ratio is None:
        return True
    unk_ratio = (
        float(numberized_sentence.count(unk_token_id))
        / len(numberized_sentence)
    )
    return unk_ratio <= max_unk_ratio


def main(args):
    start_time = time.time()
    vocab = WordVocabulary.from_file(args.vocab)
    dialogs = []
    filtered_sentences, dialog_sentences = 0, 0
    original_dialogs, original_sentences = 0, 0

    with codecs.open(args.data, 'r', 'utf-8') as data_f:
        for line in tqdm.tqdm(data_f, desc='Parsing corpus'):
            dialog = line[:-1].split('\t')
            if len(dialog) < 2:
                tqdm.tqdm.write('Cannot parse line: ' + line[:-1])
                continue
            original_dialogs += 1

            if args.preprocess:
                dialog = map(
                    lambda x: ' '.join(word_tokenizer.tokenize(x.lower())),
                    dialog,
                )
            dialog = list(map(lambda x: vocab.numberize(x), dialog))
            original_sentences += len(dialog)
            should_keep = map(
                lambda x: filter_sentence(
                    x,
                    vocab.unk_idx,
                    args.max_unk_ratio,
                    args.max_length,
                ),
                dialog,
            )

            current_dialog = []
            for d, k in zip(dialog, should_keep):
                if k:
                    current_dialog.append(d)
                else:
                    filtered_sentences += 1
                    if len(current_dialog) > 1:
                        dialogs.append(current_dialog)
                        dialog_sentences += len(current_dialog)
                    current_dialog = []
            if len(current_dialog) > 1:
                dialogs.append(current_dialog)
                dialog_sentences += len(current_dialog)

    print(
        (
            'Numberized %d dialogs (%d were originally), '
            '%d sentences (%d were originally, %d were filtered) '
            'in %d seconds'
        ) % (
            len(dialogs),
            original_dialogs,
            dialog_sentences,
            original_sentences,
            filtered_sentences,
            time.time() - start_time,
        )
    )

    DialogCorpus.pack(dialogs).to_file(args.out)

    print('Test loading the corpus')
    dialog_corpus = DialogCorpus.from_file(args.out)

    assert len(dialog_corpus) == dialog_sentences - len(dialogs)

    print('Successfully numberized the parallel corpus in %d seconds' % (
        time.time() - start_time
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the dialogs corpus text file',
    )
    parser.add_argument('--vocab', type=str, help='Path to the vocabulary file')
    parser.add_argument('--out', type=str, help='Output file')
    parser.add_argument(
        '--max-unk-ratio',
        type=float,
        help='Maximal ratio of unknown words in the sentence',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Maximal length of the sentence',
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        default=False,
        help='Additional preprocessing and lower-casing',
    )

    args = parser.parse_args()

    assert not os.path.isfile(args.out)
    assert os.path.isfile(args.data)
    assert os.path.isfile(args.vocab)

    main(args)
