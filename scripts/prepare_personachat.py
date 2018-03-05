import argparse
import codecs
from collections import Counter
import csv
from autocorrect import spell
from nltk.tokenize.treebank import TreebankWordTokenizer
import os
import pandas
import time

from ShaLab.data.corpus import PackedIndexedCorpus
from ShaLab.data.vocab import WordVocabulary


YOUR_PERSONA_STR = "your persona: "
PARTNERS_PERSONA_STR = "partner's persona: "


word_tokenizer = TreebankWordTokenizer()


# embbeddings = pandas.read_table(
#     '/home/urikz/word_vectors/glove.6B.50d.txt',
#     sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE, skiprows=0)


def prepare_line(line, autocorrect, vocab=None):
    line = line.lower().strip()
    assert len(line) > 0
    words = []
    for w in word_tokenizer.tokenize(line):
        if autocorrect:
            if vocab is not None:
                if w not in vocab:
                    w = spell(w).lower()
            # TODO: that was a hack to use autocorrect carefully,
            # basically, apply autocorrect only if the word is not in
            # embeddings file.
            # Otherwise, the autocorrect is too agressive
            # else:
            #     if w not in embbeddings.index:
            #         w = spell(w).lower()
        words.append(w)
    return words


def build_vocabulary(paths, autocorrect):
    print('Building vocabulary from %s' % ','.join(paths))

    sentences = []
    for path in paths:
        start_time = time.time()
        print('... extracting words from file %s' % path)
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as f:
            for line in f:
                space_idx = line.find(' ')
                l = line[space_idx + 1:-1]

                if l.startswith(YOUR_PERSONA_STR):
                    sentences.append(prepare_line(
                        l[len(YOUR_PERSONA_STR):],
                        autocorrect=autocorrect,
                    ))
                    continue

                if l.startswith(PARTNERS_PERSONA_STR):
                    sentences.append(prepare_line(
                        l[len(PARTNERS_PERSONA_STR):],
                        autocorrect=autocorrect,
                    ))
                    continue

                source_line, target_line, _, candidates = l.split('\t')
                sentences.append(prepare_line(source_line, autocorrect=autocorrect))
                candidates = candidates.split('|')
                assert len(candidates) > 1
                assert target_line in candidates
                for c in candidates:
                    sentences.append(prepare_line(c, autocorrect=autocorrect))

        print('..... finished in %d seconds' % (time.time() - start_time))

    print('Total number of sentences: %d' % len(sentences))
    vocab = WordVocabulary.from_corpus(sentences)
    return vocab


def verbose_numberize_corpus(corpus, vocab):
    unk_counter = Counter()
    numerized_corpus = []
    total_words, total_unks = 0, 0
    for sentence in corpus:
        unks = [word for word in sentence if word not in vocab]
        total_unks += len(unks)
        unk_counter.update(unks)
        words = vocab.numberize(sentence)
        numerized_corpus.append(words)
        total_words += len(words)
    if len(unk_counter) > 0:
        print('Found %d unique unknown words (%.2f%% = %d/%d):' % (
            len(unk_counter),
            100.0 * total_unks / total_words,
            total_unks,
            total_words,
        ))
        # for k, v in unk_counter.most_common():
        #     print('--- %s\t%d' % (k, v))
    return numerized_corpus


def prepare_file(path, output_prefix, autocorrect, vocab):
    start_time = time.time()
    print('Preparing file %s' % path)
    previous_target_line = None
    source, target = [], []
    output_debug_file = codecs.open(output_prefix + '.tsv', 'w', 'utf-8')

    with codecs.open(path, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
            space_idx = line.find(' ')
            line_id = int(line[:space_idx])
            if line_id == 1:
                previous_target_line = None

            source_line, target_line, _, _ = line[space_idx + 1:-1].split('\t')
            source_line = prepare_line(source_line, autocorrect, vocab)
            target_line = prepare_line(target_line, autocorrect, vocab)

            if previous_target_line is not None:
                source.append(previous_target_line)
                target.append(source_line)
                output_debug_file.write('%s\t%s\n' % (
                    ' '.join(previous_target_line),
                    ' '.join(source_line),
                ))

            source.append(source_line)
            target.append(target_line)
            output_debug_file.write('%s\t%s\n' % (
                ' '.join(source_line),
                ' '.join(target_line),
            ))

            previous_target_line = target_line

    output_debug_file.close()
    print('Parsed file %s in %d seconds' % (path, time.time() - start_time))

    PackedIndexedCorpus.pack_to_file(
        verbose_numberize_corpus(source, vocab),
        output_prefix + '.source.npz',
    )
    PackedIndexedCorpus.pack_to_file(
        verbose_numberize_corpus(target, vocab),
        output_prefix + '.target.npz',
    )


def main(args):
    assert os.path.isdir(args.data)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    vocab_path = os.path.join(args.out, 'vocab.tsv')
    if os.path.isfile(vocab_path):
        vocab = WordVocabulary.from_file(vocab_path)
    else:
        vocab = build_vocabulary(
            [
                os.path.join(args.data, 'train_both_original.txt'),
                # os.path.join(args.data, 'train_both_revised.txt'),
            ],
            autocorrect=False,
        )
        vocab.to_file(vocab_path)

    prepare_file(
        os.path.join(args.data, 'valid_none_original.txt'),
        os.path.join(args.out, 'tune'),
        autocorrect=False,
        vocab=vocab,
    )
    prepare_file(
        os.path.join(args.data, 'test_none_original.txt'),
        os.path.join(args.out, 'tune-test'),
        autocorrect=False,
        vocab=vocab,
    )
    # No longer needed as we're not using autocorrect in the first place
    # prepare_file(
    #     os.path.join(args.data, 'valid_none_original.txt'),
    #     os.path.join(args.out, 'tune-no-autocorrect'),
    #     autocorrect=False,
    #     vocab=vocab,
    # )
    # prepare_file(
    #     os.path.join(args.data, 'test_none_original.txt'),
    #     os.path.join(args.out, 'tune-test-no-autocorrect'),
    #     autocorrect=False,
    #     vocab=vocab,
    # )
    prepare_file(
        os.path.join(args.data, 'train_none_original.txt'),
        os.path.join(args.out, 'train'),
        autocorrect=False,
        vocab=vocab,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument('--data', type=str, help='Path to the unzipped corpus folder')
    parser.add_argument('--out', type=str, help='Path to the preprocessed folder')
    args = parser.parse_args()

    main(args)
