import argparse
import codecs
from collections import Counter
import csv
from nltk.tokenize.treebank import TreebankWordTokenizer
import os
import time

from ShaLab.data import DialogCorpusWithProfileMemory, WordVocabulary


YOUR_PERSONA_STR = "your persona: "
PARTNERS_PERSONA_STR = "partner's persona: "


word_tokenizer = TreebankWordTokenizer()


def build_vocabulary(paths):
    print('Building vocabulary from %s' % ','.join(paths))
    sentences = []
    for path in paths:
        start_time = time.time()
        print('... extracting words from file %s' % path)
        with codecs.open(path, 'r', 'utf-8', errors='ignore') as f:
            for line in f:
                space_idx = line.find(' ')
                l = line[space_idx + 1:-1].lower().strip()

                if l.startswith(YOUR_PERSONA_STR):
                    sentences.append(word_tokenizer.tokenize(
                        l[len(YOUR_PERSONA_STR):],
                    ))
                    continue

                if l.startswith(PARTNERS_PERSONA_STR):
                    sentences.append(word_tokenizer.tokenize(
                        l[len(PARTNERS_PERSONA_STR):],
                    ))
                    continue

                source_line, target_line, _, candidates = l.split('\t')
                sentences.append(word_tokenizer.tokenize(source_line))
                candidates = candidates.split('|')
                assert len(candidates) > 1
                assert target_line in candidates
                for c in candidates:
                    sentences.append(word_tokenizer.tokenize(c))

        print('..... finished in %d seconds' % (time.time() - start_time))

    print('Total number of sentences: %d' % len(sentences))
    vocab = WordVocabulary.from_corpus(sentences)
    return vocab


def verbose_numberize_dialogs(dialogs, vocab):
    unk_counter = Counter()
    numerized_dialogs = []
    total_words, total_unks = 0, 0
    for dialog in dialogs:
        numerized_dialog = []
        for sentence in dialog:
            unks = [word for word in sentence if word not in vocab]
            total_unks += len(unks)
            unk_counter.update(unks)
            words = vocab.numberize(sentence)
            numerized_dialog.append(words)
            total_words += len(words)
        numerized_dialogs.append(numerized_dialog)
    if len(unk_counter) > 0:
        print('Found %d unique unknown words (%.2f%% = %d/%d):' % (
            len(unk_counter),
            100.0 * total_unks / total_words,
            total_unks,
            total_words,
        ))
        # for k, v in unk_counter.most_common():
        #     print('--- %s\t%d' % (k, v))
    return numerized_dialogs


def prepare_file(path, output_prefix, vocab):
    start_time = time.time()
    print('Preparing file %s' % path)

    dialogs, profile_memory = [], []
    current_dialog = []
    source_profile_memories, target_profile_memories = [], []

    def flush():
        nonlocal dialogs, profile_memory, current_dialog
        nonlocal source_profile_memories, target_profile_memories
        if len(current_dialog) > 0:
            dialogs.append(current_dialog)
            assert len(source_profile_memories) in [3, 4, 5], len(source_profile_memories)
            assert len(target_profile_memories) in [3, 4, 5], len(target_profile_memories)
            profile_memory.extend([
                source_profile_memories,
                target_profile_memories,
            ])
            current_dialog = []
            source_profile_memories = []
            target_profile_memories = []

    with codecs.open(path, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
            space_idx = line.find(' ')
            if int(line[:space_idx]) == 1:
                flush()

            l = line[space_idx + 1:-1].lower().strip()

            if l.startswith(YOUR_PERSONA_STR):
                source_profile_memories.append(
                    word_tokenizer.tokenize(l[len(YOUR_PERSONA_STR):])
                )
                continue

            if l.startswith(PARTNERS_PERSONA_STR):
                target_profile_memories.append(
                    word_tokenizer.tokenize(l[len(PARTNERS_PERSONA_STR):])
                )
                continue

            source_line, target_line, _, _ = line[space_idx + 1:-1].split('\t')
            source_line = word_tokenizer.tokenize(source_line.strip())
            target_line = word_tokenizer.tokenize(target_line.strip())
            current_dialog.extend([source_line, target_line])

    flush()

    print('Parsed file %s in %d seconds' % (path, time.time() - start_time))

    DialogCorpusWithProfileMemory.pack(
        verbose_numberize_dialogs(dialogs, vocab),
        verbose_numberize_dialogs(profile_memory, vocab)
    ).to_file(output_prefix + '.npz')

    DialogCorpusWithProfileMemory.from_file(output_prefix + '.npz')


def main(args):
    assert os.path.isdir(args.data)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    vocab_path = os.path.join(args.out, 'vocab.tsv')
    if os.path.isfile(vocab_path):
        vocab = WordVocabulary.from_file(vocab_path)
    else:
        vocab = build_vocabulary([
            os.path.join(args.data, 'train_both_original.txt'),
            # TODO: we probably should use a joint vocabulary
            # for the proper training
            # os.path.join(args.data, 'train_both_revised.txt'),
        ])
        vocab.to_file(vocab_path)

    prepare_file(
        os.path.join(args.data, 'valid_both_original.txt'),
        output_prefix=os.path.join(args.out, 'valid'),
        vocab=vocab,
    )
    prepare_file(
        os.path.join(args.data, 'valid_both_revised.txt'),
        output_prefix=os.path.join(args.out, 'valid.revised'),
        vocab=vocab,
    )
    prepare_file(
        os.path.join(args.data, 'test_both_original.txt'),
        output_prefix=os.path.join(args.out, 'test.original'),
        vocab=vocab,
    )
    prepare_file(
        os.path.join(args.data, 'test_both_revised.txt'),
        output_prefix=os.path.join(args.out, 'test.revised'),
        vocab=vocab,
    )

    prepare_file(
        os.path.join(args.data, 'train_both_original.txt'),
        os.path.join(args.out, 'train'),
        vocab=vocab,
    )
    # TODO: Shall we use both original and revised for training?
    # prepare_file(
    #     os.path.join(args.data, 'train_both_revised.txt'),
    #     os.path.join(args.out, 'train.revised'),
    #     vocab=vocab,
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument('--data', type=str, help='Path to the unzipped corpus folder')
    parser.add_argument('--out', type=str, help='Path to the preprocessed folder')
    args = parser.parse_args()

    main(args)
