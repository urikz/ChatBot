import argparse
from nltk.tokenize.treebank import TreebankWordTokenizer
import codecs
import os
import time
import tqdm

from vocab import Vocabulary, WordVocabulary
from corpus import PackedIndexedCorpus, PackedIndexedParallelCorpus

word_tokenizer = TreebankWordTokenizer()

def filter_sentence(
    numberized_sentence,
    unk_token_id,
    max_unk_ratio,
    max_length,
    min_length,
):
    if min_length is not None and len(numberized_sentence) < min_length:
        return False
    if max_length is not None and len(numberized_sentence) > max_length:
        return False
    if max_unk_ratio is None:
        return True
    unk_ratio = (
        float(numberized_sentence.count(unk_token_id))
        / len(numberized_sentence)
    )
    return unk_ratio <= max_unk_ratio


def preprocess_text(text):
    return ' '.join(word_tokenizer.tokenize(text.lower()))


def main(args):
    start_time = time.time()

    if args.person_vocab is not None:
        person_vocab = Vocabulary.from_file(args.person_vocab)
        source_person_corpus, target_person_corpus = [], []
    vocab = WordVocabulary.from_file(args.vocab, size=args.vocab_size)

    source_corpus, target_corpus = [], []
    filtered_sentences = 0
    with codecs.open(args.data, 'r', 'utf-8') as data_f:
        for line in tqdm.tqdm(data_f, desc='Parsing corpus'):
            columns = line[:-1].split('\t')
            if args.person_vocab is not None:
                if len(columns) != 4:
                    tqdm.tqdm.write('Cannot parse line: ' + line[:-1])
                    continue
                else:
                    (
                        source_person,
                        source_sentence,
                        target_person,
                        target_sentence,
                    ) = columns
                    source_person_id = person_vocab.numberize(source_person)
                    target_person_id = person_vocab.numberize(target_person)
            else:
                if len(columns) != 2:
                    tqdm.tqdm.write('Cannot parse line: ' + line[:-1])
                    continue
                else:
                    source_sentence, target_sentence = columns

            if args.preprocess:
                source_sentence = preprocess_text(source_sentence)
            source_sentence_numberized = vocab.numberize(source_sentence)
            assert len(source_sentence_numberized) > 0
            if not filter_sentence(
                source_sentence_numberized,
                vocab.unk_idx,
                args.max_unk_ratio,
                args.max_length,
                None,
            ):
                filtered_sentences += 1
                continue

            if args.preprocess:
                target_sentence = preprocess_text(target_sentence)
            target_sentence_numberized = vocab.numberize(target_sentence)
            assert len(target_sentence_numberized) > 0
            if not filter_sentence(
                target_sentence_numberized,
                vocab.unk_idx,
                args.max_unk_ratio,
                args.max_length,
                args.target_min_length,
            ):
                filtered_sentences += 1
                continue

            source_corpus.append(source_sentence_numberized)
            target_corpus.append(target_sentence_numberized)

            if args.person_vocab is not None:
                source_person_corpus.append(source_person_id)
                target_person_corpus.append(target_person_id)

    assert len(source_corpus) == len(target_corpus)
    if args.person_vocab is not None:
        assert len(source_corpus) == len(source_person_corpus)
        assert len(source_corpus) == len(target_person_corpus)

    print(
        'Numberized %d pairs of sentences (%d were filtered) in %d seconds' % (
            len(source_corpus),
            filtered_sentences,
            time.time() - start_time,
        )
    )

    PackedIndexedCorpus.pack_to_file(source_corpus, args.source_out)
    PackedIndexedCorpus.pack_to_file(target_corpus, args.target_out)
    if args.person_vocab:
        PackedIndexedCorpus.pack_to_file(
            source_person_corpus,
            args.person_source_out,
        )
        PackedIndexedCorpus.pack_to_file(
            target_person_corpus,
            args.person_target_out,
        )

    print('Test loading the corpus')
    parallel_corpus = PackedIndexedParallelCorpus.from_file(
        args.source_out,
        args.target_out,
    )
    assert len(parallel_corpus) == len(source_corpus)

    print('Successfully numberized the parallel corpus in %d seconds' % (
        time.time() - start_time
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the parallel corpus file',
    )
    parser.add_argument('--vocab', type=str, help='Path to the vocabulary file')
    parser.add_argument('--vocab-size', type=int, help='Limit vocabulary size')
    parser.add_argument(
        '--out',
        type=str,
        help='Prefix for the output files',
    )
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
        '--target-min-length',
        type=int,
        default=None,
        help='Minimal length of the sentence',
    )
    parser.add_argument(
        '--person-vocab',
        type=str,
        default=None,
        help='Path to the person vocabulary file',
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        default=False,
        help='Additional preprocessing and lower-casing',
    )

    args = parser.parse_args()

    args.source_out = args.out + '.source.npz'
    args.target_out = args.out + '.target.npz'
    assert not os.path.isfile(args.source_out)
    assert not os.path.isfile(args.target_out)

    if args.person_vocab:
        assert os.path.isfile(args.person_vocab)
        args.person_source_out = args.out + '.person.source.npz'
        args.person_target_out = args.out + '.person.target.npz'
        assert not os.path.isfile(args.person_source_out)
        assert not os.path.isfile(args.person_target_out)

    assert os.path.isfile(args.data)
    assert os.path.isfile(args.vocab)

    main(args)
