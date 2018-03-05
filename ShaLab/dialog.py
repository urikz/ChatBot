import argparse
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import os
import sys
import torch
from torch.autograd import Variable

from data.vocab import WordVocabulary
from models.model import DialogModel

def main(args):
    vocab = WordVocabulary.from_file(args.vocab)
    model = DialogModel.create_from_checkpoint(args.model, args.gpu)
    model.eval()
    word_tokenizer = TreebankWordTokenizer()

    sys.stdout.write('YOU > ')
    sys.stdout.flush()

    if args.person is not None:
        person = Variable(model.to_device(torch.LongTensor([args.person])))
    else:
        person = None
    for line in sys.stdin:
        line = line.strip().lower()
        if len(line) != 0:
            words = word_tokenizer.tokenize(line)
            input_source = vocab.numberize(words)
            print(
                '----- PREPROCESSED: ' +
                ' '.join([str(i) for i in vocab.denumberize(input_source)])
            )
            input_source = Variable(model.to_device(
                torch.LongTensor([[x] for x in reversed(input_source)])
            ))
            if args.beam_size == 1:
                _, output_target = model.generate(
                    input_source=input_source,
                    max_length=args.max_length,
                    policy=args.policy,
                    person=person,
                )
                output_words = vocab.denumberize(output_target[:, 0].data)
                sys.stdout.write('BOT > ' + ' '.join(output_words) + '\n')
            else:
                output_targets = model.beam_search(
                    input_source=input_source,
                    max_length=args.max_length,
                    beam_size=args.beam_size,
                    length_normalization_factor=args.length_normalization_factor,
                    length_normalization_const=args.length_normalization_const,
                    person=person,
                )
                for output_score, output_target in output_targets:
                    output_words = vocab.denumberize(output_target)
                    sys.stdout.write(
                        'BOT (%.3f) > %s\n' % (
                            output_score,
                            ' '.join(output_words),
                        )
                    )

        sys.stdout.write('YOU > ')
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot dialog script')
    parser.add_argument(
        '-d', '--data',
        type=str,
        help=(
            'Prefix for path to the data. '
            'When specified, chatbot with take vocabulary file from there'
        ),
    )
    parser.add_argument('--vocab', type=str, help='Path to the vocab file.')
    parser.add_argument('--model', type=str, help='Path to the model file.')
    parser.add_argument('--person', type=int, default=None, help='Person ID')
    parser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size. (Default is 1 - greedy search)',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=20,
        help='Max length',
    )
    parser.add_argument(
        '--length-normalization-factor',
        type=float,
        default=0,
        help='Length normalization factor',
    )
    parser.add_argument(
        '--length-normalization-const',
        type=float,
        default=0,
        help='Length normalization const',
    )
    parser.add_argument('--seed', type=int, default=31415, help='Random seed')
    parser.add_argument(
        '--policy',
        type=str,
        help='sample or greedy',
        default='greedy',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID (default - CPU)',
    )

    args = parser.parse_args()

    assert not(args.vocab is not None and args.data is not None)
    if args.data is not None:
        args.vocab = os.path.join(args.data, 'vocab.tsv')
    assert os.path.isfile(args.vocab)
    assert os.path.isfile(args.model)
    if args.policy == 'greedy':
        assert args.beam_size >= 1
    elif args.policy == 'sample':
        assert args.beam_size == 1
    else:
        raise Exception('Unknown generation policy %s' % args.policy)
    assert args.max_length >= 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
