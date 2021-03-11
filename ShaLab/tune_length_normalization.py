import argparse
from itertools import product
import logging
import nltk
import numpy as np
import os
import time
import tqdm
import torch

from ShaLab.data import (
    DialogCorpusWithProfileMemory,
    DialogCorpus,
    WordVocabulary,
    prepare_batch,
)
from ShaLab.engine import Engine
from ShaLab.models import (
    Seq2SeqModel,
    ProfileMemoryModel,
    create_from_checkpoint,
)
from ShaLab.agents import ModelBasedAgent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def load_corpus(corpus_path):
    try:
        return DialogCorpusWithProfileMemory.from_file(corpus_path)
    except:
        return DialogCorpus.from_file(corpus_path)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vocab = WordVocabulary.from_file(args.vocab)
    corpus = load_corpus(args.corpus)
    model = create_from_checkpoint(args.model, args.gpu)
    model.eval()

    for lnf, lnc in product(
        args.length_normalization_factor,
        args.length_normalization_const,
    ):
        agent = ModelBasedAgent(
            model=model,
            max_length=args.max_length,
            policy='greedy',
            num_candidates=args.beam_size,
            length_normalization_factor=lnf,
            length_normalization_const=lnc,
            context=None,
        )

        list_of_references = []
        hypotheses = []
        for i in tqdm.tqdm(range(0, len(corpus), args.step)):
            item = corpus[i]
            if 'profile_memory' in item:
                agent.set_context([item['profile_memory']])
            input_source = model.to_device(
                torch.LongTensor([[x] for x in item['input_source']])
            )
            list_of_references.append(
                [item['input_target'].tolist()]
            )
            hypotheses.append(
                agent.generate(input_source)[0].output[:, 0].cpu().numpy().tolist(),
            )

        print('LN factor = %.3f, LN const = %.3f, BLEU = %.3f' % (
            lnf,
            lnc,
            100.0 * nltk.translate.bleu_score.corpus_bleu(
                list_of_references,
                hypotheses,
            ),
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot dialog script')
    parser.add_argument('--vocab', type=str, help='Path to the vocab file.')
    parser.add_argument(
        '--corpus',
        type=str,
        help='Path to the corpus file.',
    )
    parser.add_argument('--model', type=str, help='Path to the model file.')
    parser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam size. (Default is 1 - greedy search)',
    )
    parser.add_argument('--max-length', type=int, default=20, help='Max length')
    parser.add_argument(
        '--length-normalization-factor',
        type=float,
        default=[0],
        nargs='+',
        help='Length normalization factors',
    )
    parser.add_argument(
        '--length-normalization-const',
        type=float,
        default=[0],
        nargs='+',
        help='Length normalization consts',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--step', type=int, default=1, help='step')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID (default - CPU)',
    )
    args = parser.parse_args()

    assert args.vocab is not None
    assert args.corpus is not None
    if os.path.isdir(args.model):
        args.model = Engine.get_best_chechpoint(args.model)
    assert os.path.isfile(args.model)
    assert args.beam_size > 0
    assert args.max_length >= 1

    if args.seed is None:
        args.seed = int(time.time())
    logging.info('Random seed: %d' % args.seed)

    with torch.no_grad():
        main(args)
