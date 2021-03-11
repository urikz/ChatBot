import argparse
from itertools import product
import logging
import numpy as np
import os
import time
import torch
import tqdm

from data import Dataset
import embeddings
from models import Seq2SeqModel, ProfileMemoryModel
from engine import Engine


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]'
)


SWEEPABLE_ARGS = [
    'learning_rate',
    'momentum',
    'num_layers',
    'dropout',
    'profile_memory_estimation_weight',
]


def get_model_args_list(args, vocab):
    training_args = {
        'device_id': args.gpu,
        'vocab_size': len(vocab),
        'pad_token_id': vocab.pad_idx,
        'unk_token_id': vocab.unk_idx,
        'go_token_id': vocab.go_idx,
        'eos_token_id': vocab.eos_idx,
        'embedding_size': args.embedding_size,
        'hidden_size': args.hidden_size,
        'profile_memory_estimation_weight': args.profile_memory_estimation_weight,
    }

    if args.profile_memory_attention is not None:
        model_cls = ProfileMemoryModel
        training_args['attention_type'] = args.profile_memory_attention
        training_args['use_default_memory'] = args.use_default_memory
        training_args['use_final_attention'] = not (
            len(args.profile_memory_estimation_weight) == 1
            and args.profile_memory_estimation_weight[0] == 0
        )
    else:
        model_cls = Seq2SeqModel
        assert not args.use_default_memory

    training_args_map = {}
    args_dict = dict(args._get_kwargs())
    for p in product(*[args_dict[arg_name] for arg_name in SWEEPABLE_ARGS]):
        args_name = []
        for i, arg_name in enumerate(SWEEPABLE_ARGS):
            training_args[arg_name] = p[i]
            if isinstance(p[i], int):
                args_name.append('%s-%d' % (arg_name, p[i]))
            elif isinstance(p[i], float):
                args_name.append('%s-%.3f' % (arg_name, p[i]))
            else:
                raise Exception('Unknown type for argument: %s' % arg_name)
        training_args_map['.'.join(args_name)] = training_args.copy()

    return model_cls, training_args_map


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = Dataset(args)
    vocab = dataset.get_vocab()

    if not os.path.isdir(args.output):
        assert not os.path.exists(args.output)
        os.makedirs(args.output)

    if args.glove is not None:
        glove_embeddings, glove_index = embeddings.load_glove_embeddings(
            path=args.glove,
            vocab=vocab,
            embedding_size=args.embedding_size,
        )

    model_cls, training_args_map = get_model_args_list(args, vocab)
    logging.info('Sweeper discovered %d different model configurations' % len(
        training_args_map)
    )

    best_valid_ppl = float('inf')
    best_training_args_name = None

    for training_args_name, training_args_value in tqdm.tqdm(
        training_args_map.items(),
        desc='Model Configurations',
        bar_format=TRAINING_TQDM_BAD_FORMAT,
    ):
        model_args_value = training_args_value.copy()
        del model_args_value['learning_rate']
        del model_args_value['momentum']
        del model_args_value['profile_memory_estimation_weight']
        model = model_cls(**model_args_value)

        if args.glove is not None:
            model.set_embeddings(glove_embeddings)
            if (
                args.profile_memory_attention is not None and
                args.init_profile_memory_weights
            ):
                model.init_embeddings_weights_using_glove_index(glove_index)

        engine = Engine(
            model=model,
            vocab=vocab,
            log_interval=None,
            optimizer_params={
                'optim': args.optimizer,
                'learning_rate': training_args_value['learning_rate'],
                'momentum': training_args_value['momentum'],
            },
            verbose=False,
            profile_memory_estimation_weight=training_args_value['profile_memory_estimation_weight']
        )
        engine.set_checkpoint_dir(
            checkpoint_dir=os.path.join(args.output, training_args_name),
            verbose=False,
        )

        valid_ppl = engine.full_training(
            num_epochs=args.num_epochs,
            dataset=dataset,
            verbose=False,
        )
        tqdm.tqdm.write('%s: %.5f' % (training_args_name, valid_ppl))

        if best_valid_ppl > valid_ppl:
            best_valid_ppl = valid_ppl
            best_training_args_name = training_args_name

    logging.info(
        'Sweeping has finished with the best validation ppl %.5f' % (
            best_valid_ppl
        )
    )

    best_checkpoint_path = Engine.get_best_chechpoint(
        os.path.join(args.output, best_training_args_name)
    )
    logging.info(
        'The best checkpoint %s. Picking up the model from there',
        best_checkpoint_path,
    )
    model = model_cls.create_from_checkpoint(best_checkpoint_path, args.gpu)
    engine.model = model
    for corpus, dl in dataset.get_test_and_valid_data_loaders_map().items():
        engine.valid(dl, corpus, use_progress_bar=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    Dataset.add_cmd_arguments(parser)

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Prefix for path to the checkpointing dir.',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID (default - CPU)',
    )
    parser.add_argument(
        '--glove',
        type=str,
        default=None,
        help='Initialize embeddings with GloVe embeddings file',
    )
    parser.add_argument(
        '-optim', '--optimizer',
        type=str,
        default='sgd',
        help='Type of the optimizer. Possible values - sgd,adam',
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        nargs='+',
        type=float,
        default=[1.0],
        help='Learning rate',
    )
    parser.add_argument(
        '-m', '--momentum',
        nargs='+',
        type=float,
        default=[0.0],
        help='Momentum',
    )
    parser.add_argument(
        '-emsz', '--embedding-size',
        type=int,
        help='Dim of the hidden layer of RNN',
    )
    parser.add_argument(
        '-hsz', '--hidden-size',
        type=int,
        help='Dim of the hidden layer of RNN',
    )
    parser.add_argument(
        '-nlayers', '--num-layers',
        nargs='+',
        type=int,
        default=[1],
        help='Number of layers for RNNs',
    )
    parser.add_argument(
        '--profile-memory-estimation-weight',
        nargs='+',
        type=float,
        default=[0],
        help='Weight for profile memory estimation objective',
    )
    parser.add_argument(
        '-dp', '--dropout',
        nargs='+',
        type=float,
        default=[0.0],
        help='Dropout rate',
    )
    parser.add_argument('--seed', type=int, default=31415, help='Random seed')

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1,
        help='Number of epochs',
    )
    parser.add_argument(
        '--profile-memory-attention',
        default=None,
        help=(
            'Attention type for ProfileMemoryModel: concat or general. '
            'If not specified, vanilla Seq2SeqModel will be used'
        ),
    )
    parser.add_argument(
        '--use-default-memory',
        action='store_true',
        default=False,
        help='Use default memory/context',
    )
    parser.add_argument(
        '--init-profile-memory-weights',
        action='store_true',
        default=False,
        help='Initialize ProfileMemory weights using GloVe index',
    )
    args = parser.parse_args()

    if args.embedding_size is None:
        args.embedding_size = args.hidden_size

    assert args.num_epochs > 0

    main(args)
