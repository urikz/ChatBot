import logging

from .seq2seq_model import Seq2SeqModel
from .profile_memory_model import ProfileMemoryModel
from .single_attention_profile_memory_model import (
    SingleAttentionProfileMemoryModel
)
from .generator import Generator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def create_from_checkpoint(path, gpu):
    try:
        return ProfileMemoryModel.create_from_checkpoint(path, gpu)
    except:
        try:
            return Seq2SeqModel.create_from_checkpoint(path, gpu)
        except:
            return None


class ModelFactory(object):
    @staticmethod
    def model(args, vocab):
        model_args = {
            'device_id': args.gpu,
            'vocab_size': len(vocab),
            'pad_token_id': vocab.pad_idx,
            'unk_token_id': vocab.unk_idx,
            'go_token_id': vocab.go_idx,
            'eos_token_id': vocab.eos_idx,
            'num_layers': args.num_layers,
            'embedding_size': args.embedding_size,
            'hidden_size': args.hidden_size,
            'dropout': args.dropout,
        }

        if args.model_type == 'seq2seq':
            assert args.profile_memory_attention is None
            assert args.use_default_memory is None
            assert args.profile_memory_estimation_weight == 0
            model_cls = Seq2SeqModel
        elif args.model_type == 'profile-memory':
            assert args.profile_memory_attention is not None
            model_args['attention_type'] = args.profile_memory_attention
            model_args['use_default_memory'] = args.use_default_memory
            if args.profile_memory_estimation_weight > 0:
                model_args['use_final_attention'] = True
            model_cls = ProfileMemoryModel
        elif args.model_type == 'single-attention-profile-memory':
            assert args.profile_memory_attention is not None
            model_args['attention_type'] = args.profile_memory_attention
            model_args['use_default_memory'] = args.use_default_memory
            model_cls = SingleAttentionProfileMemoryModel
        else:
            raise Exception('Unknown model type: %s' % args.model_type)
        return model_cls(**model_args)

    @staticmethod
    def add_cmd_arguments(parser):
        parser.add_argument(
            '--model-type',
            type=str,
            help='seq2seq, profile-memory or single-attention-profile-memory',
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
            type=int,
            default=1,
            help='Number of layers for RNNs',
        )
        parser.add_argument(
            '-dp', '--dropout',
            type=float,
            default=0,
            help='Dropout rate',
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
