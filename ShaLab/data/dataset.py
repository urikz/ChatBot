import argparse
import glob
import os

from .vocab import WordVocabulary
from .corpus import (
    ConcatDataset,
    DialogCorpus,
    DialogCorpusWithProfileMemory,
    data_loader,
)


def load_profile_memory_from_files(paths):
    if len(paths) == 1:
        return DialogCorpusWithProfileMemory.from_file(
            paths[0]
        ).profile_memory.sentences
    else:
        return ConcatDataset([
            DialogCorpusWithProfileMemory.from_file(
                path
            ).profile_memory.sentences
            for path in paths
        ])


def load_corpora_from_files(corpus_cls, paths):
    datasets = [corpus_cls.from_file(path) for path in paths]
    if len(datasets) > 0:
        return ConcatDataset(datasets)
    else:
        return datasets[0]


class Dataset(object):

    VALID_CORPUS_DEFAULT_NAME = 'valid'

    def _path(self, filename):
        return os.path.join(self.data_dir, filename)

    def _get_corpora_map(self, corpus_cls, prefix, force_corpus_cls):
        datasets = {}
        for c in glob.glob(self._path(prefix + '*.npz')):
            dataset_name = os.path.basename(c)[:-len('.npz')]
            try:
                d = corpus_cls.from_file(c)
            except:
                if force_corpus_cls:
                    continue
                d = DialogCorpus.from_file(c)
            datasets[dataset_name] = d
        return datasets

    def _get_train_corpus(
        self,
        corpus_class,
        num_fake_memories=None,
        false_opponent_profile_memory=None,
    ):
        main_train_corpus = corpus_class.from_file(self._path('train.npz'))
        main_train_corpus.set_num_random_memories(num_fake_memories)
        if false_opponent_profile_memory is not None:
            main_train_corpus.false_opponent_profile_memory = false_opponent_profile_memory

        datasets = []
        for c in glob.glob(self._path('train?*.npz')):
            dataset_name = os.path.basename(c)[:-len('.npz')]
            try:
                d = corpus_class.from_file(c)
                d.set_num_random_memories(num_fake_memories)
            except:
                d = DialogCorpus.from_file(c)
                d = corpus_class(
                    sentences=d.sentences,
                    dialog_start_index=d.dialog_start_index,
                    profile_memory=main_train_corpus.profile_memory,
                    is_profile_memory_fake=True,
                )
                # TODO: we are not using non-personalized samples
                # to perform a profile memory estimations
                # Thus, we don't need to fetch too many random samples
                d.set_num_random_memories(5 + num_fake_memories)

            if false_opponent_profile_memory is not None:
                d.false_opponent_profile_memory = false_opponent_profile_memory

            datasets.append(d)
        if len(datasets) > 0:
            return ConcatDataset([main_train_corpus] + datasets)
        else:
            return main_train_corpus

    def __init__(self, args):
        assert os.path.isdir(args.data)
        self.data_dir = args.data
        self.vocab = WordVocabulary.from_file(self._path('vocab.tsv'))

        corpus_class = (
            DialogCorpusWithProfileMemory
            if args.profile_memory_attention is not None
            else DialogCorpus
        )

        self.train_corpus = self._get_train_corpus(
            corpus_class=corpus_class,
            num_fake_memories=args.num_false_memories,
            false_opponent_profile_memory=args.false_opponent_profile_memory
        )
        self.valid_corpora = self._get_corpora_map(
            corpus_class,
            'valid',
            force_corpus_cls=not args.use_default_memory,
        )
        assert Dataset.VALID_CORPUS_DEFAULT_NAME in self.valid_corpora
        self.test_corpora = self._get_corpora_map(
            corpus_class,
            'test',
            force_corpus_cls=not args.use_default_memory,
        )

        self.batch_size = args.batch_size
        self.valid_batch_size = (
            args.valid_batch_size
            if args.valid_batch_size is not None
            else self.batch_size
        )
        self.sort_batches = args.sort_batches
        self.num_data_workers = args.num_data_workers

    def get_train_data_loader(self, verbose=True):
        return data_loader(
            corpus=self.train_corpus,
            vocab=self.vocab,
            batch_size=self.batch_size,
            sort_batches=self.sort_batches,
            num_data_workers=self.num_data_workers,
            verbose=verbose,
        )

    def get_valid_data_loaders_map(self, verbose=False):
        return {
            corpus_name: data_loader(
                corpus=corpus,
                vocab=self.vocab,
                batch_size=self.valid_batch_size,
                sort_batches=True,
                num_data_workers=self.num_data_workers,
                verbose=verbose,
            )
            for corpus_name, corpus in self.valid_corpora.items()
        }

    def get_test_data_loaders_map(self, verbose=False):
        return {
            corpus_name: data_loader(
                corpus=corpus,
                vocab=self.vocab,
                batch_size=self.valid_batch_size,
                sort_batches=True,
                num_data_workers=self.num_data_workers,
                verbose=verbose,
            )
            for corpus_name, corpus in self.test_corpora.items()
        }

    def get_test_and_valid_data_loaders_map(self, verbose=False):
        data_loaders_map = self.get_valid_data_loaders_map(verbose)
        data_loaders_map.update(self.get_test_data_loaders_map(verbose))
        return data_loaders_map

    def get_vocab(self):
        return self.vocab

    @classmethod
    def add_cmd_arguments(cls, argparser):
        argparser.add_argument(
            '-d', '--data',
            type=str,
            help=(
                'Prefix for path to the data. '
                'It should contains at least these files: '
                'vocab.tsv, train.npz, eval-valid.npz'
            ),
        )
        argparser.add_argument(
            '--num-data-workers',
            type=int,
            default=1,
            help='Number of workers for data loader',
        )
        argparser.add_argument(
            '--num-false-memories',
            type=int,
            default=0,
            help='Number of false (random) memories',
        )
        argparser.add_argument(
            '--sort-batches',
            action='store_true',
            default=False,
            help=(
                'Will sort batches by length '
                'to group sentences with similar length. '
                'Use this option for larger (>1M pair of sentences) datasets.'
            ),
        )
        argparser.add_argument(
            '-bs', '--batch-size',
            type=int,
            default=32,
            help='Batch size',
        )
        argparser.add_argument(
            '-vbs', '--valid-batch-size',
            type=int,
            default=None,
            help=(
                'Batch size for validation '
                '(default is the same as the regular batch size)'
            ),
        )
        argparser.add_argument(
            '--false-opponent-profile-memory',
            type=int,
            default=None,
            help=(
                'Number of negative exampels for '
                'opponent profile memory prediction'
            ),
        )
