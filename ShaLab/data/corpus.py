import logging
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class PackedIndexedCorpus(Dataset):
    def __init__(self):
        self.data = None
        self.index_start = None
        self.index_length = None

    def __getitem__(self, index):
        return self.data[
            self.index_start[index]:
            self.index_start[index] + self.index_length[index]
        ]

    def __len__(self):
        return self.index_start.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_file(self, path):
        start_time = time.time()
        np.savez(
            path,
            data=self.data,
            index_start=self.index_start,
            index_length=self.index_length,
        )
        logging.info(
            'Serialized PackedIndexedCorpus to %s in %d seconds',
            path,
            time.time() - start_time,
        )

    def sample_random(self):
        return self.__getitem__(np.random.randint(len(self)))

    @staticmethod
    def from_file(path):
        start_time = time.time()
        corpus = PackedIndexedCorpus()
        f = open(path, 'rb')
        raw_data = np.load(f)
        corpus.data = raw_data['data']
        corpus.index_start = raw_data['index_start']
        corpus.index_length = raw_data['index_length']
        f.close()
        assert corpus.index_start.shape[0] == corpus.index_length.shape[0]
        logging.info(
            'Loaded %d sentences from file %s in %d seconds',
            len(corpus),
            path,
            time.time() - start_time,
        )
        return corpus

    @staticmethod
    def pack(sentences):
        start_time = time.time()
        corpus = PackedIndexedCorpus()
        corpus.index_start = np.zeros(len(sentences), dtype=np.int32)
        corpus.index_length = np.zeros(len(sentences), dtype=np.int32)
        total_length = 0
        for i, sentence in enumerate(sentences):
            corpus.index_start[i] = total_length
            if isinstance(sentence, np.ndarray):
                assert sentence.ndim == 1
                corpus.index_length[i] = sentence.size
            elif isinstance(sentence, list):
                corpus.index_length[i] = len(sentence)
            else:
                raise Exception('Unknown type for sentence: ' + type(sentence))
            total_length += corpus.index_length[i]
        corpus.data = np.zeros(total_length, dtype=np.int32)
        for i, sentence in enumerate(sentences):
            corpus.data[
                corpus.index_start[i]:
                corpus.index_start[i] + corpus.index_length[i]
            ] = sentence
        logging.info(
            'Built a PackedIndexedCorpus from %d sentences in %d seconds',
            len(corpus),
            time.time() - start_time,
        )
        return corpus

    @staticmethod
    def pack_to_file(sentences, path):
        PackedIndexedCorpus.pack(sentences).to_file(path)


class PackedIndexedParallelCorpus(Dataset):
    def __init__(self):
        self.source_corpus = None
        self.target_corpus = None
        self.target_person_corpus = None

    def has_person_feature(self):
        return self.target_person_corpus is not None

    def __getitem__(self, index):
        if self.target_person_corpus is None:
            return (
                self.source_corpus[index],
                self.target_corpus[index],
                self.target_corpus[index],
            )
        else:
            return (
                self.source_corpus[index],
                self.target_corpus[index],
                self.target_person_corpus[index],
                self.target_corpus[index],
            )

    def __len__(self):
        return len(self.source_corpus)

    def __iter__(self):
        for i in len(self):
            yield self[i]

    @staticmethod
    def from_file(source_path, target_path, target_person_path=None):
        corpus = PackedIndexedParallelCorpus()
        corpus.source_corpus = PackedIndexedCorpus.from_file(source_path)
        corpus.target_corpus = PackedIndexedCorpus.from_file(target_path)
        assert len(corpus.source_corpus) == len(corpus.target_corpus)
        if target_person_path is not None:
            corpus.target_person_corpus = PackedIndexedCorpus.from_file(
                target_person_path,
            )
            assert len(corpus.source_corpus) == len(corpus.target_person_corpus)
        return corpus


def prepare_batch(
    samples,
    pad_token_id,
    bos_token_id=None,
    eos_token_id=None,
    place_pad_token_first=False,
    reverse=False,
):
    max_samples_length = max([sample.shape[0] for sample in samples])
    total_length = (
        (1 if bos_token_id is not None else 0) +
        max_samples_length +
        (1 if eos_token_id is not None else 0)
    )
    batch = np.full((len(samples), total_length), pad_token_id, dtype=np.int64)
    index_direction = 1 if not reverse else -1
    for i in range(len(samples)):
        current_length = samples[i].shape[0]
        if place_pad_token_first:
            index = max_samples_length - current_length
        else:
            index = 0

        if bos_token_id is not None:
            batch[i][index] = bos_token_id
            index += 1

        batch[i][index:index + current_length] = (
            samples[i][::index_direction]
        )
        index += current_length
        if eos_token_id is not None:
            batch[i][index] = eos_token_id
    return torch.from_numpy(batch.transpose().copy())


def prepare_batch_from_parallel_samples(
    parallel_samples,
    pad_token_id,
    eos_token_id,
    go_token_id,
    reverse_source=False,
    use_person_feature=False,
):
    if use_person_feature:
        return (
            prepare_batch(
                samples=[sample[0] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                place_pad_token_first=True,
                reverse=reverse_source,
            ),
            prepare_batch(
                samples=[sample[1] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                bos_token_id=go_token_id,
            ),
            torch.from_numpy(
                np.array([sample[2] for sample in parallel_samples], dtype=np.int64).reshape(-1)
            ),
            prepare_batch(
                samples=[sample[3] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            ),
        )
    else:
        return (
            prepare_batch(
                samples=[sample[0] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                place_pad_token_first=True,
                reverse=reverse_source,
            ),
            prepare_batch(
                samples=[sample[1] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                bos_token_id=go_token_id,
            ),
            prepare_batch(
                samples=[sample[2] for sample in parallel_samples],
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            ),
        )


def reverse_batch(batch, device_id):
    inv_idx = torch.arange(batch.size(0) - 1, -1, -1).long().cuda(device_id)
    return batch.index_select(0, inv_idx)


def insert_go_token(batch, go_token_id, device_id):
    data = batch.cpu().numpy().transpose()
    result = np.zeros((batch.size(1), batch.size(0) + 1), dtype=np.int64)
    for i in range(batch.size(1)):
        word_indexes = np.nonzero(data[i])[0]
        result[i][0] = go_token_id
        result[i][1:word_indexes.size + 1] = data[i][word_indexes]
    return Variable(torch.from_numpy(result.transpose().copy())).cuda(device_id)


def append_eos_token(batch, eos_token_id, device_id):
    data = batch.cpu().numpy().transpose()
    result = np.zeros((batch.size(1), batch.size(0) + 1), dtype=np.int64)
    for i in range(batch.size(1)):
        word_indexes = np.nonzero(data[i])[0]
        result[i][:word_indexes.size] = data[i][word_indexes]
        result[i][word_indexes.size] = eos_token_id
    return Variable(torch.from_numpy(result.transpose().copy())).cuda(device_id)


class SortedBatchSampler(object):

    def __init__(
        self,
        source_lengths,
        target_lengths,
        batch_size,
    ):
        start_time = time.time()
        assert len(source_lengths) == len(target_lengths)
        lengths = np.core.records.fromarrays(
            [
                source_lengths,
                target_lengths,
                np.random.randint(
                    1000000,
                    size=len(source_lengths),
                    dtype=np.int32,
                ),
            ],
            names='s,t,r',
        )
        indices = np.argsort(lengths, order=('s', 't', 'r'), kind='mergesort')
        self.batches = np.split(
            indices[:-(len(indices) % batch_size)],
            len(indices) // batch_size,
        )
        np.random.shuffle(self.batches)
        logging.info(
            (
                'Shuffling the dataset (%d pairs of sentences, %d batches) '
                'is completed in %d seconds'
            ),
            len(source_lengths),
            len(self.batches),
            time.time() - start_time,
        )

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def data_loader(
    corpus,
    vocab,
    batch_size,
    sort_batches,
    reverse_source,
    num_data_workers=0,
    verbose=True,
):
    if sort_batches:
        if verbose:
            logging.info(
                'Using sort batch sampler. '
                'WARNING: While being more more efficient '
                'than the standard one, the loss could be higher (as the data '
                'is not completely random)'
            )
        batch_sampler = SortedBatchSampler(
            source_lengths=corpus.source_corpus.index_length,
            target_lengths=corpus.target_corpus.index_length,
            batch_size=batch_size,
        )
    else:
        if verbose:
            logging.info(
                'Using standard random batch sampler. '
                'WARNING: That might be inefficient as sentences in the batch '
                'might be of drastically different length'
            )
        batch_sampler = BatchSampler(
            sampler=RandomSampler(corpus),
            batch_size=batch_size,
            drop_last=True,
        )

    return torch.utils.data.DataLoader(
        corpus,
        batch_sampler=batch_sampler,
        num_workers=num_data_workers,
        collate_fn=lambda samples: prepare_batch_from_parallel_samples(
            parallel_samples=samples,
            pad_token_id=vocab.pad_idx,
            eos_token_id=vocab.eos_idx,
            go_token_id=vocab.go_idx,
            reverse_source=reverse_source,
            use_person_feature=corpus.has_person_feature(),
        ),
        pin_memory=True,
    )
