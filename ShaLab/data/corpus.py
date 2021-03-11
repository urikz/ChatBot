import logging
import numpy as np
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class PackedCorpus(Dataset):
    def __init__(self):
        super(PackedCorpus, self).__init__()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_file(self, path, compress=False):
        start_time = time.time()
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(path, **self.to_numpy())
        logging.info(
            'Serialized %s to %s in %d seconds',
            str(self),
            path,
            time.time() - start_time,
        )

    @classmethod
    def from_file(cls, path):
        start_time = time.time()
        with open(path, 'rb') as f:
            raw_data = np.load(f)
            corpus = cls.from_numpy(raw_data)
        logging.info(
            'Loaded %s from file %s in %d seconds',
            str(corpus),
            path,
            time.time() - start_time,
        )
        return corpus


class PackedIndexedCorpus(PackedCorpus):
    def __init__(self, data, index_start, prefix=''):
        super(PackedIndexedCorpus, self).__init__()
        self.data = data
        self.index_start = index_start
        self.prefix = prefix
        assert len(self.index_start) > 0
        assert self.index_start[-1] <= len(self.data)
        self.index_length = np.ediff1d(
            self.index_start,
            to_end=len(self.data) - self.index_start[-1],
        )
        assert np.all(self.index_length >= 0)

    def __getitem__(self, index):
        return self.data[
            self.index_start[index]:
            self.index_start[index] + self.index_length[index]
        ]

    def get_random_sentences(self, n):
        indices = np.random.randint(len(self), size=n)
        return [self[i] for i in indices]

    def __len__(self):
        return len(self.index_start)

    def __str__(self):
        return 'PackedIndexedCorpus[%d sentences]' % len(self)

    def to_numpy(self):
        return {
            self.prefix + 'data': self.data,
            self.prefix + 'index_start': self.index_start,
        }

    def get_lengths(self):
        return (self.index_length, )

    @classmethod
    def from_numpy(cls, raw_data, prefix=''):
        return cls(
            data=raw_data[prefix + 'data'],
            index_start=raw_data[prefix + 'index_start'],
            prefix=prefix,
        )

    @staticmethod
    def pack(sentences, prefix='', verbose=True):
        start_time = time.time()
        index_start = np.zeros(len(sentences), dtype=np.int32)
        index_length = np.zeros(len(sentences), dtype=np.int32)
        total_length = 0
        for i, sentence in enumerate(sentences):
            index_start[i] = total_length
            if isinstance(sentence, np.ndarray):
                assert sentence.ndim == 1
                index_length[i] = sentence.size
            elif isinstance(sentence, list) or isinstance(sentence, tuple):
                index_length[i] = len(sentence)
            else:
                raise Exception('Unknown type for sentence: ' + type(sentence))
            total_length += index_length[i]

        data = np.zeros(total_length, dtype=np.int32)
        for i, sentence in enumerate(sentences):
            data[index_start[i]: index_start[i] + index_length[i]] = sentence

        corpus = PackedIndexedCorpus(data, index_start, prefix)

        if verbose:
            logging.info(
                'Built a %s in %d seconds',
                str(corpus),
                time.time() - start_time,
            )
        return corpus


class DialogCorpus(PackedCorpus):
    def __init__(self, sentences, dialog_start_index):
        super(DialogCorpus, self).__init__()
        self.sentences = sentences
        self.dialog_start_index = dialog_start_index
        can_be_training_example = np.ones(len(self.sentences), dtype=np.bool)
        can_be_training_example[self.dialog_start_index] = False
        self.example_to_sentence_index = np.flatnonzero(can_be_training_example)

    def __getitem__(self, index):
        # TODO: support longer dialog history
        return {
            "input_source": self.sentences[self.example_to_sentence_index[index] - 1],
            "input_target": self.sentences[self.example_to_sentence_index[index]],
            "output_target": self.sentences[self.example_to_sentence_index[index]],
        }

    def get_random_sentences(self, n):
        return self.sentences.get_random_sentences(n)

    def get_num_dialogs(self):
        return len(self.dialog_start_index)

    def get_dialog(self, index):
        start_index = self.dialog_start_index[index]
        end_index = (
            self.dialog_start_index[index + 1]
            if index + 1 < self.get_num_dialogs()
            else len(self.sentences)
        )
        return [self.sentences[i] for i in range(start_index, end_index)]

    def __len__(self):
        return len(self.example_to_sentence_index)

    def __str__(self):
        return (
            'DialogCorpus[%d dialogs, %d sentences, avg %.2f turns per dialog]'
        ) % (
            self.get_num_dialogs(),
            len(self.sentences),
            len(self.sentences) / len(self.dialog_start_index),
        )

    def get_lengths(self):
        return (
            self.sentences.index_length[self.example_to_sentence_index - 1],
            self.sentences.index_length[self.example_to_sentence_index],
        )

    def to_numpy(self):
        params = self.sentences.to_numpy()
        params.update({
            'dialog_start_index': self.dialog_start_index,
        })
        return params

    @classmethod
    def from_numpy(cls, raw_data):
        return DialogCorpus(
            sentences=PackedIndexedCorpus.from_numpy(raw_data),
            dialog_start_index=raw_data['dialog_start_index'],
        )

    @staticmethod
    def pack(dialogs, verbose=True):
        start_time = time.time()
        sentences = PackedIndexedCorpus.pack(
            [sentence for dialog in dialogs for sentence in dialog],
            verbose=False,
        )
        dialog_start_index = np.array([len(dialog) for dialog in dialogs])
        if len(dialogs) > 0:
            dialog_start_index = np.roll(dialog_start_index, 1)
            dialog_start_index[0] = 0
            dialog_start_index = np.cumsum(dialog_start_index)

        dialog_corpus = DialogCorpus(sentences, dialog_start_index)

        if verbose:
            logging.info(
                'Built a %s in %d seconds',
                str(dialog_corpus),
                time.time() - start_time,
            )
        return dialog_corpus


class PackedProfileMemory(PackedCorpus):
    def __init__(self, sentences, profile_memory_index):
        super(PackedProfileMemory, self).__init__()
        self.sentences = sentences
        self.profile_memory_index = profile_memory_index
        assert self.sentences.prefix == 'profile_memory_'
        assert self.profile_memory_index.prefix == 'profile_memory_index_'
        for i in range(len(self.profile_memory_index)):
            for index in self.profile_memory_index[i]:
                assert index >= 0 and index <= len(self.sentences)

    def get_random_memories(self, n):
        indices = np.random.randint(len(self.sentences), size=n)
        return indices, [self.sentences[i] for i in indices]

    def __getitem__(self, index):
        return [
            self.sentences[i]
            for i in self.profile_memory_index[index]
        ]

    def __len__(self):
        return len(self.profile_memory_index)

    def __str__(self):
        return 'PackedProfileMemory[%d unique memories, %d profiles]' % (
            len(self.sentences),
            len(self.profile_memory_index),
        )

    def to_numpy(self):
        params = self.sentences.to_numpy()
        params.update(self.profile_memory_index.to_numpy())
        return params

    @classmethod
    def from_numpy(cls, raw_data):
        return PackedProfileMemory(
            sentences=PackedIndexedCorpus.from_numpy(
                raw_data=raw_data,
                prefix='profile_memory_',
            ),
            profile_memory_index=PackedIndexedCorpus.from_numpy(
                raw_data=raw_data,
                prefix='profile_memory_index_',
            ),
        )

    @staticmethod
    def pack(profile_memory, verbose=True):
        start_time = time.time()
        sentences = list(set([
            tuple(s) for single_memory in profile_memory for s in single_memory
        ]))
        sentences_dict = dict(zip(sentences, range(len(sentences))))
        sentences = PackedIndexedCorpus.pack(
            sentences,
            prefix='profile_memory_',
            verbose=False,
        )
        index = PackedIndexedCorpus.pack(
            [
                [np.array(sentences_dict[tuple(s)]) for s in single_memory]
                for single_memory in profile_memory
            ],
            prefix='profile_memory_index_',
            verbose=False,
        )
        pm = PackedProfileMemory(sentences, index)
        if verbose:
            logging.info(
                'Built a %s in %d seconds',
                str(pm),
                time.time() - start_time,
            )
        return pm


class DialogCorpusWithProfileMemory(DialogCorpus):
    def __init__(
        self,
        sentences,
        dialog_start_index,
        profile_memory,
        is_profile_memory_fake=False,
        false_opponent_profile_memory=None,
    ):
        super(DialogCorpusWithProfileMemory, self).__init__(
            sentences,
            dialog_start_index,
        )
        self.profile_memory = profile_memory
        self.num_random_memories = 0
        self.prepare_opponents_memory = True
        self.false_opponent_profile_memory = false_opponent_profile_memory

        if is_profile_memory_fake:
            self.example_to_profile_memory = None
        else:
            assert len(self.profile_memory) == 2 * self.get_num_dialogs()
            sentence_to_target_profile = np.zeros(
                len(self.sentences),
                dtype=np.int32,
            )
            dialog_index, profile_index = -1, 1
            for i in range(len(self.sentences)):
                if (
                    dialog_index + 1 < len(self.dialog_start_index) and
                    i >= self.dialog_start_index[dialog_index + 1]
                ):
                    dialog_index += 1
                    profile_index = 1
                sentence_to_target_profile[i] = 2 * dialog_index + profile_index
                assert sentence_to_target_profile[i] < len(self.profile_memory)
                profile_index = (profile_index + 1) % 2

            self.example_to_profile_memory = np.delete(
                sentence_to_target_profile,
                self.dialog_start_index,
            )

    def set_num_random_memories(self, num_random_memories):
        self.num_random_memories = num_random_memories

    def get_dialog_profile_memories(self, index):
        return (
            self.profile_memory[2 * index + 1],
            self.profile_memory[2 * index],
        )

    def get_opponent_memory(self, index):
        if index % 2 == 0:
            return index + 1
        else:
            return index - 1

    def _prepare_profile_memory(self, index, num_random_memories):
        profile_memories = []
        if index is not None:
            profile_memories.extend(self.profile_memory[index])
        num_true_profile_memories = len(profile_memories)

        if self.num_random_memories > 0:
            _, random_memories = self.profile_memory.get_random_memories(
                num_random_memories
            )
            profile_memories.extend(random_memories)

        profile_memory_mask = np.zeros(len(profile_memories), dtype=np.float32)
        profile_memory_mask[:num_true_profile_memories] = 1
        return profile_memory_mask, profile_memories

    def __getitem__(self, index):
        item = super(DialogCorpusWithProfileMemory, self).__getitem__(index)
        profile_memory_mask, profile_memories = self._prepare_profile_memory(
            (
                self.example_to_profile_memory[index]
                if self.example_to_profile_memory is not None
                else None
            ),
            self.num_random_memories,
        )
        item["profile_memory"] = profile_memories
        item["profile_memory_mask"] = profile_memory_mask
        if self.false_opponent_profile_memory is not None:
            opponent_profile_memory = self._prepare_profile_memory(
                (
                    self.get_opponent_memory(
                        self.example_to_profile_memory[index]
                    )
                    if self.example_to_profile_memory is not None
                    else None
                ),
                self.false_opponent_profile_memory,
            )
            item["opponent_profile_memory_candidates"] = opponent_profile_memory
            item["opponent_profile_memory_correct"] = range(
                0,
                len(opponent_profile_memory) - self.false_opponent_profile_memory,
            )
        return item

    def __str__(self):
        return (
            'DialogCorpusWithProfileMemory[%d dialogs, %d sentences, '
            'avg %.2f turns per dialog, %d unique profile memories]'
        ) % (
            len(self.dialog_start_index),
            len(self.sentences),
            len(self.sentences) / len(self.dialog_start_index),
            len(self.profile_memory.sentences),
        )

    def to_numpy(self):
        params = super(DialogCorpusWithProfileMemory, self).to_numpy()
        params.update(self.profile_memory.to_numpy())
        return params

    @classmethod
    def from_numpy(cls, raw_data):
        return DialogCorpusWithProfileMemory(
            sentences=PackedIndexedCorpus.from_numpy(raw_data),
            dialog_start_index=raw_data['dialog_start_index'],
            profile_memory=PackedProfileMemory.from_numpy(raw_data),
        )

    @staticmethod
    def pack(dialogs, profile_memory, verbose=True):
        start_time = time.time()
        dialog_corpus = DialogCorpus.pack(dialogs, verbose=False)
        profile_memory = PackedProfileMemory.pack(profile_memory, verbose=False)

        dialog_corpus_with_profile_memory = DialogCorpusWithProfileMemory(
            sentences=dialog_corpus.sentences,
            dialog_start_index=dialog_corpus.dialog_start_index,
            profile_memory=profile_memory,
        )

        if verbose:
            logging.info(
                'Built a %s in %d seconds',
                str(dialog_corpus_with_profile_memory),
                time.time() - start_time,
            )
        return dialog_corpus_with_profile_memory


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0
        self.datasets = list(datasets)

        total_size = sum([len(d) for d in self.datasets])
        self.dataset_index = np.zeros(total_size, dtype=np.int32)
        self.sample_index = np.zeros(total_size, dtype=np.int32)
        dataset_index, sample_index = 0, 0
        for i in range(total_size):
            if sample_index >= len(self.datasets[dataset_index]):
                dataset_index += 1
                sample_index = 0
            self.dataset_index[i] = dataset_index
            self.sample_index[i] = sample_index
            sample_index += 1

        lengths = [d.get_lengths() for d in self.datasets]
        num_lengths = [len(l) for l in lengths]
        assert max(num_lengths) == min(num_lengths)
        self.lengths = tuple(
            np.concatenate([l[i] for l in lengths])
            for i in range(max(num_lengths))
        )

    def __len__(self):
        return len(self.dataset_index)

    def __str__(self):
        return 'ConcatDataset[%d samples from %s]' % (
            len(self),
            ', '.join([str(d) for d in self.datasets])
        )

    def __getitem__(self, idx):
        return self.datasets[self.dataset_index[idx]][self.sample_index[idx]]

    def get_lengths(self):
        return self.lengths

    def get_random_sentences(self, n):
        indices = np.random.randint(len(self), size=n)
        return [self[i] for i in indices]


def prepare_batch(
    samples,
    pad_token_id,
    bos_token_id=None,
    eos_token_id=None,
    place_pad_token_first=False,
):
    max_samples_length = max([sample.shape[0] for sample in samples])
    total_length = (
        (1 if bos_token_id is not None else 0) +
        max_samples_length +
        (1 if eos_token_id is not None else 0)
    )
    batch = np.full((len(samples), total_length), pad_token_id, dtype=np.int64)
    for i in range(len(samples)):
        current_length = samples[i].shape[0]
        if place_pad_token_first:
            index = max_samples_length - current_length
        else:
            index = 0

        if bos_token_id is not None:
            batch[i][index] = bos_token_id
            index += 1

        batch[i][index:index + current_length] = samples[i]
        index += current_length
        if eos_token_id is not None:
            batch[i][index] = eos_token_id
    return torch.from_numpy(batch.transpose().copy())


def prepare_profile_memory(samples, pad_token_id):
    profile_memory_per_samples = max([len(pm) for pm in samples])
    profile_memory_flatten = []
    for i in range(len(samples)):
        profile_memory_flatten.extend(samples[i])
        empty_memories = profile_memory_per_samples - len(samples[i])
        profile_memory_flatten.extend([np.zeros(0)] * empty_memories)
    return prepare_batch(
        samples=profile_memory_flatten,
        pad_token_id=pad_token_id,
    ).view(-1, len(samples), profile_memory_per_samples)


def prepare_batch_from_parallel_samples(
    parallel_samples,
    pad_token_id,
    eos_token_id,
    go_token_id,
):
    batch_size = len(parallel_samples)
    batch = {
        "input_source": prepare_batch(
            samples=[sample["input_source"] for sample in parallel_samples],
            pad_token_id=pad_token_id,
            place_pad_token_first=True,
        ),
        "input_target": prepare_batch(
            samples=[sample["input_target"] for sample in parallel_samples],
            pad_token_id=pad_token_id,
            bos_token_id=go_token_id,
        ),
        "output_target": prepare_batch(
            samples=[sample["output_target"] for sample in parallel_samples],
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        ),
    }
    if "profile_memory" in parallel_samples[0]:
        batch["profile_memory"] = prepare_profile_memory(
            samples=[sample["profile_memory"] for sample in parallel_samples],
            pad_token_id=pad_token_id,
        )
        profile_memory_per_samples = batch["profile_memory"].size(2)
        profile_memory_mask = np.zeros(
            (batch_size, profile_memory_per_samples),
            dtype=np.float32,
        )
        for i in range(batch_size):
            assert "profile_memory_mask" in parallel_samples[i]
            current_mask = parallel_samples[i]["profile_memory_mask"]
            profile_memory_mask[i][:len(current_mask)] = current_mask
        batch["profile_memory_mask"] = torch.from_numpy(profile_memory_mask)

    if "opponent_profile_memory_candidates" in parallel_samples[0]:
        batch["opponent_profile_memory_candidates"] = prepare_profile_memory(
            samples=[
                sample["opponent_profile_memory_candidates"]
                for sample in parallel_samples
            ],
            pad_token_id=pad_token_id,
        )
        # TODO: prepare opponent_profile_memory

    return batch


class SortedBatchSampler(object):
    def __init__(
        self,
        source_lengths,
        target_lengths,
        batch_size,
        verbose=True,
    ):
        start_time = time.time()
        assert len(source_lengths) == len(target_lengths)
        # TODO: Shall we sort by target length first?
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
        if batch_size > 1:
            indices = indices[:-(len(indices) % batch_size)]
        self.batches = np.split(indices, len(indices) // batch_size)

        np.random.shuffle(self.batches)
        if verbose:
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


class BatchRandomSampler(object):
    def __init__(self, dataset, batch_size, length):
        self.length = length
        self.indices = torch.randint(
            len(dataset),
            size=(self.length, batch_size),
            dtype=torch.long,
            requires_grad=False,
        )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.length


def data_loader(
    corpus,
    vocab,
    batch_size,
    sort_batches,
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
        source_lengths, target_lengths = corpus.get_lengths()
        batch_sampler = SortedBatchSampler(
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            batch_size=batch_size,
            verbose=verbose,
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
        ),
        pin_memory=True,
    )


def data_loader_for_corpus(
    corpus,
    vocab,
    batch_size,
    length=None,
    num_data_workers=0,
):
    if length is not None:
        batch_sampler = BatchRandomSampler(
            dataset=corpus,
            batch_size=batch_size,
            length=length,
        )
    else:
        batch_sampler = BatchSampler(
            sampler=RandomSampler(corpus),
            batch_size=batch_size,
            drop_last=True,
        )

    return torch.utils.data.DataLoader(
        corpus,
        batch_sampler=batch_sampler,
        num_workers=num_data_workers,
        collate_fn=lambda samples: prepare_batch(
            samples=samples,
            pad_token_id=vocab.pad_idx,
        ),
        pin_memory=True,
    )
