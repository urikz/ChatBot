import numpy as np
import sys
import torch

from ShaLab.models import Generator
from ShaLab.data import prepare_batch
from .base_agent import BaseAgent

class RandomWordsAgent(BaseAgent):
    def __init__(
        self,
        device_id,
        vocab,
        max_length,
        num_candidates=1,
        context=None,
    ):
        super(RandomWordsAgent, self).__init__(device_id, num_candidates, context)
        self.word_options = np.setdiff1d(
            np.arange(len(vocab)),
            np.array([
                vocab.pad_idx,
                vocab.go_idx,
                vocab.eos_idx,
                vocab.unk_idx,
            ]),
        )
        self.max_length = max_length

    def __iter__(self):
        while True:
            yield self.generate(None)[0].output

    def __len__(self):
        return sys.maxsize

    def generate(self, input_source):
        random_sentences = torch.from_numpy(self.word_options[
            np.random.randint(
                len(self.word_options),
                size=(self.max_length, self.num_candidates)
            )
        ])
        if self.device_id is not None:
            random_sentences = random_sentences.cuda(
                self.device_id,
                async=True,
            )
        return [Generator.SingleGenerationResult(
            log_prob=None,
            output=random_sentences,
            attention=None,
            score=None,
        )]


class RandomSentencesAgent(BaseAgent):
    def __init__(self, device_id, corpus, pad_token_id, num_candidates=1, context=None):
        super(RandomSentencesAgent, self).__init__(device_id, num_candidates, context)
        self.corpus = corpus
        self.pad_token_id = pad_token_id

    def generate(self, input_source):
        random_sentences = prepare_batch(
            self.corpus.get_random_sentences(self.num_candidates),
            self.pad_token_id,
        )
        if self.device_id is not None:
            random_sentences = random_sentences.cuda(
                self.device_id,
                async=True,
            )
        return [Generator.SingleGenerationResult(
            log_prob=None,
            output=random_sentences,
            attention=None,
            score=None,
        )]
