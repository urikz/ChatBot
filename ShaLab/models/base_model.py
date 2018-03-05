from collections import namedtuple
import heapq
import logging
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

BIG_NEGATIVE_SCORE = -10000


def get_map_location(device_id):
    if device_id is None:
        return lambda storage, location: storage
    else:
        return lambda storage, location: storage.cuda(device_id)


class BaseModel(nn.Module):
    def __init__(
        self,
        device_id,
        vocab_size,
        pad_token_id,
        unk_token_id,
        go_token_id,
        eos_token_id,
    ):
        super(BaseModel, self).__init__()

        self.device_id = device_id
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.go_token_id = go_token_id
        self.eos_token_id = eos_token_id

        # Special penalty to avoid generating these tokens
        self.special_token_mask = torch.zeros(self.vocab_size)
        self.special_token_mask[self.pad_token_id] = BIG_NEGATIVE_SCORE
        self.special_token_mask[self.unk_token_id] = BIG_NEGATIVE_SCORE
        self.special_token_mask[self.go_token_id] = BIG_NEGATIVE_SCORE
        self.special_token_mask = self.to_device(self.special_token_mask)
        self.special_token_mask = Variable(
            self.special_token_mask,
            requires_grad=False,
        )
        # Special penalty to avoid generating EOS token
        self.eos_token_mask = torch.zeros(self.vocab_size)
        self.eos_token_mask[self.eos_token_id] = BIG_NEGATIVE_SCORE
        self.eos_token_mask = self.to_device(self.eos_token_mask)
        self.eos_token_mask = Variable(self.eos_token_mask, requires_grad=False)

    def to_device(self, m, async=True):
        if self.device_id is not None:
            return m.cuda(self.device_id, async=async)
        return m.cpu()

    def get_device_id(self):
        return self.device_id

    def get_map_location(self):
        return get_map_location(self.device_id)

    def get_parameters(self):
        return {
            'device_id': self.device_id,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'go_token_id': self.go_token_id,
            'eos_token_id': self.eos_token_id,
        }

    def generate(
        self,
        input_source,
        max_length,
        policy,
        person=None,
    ):
        batch_size = input_source.size(1)
        outs = []
        log_probs = 0

        if self.device_id is not None:
            with torch.cuda.device(self.device_id):
                previous_target_words = torch.cuda.LongTensor(1, batch_size)
                has_finished_mask = torch.cuda.ByteTensor(batch_size)
        else:
            previous_target_words = torch.LongTensor(1, batch_size)
            has_finished_mask = torch.ByteTensor(batch_size)

        previous_target_words = Variable(
            previous_target_words.fill_(self.go_token_id)
        )
        has_finished_mask = Variable(has_finished_mask.zero_())

        previous_state = self.encode(input_source)

        for i in range(max_length):
            output, current_state = self.decode(
                previous_target_words,
                previous_state,
                person,
            )
            # avoid generating some special tokens
            output = torch.squeeze(output, dim=0) + self.special_token_mask
            if i == 0:
                # avoid generating end-of-sentence token at the first step
                output = output + self.eos_token_mask

            # TODO: subtract max in order to make softmax+multinomial more stable
            probs = F.softmax(output, dim=1)

            if policy == 'greedy':
                # Shape: [batch_size]
                words = probs.max(dim=1)[1].detach()
            elif policy == 'sample':
                words = probs.multinomial().squeeze(1).detach()
            else:
                raise Exception('Unknown generation policy: ' + policy)

            has_finished_mask = (
                has_finished_mask + (words == self.eos_token_id)
            ) > 0
            # if the sentence has already finished - generate only PAD tokens
            words = (
                (has_finished_mask == 0).long() * words +
                has_finished_mask.long() * self.pad_token_id
            )
            if has_finished_mask.sum().data[0] == batch_size:
                # all sentences have been finished. Abort further generation.
                break

            log_probs += -F.cross_entropy(
                output,
                words,
                ignore_index=self.pad_token_id,
                reduce=False,
            )
            # Shape: [1, batch_size]
            previous_target_words = torch.unsqueeze(words, 0)
            outs.append(previous_target_words)
            previous_state = current_state

        # Shape: [batch_size, 1]
        # Shape: [max_length, batch_size]
        return log_probs, torch.cat(outs, 0)

    def beam_search(
        self,
        input_source,
        max_length,
        beam_size,
        # See https://arxiv.org/abs/1609.08144
        length_normalization_factor=0,
        length_normalization_const=0,
        person=None,
    ):
        batch_size = input_source.size(1)

        assert batch_size == 1
        assert max_length > 0

        previous_target_words = Variable(self.to_device(
            torch.LongTensor(1, batch_size).fill_(self.go_token_id)
        ))
        previous_scores = Variable(self.to_device(torch.zeros(beam_size)))
        prev_hypo_index, words, scores = [], [], []

        previous_states = self.encode(input_source)

        for i in range(max_length):
            output, new_states = self.decode(
                previous_target_words,
                previous_states,
                person,
            )
            output = torch.squeeze(output, dim=0) + self.special_token_mask
            if i == 0:
                # avoid generating end-of-sentence token at the first step
                output = output + self.eos_token_mask

            log_probs = F.log_softmax(output)

            best_logprobs_per_hypo, best_words_per_hypo = log_probs.topk(
                beam_size,
                dim=1,
            )
            words_after_eos_penalty = torch.unsqueeze(
                torch.index_select(
                    self.eos_token_mask,
                    0,
                    torch.squeeze(previous_target_words, 0),
                ),
                1,
            )
            best_score_per_hypo = (
                best_logprobs_per_hypo +
                previous_scores +
                words_after_eos_penalty
            )
            current_scores, best_indices = best_score_per_hypo.view(-1).topk(beam_size)
            current_hypo = best_indices.div(beam_size)
            current_words = torch.index_select(best_words_per_hypo.view(-1), 0, best_indices)
            current_states = tuple([
                torch.index_select(new_state, 1, current_hypo)
                for new_state in new_states
            ])

            prev_hypo_index.append(current_hypo.cpu().data.numpy().copy())
            words.append(current_words.cpu().data.numpy().copy())
            scores.append(current_scores.cpu().data.numpy().copy())

            previous_scores = torch.unsqueeze(current_scores, 1)
            previous_target_words = torch.unsqueeze(current_words, 0)
            previous_states = current_states

        heap_of_hypo = []
        for i in range(max_length):
            for j in range(beam_size):
                if i == max_length - 1 or words[i][j] == self.eos_token_id:
                    score = scores[i][j]
                    if length_normalization_factor > 0:
                        length_penalty = (
                            (length_normalization_const + i + 1) /
                            (length_normalization_const + 1)
                        )
                        score /= (length_penalty ** length_normalization_factor)
                    heapq.heappush(heap_of_hypo, (-score, (i, j)))

        results = []
        while len(heap_of_hypo) > 0 and len(results) < beam_size:
            result = []
            score, (l, hypo_index) = heapq.heappop(heap_of_hypo)
            while l >= 0:
                result.append(words[l][hypo_index])
                hypo_index = prev_hypo_index[l][hypo_index]
                l -= 1
            result = list(reversed(result))
            if (
                result.count(self.eos_token_id) > 1 or
                (
                    result.count(self.eos_token_id) == 1 and
                    result[-1] != self.eos_token_id
                )
                # or result.count(self.unk_token_id) > 0
                # alright, we're fune with UNKs
            ):
                continue

            results.append(
                (-score, self.to_device(torch.from_numpy(np.array(result))))
            )

        return sorted(results, reverse=True)
