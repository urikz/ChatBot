from collections import namedtuple
import heapq
import numpy as np
import torch
import torch.nn.functional as F


BIG_NEGATIVE_SCORE = -10000


class Generator(object):

    SingleGenerationResult = namedtuple(
        'SingleGenerationResult',
        ['log_prob', 'output', 'attention', 'score'],
    )

    def __init__(self, model):
        self.model = model

        # Special penalty to avoid generating these tokens
        self.special_token_mask = torch.zeros(
            self.model.vocab_size,
            device=self.model.get_device_id(),
            requires_grad=False,
        )
        self.special_token_mask[self.model.pad_token_id] = BIG_NEGATIVE_SCORE
        self.special_token_mask[self.model.unk_token_id] = BIG_NEGATIVE_SCORE
        self.special_token_mask[self.model.go_token_id] = BIG_NEGATIVE_SCORE

        # Special penalty to avoid generating EOS token
        self.eos_token_mask = torch.zeros(
            self.model.vocab_size,
            device=self.model.get_device_id(),
            requires_grad=False,
        )
        self.eos_token_mask[self.model.eos_token_id] = BIG_NEGATIVE_SCORE

        self.initial_go_token = torch.full(
            (1, 1),
            fill_value=self.model.go_token_id,
            dtype=torch.long,
            device=self.model.get_device_id(),
            requires_grad=False,
        )
        self.has_finished_mask = torch.zeros(
            1,
            dtype=torch.uint8,
            device=self.model.get_device_id(),
            requires_grad=False,
        )
        self.initial_beam_scores = torch.zeros(
            1,
            device=self.model.get_device_id(),
            requires_grad=False,
        )

    # TODO: Hack this function to support multiple outputs for a
    # single input_source and/or single context
    def greedy_search(
        self,
        input_source,
        max_length,
        policy,
        context,
    ):
        if context is not None:
            batch_size = context.size(1)
        else:
            batch_size = input_source.size(1)
        outs, attentions = [], []
        log_probs = 0

        if self.initial_go_token.size(1) != batch_size:
            self.initial_go_token.resize_(
                self.initial_go_token.size(0),
                batch_size,
            ).fill_(self.model.go_token_id)
            self.has_finished_mask.resize_(batch_size).fill_(0)

        previous_target_words = self.initial_go_token
        has_finished_mask = self.has_finished_mask

        previous_state = self.model.encode(input_source)
        if input_source.size(1) == 1 and batch_size > 1:
            previous_state = [
                s.expand(s.size(0), batch_size, s.size(2))
                for s in previous_state
            ]
        encoded_context = self.model.encode_context(context, previous_state)

        for i in range(max_length):
            forward_result = self.model.decode(
                previous_target_words,
                previous_state,
                encoded_context,
            )
            # avoid generating some special tokens
            output = (
                torch.squeeze(forward_result.logits, dim=0)
                + self.special_token_mask
            )
            if i == 0:
                # avoid generating end-of-sentence token at the first step
                output = output + self.eos_token_mask

            # TODO: subtract max in order to make softmax+multinomial more stable
            # (probably that is already done by F.softmax)
            probs = F.softmax(output, dim=1)

            if policy == 'greedy':
                # Shape: [batch_size]
                current_probs, words = probs.max(dim=1)
                current_log_probs = current_probs.log()
                words = words.detach()
            elif policy == 'sample':
                words = probs.multinomial(num_samples=1).detach()
                current_log_probs = torch.gather(probs, 1, words).log().squeeze(1)
                words = words.squeeze(1)
            else:
                raise Exception('Unknown generation policy: ' + policy)

            has_finished_mask = (
                has_finished_mask + (words == self.model.eos_token_id)
            ) > 0
            # if the sentence has already finished - generate only PAD tokens
            words = (
                (has_finished_mask == 0).long() * words +
                has_finished_mask.long() * self.model.pad_token_id
            )
            if has_finished_mask.sum() == batch_size:
                # all sentences have been finished. Abort further generation.
                break

            log_probs += current_log_probs
            # Shape: [1, batch_size]
            previous_target_words = torch.unsqueeze(words, 0)
            outs.append(previous_target_words)
            if hasattr(forward_result, "attention"):
                attentions.append(forward_result.attention)
            previous_state = forward_result.state

        # Shape: [batch_size, 1]
        # Shape: [max_length, batch_size]
        # Shape: [max_length, batch_size, num_memories (+1)]
        return [Generator.SingleGenerationResult(
            log_prob=log_probs,
            output=torch.cat(outs, 0).detach(),
            attention=(
                torch.cat(attentions, 0).detach()
                if len(attentions) > 0
                else None
            ),
            score=log_probs.detach(),
        )]

    def beam_search(
        self,
        input_source,
        max_length,
        beam_size,
        context,
        # See https://arxiv.org/abs/1609.08144
        length_normalization_factor=0,
        length_normalization_const=0,
    ):
        batch_size = input_source.size(1)

        assert batch_size == 1
        assert max_length > 0

        if self.initial_go_token.size(1) != batch_size:
            self.initial_go_token.resize_(
                self.initial_go_token.size(0),
                batch_size,
            ).fill_(self.model.go_token_id)
        previous_target_words = self.initial_go_token

        if self.initial_beam_scores.size(0) != beam_size:
            self.initial_beam_scores.resize_(beam_size).fill_(0)
        previous_scores = self.initial_beam_scores

        prev_hypo_index, words, scores = [], [], []

        previous_states = self.model.encode(input_source)
        encoded_context = self.model.encode_context(context, previous_states)

        for i in range(max_length):
            if i > 0 and encoded_context is not None:
                encoded_context = (
                    encoded_context[0].expand(beam_size, -1, -1),
                    encoded_context[1].expand(beam_size, -1, -1),
                )
            forward_result = self.model.decode(
                previous_target_words,
                previous_states,
                encoded_context,
            )
            output = (
                torch.squeeze(forward_result.logits, dim=0)
                + self.special_token_mask
            )
            if i == 0:
                # avoid generating end-of-sentence token at the first step
                output = output + self.eos_token_mask

            log_probs = F.log_softmax(output, dim=1)

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
                for new_state in forward_result.state
            ])

            prev_hypo_index.append(current_hypo.detach().cpu().numpy())
            words.append(current_words.detach().cpu().numpy())
            scores.append(current_scores.detach().cpu().numpy())

            previous_scores = torch.unsqueeze(current_scores, 1)
            previous_target_words = torch.unsqueeze(current_words, 0)
            previous_states = current_states

        heap_of_hypo = []
        for i in range(max_length):
            for j in range(beam_size):
                if i == max_length - 1 or words[i][j] == self.model.eos_token_id:
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
                result.count(self.model.eos_token_id) > 1 or
                (
                    result.count(self.model.eos_token_id) == 1 and
                    result[-1] != self.model.eos_token_id
                )
            ):
                # Additional safety check on the correctness of the generated
                # sequence. However, things like that should not happen,
                # since we're using special self.eos_token_mask
                # during generation.
                continue

            results.append(
                (
                    -score,
                    self.model.to_device(torch.from_numpy(np.array(result))).unsqueeze(1),
                )
            )
        return [
            Generator.SingleGenerationResult(
                log_prob=r[0],
                output=r[1],
                attention=None,
                score=r[0],
            )
            for r in sorted(results, reverse=True, key=lambda x: x[0])
        ]

    def generate(
        self,
        input_source,
        max_length,
        policy,
        num_candidates,
        context,
        # See https://arxiv.org/abs/1609.08144
        length_normalization_factor=0,
        length_normalization_const=0,
    ):
        batch_size = input_source.size(1)

        if policy == 'sample':
            if num_candidates > 1:
                assert batch_size == 1
                # TODO: Implement sampling multiple responses from
                # a single input. We probably need that only for a single
                # input (original batch_size == 1)
                raise Exception('Not yet implemented')
            results = self.greedy_search(
                input_source=input_source,
                max_length=max_length,
                policy=policy,
                context=context,
            )
        elif policy == 'greedy':
            if num_candidates == 1:
                results = self.greedy_search(
                    input_source=input_source,
                    max_length=max_length,
                    policy=policy,
                    context=context,
                )
            else:
                assert batch_size == 1
                results = self.beam_search(
                    input_source=input_source,
                    max_length=max_length,
                    beam_size=num_candidates,
                    length_normalization_factor=length_normalization_factor,
                    length_normalization_const=length_normalization_const,
                    context=context,
                )
        else:
            raise Exception('Unknown generation policy ' + policy)

        return results
