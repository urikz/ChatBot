from collections import namedtuple, OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn_encoder_model import RnnEncoderModel


class SingleAttentionProfileMemoryModel(RnnEncoderModel):

    ForwardResult = namedtuple(
        'ForwardResult',
        ['logits', 'output_state', 'state', 'attention', 'final_attention'],
    )

    def __init__(
        self,
        device_id,
        vocab_size,
        pad_token_id,
        unk_token_id,
        go_token_id,
        eos_token_id,
        embedding_size,
        num_layers,
        hidden_size,
        dropout,
        attention_type,
        use_default_memory,
    ):
        super(SingleAttentionProfileMemoryModel, self).__init__(
            device_id=device_id,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            go_token_id=go_token_id,
            eos_token_id=eos_token_id,
            embedding_size=embedding_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.profile_memory_embeddings_weights = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=1,
            padding_idx=self.pad_token_id,
            sparse=True,
        )
        self.profile_memory_embeddings_weights.weight.data.fill_(1)
        self.profile_memory_embeddings_weights.weight.data[self.pad_token_id] = 0
        self.profile_memory_embeddings_weights.weight.data[self.eos_token_id] = 0
        self.profile_memory_embeddings_weights.weight.data[self.go_token_id] = 0
        self.profile_memory_embeddings_weights.weight.data[:100] = 0.01

        self.use_default_memory = use_default_memory
        if self.use_default_memory:
            self.default_memory = nn.Parameter(
                nn.init.xavier_uniform_(torch.Tensor(1, 1, self.embedding_size))
            )
            self.default_memory_mask = torch.ones(1, 1, 1, requires_grad=False)

        self.decoder = nn.LSTM(
            input_size=2 * self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bias=True,
        )
        self.attention_type = attention_type

        if self.attention_type == 'concat':
            self.attention = nn.Sequential(OrderedDict([
                ('attention_input_combination', nn.Linear(
                    self.hidden_size + self.embedding_size,
                    self.embedding_size,
                    bias=False,
                )),
                ('tanh', nn.Tanh()),
                ('attention_v', nn.Linear(self.embedding_size, 1, bias=False)),
            ]))
        elif self.attention_type == 'general':
            self.attention = nn.Linear(
                self.hidden_size,
                self.embedding_size,
                bias=False,
            )
        else:
            raise Exception('Unknown attention type: %s' % self.attention_type)

        self.attention_combination = nn.Sequential(OrderedDict([
            ('attention_combination', nn.Linear(
                self.hidden_size + self.embedding_size,
                self.embedding_size,
                bias=False,
            )),
            ('tanh', nn.Tanh()),
        ]))

        self.output_projection = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )
        if self.device_id is not None:
            self.cuda(self.device_id)
            if self.use_default_memory:
                self.default_memory_mask = self.to_device(
                    self.default_memory_mask
                )

    # Based on https://github.com/facebookresearch/ParlAI/blob/master/projects/personachat/persona_seq2seq.py#L1246
    def init_embeddings_weights_using_glove_index(self, glove_index):
        glove_index = glove_index.copy()
        for i in range(self.vocab_size):
            if glove_index[i] > 0:
                glove_index[i] = 1e6 * 1 / glove_index[i]**1.07
            glove_index[i] = 1.0 / (1.0 + np.log(1.0 + glove_index[i]))

        self.profile_memory_embeddings_weights.weight.data = self.to_device(
            torch.from_numpy(glove_index).unsqueeze(1)
        )
        self.profile_memory_embeddings_weights.weight.data[self.pad_token_id] = 0
        self.profile_memory_embeddings_weights.weight.data[self.eos_token_id] = 0
        self.profile_memory_embeddings_weights.weight.data[self.go_token_id] = 0

    def encode_context(self, profile_memory, encoded_input_source):
        # TODO: might have to fix it for more layers
        assert self.num_layers == 1
        batch_size = encoded_input_source[0].size(1)
        if profile_memory is not None:
            num_memories_per_sentence = profile_memory.size(2)
            profile_memory_flatten = profile_memory.view(
                -1,
                batch_size * num_memories_per_sentence,
            )
            emb_profile_memory = F.dropout(
                self.embeddings(profile_memory_flatten),
                p=self.dropout,
                training=self.training,
            )
            profile_memory_weights = self.profile_memory_embeddings_weights(
                profile_memory_flatten
            )
            profile_memory_weights_norm = profile_memory_weights.sum(
                dim=0,
                keepdim=True,
            )
            memory_mask = profile_memory_weights_norm.ne(0).float()
            profile_memory_weights_norm += (1. - memory_mask)
            profile_memory_weights /= profile_memory_weights_norm

            weighted_emb_profile_memory = (
                emb_profile_memory * profile_memory_weights
            ).sum(0).view(batch_size, num_memories_per_sentence, -1)
            memory_mask = memory_mask.view(
                batch_size,
                num_memories_per_sentence,
                1,
            )
            context = (weighted_emb_profile_memory, memory_mask)
        else:
            context = None
        return self.apply_attention(
            decoder_output=encoded_input_source[0].squeeze(0),
            context=context,
            batch_size=batch_size,
            use_default_memory=self.use_default_memory,
        )

    def apply_attention(
        self,
        decoder_output,
        context,
        batch_size,
        use_default_memory,
    ):
        assert batch_size == decoder_output.size(0)
        hidden_size = decoder_output.size(1)

        if context is None:
            assert use_default_memory
            return (
                self.default_memory.expand(batch_size, 1, -1).squeeze(1),
                self.default_memory_mask.expand(batch_size, 1, 1),
            )
        else:
            encoded_profile_memory, memory_mask = context

        num_memories_per_sentence = encoded_profile_memory.size(1)

        if use_default_memory:
            encoded_profile_memory = torch.cat(
                (
                    encoded_profile_memory,
                    self.default_memory.expand(batch_size, 1, -1),
                ),
                dim=1,
            )
            memory_mask = torch.cat(
                (
                    memory_mask,
                    self.default_memory_mask.expand(batch_size, 1, 1),
                ),
                dim=1,
            )
            num_memories_per_sentence += 1

        if self.attention_type == 'concat':
            decoder_output_expanded = (decoder_output
                .view(batch_size, 1, hidden_size)
                .expand(batch_size, num_memories_per_sentence, hidden_size))
            attention_logits = self.attention(
                torch.cat(
                    (decoder_output_expanded, encoded_profile_memory),
                    dim=2,
                )
            )
        elif self.attention_type == 'general':
            decoder_output_transformed = self.attention(decoder_output)
            attention_logits = torch.bmm(
                encoded_profile_memory,
                decoder_output_transformed.unsqueeze(2),
            )
        else:
            raise Exception('Unknown attention type: %s' % self.attention_type)

        attention_weights = F.softmax(
            attention_logits * memory_mask - (1. - memory_mask) * 1e20,
            dim=1,
        )
        return (
            (encoded_profile_memory * attention_weights).sum(dim=1),
            attention_weights,
        )

    def decode(
        self,
        input_target,
        input_state,
        context,
    ):
        max_length = input_target.size(0)
        batch_size = input_target.size(1)
        profile_memory_context, attention_weights = context
        assert batch_size == profile_memory_context.size(0)
        assert batch_size == attention_weights.size(0)
        attention_weights = attention_weights.squeeze(2)

        emb_input_target = F.dropout(
            self.embeddings(input_target),
            p=self.dropout,
            training=self.training,
        )
        profile_memory_context = profile_memory_context.unsqueeze(0)
        profile_memory_context = profile_memory_context.expand((
            emb_input_target.size(0),
            batch_size,
            profile_memory_context.size(2),
        ))
        decoder_input = torch.cat(
            (emb_input_target, profile_memory_context),
            2,
        )
        input_state = [
            F.dropout(s, p=0.5, training=self.training)
            for s in input_state
        ]

        decoder_output, decoder_state = self.decoder(
            decoder_input,
            input_state,
        )
        logits = self.output_projection(decoder_output)
        return SingleAttentionProfileMemoryModel.ForwardResult(
            logits=logits,
            output_state=decoder_output,
            state=decoder_state,
            # TODO: might have to expand it
            attention=attention_weights,
            # TODO: might have to transpose it
            final_attention=(attention_weights[:, :-1] + 1e-8).log(),
        )

    def get_parameters(self):
        parameters = super(
            SingleAttentionProfileMemoryModel,
            self,
        ).get_parameters()
        parameters.update({
            'attention_type': self.attention_type,
            'use_default_memory': self.use_default_memory,
        })
        return parameters
