import logging
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        embedding_size,
    ):
        super(BaseModel, self).__init__()

        self.device_id = device_id
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.go_token_id = go_token_id
        self.eos_token_id = eos_token_id
        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_token_id,
            sparse=True,
        )

        if self.device_id is not None:
            self.cuda(self.device_id)

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
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'go_token_id': self.go_token_id,
            'eos_token_id': self.eos_token_id,
            'embedding_size': self.embedding_size,
        }

    def get_embeddings(self):
        return self.embeddings

    def set_embeddings(self, embeddings):
        self.embeddings.weight.data = self.to_device(
            torch.from_numpy(embeddings)
        )
        self.embeddings.weight.data[self.pad_token_id].fill_(0)

    def serialize(self):
        return {
            'state_dict': self.state_dict(),
            'params': self.get_parameters(),
        }

    def load_from_checkpoint(self, checkpoint, verbose=True):
        start_time = time.time()
        if isinstance(checkpoint, str):
            # We need to load checkpoint from the file first
            checkpoint = torch.load(
                checkpoint,
                map_location=self.get_map_location(),
            )

        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        self.load_state_dict(checkpoint['state_dict'])
        if verbose:
            logging.info(
                'Loaded dialog model from checkpoint in %d seconds',
                time.time() - start_time,
            )

    @classmethod
    def create_from_checkpoint(cls, path, device_id):
        start_time = time.time()
        checkpoint = torch.load(
            path,
            map_location=get_map_location(device_id),
        )
        checkpoint['model']['params']['device_id'] = device_id
        model = cls(**checkpoint['model']['params'])
        model.load_state_dict(checkpoint['model']['state_dict'])
        logging.info(
            'Loaded dialog model from file %s in %d seconds.',
            path,
            time.time() - start_time,
        )
        print('Params: {\n%s\n}' % '\n'.join(
            [
                '  ' + k + ': ' + str(v)
                for (k, v) in checkpoint['model']['params'].items()
            ]
        ))
        print('Model: %s' % model)
        return model

    def encode(self, input_source):
        raise NotImplementedError()

    def encode_context(self, context, encoded_input_source=None):
        return None

    def decode(self, input_source, input_target, context):
        raise NotImplementedError()

    def forward(
        self,
        input_source,
        input_target,
        context,
        apply_final_attention=False,
    ):
        batch_size = input_target.size(1)
        final_encoder_state = self.encode(input_source)
        if input_source.size(1) == 1 and batch_size > 1:
            final_encoder_state = [
                s.expand(s.size(0), batch_size, s.size(2)).clone()
                for s in final_encoder_state
            ]
        encoded_context = self.encode_context(context, final_encoder_state)
        forward_result = self.decode(
            input_target,
            final_encoder_state,
            encoded_context,
        )
        if apply_final_attention and forward_result.final_attention is None:
            _, final_attention = self.apply_attention(
                decoder_output=forward_result.output_state,
                context=encoded_context,
                batch_size=batch_size,
                use_default_memory=False,
            )
            forward_result = forward_result._replace(
                final_attention=(final_attention + 1e-8).log().squeeze(2)
            )
        return forward_result

    def log_prob(self, input_source, input_target, output_target, context):
        batch_size = input_target.size(1)
        assert batch_size == output_target.size(1)
        assert batch_size == context.size(1)

        logits = self.forward(input_source, input_target, context).logits
        return -F.cross_entropy(
            logits.view(-1, self.vocab_size),
            output_target.view(-1),
            ignore_index=self.pad_token_id,
            reduce=False,
        ).view(-1, batch_size).sum(dim=0)
