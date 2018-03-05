from collections import namedtuple
import heapq
import logging
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ShaLab.models.base_model import BaseModel, get_map_location

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

BIG_NEGATIVE_SCORE = -10000

class DialogModel(BaseModel):
    def __init__(
        self,
        device_id,
        vocab_size,
        pad_token_id,
        unk_token_id,
        go_token_id,
        eos_token_id,
        num_layers,
        embedding_size,
        hidden_size,
        dropout,
        person_vocab_size=None,
        person_embedding_size=None,
    ):
        super(DialogModel, self).__init__(
            device_id=device_id,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            go_token_id=go_token_id,
            eos_token_id=eos_token_id,
        )

        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=self.pad_token_id,
            sparse=True,
        )
        decoder_input_size = self.embedding_size
        self.person_vocab_size = person_vocab_size
        self.person_embedding_size = person_embedding_size
        if self.person_vocab_size is not None:
            assert self.person_embedding_size is not None
            self.person_embeddings = nn.Embedding(
                num_embeddings=self.person_vocab_size,
                embedding_dim=self.person_embedding_size,
                sparse=True,
            )
            self.person_embeddings_dropout = nn.Dropout(self.dropout)
            decoder_input_size += self.person_embedding_size

        self.encoder_input_dropout = nn.Dropout(self.dropout)
        self.decoder_input_dropout = nn.Dropout(self.dropout)

        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=True,
        )
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=True,
        )
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        if self.device_id is not None:
            self.cuda(self.device_id)
        self.init_weights()

    def init_weights(self):
        self.output_projection.bias.data.fill_(0)

    def get_embeddings(self):
        return self.embeddings

    def set_embeddings(self, embeddings):
        self.embeddings.weight = nn.Parameter(
            self.to_device(torch.from_numpy(embeddings))
        )

    def get_person_vocab_size(self):
        return self.person_vocab_size

    def encode(self, input_source):
        emb_input_source = self.encoder_input_dropout(
            self.embeddings(input_source)
        )
        _, final_encoder_state = self.encoder(emb_input_source, None)
        return final_encoder_state

    def decode(self, input_target, input_state, input_person_target=None):
        emb_input_target = self.decoder_input_dropout(
            self.embeddings(input_target)
        )
        if input_person_target is not None:
            emb_input_person_target = self.person_embeddings_dropout(
                self.person_embeddings(input_person_target)
            )
            emb_input_person_target = emb_input_person_target.unsqueeze(0)
            emb_input_person_target = emb_input_person_target.expand((
                emb_input_target.size(0),
                emb_input_target.size(1),
                emb_input_person_target.size(2),
            ))
            emb_input_target = torch.cat(
                (emb_input_target, emb_input_person_target),
                2,
            )

        decoder_output, output_state = self.decoder(
            emb_input_target,
            input_state,
        )
        logits = self.output_projection(decoder_output)
        return (logits, output_state)

    def forward(self, input_source, input_target, input_person_target=None):
        final_encoder_state = self.encode(input_source)
        logits, _ = self.decode(
            input_target,
            final_encoder_state,
            input_person_target,
        )
        return logits

    def log_prob(
        self,
        input_source,
        input_target,
        output_target,
        person,
    ):
        batch_size = input_source.size(1)
        assert batch_size == input_target.size(1)
        assert input_target.size() == output_target.size()
        logits = self.forward(
            input_source=input_source,
            input_target=input_target,
            input_person_target=person,
        )
        return -F.cross_entropy(
            logits.view(-1, self.vocab_size),
            output_target.view(-1),
            ignore_index=self.pad_token_id,
            reduce=False,
        ).view(-1, batch_size).sum(0)

    def get_parameters(self):
        parameters = super(DialogModel, self).get_parameters()
        parameters.update({
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'person_vocab_size': self.person_vocab_size,
            'person_embedding_size': self.person_embedding_size,
        })
        return parameters

    def serialize(self):
        return {
            'state_dict': self.state_dict(),
            'params': self.get_parameters(),
        }

    def load_from_checkpoint(self, checkpoint):
        start_time = time.time()
        if isinstance(checkpoint, str):
            # We need to load checkpoint from the file first
            checkpoint = torch.load(
                checkpoint,
                map_location=self.get_map_location(),
            )

        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        weight_name = 'decoder.weight_ih_l0'
        weight_numpy = checkpoint['state_dict'][weight_name].cpu().numpy()
        if (
            weight_numpy.shape !=
            (4 * self.hidden_size, self.embedding_size + self.person_embedding_size)
        ):
            logging.info(
                'Adjusting parameter %s for personalized model',
                weight_name,
            )
            fixed_weight_t = np.random.uniform(
                low=-0.1,
                high=0.1,
                size=(
                    self.embedding_size + self.person_embedding_size,
                    4 * self.hidden_size,
                ),
            )
            fixed_weight_t[:self.embedding_size][:] = weight_numpy.transpose().copy()
            checkpoint['state_dict'][weight_name] = self.to_device(
                torch.from_numpy(fixed_weight_t.transpose().copy())
            )
        self.load_state_dict(checkpoint['state_dict'])
        logging.info(
            'Loaded dialog model from checkpoint in %d seconds',
            time.time() - start_time,
        )

    @staticmethod
    def create_from_checkpoint(path, device_id):
        start_time = time.time()
        checkpoint = torch.load(
            path,
            map_location=get_map_location(device_id),
        )
        checkpoint['model']['params']['device_id'] = device_id
        model = DialogModel(**checkpoint['model']['params'])
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
