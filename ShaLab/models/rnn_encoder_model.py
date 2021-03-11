import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


class RnnEncoderModel(BaseModel):
    def __init__(
        self,
        device_id,
        vocab_size,
        pad_token_id,
        unk_token_id,
        go_token_id,
        eos_token_id,
        embedding_size,
        dropout,
        num_layers,
        hidden_size,
    ):
        super(RnnEncoderModel, self).__init__(
            device_id=device_id,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            go_token_id=go_token_id,
            eos_token_id=eos_token_id,
            embedding_size=embedding_size,
        )
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.initial_encoder_state = torch.zeros(
            self.num_layers,
            1,
            self.hidden_size,
            requires_grad=False,
        )
        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bias=True,
        )

        if self.device_id is not None:
            self.cuda(self.device_id)
            self.initial_encoder_state = self.to_device(
                self.initial_encoder_state
            )

    def get_parameters(self):
        parameters = super(RnnEncoderModel, self).get_parameters()
        parameters.update({
            'dropout': self.dropout,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
        })
        return parameters

    def encode(self, input_source):
        emb_input_source = F.dropout(
            self.embeddings(input_source),
            p=self.dropout,
            training=self.training,
        )
        batch_size = input_source.size(1)
        if self.initial_encoder_state.size(1) != batch_size:
            self.initial_encoder_state.resize_(
                self.initial_encoder_state.size(0),
                batch_size,
                self.initial_encoder_state.size(2),
            ).fill_(0)

        _, final_encoder_state = self.encoder(
            emb_input_source,
            (self.initial_encoder_state, self.initial_encoder_state),
        )
        return final_encoder_state
