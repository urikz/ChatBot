from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F

from .rnn_encoder_model import RnnEncoderModel


class Seq2SeqModel(RnnEncoderModel):

    ForwardResult = namedtuple('ForwardResult', ['logits', 'state'])

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
    ):
        super(Seq2SeqModel, self).__init__(
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

        self.decoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bias=True,
        )
        self.output_projection = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )
        if self.device_id is not None:
            self.cuda(self.device_id)

    def decode(self, input_target, input_state, context):
        emb_input_target = F.dropout(
            self.embeddings(input_target),
            p=self.dropout,
            training=self.training,
        )
        decoder_output, output_state = self.decoder(
            emb_input_target,
            input_state,
        )
        logits = self.output_projection(decoder_output)
        return Seq2SeqModel.ForwardResult(logits=logits, state=output_state)
