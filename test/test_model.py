import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import unittest

from teds.model import DialogModel
from teds.data import prepare_batch_from_parallel_samples

class CopyDataParalleCorpus(torch.utils.data.Dataset):
    GO_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    PAD_TOKEN_ID = 2
    NUM_SPECIAL_TOKENS = 3

    def __init__(self, num_values, max_length, dataset_size):
        assert num_values > 0
        self.max_value = CopyDataParalleCorpus.NUM_SPECIAL_TOKENS + num_values
        self.max_length = max_length
        self.dataset_size = dataset_size

    def __getitem__(self, index):
        length = np.random.randint(low=1, high=self.max_length + 1)
        x = np.random.randint(
            low=CopyDataParalleCorpus.NUM_SPECIAL_TOKENS,
            high=self.max_value,
            size=length,
        )
        return (x, x, x)

    def __len__(self):
        return self.dataset_size

    def get_max_value(self):
        return self.max_value


class TestDialogModel(unittest.TestCase):
    def run_test_dialog_model(
        self,
        hidden_size,
        num_values,
        max_length,
        dataset_size,
        batch_size,
        test_dataset_size,
    ):
        corpus = CopyDataParalleCorpus(
            num_values=num_values,
            max_length=max_length,
            dataset_size=dataset_size,
        )
        dataloader = torch.utils.data.DataLoader(
            corpus,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda samples: prepare_batch_from_parallel_samples(
                parallel_samples=samples,
                pad_token_id=CopyDataParalleCorpus.PAD_TOKEN_ID,
                eos_token_id=CopyDataParalleCorpus.EOS_TOKEN_ID,
                go_token_id=CopyDataParalleCorpus.GO_TOKEN_ID,
                reverse_source=False,
            ),
        )
        model = DialogModel(
            vocab_size=corpus.get_max_value(),
            num_layers=1,
            hidden_size=hidden_size,
            dropout=0,
            pad_token_id=CopyDataParalleCorpus.PAD_TOKEN_ID,
        )
        crit = nn.CrossEntropyLoss(
            size_average=False,
            ignore_index=CopyDataParalleCorpus.PAD_TOKEN_ID,
        )
        opt = optim.SGD(model.parameters(), lr=1)

        for batch in dataloader:
            (input_source, input_target, output_target) = batch
            opt.zero_grad()
            logits = model.forward(
                input_source=Variable(input_source),
                input_target=Variable(input_target),
            )
            loss = crit(
                logits.view(-1, corpus.get_max_value()),
                Variable(output_target.contiguous()).view(-1),
            )
            loss /= batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
            opt.step()

        test_corpus = CopyDataParalleCorpus(
            num_values=num_values,
            max_length=max_length,
            dataset_size=test_dataset_size,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_corpus,
            batch_size=test_dataset_size,
            shuffle=False,
            collate_fn=lambda samples: prepare_batch_from_parallel_samples(
                parallel_samples=samples,
                pad_token_id=CopyDataParalleCorpus.PAD_TOKEN_ID,
                eos_token_id=CopyDataParalleCorpus.EOS_TOKEN_ID,
                go_token_id=CopyDataParalleCorpus.GO_TOKEN_ID,
                reverse_source=False,
            ),
        )
        test_loss = 0
        for batch in test_dataloader:
            (input_source, input_target, output_target) = batch
            logits = model.forward(
                input_source=Variable(input_source),
                input_target=Variable(input_target),
            )
            loss = crit(
                logits.view(-1, corpus.get_max_value()),
                Variable(output_target.contiguous()).view(-1),
            )
            test_loss += loss.data[0]
        return test_loss / test_dataset_size

    def test_copy_length(self):
        loss = self.run_test_dialog_model(
            hidden_size=32,
            num_values=1,
            max_length=5,
            dataset_size=16 * 200,
            batch_size=16,
            test_dataset_size=32,
        )
        print('\nLoss for copy-length test examples: %f' % loss)
        self.assertLess(loss, 0.1)

    def test_copy_binary(self):
        loss = self.run_test_dialog_model(
            hidden_size=32,
            num_values=2,
            max_length=4,
            dataset_size=16 * 1000,
            batch_size=16,
            test_dataset_size=32,
        )
        print('\nLoss for copy-binary test examples: %f' % loss)
        self.assertLess(loss, 0.1)


if __name__ == '__main__':
    unittest.main()
