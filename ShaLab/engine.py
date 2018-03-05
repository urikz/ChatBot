import argparse
from collections import namedtuple
import glob
import logging
import numpy as np
import re
import os
import time
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import tqdm

from data.corpus import PackedIndexedParallelCorpus, data_loader
from data.vocab import Vocabulary, WordVocabulary
import embeddings
from models.model import DialogModel
import torch_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]'
)

class Engine(object):

    StepResult = namedtuple(
        'StepResult',
        [
            'loss',
            'ppl',
            'grad_norm',
            'num_samples',
            'target_length',
            'target_pads',
            'source_length',
            'source_pads',
        ],
    )

    def _get_optimizer(optimizer_params, model):
        if optimizer_params['optim'] == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=optimizer_params['learning_rate'],
                momentum=optimizer_params['momentum'],
                nesterov=False,
            )
        elif optimizer_params['optim'] == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=optimizer_params['learning_rate'],
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        else:
            raise Exception(
                'Unknown type of the optimizer %s' % str(optimizer_params)
            )

    def _restore_optimizer(self, checkpoint):
        if 'optimizer' in checkpoint:
            checkpoint = checkpoint['optimizer']
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        # Manually fix the device.
        # See https://github.com/pytorch/pytorch/issues/2830 for details
        self.optimizer.load_state_dict(checkpoint)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                state[k] = self.model.to_device(v, async=False)

    def __init__(self, model, vocab, log_interval, optimizer_params):
        self.model = model
        self.vocab = vocab
        self.log_interval = log_interval
        self.criterion = nn.CrossEntropyLoss(
            size_average=False,
            ignore_index=self.vocab.pad_idx,
        )
        if self.model.get_device_id() is not None:
            self.criterion = self.criterion.cuda(self.model.get_device_id())
        self.optimizer_params = optimizer_params
        self.optimizer = Engine._get_optimizer(self.optimizer_params, model)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=0.25,
            patience=0,
            cooldown=1,
            verbose=True,
        )
        self.log_interval = log_interval
        self.global_step = 0
        self.global_num_samples = 0
        self.epoch = 0
        self.best_validation_ppl = float('inf')

    def get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def lr_update(self, validation_ppl):
        self.best_validation_ppl = min(self.best_validation_ppl, validation_ppl)
        self.lr_scheduler.step(validation_ppl, self.epoch)

    @staticmethod
    def get_latest_chechpoint(dirpath):
        latest_checkpoint = (None, None)
        for f in os.scandir(dirpath):
            m = re.match(r'(.*)\.epoch-([0-9]+)\.pth\.tar$', f.name)
            if m is not None:
                epoch_index = int(m.group(2))
                if (
                    latest_checkpoint[1] is None or
                    latest_checkpoint[1] < epoch_index
                ):
                    latest_checkpoint = (f.path, epoch_index)
        return latest_checkpoint[0]

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        logging.info('Set a directory for checkpoints: %s' % checkpoint_dir)
        latest_checkpoint = Engine.get_latest_chechpoint(self.checkpoint_dir)

        if latest_checkpoint is not None:
            start_time = time.time()
            logging.info(
                'Found previous checkpoint %s. Picking up the training from there',
                latest_checkpoint,
            )
            checkpoint = torch.load(
                latest_checkpoint,
                map_location=self.model.get_map_location(),
            )
            self.model.load_from_checkpoint(checkpoint)
            self._restore_optimizer(checkpoint)

            self.epoch = checkpoint['epoch']
            validation_ppl = checkpoint['validation_ppl']
            self.best_validation_ppl = checkpoint['best_validation_ppl']
            self.lr_scheduler.step(self.best_validation_ppl, self.epoch)
            logging.info(
                (
                    'Training successfully picked up from the previous checkpoint. '
                    'Epoch %d, best valid ppl so far %.2f, valid ppl %s'
                ) % (
                    self.epoch,
                    self.best_validation_ppl,
                    str(validation_ppl),
                )
            )

    def step(
        self,
        use_optimizer,
        input_source,
        input_target,
        output_target,
        input_person_target=None,
    ):
        batch_size = input_source.size(1)
        assert batch_size == input_target.size(1)
        assert batch_size == output_target.size(1)

        input_source = self.model.to_device(input_source)
        input_target = self.model.to_device(input_target)
        output_target = self.model.to_device(output_target)

        target_length = (output_target != self.vocab.pad_idx).sum()
        target_pads = (output_target == self.vocab.pad_idx).sum()
        source_length = (input_source != self.vocab.pad_idx).sum()
        source_pads = (input_source == self.vocab.pad_idx).sum()

        if input_person_target is not None:
            input_person_target = Variable(
                self.model.to_device(input_person_target)
            )

        input_source = Variable(input_source)
        input_target = Variable(input_target)
        output_target = Variable(output_target)

        if use_optimizer:
            self.optimizer.zero_grad()

        logits = self.model.forward(
            input_source=input_source,
            input_target=input_target,
            input_person_target=input_person_target,
        )
        loss = self.criterion(
            logits.view(-1, len(self.vocab)),
            output_target.view(-1),
        )
        loss_scalar = loss.data[0]
        if use_optimizer:
            loss /= batch_size
            loss.backward()
            grad_norm = torch_utils.clip_grad_norm(
                self.model.parameters(),
                max_norm=5.0,
            )
            self.optimizer.step()
        else:
            grad_norm = 0

        return Engine.StepResult(
            loss=loss_scalar,
            ppl=np.exp(loss_scalar / target_length),
            grad_norm=grad_norm,
            num_samples=batch_size,
            target_length=target_length,
            target_pads=target_pads,
            source_length=source_length,
            source_pads=source_pads,
        )

    def step_train(self, batch):
        self.global_step += 1
        result = self.step(*([True] + batch))
        self.global_num_samples += result.num_samples
        return result

    def step_valid(self, batch):
        return self.step(*([False] + batch))

    def run(self, data_loader, tqdm_message, log_interval, step_fn):
        if tqdm_message is not None:
            data_loader = tqdm.tqdm(
                data_loader,
                desc=tqdm_message,
                bar_format=TRAINING_TQDM_BAD_FORMAT,
            )

        def log_progress(message):
            if tqdm_message is not None:
                tqdm.tqdm.write(message)

        total_loss, total_grad_norm, total_num_samples = 0, 0, 0
        total_source_length, total_source_pads = 0, 0
        total_target_length, total_target_pads = 0, 0

        for batch_id, batch in enumerate(data_loader, start=1):
            step_result = step_fn(batch)

            total_loss += step_result.loss
            total_grad_norm += step_result.grad_norm
            total_num_samples += step_result.num_samples
            total_source_length += step_result.source_length
            total_source_pads += step_result.source_pads
            total_target_length += step_result.target_length
            total_target_pads += step_result.target_pads

            if log_interval is not None and batch_id % log_interval == 0:
                log_progress(
                    (
                        '----- Batch %d (%d samples, '
                        '%.2f avg source length, %.2f avg target length) '
                        'ppl %.3f, avg grad norm %.5f'
                    ) % (
                        self.global_step,
                        self.global_num_samples,
                        float(total_source_length) / total_num_samples,
                        float(total_target_length) / total_num_samples,
                        np.exp(total_loss / total_target_length),
                        total_grad_norm / log_interval,
                    )
                )
                source_pads_pct = 100.0 * total_source_pads / (total_source_pads + total_source_length)
                target_pads_pct = 100.0 * total_target_pads / (total_target_pads + total_target_length)
                if (max(source_pads_pct, target_pads_pct) > 5.0):
                    log_progress(
                        (
                            'WARNING: There are too many pad tokens: '
                            '%.2f%% source pads, %.2f%% target pads'
                        ) % (
                            source_pads_pct,
                            target_pads_pct,
                        )
                    )

                total_loss, total_grad_norm, total_num_samples = 0, 0, 0
                total_source_length, total_source_pads = 0, 0
                total_target_length, total_target_pads = 0, 0
        if tqdm_message is not None:
            data_loader.close()

        return Engine.StepResult(
            loss=total_loss,
            ppl=np.exp(total_loss / total_target_length),
            grad_norm=total_grad_norm,
            num_samples=total_num_samples,
            source_length=total_source_length,
            source_pads=total_source_pads,
            target_length=total_target_length,
            target_pads=total_target_pads,
        )

    def train(self, data_loader):
        self.epoch += 1
        self.model.train()
        return self.run(
            data_loader=data_loader,
            tqdm_message='Training epoch %d' % self.epoch,
            log_interval=self.log_interval,
            step_fn=lambda batch: self.step_train(batch),
        )

    def valid(self, data_loader, tune_corpus_name, use_progress_bar):
        self.model.eval()
        tqdm_message = (
            'Validation [%s] after epoch %d' % (tune_corpus_name, self.epoch)
            if use_progress_bar
            else None
        )
        result = self.run(
            data_loader=data_loader,
            tqdm_message=tqdm_message,
            log_interval=None,
            step_fn=lambda batch: self.step_valid(batch),
        )
        print(
            (
                '----- Validation [%s] after epoch %d (%d samples, '
                '%.2f avg source length, %.2f avg target length) '
                'perplexity %.3f'
            ) % (
                tune_corpus_name,
                self.epoch,
                result.num_samples,
                float(result.source_length) / result.num_samples,
                float(result.target_length) / result.num_samples,
                np.exp(result.loss / result.target_length),
            ),
        )
        return result

    def save_checkpoint(self, validation_ppl, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        output_file = os.path.join(
            checkpoint_dir,
            'model.checkpoint.epoch-%d.pth.tar' % self.epoch
        )
        torch.save(
            {
                'model': self.model.serialize(),
                'optimizer': {
                    'params': self.optimizer_params,
                    'state_dict': self.optimizer.state_dict(),
                },
                'epoch': self.epoch,
                'timestamp': time.time(),
                'validation_ppl': validation_ppl,
                'best_validation_ppl': self.best_validation_ppl
            },
            output_file,
        )
        logging.info(
            'Checkpoint has been successfully saved to %s' % output_file
        )


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vocab = WordVocabulary.from_file(os.path.join(args.data, 'vocab.tsv'))
    if args.person_embedding_size > 0:
        person_vocab = Vocabulary.from_file(
            os.path.join(args.data, 'person.vocab.tsv')
        )
    else:
        person_vocab = None

    train_corpus = PackedIndexedParallelCorpus.from_file(
        os.path.join(args.data, 'train.source.npz'),
        os.path.join(args.data, 'train.target.npz'),
        (
            os.path.join(args.data, 'train.person.target.npz')
            if os.path.isfile(os.path.join(args.data, 'train.person.target.npz'))
            else None
        ),
    )
    tune_corpora = {}
    tune_corpus_candidates = glob.glob(
        os.path.join(args.data, 'tune*.source.npz')
    )
    for tune_corpus_candidate_s in tune_corpus_candidates:
        if not tune_corpus_candidate_s.endswith('person.source.npz'):
            corpus_name = os.path.basename(tune_corpus_candidate_s)[:-len('.source.npz')]
            logging.info('Detected another tune corpus "%s"', corpus_name)
            assert corpus_name not in tune_corpora
            tune_corpus_candidate_t = (
                os.path.join(args.data, corpus_name) + '.target.npz'
            )
            if args.person_embedding_size > 0:
                tune_corpus_candidate_t_p = (
                    os.path.join(args.data, corpus_name) + '.person.target.npz'
                )
                tune_corpora[corpus_name] = PackedIndexedParallelCorpus.from_file(
                    tune_corpus_candidate_s,
                    tune_corpus_candidate_t,
                    tune_corpus_candidate_t_p,
                )
            else:
                tune_corpora[corpus_name] = PackedIndexedParallelCorpus.from_file(
                    tune_corpus_candidate_s,
                    tune_corpus_candidate_t,
                )
    assert 'tune' in tune_corpora

    if not os.path.isdir(args.output):
        assert not os.path.exists(args.output)
        os.makedirs(args.output)

    model = DialogModel(
        device_id=args.gpu,
        vocab_size=len(vocab),
        pad_token_id=vocab.pad_idx,
        unk_token_id=vocab.unk_idx,
        go_token_id=vocab.go_idx,
        eos_token_id=vocab.eos_idx,
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        person_vocab_size=(
            len(person_vocab)
            if person_vocab is not None
            else None
        ),
        person_embedding_size=args.person_embedding_size,
    )
    if args.glove is not None:
        model.set_embeddings(embeddings.load_glove_embeddings(
            path=args.glove,
            vocab=vocab,
            embedding_size=args.embedding_size,
        ))

    engine = Engine(
        model=model,
        vocab=vocab,
        log_interval=args.log_interval,
        optimizer_params={
            'optim': args.optimizer,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
        },
    )
    engine.set_checkpoint_dir(args.output)

    while engine.epoch < args.num_epochs:
        print(
            '----- Epoch %d/%d, learing rate = %.5lf' % (
                engine.epoch + 1,
                args.num_epochs,
                engine.get_learning_rate(),
            ),
        )
        engine.train(data_loader=data_loader(
            corpus=train_corpus,
            vocab=vocab,
            batch_size=args.batch_size,
            sort_batches=args.sort_batches,
            reverse_source=False,
            num_data_workers=args.num_data_workers,
            verbose=True,
        ))
        validation_ppl = {
             tune_corpus_name: engine.valid(
                data_loader=data_loader(
                    corpus=tune_corpus,
                    vocab=vocab,
                    batch_size=args.batch_size,
                    sort_batches=False,
                    reverse_source=False,
                    num_data_workers=args.num_data_workers,
                    verbose=False,
                ),
                tune_corpus_name=tune_corpus_name,
                use_progress_bar=False,
            ).ppl
            for tune_corpus_name, tune_corpus in tune_corpora.items()
        }

        engine.lr_update(validation_ppl['tune'])
        engine.save_checkpoint(validation_ppl)

    print(
        'Training has finished with the best validation ppl %.5f' % (
            engine.best_validation_ppl
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    parser.add_argument(
        '-d', '--data',
        type=str,
        help=(
            'Prefix for path to the data. '
            'It should contains these files: vocab.tsv, train.source.npz, '
            'train.target.npz, tune.source.npz, tune.target.npz'
        ),
    )
    # parser.add_argument(
    #     '--base-model',
    #     type=str,
    #     defult=None,
    #     help='Prefix for path to the initial model dir (default = None).',
    # )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Prefix for path to the checkpointing dir.',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU ID (default - CPU)',
    )
    parser.add_argument(
        '--num-data-workers',
        type=int,
        default=1,
        help='Number of workers for data loader',
    )
    parser.add_argument(
        '--sort-batches',
        action='store_true',
        default=False,
        help=(
            'Will sort batches by length '
            'to group sentences with similar length. '
            'Use this option for larger (>1M pair of sentences) datasets.'
        ),
    )
    parser.add_argument(
        '--glove',
        type=str,
        default=None,
        help='Initialize embeddings with GloVe embeddings file',
    )
    parser.add_argument(
        '-optim', '--optimizer',
        type=str,
        default='sgd',
        help='Type of the optimizer. Possible values - sgd,adam',
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        type=float,
        default=1.0,
        help='Learning rate',
    )
    parser.add_argument(
        '-m', '--momentum',
        type=float,
        default=0,
        help='Momentum',
    )
    parser.add_argument(
        '-emsz', '--embedding-size',
        type=int,
        help='Dim of the hidden layer of RNN',
    )
    parser.add_argument(
        '-hsz', '--hidden-size',
        type=int,
        help='Dim of the hidden layer of RNN',
    )
    parser.add_argument(
        '-nlayers', '--num-layers',
        type=int,
        default=1,
        help='Number of layers for RNNs',
    )
    parser.add_argument(
        '-bs', '--batch-size',
        type=int,
        default=32,
        help='Batch size',
    )
    parser.add_argument(
        '-dp', '--dropout',
        type=float,
        default=0,
        help='Dropout rate',
    )
    parser.add_argument('--seed', type=int, default=31415, help='Random seed')

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1,
        help='Number of epochs',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='How many batches to wait before logging training status',
    )
    parser.add_argument(
        '--person-embedding-size',
        type=int,
        default=0,
        help='Size of persona emebddings',
    )
    args = parser.parse_args()

    if args.embedding_size is None:
        args.embedding_size = args.hidden_size

    assert args.num_epochs > 0

    main(args)
