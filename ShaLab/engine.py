import argparse
from collections import namedtuple
import logging
import numpy as np
import re
import os
import time
import torch
from torch import optim
import torch.nn as nn
import tqdm

from ShaLab.data import Dataset
import ShaLab.embeddings as embeddings
from ShaLab.models import ModelFactory
import ShaLab.torch_utils as torch_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]'
)

BIG_NUMBER = 1000000


class Engine(object):

    StepResult = namedtuple(
        'StepResult',
        [
            'loss',
            'prediction_loss',
            'estimation_loss',
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
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=optimizer_params['learning_rate'],
                momentum=optimizer_params['momentum'],
                nesterov=False,
            )
        elif optimizer_params['optim'] == 'adam':
            return optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
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

    def __init__(
        self,
        model,
        vocab,
        log_interval,
        optimizer_params,
        profile_memory_estimation_weight=0,
        profile_memory_estimation_best_match=False,
        verbose=True,
    ):
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
            verbose=verbose,
        )
        self.profile_memory_estimation_weight = profile_memory_estimation_weight
        self.profile_memory_estimation_best_match = profile_memory_estimation_best_match
        self.log_interval = log_interval
        self.global_step = 0
        self.global_num_samples = 0
        self.epoch = 0
        self.best_valid_ppl = float('inf')

    def get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def lr_update(self, valid_ppl):
        self.best_valid_ppl = min(self.best_valid_ppl, valid_ppl)
        self.lr_scheduler.step(valid_ppl, self.epoch)

    @staticmethod
    def get_chechpoints(dirpath):
        checkpoints = []
        for f in os.scandir(dirpath):
            m = re.match(r'(.*)\.epoch-([0-9]+)\.pth\.tar$', f.name)
            if m is not None:
                epoch_index = int(m.group(2))
                checkpoints.append((f.path, epoch_index))
        return checkpoints

    @staticmethod
    def get_latest_chechpoint(dirpath):
        checkpoints = Engine.get_chechpoints(dirpath)
        if len(checkpoints) == 0:
            return None
        else:
            return max(
                checkpoints,
                key=lambda x: x[1],
            )[0]

    @staticmethod
    def get_checkpoint_ppl(path):
        return torch.load(path, 'cpu')['eval_ppl'][
            Dataset.VALID_CORPUS_DEFAULT_NAME
        ]

    @staticmethod
    def get_best_chechpoint(dirpath):
        checkpoints = Engine.get_chechpoints(dirpath)
        if len(checkpoints) == 0:
            return None
        else:
             return min(
                checkpoints,
                key=lambda x: Engine.get_checkpoint_ppl(x[0]),
            )[0]

    @staticmethod
    def get_best_checkpoint_ppl(dirpath):
        best_checkpoint = Engine.get_best_chechpoint(dirpath)
        if best_checkpoint is not None:
            return Engine.get_checkpoint_ppl(best_checkpoint)
        else:
            return None

    def set_checkpoint_dir(self, checkpoint_dir, verbose=True):
        start_time = time.time()
        self.checkpoint_dir = checkpoint_dir
        if verbose:
            logging.info('Set a directory for checkpoints: %s' % checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            assert not os.path.exists(checkpoint_dir)
            os.makedirs(checkpoint_dir)

        latest_checkpoint = Engine.get_latest_chechpoint(self.checkpoint_dir)

        if latest_checkpoint is not None:
            start_time = time.time()
            if verbose:
                logging.info(
                    (
                        'Found previous checkpoint %s in %d seconds. '
                        'Picking up the training from there'
                    ),
                    latest_checkpoint,
                    time.time() - start_time,
                )
            checkpoint = torch.load(
                latest_checkpoint,
                map_location=self.model.get_map_location(),
            )
            self.model.load_from_checkpoint(checkpoint, verbose=verbose)
            self._restore_optimizer(checkpoint)

            self.epoch = checkpoint['epoch']
            eval_ppl = checkpoint['eval_ppl']
            self.best_valid_ppl = Engine.get_best_checkpoint_ppl(checkpoint_dir)
            self.lr_scheduler.step(self.best_valid_ppl, self.epoch)
            if verbose:
                logging.info(
                    (
                        'Training successfully picked up from '
                        'the previous checkpoint in %d seconds. '
                        'Epoch %d, best valid ppl so far %.2f, valid ppl %s'
                    ) % (
                        time.time() - start_time,
                        self.epoch,
                        self.best_valid_ppl,
                        str(eval_ppl),
                    )
                )

    def step(self, batch, use_optimizer):
        input_source = batch.get('input_source')
        input_target = batch.get('input_target')
        output_target = batch.get('output_target')
        context = batch.get('profile_memory')
        profile_memory_mask = batch.get('profile_memory_mask')

        batch_size = input_source.size(1)
        assert batch_size == input_target.size(1)
        assert batch_size == output_target.size(1)

        input_source = self.model.to_device(input_source)
        input_target = self.model.to_device(input_target)
        output_target = self.model.to_device(output_target)

        if context is not None:
            context = self.model.to_device(context)
            profile_memory_mask = self.model.to_device(profile_memory_mask)

        target_length = (output_target != self.vocab.pad_idx).long().sum().item()
        target_pads = (output_target == self.vocab.pad_idx).long().sum().item()
        source_length = (input_source != self.vocab.pad_idx).long().sum().item()
        source_pads = (input_source == self.vocab.pad_idx).long().sum().item()

        if use_optimizer:
            self.optimizer.zero_grad()

        forward_result = self.model.forward(
            input_source,
            input_target,
            context,
            apply_final_attention=(context is not None),
        )
        loss = self.criterion(
            forward_result.logits.view(-1, len(self.vocab)),
            output_target.view(-1),
        )
        prediction_loss_scalar = loss.item()
        ppl = np.exp(prediction_loss_scalar / target_length),
        if use_optimizer:
            loss /= batch_size

            if forward_result.final_attention is not None:
                if self.profile_memory_estimation_best_match:
                    best_scores, best_predictions = torch.min(
                        (
                            -forward_result.final_attention * profile_memory_mask
                            + BIG_NUMBER * (1.0 - profile_memory_mask)
                        ),
                        dim=1,
                        keepdim=True,
                    )
                    effective_batch_size = (
                        (best_scores < BIG_NUMBER - 1e-8).sum().float() + 1e-8
                    )
                    estimation_loss = torch.gather(
                        -forward_result.final_attention * profile_memory_mask,
                        dim=1,
                        index=best_predictions.detach(),
                    ).sum() / effective_batch_size.detach()
                else:
                    estimation_loss = - (
                        forward_result.final_attention * profile_memory_mask
                    ).sum() / (profile_memory_mask.sum() + 1e-8)

                estimation_loss_scalar = estimation_loss.item()
                if self.profile_memory_estimation_weight > 0:
                    loss += (
                        self.profile_memory_estimation_weight * estimation_loss
                    )
            else:
                estimation_loss_scalar = 0

            loss.backward()
            grad_norm = torch_utils.clip_grad_norm(
                self.model.parameters(),
                max_norm=5.0,
            ).item()
            self.optimizer.step()
        else:
            grad_norm = 0
            estimation_loss_scalar = 0

        return Engine.StepResult(
            loss=loss.item(),
            prediction_loss=prediction_loss_scalar,
            estimation_loss=estimation_loss_scalar,
            ppl=ppl,
            grad_norm=grad_norm,
            num_samples=batch_size,
            target_length=target_length,
            target_pads=target_pads,
            source_length=source_length,
            source_pads=source_pads,
        )

    def step_train(self, batch):
        self.global_step += 1
        result = self.step(batch, use_optimizer=True)
        self.global_num_samples += result.num_samples
        return result

    def step_valid(self, batch):
        return self.step(batch, use_optimizer=False)

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

        total_batches, total_num_samples = 0, 0
        total_estimation_loss, total_prediction_loss = 0, 0
        total_loss, total_grad_norm = 0, 0
        total_source_length, total_source_pads = 0, 0
        total_target_length, total_target_pads = 0, 0

        for batch_id, batch in enumerate(data_loader, start=1):
            step_result = step_fn(batch)

            total_loss += step_result.loss
            total_prediction_loss += step_result.prediction_loss
            total_estimation_loss += step_result.estimation_loss
            total_grad_norm += step_result.grad_norm
            total_num_samples += step_result.num_samples
            total_batches += 1
            total_source_length += step_result.source_length
            total_source_pads += step_result.source_pads
            total_target_length += step_result.target_length
            total_target_pads += step_result.target_pads

            if (
                log_interval is not None and
                (batch_id % log_interval == 0 or batch_id == len(data_loader) - 1)
            ):
                log_progress(
                    (
                        '----- Batch %d (%d samples, '
                        '%.2f avg source length, %.2f avg target length) '
                        'ppl %.3f, avg grad norm %.5f, '
                        'loss %.3f, estimation loss %.3f'
                    ) % (
                        self.global_step,
                        self.global_num_samples,
                        float(total_source_length) / total_num_samples,
                        float(total_target_length) / total_num_samples,
                        np.exp(total_prediction_loss / total_target_length),
                        total_grad_norm / total_batches,
                        total_loss / total_batches,
                        total_estimation_loss / total_batches,
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

                total_batches, total_num_samples = 0, 0
                total_estimation_loss, total_prediction_loss = 0, 0
                total_loss, total_grad_norm = 0, 0
                total_source_length, total_source_pads = 0, 0
                total_target_length, total_target_pads = 0, 0

        if tqdm_message is not None:
            data_loader.close()

        return Engine.StepResult(
            loss=total_loss,
            prediction_loss=total_prediction_loss,
            estimation_loss=total_estimation_loss,
            ppl=np.exp(total_loss / total_target_length),
            grad_norm=total_grad_norm,
            num_samples=total_num_samples,
            source_length=total_source_length,
            source_pads=total_source_pads,
            target_length=total_target_length,
            target_pads=total_target_pads,
        )

    def train(self, data_loader, verbose=True):
        self.epoch += 1
        self.model.train()
        return self.run(
            data_loader=data_loader,
            tqdm_message=(
                'Training epoch %d' % self.epoch
                if verbose else None
            ),
            log_interval=self.log_interval,
            step_fn=lambda batch: self.step_train(batch),
        )

    def valid(
        self,
        data_loader,
        valid_corpus_name,
        use_progress_bar,
        verbose=True,
    ):
        self.model.eval()
        tqdm_message = (
            'Validation [%s] after epoch %d' % (valid_corpus_name, self.epoch)
            if use_progress_bar
            else None
        )
        result = self.run(
            data_loader=data_loader,
            tqdm_message=tqdm_message,
            log_interval=None,
            step_fn=lambda batch: self.step_valid(batch),
        )
        if verbose:
            print(
                (
                    '----- Validation [%s] after epoch %d (%d samples, '
                    '%.2f avg source length, %.2f avg target length) '
                    'perplexity %.3f'
                ) % (
                    valid_corpus_name,
                    self.epoch,
                    result.num_samples,
                    float(result.source_length) / result.num_samples,
                    float(result.target_length) / result.num_samples,
                    result.ppl,
                ),
            )
        return result

    def save_checkpoint(self, eval_ppl, checkpoint_dir=None, verbose=True):
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
                'eval_ppl': eval_ppl,
            },
            output_file,
        )
        if verbose:
            logging.info(
                'Checkpoint has been successfully saved to %s' % output_file
            )

    def full_training(self, num_epochs, dataset, verbose=True):
        while self.epoch < num_epochs:
            if verbose:
                print(
                    '----- Epoch %d/%d, learing rate = %.5lf' % (
                        self.epoch + 1,
                        num_epochs,
                        self.get_learning_rate(),
                    ),
                )
            self.train(
                dataset.get_train_data_loader(verbose=verbose),
                verbose=verbose,
            )
            eval_ppl = {
                 corpus_name: self.valid(
                    data_loader=dl,
                    valid_corpus_name=corpus_name,
                    use_progress_bar=False,
                    verbose=verbose,
                ).ppl
                for corpus_name, dl in dataset.get_valid_data_loaders_map().items()
            }
            self.lr_update(eval_ppl[Dataset.VALID_CORPUS_DEFAULT_NAME])
            self.save_checkpoint(eval_ppl, verbose=verbose)

        if verbose:
            logging.info(
                'Training has finished with the best validation ppl %.5f' % (
                    self.best_valid_ppl
                )
            )

        return self.best_valid_ppl


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = Dataset(args)
    vocab = dataset.get_vocab()
    model = ModelFactory.model(args, vocab)

    if args.glove is not None:
        glove_embeddings, glove_index = embeddings.load_glove_embeddings(
            path=args.glove,
            vocab=vocab,
            embedding_size=args.embedding_size,
        )
        model.set_embeddings(glove_embeddings)
        if (
            args.profile_memory_attention is not None and
            args.init_profile_memory_weights
        ):
            model.init_embeddings_weights_using_glove_index(glove_index)

    engine = Engine(
        model=model,
        vocab=vocab,
        log_interval=args.log_interval,
        optimizer_params={
            'optim': args.optimizer,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
        },
        profile_memory_estimation_weight=args.profile_memory_estimation_weight,
        profile_memory_estimation_best_match=(
            args.profile_memory_estimation_best_match
        ),
    )
    engine.set_checkpoint_dir(args.output)

    engine.full_training(num_epochs=args.num_epochs, dataset=dataset)

    best_checkpoint_path = Engine.get_best_chechpoint(args.output)
    logging.info(
        'The best checkpoint %s. Picking up the model from there',
        best_checkpoint_path,
    )
    model.load_from_checkpoint(best_checkpoint_path)
    for corpus, dl in dataset.get_test_and_valid_data_loaders_map().items():
        engine.valid(dl, corpus, use_progress_bar=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatBot training script')
    Dataset.add_cmd_arguments(parser)
    ModelFactory.add_cmd_arguments(parser)

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
        '--init-profile-memory-weights',
        action='store_true',
        default=False,
        help='Initialize ProfileMemory weights using GloVe index',
    )
    parser.add_argument(
        '--profile-memory-estimation-weight',
        type=float,
        default=0,
        help='Weight for profile memory estimation objective',
    )
    parser.add_argument(
        '--profile-memory-estimation-best-match',
        action='store_true',
        default=False,
        help='Should we train estimatio based on all labels or only using the best match',
    )

    args = parser.parse_args()

    if args.embedding_size is None:
        args.embedding_size = args.hidden_size

    assert args.num_epochs > 0

    main(args)
