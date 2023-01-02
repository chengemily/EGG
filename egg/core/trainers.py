# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
<<<<<<< HEAD
import time
=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
import pathlib
from typing import List, Optional

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.utils.data import DataLoader

from .batch import Batch
from .callbacks import (
    Callback,
    Checkpoint,
    CheckpointSaver,
    ConsoleLogger,
    TensorboardLogger,
)
from .distributed import get_preemptive_checkpoint_dir
from .interaction import Interaction
from .util import get_opts, move_to

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    pass


<<<<<<< HEAD
def get_grad_norm(agent):
    with torch.no_grad():
        total_norm = 0
        for p in agent.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
class Trainer:
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        optimizer_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        convergence_epsilon: float = None,
        aggregate_interaction_logs: bool = True,
    ):
        """
        :param game: A nn.Module that implements forward(); it is expected that forward returns a tuple of (loss, d),
            where loss is differentiable loss to be minimized and d is a dictionary (potentially empty) with auxiliary
            metrics that would be aggregated and reported
        :param optimizer: An instance of torch.optim.Optimizer
        :param optimizer_scheduler: An optimizer scheduler to adjust lr throughout training
        :param train_data: A DataLoader for the training set
        :param validation_data: A DataLoader for the validation set (can be None)
        :param device: A torch.device on which to tensors should be stored
        :param callbacks: A list of egg.core.Callback objects that can encapsulate monitoring or checkpointing
        """
        self.game = game
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.train_data = train_data
        self.validation_data = validation_data
        common_opts = get_opts()
        self.validation_freq = common_opts.validation_freq
        self.device = common_opts.device if device is None else device
        self.should_stop = False
        self.start_epoch = 0  # Can be overwritten by checkpoint loader
        self.callbacks = callbacks if callbacks else []
        self.grad_norm = grad_norm
        self.convergence_epsilon = convergence_epsilon
        self.aggregate_interaction_logs = aggregate_interaction_logs

        self.update_freq = common_opts.update_freq

        if common_opts.load_from_checkpoint is not None:
            print(
                f"# Initializing model, trainer, and optimizer from {common_opts.load_from_checkpoint}"
            )
            self.load_from_checkpoint(common_opts.load_from_checkpoint)

        self.distributed_context = common_opts.distributed_context
        if self.distributed_context.is_distributed:
            print("# Distributed context: ", self.distributed_context)
        if self.distributed_context.is_leader and not any(
            isinstance(x, CheckpointSaver) for x in self.callbacks
        ):
            if common_opts.preemptable:
                assert (
                    common_opts.checkpoint_dir
                ), "checkpointing directory has to be specified"
                d = get_preemptive_checkpoint_dir(common_opts.checkpoint_dir)
                self.checkpoint_path = d
                self.load_from_latest(d)
            else:
                self.checkpoint_path = (
                    None
                    if common_opts.checkpoint_dir is None
                    else pathlib.Path(common_opts.checkpoint_dir)
                )

            if self.checkpoint_path:
                checkpointer = CheckpointSaver(
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_freq=common_opts.checkpoint_freq,
                )
                self.callbacks.append(checkpointer)

        if self.distributed_context.is_leader and common_opts.tensorboard:
            assert (
                common_opts.tensorboard_dir
            ), "tensorboard directory has to be specified"
            tensorboard_logger = TensorboardLogger()
            self.callbacks.append(tensorboard_logger)

        if self.callbacks is None:
            self.callbacks = [
                ConsoleLogger(print_train_loss=False, as_json=False),
            ]
        if self.distributed_context.is_distributed:
            device_id = self.distributed_context.local_rank
            torch.cuda.set_device(device_id)
            self.game.to(device_id)

            # NB: here we are doing something that is a bit shady:
            # 1/ optimizer was created outside of the Trainer instance, so we don't really know
            #    what parameters it optimizes. If it holds something what is not within the Game instance
            #    then it will not participate in distributed training
            # 2/ if optimizer only holds a subset of Game parameters, it works, but somewhat non-documentedly.
            #    In fact, optimizer would hold parameters of non-DistributedDataParallel version of the Game. The
            #    forward/backward calls, however, would happen on the DistributedDataParallel wrapper.
            #    This wrapper would sync gradients of the underlying tensors - which are the ones that optimizer
            #    holds itself.  As a result it seems to work, but only because DDP doesn't take any tensor ownership.

            self.game = torch.nn.parallel.DistributedDataParallel(
                self.game,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
            )
            self.optimizer.state = move_to(self.optimizer.state, device_id)

        else:
            self.game.to(self.device)
            # NB: some optimizers pre-allocate buffers before actually doing any steps
            # since model is placed on GPU within Trainer, this leads to having optimizer's state and model parameters
            # on different devices. Here, we protect from that by moving optimizer's internal state to the proper device
            self.optimizer.state = move_to(self.optimizer.state, self.device)
        if common_opts.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def eval(self, data=None):
        mean_loss = 0.0
        interactions = []
        n_batches = 0
        validation_data = self.validation_data if data is None else data

        self.game.eval()

<<<<<<< HEAD
        # with torch.no_grad():
        for batch in validation_data:
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)
            # t = time.perf_counter()
            # print('trainer 188')
            optimized_loss, interaction = self.game(*batch)
            optimized_loss = optimized_loss.detach()
            # print('trainer 190')
            # print('val forward pass: ', time.perf_counter() - t)
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(
                    interaction
                )
            interaction = interaction.to("cpu")
            mean_loss += optimized_loss

            for callback in self.callbacks:
                callback.on_batch_end(
                    interaction, optimized_loss, n_batches, is_training=False
                )

            interactions.append(interaction)
            n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        # print('done with eval')
=======
        with torch.no_grad():
            for batch in validation_data:
                if not isinstance(batch, Batch):
                    batch = Batch(*batch)
                batch = batch.to(self.device)
                optimized_loss, interaction = self.game(*batch)
                if (
                    self.distributed_context.is_distributed
                    and self.aggregate_interaction_logs
                ):
                    interaction = Interaction.gather_distributed_interactions(
                        interaction
                    )
                interaction = interaction.to("cpu")
                mean_loss += optimized_loss

                for callback in self.callbacks:
                    callback.on_batch_end(
                        interaction, optimized_loss, n_batches, is_training=False
                    )

                interactions.append(interaction)
                n_batches += 1

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        return mean_loss.item(), full_interaction

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()
<<<<<<< HEAD
        self.last_grad_receiver, self.last_grad_sender = 0.0, 0.0
=======

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        for batch_id, batch in enumerate(self.train_data):
            if not isinstance(batch, Batch):
                batch = Batch(*batch)
            batch = batch.to(self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
<<<<<<< HEAD
                t = time.perf_counter()
                optimized_loss, interaction = self.game(*batch)
                # print('train forward pass: ', time.perf_counter() - t)
=======
                optimized_loss, interaction = self.game(*batch)

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
<<<<<<< HEAD
                # print('right before backwards on overall loss')
=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

<<<<<<< HEAD
                # self.last_grad_sender += get_grad_norm(self.game.module.sender) if self.distributed_context.is_distributed \
                #     else get_grad_norm(self.game.sender)
                # self.last_grad_receiver += get_grad_norm(self.game.module.receiver) if self.distributed_context.is_distributed \
                #     else get_grad_norm(self.game.receiver)
=======
                # self.last_grad_sender = get_grad_norm(self.game.sender)
                # self.last_grad_receiver = get_grad_norm(self.game.receiver)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
<<<<<<< HEAD

=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")
<<<<<<< HEAD
            # print('to cpu: ', time.perf_counter() - t)
            # t = time.perf_counter()
            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)
            # print('callback train on batch end: ', time.perf_counter() - t)
            interactions.append(interaction)

        self.last_grad_sender /= n_batches
        self.last_grad_receiver /= n_batches

        # print('Max grad norm: ', max(self.last_grad_sender, self.last_grad_receiver))
=======

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

        if self.optimizer_scheduler:
            self.optimizer_scheduler.step()

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

    def train(self, n_epochs):
        for callback in self.callbacks:
            callback.on_train_begin(self)
<<<<<<< HEAD

        for epoch in range(self.start_epoch, n_epochs):
            if hasattr(self.game.__class__, 'init_pairings') and callable(
                    getattr(self.game.__class__, 'init_pairings')):
                # Init communication and imitation pairings if we're in direct imitation, i.e. game is DirectImitationGame
                # Use case: pair sender and receiver agents in population for reinforcement task + imitation task.
                # Needs to be done once per epoch.
                self.game.init_pairings()

            t = time.perf_counter()
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)
            # print('on epoch begin: ', time.perf_counter() - t)

            t = time.perf_counter()
            train_loss, train_interaction = self.train_epoch()
            # print('train epoch: ', time.perf_counter() - t)

            for callback in self.callbacks:
                t = time.perf_counter()
                callback.on_epoch_end(train_loss, train_interaction, epoch + 1)
                # print('callback on epoch end: ', time.perf_counter() - t)
=======
        for epoch in range(self.start_epoch, n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch + 1)

            train_loss, train_interaction = self.train_epoch()
            for callback in self.callbacks:
                callback.on_epoch_end(train_loss, train_interaction, epoch + 1)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            validation_loss = validation_interaction = None
            if (
                self.validation_data is not None
                and self.validation_freq > 0
                and (epoch + 1) % self.validation_freq == 0
            ):
                for callback in self.callbacks:
<<<<<<< HEAD
                    t = time.perf_counter()
                    callback.on_validation_begin(epoch + 1)
                    # print('callback val begin: ', time.perf_counter() - t)

                t = time.perf_counter()
                validation_loss, validation_interaction = self.eval()
                # print('val eval: ', time.perf_counter() - t)

                for callback in self.callbacks:
                    t = time.perf_counter()
                    callback.on_validation_end(
                        validation_loss, validation_interaction, epoch + 1
                    )
                    # print('callback val end: ', time.perf_counter() - t)
=======
                    callback.on_validation_begin(epoch + 1)
                validation_loss, validation_interaction = self.eval()

                for callback in self.callbacks:
                    callback.on_validation_end(
                        validation_loss, validation_interaction, epoch + 1
                    )
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            if self.should_stop:
                for callback in self.callbacks:
                    callback.on_early_stopping(
                        train_loss,
                        train_interaction,
                        epoch + 1,
                        validation_loss,
                        validation_interaction,
                    )
                break

        for callback in self.callbacks:
            callback.on_train_end()

    def load(self, checkpoint: Checkpoint):
        self.game.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.optimizer_scheduler_state_dict:
            self.optimizer_scheduler.load_state_dict(
                checkpoint.optimizer_scheduler_state_dict
            )
        self.start_epoch = checkpoint.epoch

    def load_from_checkpoint(self, path):
        """
        Loads the game, agents, and optimizer state from a file
        :param path: Path to the file
        """
        print(f"# loading trainer state from {path}")
        checkpoint = torch.load(path)
        self.load(checkpoint)

    def load_from_latest(self, path):
        latest_file, latest_time = None, None

        for file in path.glob("*.tar"):
            creation_time = os.stat(file).st_ctime
            if latest_time is None or creation_time > latest_time:
                latest_file, latest_time = file, creation_time

        if latest_file is not None:
            self.load_from_checkpoint(latest_file)
