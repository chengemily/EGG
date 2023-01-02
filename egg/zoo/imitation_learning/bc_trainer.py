import torch
import argparse

from egg.core.batch import Batch

from egg.zoo.imitation_learning.behavioural_cloning import eval_bc_prediction, eval_bc_original_task
# from egg.zoo.imitation_learning.loader import log_performance
from egg.zoo.imitation_learning.util import get_grad_norm


def log_performance(perf_log, r_loss, s_loss, r_acc, s_acc, mean_loss, acc, acc_or, t_speaker, t_receiver):
    perf_log['r_loss'].append(r_loss)
    perf_log['s_loss'].append(s_loss)
    perf_log['r_acc'].append(r_acc)
    perf_log['s_acc'].append(s_acc)
    perf_log['mean_loss'].append(mean_loss)
    perf_log['acc'].append(acc)
    perf_log['acc_or'].append(acc_or)
    perf_log['epoch'].append(max(t_speaker, t_receiver))
    perf_log['epoch_speaker'].append(t_speaker)
    perf_log['epoch_receiver'].append(t_receiver)

class BCTrainer:
    """
    Training class for behavioural cloning.
    """
    def __init__(self,
                 bc_args: argparse.Namespace,
                 sender_aware_weight: float=0.0):
        self.bc_args = bc_args
        self.bc_epochs = bc_args.n_epochs_bc
        self.early_stopping_thr_bc = bc_args.early_stopping_thr_bc
        self.is_distributed = bc_args.distributed_context.is_distributed
        self.sender_aware_weight = sender_aware_weight
        self.device = self.bc_args.device

    def train_epoch(self, imitator_optimizer, imitator_agent, expert, train_data, aux_info={}):
        imitator_optimizer.zero_grad()
        train_data = train_data.to(self.device)
        inp = train_data.message if imitator_agent.is_receiver else train_data.sender_input

        with torch.no_grad():
            if imitator_agent.is_receiver:
                _, _, _, out = expert(inp, train_data.receiver_input, train_data.aux_input, train_data.message_length)
                out = out['sample']
            else:
                out, _, _, _ = expert(inp, train_data.aux_input)

        loss, acc, _ = imitator_agent.score(inp, out.detach(), aux_info=aux_info)
        loss = loss.mean()
        loss.backward()
        imitator_optimizer.step()

        return loss, acc, {}

    def eval(self, new_sender, new_receiver, t, interaction, new_trainer=None, perf_log=None):
        new_sender.eval()
        new_receiver.eval()
        r_loss, s_loss, r_acc, s_acc = eval_bc_prediction(new_sender, new_receiver, interaction, t)
        if new_trainer is not None:
            mean_loss, acc, acc_or = eval_bc_original_task(new_trainer, t)

        if perf_log is not None:
            log_performance(perf_log, r_loss.item(),
                            s_loss.item(), r_acc.item(), s_acc.item(), mean_loss,
                            acc.item(), acc_or.item(), sender_converged_epoch, receiver_converged_epoch)

    def train_agent(self, new_agent, optimizer, interaction, expert):
        converged = False
        converged_epoch = 0

        # Logging
        cumu_loss = torch.zeros(self.bc_epochs)
        cumu_acc = torch.zeros(self.bc_epochs)
        acc = torch.Tensor([0.0])

        train_data = interaction

        # Train imitators
        for t in range(self.bc_epochs):
            # val = t % self.bc_args.val_interval == 0
            # if val:
            #     self.eval(new_sender, new_receiver, t, interaction, new_trainer=new_trainer, perf_log=perf_log)

            if not converged:
                new_agent.train()
                loss, acc, _ = self.train_epoch(
                    optimizer, new_agent, expert, train_data
                )
                cumu_loss[t] = loss
                cumu_acc[t] = acc
                converged = \
                    get_grad_norm(new_agent) < self.bc_args.convergence_epsilon or \
                    acc > self.early_stopping_thr_bc
                converged_epoch = t

            if converged:
                print('Gradients < epsilon={}'.format(self.bc_args.convergence_epsilon))
                print('acc > acc={}'.format(self.early_stopping_thr_bc))
                break
            # print('Epoch: {}; R loss: {}; S loss: {}; R acc: {}; S acc: {}'.format(t, r_loss, s_loss,
            #                                                                                    r_acc, s_acc))

        cumu_loss = cumu_loss.sum().detach()
        cumu_acc = cumu_acc.sum()

        # print('S_t: {}; R_t: {}; R loss: {}; S loss: {}; R acc: {}; S acc: {}; S SOL: {}; R SOL: {}'.format(
        #     sender_converged_epoch, receiver_converged_epoch, cumu_r_loss, cumu_s_loss,
        #     r_acc, s_acc, cumu_s_acc, cumu_r_acc))

        return cumu_loss, converged_epoch, acc, cumu_acc

    def train(self, new_sender, new_receiver, optimizer_r, optimizer_s,
              interaction=None, expert_sender=None, expert_receiver=None, new_trainer=None, perf_log=None):
        # Get training data
        train_data = interaction

        # Train imitators
        cumu_r_loss, receiver_converged_epoch, r_acc, cumu_r_acc = self.train_agent(new_receiver, optimizer_r, train_data, expert_receiver)
        cumu_s_loss, sender_converged_epoch, s_acc, cumu_s_acc = self.train_agent(new_sender, optimizer_s, train_data, expert_sender)

        # print('S_t: {}; R_t: {}; R loss: {}; S loss: {}; R acc: {}; S acc: {}; S SOL: {}; R SOL: {}'.format(
        #     sender_converged_epoch, receiver_converged_epoch, cumu_r_loss, cumu_s_loss,
        #     r_acc, s_acc, cumu_s_acc, cumu_r_acc))

        return cumu_s_loss, cumu_r_loss, \
               sender_converged_epoch, receiver_converged_epoch, \
               s_acc, r_acc, cumu_s_acc, cumu_r_acc

