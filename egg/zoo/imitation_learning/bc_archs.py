import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
from torch.distributions import Categorical

from egg.zoo.imitation_learning.archs import *
from egg.zoo.imitation_learning.imitator_agent_with_forcing import *
from egg.core import RnnReceiverReinforce


def bc_agents_setup(opts, bc_opts, device, new_sender, new_receiver):
    print(12)
    # new sender should be imitator agent with forcing
    if bc_opts.imitation_reinforce:
        new_sender = RnnSenderReinforceImitator(
            new_sender.wrapped.agent,
            opts.vocab_size,
            opts.sender_emb,
            opts.hidden,
            opts.max_len,
            cell=opts.sender_cell
        )
        new_sender = PlusOneWrapper(new_sender)

        new_receiver = RnnReceiverReinforce(new_receiver.agent,
                                            opts.vocab_size,
                                            opts.receiver_emb,
                                            opts.hidden,
                                            cell=opts.receiver_cell)
    print(30)
    return RnnSenderBC(new_sender, opts, bc_opts, device), \
           RnnReceiverBC(new_receiver, opts, bc_opts, device)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        """
        Assumes non-softmaxed, non-logged input.
        logits: if True, we're not in softmax space.
        :return: entropy of X (differentiable)
        """
        b = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
        b = -1.0 * b.sum(dim=-1)
        return b
=======

from egg.zoo.imitation_learning.archs import *


def bc_agents_setup(opts, device, new_sender, new_receiver):
    return RnnSenderBC(new_sender, opts, device), RnnReceiverBC(new_receiver, opts, device)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0


class RnnReceiverBC(nn.Module):
    def __init__(
<<<<<<< HEAD
            self, agent, opts, bc_opts, device
=======
            self, agent, opts, device
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
    ):
        super(RnnReceiverBC, self).__init__()
        self.agent = agent
        self.opts = opts
<<<<<<< HEAD
        self.bc_opts = bc_opts
        self.device = device
        self.to(device)

        # for REINFORCE
        self.use_reinforce = self.bc_opts.imitation_reinforce
        self.counter = 0
        self.mean_baseline = torch.Tensor([0.0]).to(self.device)

    # For REINFORCE
    def update_baseline(self, loss):
        self.counter += 1
        self.mean_baseline += (loss.detach().mean(dim=-1).mean() - self.mean_baseline) / self.counter

    # REINFORCE
    def get_reinforce_loss(self, loss, log_probs, entropy):
        rf_loss = ((loss.detach() - self.mean_baseline) * log_probs)
        if self.training: self.update_baseline(loss)

        return rf_loss - self.opts.entropy_weight * entropy

    def forward(self, message):
        message = message.to(self.device)
        print(80)
        receiver_output, log_prob_r, entropy_r, metadata = self.agent(message - 1)
        print(82)
=======
        self.device = device
        self.to(device)

    def forward(self, message):
        message = message.to(self.device)
        receiver_output, log_prob_r, entropy_r = self.agent(message)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        batch_size = receiver_output.shape[0]
        receiver_output = receiver_output.view(
            batch_size, self.opts.n_attributes, self.opts.n_values
        )

<<<<<<< HEAD
        return receiver_output, log_prob_r, entropy_r
=======
        return receiver_output, log_prob_r, entropy_r, batch_size
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

    def score(self, interaction, val=False, expert=None, imitation=False, aux_info={}):
        """
        Returns cross-entropy loss between the output of the BC receiver and the output
        of the expert receiver.
        :param interaction: (Interaction)
        :param val: (bool) whether in evaluation mode
        :return: cross entropy loss
        """
<<<<<<< HEAD
        print(96)
        if val:
            with torch.no_grad():
                receiver_output, log_prob_r, entropy_r = self.forward(interaction.message)
        else:
            receiver_output, log_prob_r, entropy_r = self.forward(interaction.message)

        print(103)
        batch_size = receiver_output.shape[0]

        interaction.receiver_output = interaction.receiver_output.to(self.device)
        interaction.receiver_output = interaction.receiver_output.view(
            batch_size, self.opts.n_attributes, self.opts.n_values
        )
        interaction_output_labels = interaction.receiver_output.argmax(dim=-1)
        print(111)
        accuracy = torch.eq(
            receiver_output.argmax(dim=-1), interaction_output_labels
        ).sum() / torch.numel(interaction_output_labels)
        print(115)
=======
        if val:
            with torch.no_grad():
                receiver_output, log_prob_r, entropy_r, batch_size = self.forward(interaction.message)
        else:
            receiver_output, log_prob_r, entropy_r, batch_size = self.forward(interaction.message)

        batch_size = receiver_output.shape[0]

        interaction.receiver_output = interaction.receiver_output.cuda()
        interaction.receiver_output = interaction.receiver_output.view(
            batch_size, self.opts.n_attributes, self.opts.n_values
        ).argmax(dim=-1)

        accuracy = torch.eq(
            receiver_output.argmax(dim=-1), interaction.receiver_output
        ).sum() / torch.numel(interaction.receiver_output)

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        sender_aware = 'sender_aware' in aux_info and aux_info['sender_aware']
        imitation_reinforce_for_sender = 0

        if expert is None:
<<<<<<< HEAD
            if True:#self.opts.loss == 'cross_entropy':
                # CE loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html on class probabilities.
                if not self.use_reinforce:
                    r_loss = F.cross_entropy(receiver_output.transpose(1, 2), interaction_output_labels)
                else:
                    loss = F.cross_entropy(receiver_output.transpose(1, 2), interaction_output_labels, reduction='none')
                    r_loss = self.get_reinforce_loss(loss, log_prob_r, entropy_r)
            else:
                #elif self.opts.loss == 'kl':
                r_loss = F.kl_div(
                    nn.LogSoftmax(dim=-1)(receiver_output),
                    F.softmax(interaction.receiver_output, dim=-1),
                    reduction='batchmean'
                )
=======
            if self.opts.loss == 'cross_entropy':
                # CE loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html on class probabilities.
                r_loss = F.cross_entropy(receiver_output.transpose(1, 2), interaction.receiver_output)
            elif self.opts.loss == 'kl':
                # TODO: transform to log probs.
                print(receiver_output)
                input()
                r_loss = F.kl_div(receiver_output.transpose(1, 2), interaction.receiver_output)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        else:
            if sender_aware:
                expert_sender = aux_info['expert_sender']
                rcvr_input, log_prob_s, _, _ = expert_sender(interaction.sender_input.to(self.device))
            else:
                rcvr_input = interaction.message.to(self.device)

            expert_output, log_prob_r, _ = expert(rcvr_input)
<<<<<<< HEAD

=======
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
            if not imitation:
                expert_output = expert_output.detach()

            # Calculate loss for receiver
            expert_output = expert_output.view(
                batch_size, self.opts.n_attributes, self.opts.n_values
            )
<<<<<<< HEAD

            # In case we need to scale losses differently
            r_loss_expert = F.kl_div(nn.LogSoftmax(dim=-1)(receiver_output.detach()),
                                 F.softmax(expert_output, dim=-1), reduction='batchmean')
            if self.opts.loss == 'cross_entropy':
                # add entropy term of expert.
                entropy = HLoss()
                r_loss_expert += entropy(expert_output).mean()

            r_loss_student = F.kl_div(nn.LogSoftmax(dim=-1)(receiver_output),
                                 F.softmax(expert_output.detach(), dim=-1), reduction='batchmean')

            r_loss = r_loss_student + self.opts.imitation_weight
=======
            if self.opts.loss == 'cross_entropy':
                r_loss = F.cross_entropy(receiver_output.transpose(1,2),
                                     expert_output.transpose(1,2).softmax(dim=1), reduction='none')
            elif self.opts.loss == 'kl':
                r_loss = F.kl_div(receiver_output.transpose(1,2),
                                     expert_output.transpose(1,2).softmax(dim=1), reduction='none')
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

            # Calculate reinforce loss for sender if applicable.
            if sender_aware:
                # add the reinforce loss
                message_length = find_lengths(rcvr_input)
                effective_log_prob_s = torch.zeros_like(log_prob_r)

                for i in range(rcvr_input.size(1)):
                    not_eosed = (i < message_length).float()
                    effective_log_prob_s += log_prob_s[:, i] * not_eosed

                imitation_reinforce_for_sender = (log_prob_s * r_loss.detach()).mean()
<<<<<<< HEAD
        print(175)
=======

            r_loss = r_loss.mean()

>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        return r_loss, accuracy, {'reinforce_loss': imitation_reinforce_for_sender}


class RnnSenderBC(nn.Module):
    def __init__(
<<<<<<< HEAD
            self, agent: PlusOneWrapper, opts, bc_opts, device
=======
            self, agent:PlusOneWrapper, opts, device
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
    ):
        super(RnnSenderBC, self).__init__()
        self.agent = agent
        self.opts = opts
<<<<<<< HEAD
        self.bc_opts = bc_opts
        self.use_reinforce = bc_opts.imitation_reinforce
        self.device = device
        self.to(device)

        # For REINFORCE
        self.counter = 0
        print('using rf: ', self.use_reinforce)
        self.mean_baseline = torch.Tensor([0.0]).to(self.device)

    # For REINFORCE
    def update_baseline(self, loss):
        self.counter += 1
        self.mean_baseline += (loss.detach().mean(dim=-1).mean() - self.mean_baseline) / self.counter

    # REINFORCE
    def get_reinforce_loss(self, loss, log_probs, entropy):
        rf_loss = ((loss.detach() - self.mean_baseline) * log_probs)
        if self.training: self.update_baseline(loss)

        return rf_loss - 0.1 * entropy

    def forward(self, sender_input, ground_truth_sequence=None):
        sender_input = sender_input.to(self.device)

        if ground_truth_sequence is not None:
            # imitator forcing
            sender_output, log_prob_s, entropy_s, class_proba_s = self.agent(sender_input, ground_truth_sequence)
        else:
            # no imitator forcing
            sender_output, log_prob_s, entropy_s, class_proba_s = self.agent(sender_input)

=======
        self.device = device
        self.to(device)

    def forward(self, sender_input):
        sender_input = sender_input.to(self.device)
        sender_output, log_prob_s, entropy_s, class_proba_s = self.agent(sender_input)
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        batch_size = sender_output.shape[0]
        class_proba_s = class_proba_s.reshape(
            batch_size * self.opts.max_len, self.opts.vocab_size
        )

        return sender_output, log_prob_s, entropy_s, class_proba_s, batch_size

    def score(self, interaction, val=False, expert=None, imitation=False, aux_info={}):
        """
        Returns cross-entropy loss between the output of the BC sender and the output
        of the expert sender given the same input.
        :param interaction: (Interaction)
        :param val: (bool) whether in evaluation mode
        :return: cross entropy loss
        """
<<<<<<< HEAD
        # remove end token
        interaction.message = interaction.message.to(self.device)
        interaction.message = interaction.message[:, :self.opts.max_len]
        msg = interaction.message - 1  # reset to 0 - 99; in PlusOneWrapper they add one.
        if val:
            with torch.no_grad():
                sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(
                    interaction.sender_input, ground_truth_sequence=msg
                )
        else:
            sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(
                interaction.sender_input, ground_truth_sequence=msg
            )
        msg = msg.reshape(
            batch_size * self.opts.max_len
        )
        class_probas_e = interaction.aux['probas_s'].to(self.device)

        if expert is None: # regular BC
            if True: #self.opts.loss == 'cross_entropy':
                if not self.use_reinforce:
                    s_loss = F.cross_entropy(class_proba_s, msg, reduction='mean')
                else:
                    if self.bc_opts.sender_reward == 'cross_entropy':
                        loss = F.cross_entropy(class_proba_s, msg, reduction='mean')
                    elif self.bc_opts.sender_reward == 'accuracy':
                        loss = -(sender_output[:, :self.opts.max_len] == interaction.message).float()

                    s_loss = self.get_reinforce_loss(loss, log_prob_s[:, :-1], entropy_s[:, :-1])
                    s_loss = s_loss.mean(dim=-1).mean()
            else:
                # elif self.opts.loss == 'kl':
                s_loss = F.kl_div(
                    nn.LogSoftmax(dim=2)(class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size)),
                    F.softmax(class_probas_e, dim=2),
                    reduction='batchmean'
                )
=======
        if val:
            with torch.no_grad():
                sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(
                    interaction.sender_input
                )
        else:
            sender_output, log_prob_s, entropy_s, class_proba_s, batch_size = self.forward(
                interaction.sender_input
            )

        # remove end token
        interaction.message = interaction.message.cuda()
        interaction.message = interaction.message[:, :self.opts.max_len]
        msg = interaction.message.reshape(
            batch_size * self.opts.max_len
        ) - 1 # reset to 0 - 99; in PlusOneWrapper they add one.

        if expert is None: # regular BC
            if self.opts.loss == 'cross_entropy':
                s_loss = F.cross_entropy(class_proba_s, msg, reduction='mean')
            elif self.opts.loss == 'kl':
                s_loss = F.kl_div(class_proba_s, msg, reduction='mean')
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0
        else: # imitation pressure
            _, _, _, class_probas_e = expert(interaction.sender_input.to(self.device))
            if not imitation:
                class_probas_e = class_probas_e.detach()

<<<<<<< HEAD
            # Calculate loss separately.
            s_loss_expert = F.kl_div(
                nn.LogSoftmax(dim=2)(
                    class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size).detach()
                ),
                class_probas_e.softmax(dim=2),
                reduction='batchmean'
            )
            if self.opts.loss == 'cross_entropy':
                entropy = HLoss()
                s_loss_expert += entropy(class_probas_e).mean()

            s_loss_student = F.kl_div(
                nn.LogSoftmax(dim=2)(
                    class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size)
                ),
                class_probas_e.softmax(dim=2).detach(),
                reduction='batchmean'
            )

            s_loss = s_loss_student + self.opts.imitation_weight * s_loss_expert
=======
            if self.opts.loss == 'cross_entropy':
                s_loss = F.cross_entropy(class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size).transpose(1,2),
                                     class_probas_e.transpose(1,2).softmax(dim=1))
            elif self.opts.loss == 'kl':
                s_loss = F.kl_div(
                    class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size).transpose(1, 2),
                    class_probas_e.transpose(1, 2).softmax(dim=1))
>>>>>>> 9c4732ffb57be8aa6b1e3bb7bcfb6aa4488225a0

        accuracy = torch.eq(sender_output[:, :self.opts.max_len], interaction.message).sum() / torch.numel(interaction.message)

        return s_loss, accuracy, {}




