import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.zoo.compo_vs_generalization.archs import *


def bc_agents_setup(opts, device, new_sender, new_receiver):
    return RnnSenderBC(new_sender, opts, device), RnnReceiverBC(new_receiver, opts, device)


class RnnReceiverBC(nn.Module):
    def __init__(
            self, agent, opts, device
    ):
        super(RnnReceiverBC, self).__init__()
        self.agent = agent
        self.opts = opts
        self.device = device
        self.to(device)

    def forward(self, message):
        message = message.cuda()
        receiver_output, log_prob_r, entropy_r = self.agent(message)
        batch_size = receiver_output.shape[0]
        receiver_output = receiver_output.view(
            batch_size, self.opts.n_attributes, self.opts.n_values
        )

        return receiver_output, log_prob_r, entropy_r, batch_size

    def score(self, interaction, val=False, expert=None, imitation=False, aux_info={}):
        """
        Returns cross-entropy loss between the output of the BC receiver and the output
        of the expert receiver.
        :param interaction: (Interaction)
        :param val: (bool) whether in evaluation mode
        :return: cross entropy loss
        """
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

        sender_aware = 'sender_aware' in aux_info and aux_info['sender_aware']
        imitation_reinforce_for_sender = 0

        if expert is None:
            # CE loss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html on class probabilities.
            r_loss = F.cross_entropy(receiver_output.transpose(1, 2), interaction.receiver_output)
        else:
            if sender_aware:
                expert_sender = aux_info['expert_sender']
                rcvr_input, log_prob_s, _, _ = expert_sender(interaction.sender_input.to(self.device))
            else:
                rcvr_input = interaction.message.to(self.device)

            expert_output, log_prob_r, _ = expert(rcvr_input)
            if not imitation:
                expert_output = expert_output.detach()

            # Calculate loss for receiver
            expert_output = expert_output.view(
                batch_size, self.opts.n_attributes, self.opts.n_values
            )
            r_loss = F.cross_entropy(receiver_output.transpose(1,2),
                                     expert_output.transpose(1,2).softmax(dim=1), reduction='none')

            # Calculate reinforce loss for sender if applicable.
            if sender_aware:
                # add the reinforce loss
                message_length = find_lengths(rcvr_input)
                effective_log_prob_s = torch.zeros_like(log_prob_r)

                for i in range(rcvr_input.size(1)):
                    not_eosed = (i < message_length).float()
                    effective_log_prob_s += log_prob_s[:, i] * not_eosed

                imitation_reinforce_for_sender = (log_prob_s * r_loss.detach()).mean()

            r_loss = r_loss.mean()

        return r_loss, accuracy, {'reinforce_loss': imitation_reinforce_for_sender}


class RnnSenderBC(nn.Module):
    def __init__(
            self, agent:PlusOneWrapper, opts, device
    ):
        super(RnnSenderBC, self).__init__()
        self.agent = agent
        self.opts = opts
        self.device = device
        self.to(device)

    def forward(self, sender_input):
        sender_input = sender_input.cuda()
        sender_output, log_prob_s, entropy_s, class_proba_s = self.agent(sender_input)
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
            s_loss = F.cross_entropy(class_proba_s, msg, reduction='mean')
        else: # imitation pressure
            _, _, _, class_probas_e = expert(interaction.sender_input.to(self.device))
            if not imitation:
                class_probas_e = class_probas_e.detach()
            s_loss = F.cross_entropy(class_proba_s.view(batch_size, self.opts.max_len, self.opts.vocab_size).transpose(1,2),
                                     class_probas_e.transpose(1,2).softmax(dim=1))

        accuracy = torch.eq(sender_output[:, :self.opts.max_len], interaction.message).sum() / torch.numel(interaction.message)

        return s_loss, accuracy, {}




