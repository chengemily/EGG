import torch
import torch.nn as nn
import torch.nn.functional as F

import egg.core as core

def define_agents(opts):
    n_dim = opts.n_attributes * opts.n_values

    if opts.receiver_cell in ["lstm", "rnn", "gru"]:
        receiver = Receiver(n_hidden=opts.hidden, n_outputs=n_dim)
        receiver = core.RnnReceiverReinforceDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.hidden,
            cell=opts.receiver_cell,
        )
    elif opts.receiver_cell == 'transformer':
        receiver = Receiver(n_hidden=opts.hidden, n_outputs=n_dim)
        receiver = core.TransformerReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.max_len,
            opts.receiver_emb,
            1,
            opts.hidden,
            10
        )
    else:
        raise ValueError(f"Unknown receiver cell, {opts.receiver_cell}")

    if opts.sender_cell in ["lstm", "rnn", "gru"]:
        sender = Sender(n_inputs=n_dim, n_hidden=opts.hidden)
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.hidden,
            max_len=opts.max_len,
            cell=opts.sender_cell,
        )
    elif opts.sender_cell == 'transformer':
        sender = Sender(n_inputs=n_dim, n_hidden=opts.hidden)
        sender = core.TransformerSenderReinforce(
            sender,
            opts.vocab_size,
            opts.sender_emb,
            opts.max_len,
            10,
            1,
            opts.hidden,
            generate_style="standard",
            causal=True,
        )
    else:
        raise ValueError(f"Unknown sender cell, {opts.sender_cell}")

    sender = PlusOneWrapper(sender)
    return sender, receiver


class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.fc = nn.Linear(n_hidden, n_outputs)

    def forward(self, x, _input, _aux_input):
        return self.fc(x)


class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden)

    def forward(self, x, _aux_input):
        x = self.fc1(x)
        return x


class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *input):
        r1, r2, r3, r4 = self.wrapped(*input)
        return r1 + 1, r2, r3, r4
