import math
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RnnSenderReinforceImitator(nn.Module):
    """
    Reinforce Sender with forcing (learns conditional probabilities)
    """

    def __init__(
        self,
        agent,
        vocab_size,
        embed_dim,
        hidden_size,
        max_len,
        num_layers=1,
        cell="rnn",
    ):
        """
        :param agent: the agent to be wrapped
        :param vocab_size: the communication vocabulary size
        :param embed_dim: the size of the embedding used to embed the output symbols
        :param hidden_size: the RNN cell's hidden state size
        :param max_len: maximal length of the output messages
        :param cell: type of the cell used (rnn, gru, lstm)
        """
        super(RnnSenderReinforceImitator, self).__init__()
        self.agent = agent

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len
        # self.layer_norm_h = nn.LayerNorm(hidden_size)
        # self.layer_norm_c = nn.LayerNorm(hidden_size)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.batch_norm = nn.BatchNorm1d(embed_dim, momentum=5.0, track_running_stats=False)

        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cells = None

        cell = cell.lower()
        cell_types = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}

        if cell not in cell_types:
            raise ValueError(f"Unknown RNN Cell: {cell}")

        cell_type = cell_types[cell]
        self.cells = nn.ModuleList(
            [
                cell_type(input_size=embed_dim, hidden_size=hidden_size)
                if i == 0
                else cell_type(input_size=hidden_size, hidden_size=hidden_size)
                for i in range(self.num_layers)
            ]
        )  # noqa: E502

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x, ground_truth_sequence, aux_input=None):
        prev_hidden = [self.agent(x, aux_input)]
        prev_hidden.extend(
            [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)]
        )

        prev_c = [
            torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)
        ]  # only used for LSTM

        input = torch.stack([self.sos_embedding] * x.size(0))

        sequence = []
        predicted_dist = []
        logits = []
        entropy = []

        for step in range(self.max_len):
            for i, layer in enumerate(self.cells):
                if isinstance(layer, nn.LSTMCell):
                    h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
                    c_t = self.layer_norm_c(c_t)
                    prev_c[i] = c_t
                else:
                    h_t = layer(input, prev_hidden[i])

                # h_t = self.layer_norm_h(h_t)
                prev_hidden[i] = h_t
                input = h_t

            step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
            predicted_dist.append(step_logits)

            distr = Categorical(logits=step_logits)
            entropy.append(distr.entropy())

            if self.training:
                x = distr.sample()
            else:
                x = step_logits.argmax(dim=1)
            logits.append(distr.log_prob(x))

            # Replace next input with the ground truth
            if self.training and ground_truth_sequence is not None:
                input = self.embedding(ground_truth_sequence[:, step])

            else:
                input = self.embedding(x)

            sequence.append(x)
            # input = self.batch_norm(input)

        predicted_dist = torch.stack(predicted_dist).transpose(1, 0)
        sequence = torch.stack(sequence).permute(1, 0)
        logits = torch.stack(logits).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

        sequence = torch.cat([sequence, zeros.long()], dim=1)
        logits = torch.cat([logits, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return sequence, logits, entropy, predicted_dist
