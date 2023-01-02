import torch
import torch.nn as nn
import random

from egg.zoo.compo_vs_generalization.data import (
    ScaledDataset,
    enumerate_attribute_value,
    one_hotify,
    select_subset_V1,
    select_subset_V2,
    split_holdout,
    split_train_test,
)


def get_experts(opts):
    EXPERTS = {
        'Random': RandomSender,
        'Deterministic': DeterministicSender,
        'Compositional': CompositionalSender,
        'Noncompositional': NoncompositionalSender,
        'VariableLength': VariableLengthSender
    }

    return [
        EXPERTS[name](
            opts.max_len,
            opts.max_len if name != 'VariableLength' else int(opts.expert_lengths[i]),
            opts.vocab_size,
            opts.n_attributes,
            opts.n_values
        ) for i, name in enumerate(opts.experts)
    ]


class RandomSender:
    def __init__(self, message_length, _, vocab_size, n_attributes=None, n_values=None):
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.name = 'Random'

    def forward(self, sender_input):
        batch_size = sender_input.shape[0]
        return torch.randint(0, self.vocab_size, (batch_size, self.message_length))


class DeterministicSender:
    def __init__(self, message_length, _, vocab_size, n_attributes=None, n_values=None):
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.token = random.randint(0, self.vocab_size - 1)
        self.name = 'Deterministic'

    def forward(self, sender_input):
        batch_size = sender_input.shape[0]
        return torch.ones((batch_size, self.message_length), dtype=torch.long) * self.token


class CompositionalSender:
    def __init__(self, message_length, _, vocab_size, n_attributes, n_values):
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.name = 'Compositional'

        assert self.vocab_size >= self.n_values
        assert self.message_length == self.n_attributes

        # Make a deterministic mapping from value to symbol
        self.value_symbol_lookup = list(range(self.vocab_size))
        random.shuffle(self.value_symbol_lookup)
        print(self.value_symbol_lookup)

    def forward(self, sender_input):
        batch_size = sender_input.shape[0]
        sender_input_att_val = sender_input.reshape((batch_size, self.n_attributes, self.n_values))
        sender_input_att = torch.argmax(sender_input_att_val, dim=-1).cpu()
        sender_input_att.apply_(lambda x: self.value_symbol_lookup[int(x)])
        return torch.LongTensor(sender_input_att)


class NoncompositionalSender:
    def __init__(self, max_length, message_length, vocab_size, n_attributes, n_values):
        print('max len: ', max_length)
        print('message len: ', message_length)
        self.max_len = max_length
        self.message_length = message_length

        self.vocab_size = vocab_size
        self.name = 'Noncompositional'
        self.n_attributes = n_attributes
        self.n_values = n_values

        # Make a deterministic mapping from entire input to a message
        self.input_symbol_lookup = {}
        for inp in enumerate_attribute_value(n_attributes, n_values):
            self.input_symbol_lookup[inp] = self.generate_random_message()

    def generate_random_message(self):
        message = torch.randint(self.vocab_size, (self.message_length,), dtype=torch.long)
        return message

    def forward(self, sender_input):
        batch_size = sender_input.shape[0]
        sender_input_att_val = torch.argmax(sender_input.reshape((batch_size, self.n_attributes, self.n_values)), dim=-1)
        sender_input_att_val = sender_input_att_val.numpy()
        messages = torch.stack([self.input_symbol_lookup[tuple(inp)] for inp in sender_input_att_val])
        return messages


class VariableLengthSender(NoncompositionalSender):
    def __init__(self, max_length, message_length, vocab_size, n_attributes, n_values):
        super(VariableLengthSender, self).__init__(max_length, message_length, vocab_size, n_attributes, n_values)
        self.name = 'VariableLength' + str(self.message_length)

    def generate_random_message(self):
        message = torch.zeros((self.max_len,), dtype=torch.long)
        message_to_fill = torch.randint(self.vocab_size, (self.message_length,), dtype=torch.long) + 1 # 0 is EOS
        message[:self.message_length] = message_to_fill
        return message