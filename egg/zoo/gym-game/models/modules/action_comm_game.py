import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modules.processing import ProcessingModule

"""
    An ActionModule takes in the physical observation feature vector, the
    utterance observation feature vector and the individual goal of an agent
    (alongside the memory for the module), processes the goal to turn it into
    a goal feature vector, and runs the concatenation of all three feature
    vectors through a processing module. The output of the processing module
    is then fed into two independent fully connected networks to output
    utterance and movement actions
"""
class ActionModule(nn.Module):
    def __init__(self, config, predict_F1=False):
        super(ActionModule, self).__init__()
        self.using_cuda = config.use_cuda
        self.gru = ProcessingModule(config.recurrent_processor)
        self.predict_F1 = predict_F1
        self.action_decoder = nn.Sequential(
                nn.Linear(config.action_decoder.input_size, config.action_decoder.hidden_size),
                nn.ELU(),
                nn.Linear(config.action_decoder.hidden_size, config.action_decoder.out_size),
                nn.Tanh() if predict_F1 else nn.Softmax())
        self.base_sample_prob = 0.05
        self.gamma = 0.99

    def forward(self, map_features, movement_features, mem, epoch, fol_embed=None, training=True):
        # define x
        if fol_embed is not None:
            x = torch.cat([map_features, movement_features, fol_embed.unsqueeze(1)], 2)
        else:
            x = torch.cat([map_features, movement_features], 2)

        processed, mem = self.gru(x, mem)
        action_out = self.action_decoder(processed)

        if self.predict_F1:
            return action_out, mem

        m = Categorical(action_out)
        entropy = m.entropy()

        if training:
            random_sample_prob = self.base_sample_prob * self.gamma ** epoch
            if torch.rand(1) < random_sample_prob:
                action = torch.randint(m.param_shape[-1], (m.param_shape[0],))
            else:
                action = m.sample()
        else:
            action = torch.argmax(action_out, dim=1)

        log_prob = m.log_prob(action)

        return action, log_prob, mem, entropy

    def reset_memory(self):
        # resets the GRU hidden state (use when doing a new task)
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        return (hidden_state, cell_state)


class ActionDecoderModule(nn.Module):
    def __init__(self, config):
        super(ActionDecoderModule, self).__init__()
        self.using_cuda = config.use_cuda

        # Attention
        self.attn = nn.Linear(config.attention_input, config.recurrent_processor.input_size) # out = max teach iters
        self.attn_combine = nn.Linear(config.attention_input, config.recurrent_processor.input_size)

        self.gru = ProcessingModule(config.recurrent_processor)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.action_decoder.input_size, config.action_decoder.hidden_size),
            nn.ELU(),
            nn.Linear(config.action_decoder.hidden_size, config.action_decoder.out_size),
            nn.Softmax())

    def forward(self, map_features, movement_features, mem, encoder_hiddens, training=True):
        input = torch.cat([map_features, movement_features], 2)

        # Attention: concatenate input vec and hidden vec, then apply to the encoder hidden layers
        attn_weights = F.softmax(
            self.attn(torch.cat((input, mem.unsqueeze(1)), 2)), dim=2)

        attn_applied = torch.bmm(attn_weights,
                                 encoder_hiddens)

        # The new input to the model is the input concat with the attention-ed hidden layers
        output = torch.cat((input, attn_applied), 2)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output).squeeze(0)

        output, hidden = self.gru(output, mem)
        action_out = self.action_decoder(output)
        m = Categorical(action_out)
        if training:
            action = m.sample()
        else:
            action = torch.argmax(action_out, dim=1)

        log_prob = m.log_prob(action)

        return action, log_prob, mem, attn_weights
