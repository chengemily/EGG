import torch
import torch.nn as nn
import pdb
"""
    A Processing module takes an input from a stream and the independent memory
    of that stream and runs a single timestep of a GRU cell, followed by
    dropout and finally a linear ELU layer on top of the GRU output.
    It returns the output of the fully connected layer as well as the update to
    the independent memory.
"""
class ProcessingModule(nn.Module):
    def __init__(self, config):
        super(ProcessingModule, self).__init__()
        self.input_size = config.input_size
        self.cell = nn.GRUCell(config.input_size, config.hidden_size)
        self.fully_connected = nn.Sequential(
                nn.BatchNorm1d(num_features=config.hidden_size),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ELU())

    def forward(self, x, m):
        m = self.cell(x.squeeze(1), m)

        return self.fully_connected(m), m


"""
    A CNN module takes a map tensor input, convolves it multiple times, then
    puts it through a feed-forward layer to get a map feature vector.
"""
class CNNModule(nn.Module):
    def __init(self, config):
        super(CNNModule, self).__init__()



"""
    A FOLEncModule takes a sequence (seq_len, feat_len, -1)? and runs it through an lstm
"""
class FOLEncModule(nn.Module):
    def __init__(self, config):
        super(FOLEncModule, self).__init__()
        # print('here')
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        # print('make lstm')
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim)

    def forward(self, fol_as_indices):
        """

        :param fol_as_indices: torch (1, 9) LongTensor with indices
        :return:
        """
        embeds = self.word_embeddings(fol_as_indices.long())
        lstm_out, _ = self.lstm(embeds.view(len(fol_as_indices), 1, -1))
        return lstm_out

"""
    An FC module takes a vector, runs it through 3 layers, and returns the output.
    Ex. use to process 
"""
class FCModule(nn.Module):
    def __init__(self, config):
        super(FCModule, self).__init__()
        self.input_size = config.input_size
        self.fully_connected = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ELU(),
            nn.BatchNorm1d(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.out_size),
            nn.ELU()
        )

    def forward(self, x):
        return self.fully_connected(x)



