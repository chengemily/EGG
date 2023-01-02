import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.processing import ProcessingModule, FCModule, FOLEncModule
from modules.action_comm_game import ActionModule, ActionDecoderModule
import pdb

"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class RandomAgentModule(nn.Module):
    def __init__(self, config):
        super(RandomAgentModule, self).__init__()
        self.init_from_config(config)

        self.total_rwd = Variable(self.Tensor(1).zero_())

        self.student = 1

    def init_from_config(self, config):
        self.using_cuda = config.use_cuda
        self.Tensor = torch.cuda.FloatTensor if self.using_cuda else torch.FloatTensor

    def reset(self):
        self.total_rwd = torch.zeros_like(self.total_rwd)

    def forward(self, game):
        done_with_task = False

        while not done_with_task:
            # Actions placeholder (gets populated in get_action)
            actions = Variable(self.Tensor(game.batch_size, game.num_agents).zero_())

            if game.is_teaching_stage():
                # submit a no-op
                game(actions)
            else:
                actions[:, self.student] = torch.randint(4, (1, game.batch_size))

                # STEP
                _, _, reward, done_with_task, next_map = game(actions)

                if next_map:
                    # reset the hidden state of the decoder
                    # input('done')
                    game.reset_memory(new_test_map=True, new_task=False)

                self.total_rwd += reward

        return self.total_rwd, None, None, None




