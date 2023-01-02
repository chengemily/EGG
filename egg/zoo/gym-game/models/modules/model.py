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
class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.init_from_config(config)
        self.epoch = 0
        self.total_loss, \
        self.total_student_loss, \
        self.total_teacher_loss = Variable(self.Tensor(1).zero_()), \
                                  Variable(self.Tensor(1).zero_()), \
                                  Variable(self.Tensor(1).zero_())
        self.teacher = 0
        self.student = 1

        # fol processor
        self.fol_processor = FOLEncModule(config.fol_processor)

        # need one physical processor per agent: 0 = teacher, 1 = student
        self.physical_processors = nn.ModuleList([FCModule(config.physical_processor), FCModule(config.physical_processor)])
        self.physical_pooling = nn.AdaptiveAvgPool2d((1, config.feat_vec_size))

        # Need one action processor per agent per stage
        self.stages = list(config.action_processors.keys()) # [teach, test]
        if 'teach' in self.stages:
            self.action_processors = nn.ModuleDict({
                "teach": nn.ModuleList([
                    ActionModule(config.action_processors['teach']['teacher']),
                    ActionModule(config.action_processors['teach']['student'])
                ]),
                "test": ActionDecoderModule(config.action_processors['test']['student'])
            })
            self.num_agents = 2
        else:
            self.action_processors = nn.ModuleDict({
                "test": ActionModule(config.action_processors['test']['student'])
            })
            self.num_agents = 1

    def init_from_config(self, config):
        self.training = True
        self.using_cuda = config.use_cuda
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.Tensor = torch.cuda.FloatTensor if self.using_cuda else torch.FloatTensor

    def reset(self):
        self.total_loss = torch.zeros_like(self.total_loss)
        self.total_teacher_loss = torch.zeros_like(self.total_teacher_loss)
        self.total_student_loss = torch.zeros_like(self.total_student_loss)

    def train(self, mode=True):
        super(AgentModule, self).train(mode)
        self.training = mode

    """
        Requires: new_mem = batch_size x k
    """
    def update_mem(self, game, mem_str, new_mem, agent, iter=None):
        new_big_mem = Variable(self.Tensor(game.memories[mem_str].data))

        if iter is not None:
            # print('target size')
            # print(new_big_mem[:, agent, iter].size)
            new_big_mem[:, agent, iter] = new_mem
        else:
            new_big_mem[:, agent] = new_mem

        game.memories[mem_str] = new_big_mem

    def get_physical_feat(self, agent, objects_on_map):
        """
        Do the softmax combinations of all the feature vectors of the map.
        :param game:
        :param agent:
        :return:
        """
        # objects on map = N objects x 9. [x, y, features]
        map_features = torch.zeros_like(objects_on_map)
        map_features[:, 2:] = self.physical_processors[agent](objects_on_map[:, 2:]) # select only the features
        map_features[:, :2] = objects_on_map[:, :2]
        return self.physical_pooling(map_features.unsqueeze(0)).squeeze(0)

    def get_fol_embed(self, fol_as_indices):
        """
        Put FOL indices through LSTM, then softmax pool
        :param fol_as_indices: tensor of indices
        :return: embedding
        """
        fol_embeddings = self.fol_processor(fol_as_indices)
        return self.physical_pooling(fol_embeddings.view(
            [fol_embeddings.shape[1], fol_embeddings.shape[0], fol_embeddings.shape[2]]))

    def get_action(self, game, agent, physical_feat, movement_feat, actions, log_probs, epoch, iter=None, fol_embed=None):
        # input()
        if game.is_teaching_stage():
            action, log_prob, new_mem, entropy = self.action_processors['teach'][agent](
                physical_feat,
                movement_feat,
                game.memories["hidden"][:,agent],
                epoch,
                fol_embed=fol_embed,
                training=self.training
            )
            self.update_mem(game, "encoder_hiddens", new_mem, agent, iter=iter)
        else:
            if type(self.action_processors['test']) == ActionDecoderModule:
                action, log_prob, new_mem, attn_weights = self.action_processors['test'](
                    physical_feat,
                    movement_feat,
                    game.memories["hidden"][:, agent],
                    game.memories["encoder_hiddens"][:, agent],
                    training=self.training
                )
            elif type(self.action_processors['test'] == ActionModule):
                action, log_prob, new_mem, entropy = self.action_processors['test'](
                    physical_feat,
                    movement_feat,
                    game.memories["hidden"][:, agent],
                    epoch,
                    training=self.training
                )
                attn_weights = None
                self.update_mem(game, "entropy", entropy, agent, iter=iter)

        # print("entropy")
        # print(entropy)
        self.update_mem(game, "hidden", new_mem, agent)

        actions[:, agent] = action # update in-place
        log_probs[:, agent] = log_prob

        if not game.is_teaching_stage():
            return attn_weights

    def forward(self, game):
        attention_weights = []
        t = 0
        done_with_task = False

        # get initial observations
        # print('model;139')
        (movement_feat, batch_of_maps), task_features = game.get_obs() # obj on map: list of torch tensors
        # print(task_features.shape)

        while not done_with_task:
            # Actions placeholder (gets populated in get_action)
            actions = Variable(self.Tensor(game.batch_size, 2).zero_())
            log_probs = Variable(self.Tensor(game.batch_size, 2).zero_())

            if game.is_teaching_stage():
                if not game.use_pretrained_embeddings:
                    task_features = self.get_fol_embed(task_features)
                # print(
                #     't:', t)
                for agent in range(game.num_agents):
                    # process batch of maps
                    physical_feat = torch.stack([self.get_physical_feat(agent, single_map) \
                                                 for single_map in batch_of_maps[agent]])
                    # print(physical_feat.shape)
                    # print('model:line102')
                    self.get_action(
                        game,
                        agent,
                        physical_feat,
                        movement_feat[agent],
                        actions,
                        log_probs,
                        self.epoch,
                        iter=t,
                        fol_embed=task_features if agent == self.teacher else None)

                    # print('actions teach: ', actions)
                    # update log probs
                    self.update_mem(game, "log_probs", log_probs[:,agent], agent, iter=t)

                movement_feat, batch_of_maps, reward, _, _ = game(actions)

                # The teacher gets a reward
                self.update_mem(game, 'rewards', reward, self.teacher, iter=t)
            else:
                # TESTING STAGE: get physical features for the student and the student's action.
                physical_feat = torch.stack([self.get_physical_feat(self.student, single_map) \
                                                 for single_map in batch_of_maps[self.student]])
                attn_weights = self.get_action(game, self.student, physical_feat,
                                               movement_feat[self.student], actions, log_probs, self.epoch, iter=t)
                if attn_weights is not None:
                    self.update_mem(game, 'attention_weights', attn_weights, 0, iter=t)

                    if t % 10 == 0:
                        attention_weights.append(attn_weights)

                # STEP
                movement_feat, batch_of_maps, reward, done_with_task, next_map = game(actions)

                self.update_mem(game, 'rewards', self.Tensor(reward), self.student, iter=t)

                # Add reward to storage
                if self.num_agents == 2:
                    self.update_mem(game, 'rewards', self.Tensor(reward), self.teacher, iter=t)

                # Add student log prob to storage
                self.update_mem(game, 'log_probs', log_probs[:, self.student], self.student, iter=t)

                # Apply reward (with attention) in the testing stage only
                if self.num_agents == 2:
                    # batch size x 1 x 20
                    teacher_log_probs = game.memories['log_probs'][:, self.teacher, :game.max_teach_iters]

                    attentioned_teacher_log_probs = torch.bmm(
                        attn_weights, teacher_log_probs.unsqueeze(2)  # batched dot product
                    ).squeeze(1).squeeze(1)

                    self.update_mem(game, 'log_probs', attentioned_teacher_log_probs, self.teacher, iter=t)

                if next_map:
                    # reset the hidden state of the decoder
                    game.reset_memory(new_test_map=True, new_task=False)

            t += 1

        self.epoch += 1

        return attention_weights



