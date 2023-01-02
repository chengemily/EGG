import gym
from gym import error, spaces, utils
from gym.utils import seeding

import random
random.seed(10)

import torch

from enum import IntEnum, Enum
import numpy as np
np.random.seed(10)

class GameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    class Actions(IntEnum):
        down = 0  # decelerate
        up = 1  # accelerate
        left = 2  # turn left
        right = 3  # turn right
        noop = 4  # use

    class Players(IntEnum):
        teacher = 0
        student = 1

    class Movements:
        # Indices in the trajectory tracking
        def __init__(self):
            self.pos = slice(2)
            self.vel = slice(2, 4)
            self.a = 4
            self.a_vel = 5

    def __init__(self):
        self.points = 0 # track point count every task
        self.F1s = [-1] # track max F1 each test map.
        self.task = None

        self.test_no = 0

        self.collision_tol = 0.15
        self.max_acc = 0.10
        self.max_angle_acc = 0.3
        self.max_velocity = 0.10
        self.max_angular_velocity = 0.5
        self.actions = GameEnv.Actions
        self.teacher = GameEnv.Players.teacher
        self.student = GameEnv.Players.student
        self.idx_of = GameEnv.Movements()
        self.dim = 1  # x, y dimensions are the same.
        self.is_done_with_task = False

        self.action_space = spaces.Box(low=-1, high=4, shape=(2,))

        self.observation_space = spaces.Dict({
            'movement_features': spaces.Box(low=-self.max_velocity, high=2*np.pi, shape=(1, 2, self.idx_of.a + 1)),
            'map': spaces.MultiDiscrete([2 for i in range(9)])  # change shape as we go
        })


    def init_data(self, max_teach_iters, max_test_iters, num_test_rounds):
        """
        Given complete data, sets up the first task and initializes positions.
        :param task: command_no, {task_str: (str), teaching_maps: list of np arrays of vectors (x, y, etc), test_maps: list of vectors}
        """
        self.max_teach_iters = max_teach_iters
        if self.max_teach_iters == 0:
            self.is_teaching = False
        self.max_test_iters = max_test_iters
        self.num_test_rounds = num_test_rounds

    def move(self, player):
        """
        Update position based on velocity.
        """
        old_pos = self.trajectories[:, player, self.idx_of.pos, self.timestep]
        vel = self.trajectories[:, player, self.idx_of.vel, self.timestep]

        # set new velocity, position
        self.trajectories[:, player, self.idx_of.pos, self.timestep] = torch.clip(old_pos + vel, min=0, max=self.dim)

    def accelerate(self, player, direction):
        """
        Apply force, then change velocity.
        :param direction: 1 for forwards, -1 for backwards
        :param player: player.student or player.teacher
        """
        angle = self.trajectories[:, player, self.idx_of.a, self.timestep]
        old_vel = self.trajectories[:, player, self.idx_of.vel, self.timestep]

        # apply force and change velocity
        acc = self.max_acc * direction
        acc_x = acc * np.cos(angle)
        acc_y = acc * np.sin(angle)

        # bounds for velocity
        l = torch.Tensor([self.max_velocity * np.cos(angle), self.max_velocity * np.sin(angle)])
        u = torch.Tensor([-self.max_velocity * np.cos(angle), -self.max_velocity * np.sin(angle)])
        new_vel = torch.max(torch.min(old_vel + torch.Tensor([acc_x, acc_y]), u), l)

        self.trajectories[:, player, self.idx_of.vel, self.timestep] = new_vel

    def turn(self, player, direction):
        """
        Apply angular force, change angular velocity, change angle, then match velocity to angle
        :param player:
        :param direction: 1 for right and -1 for left
        :return:
        """
        angle = self.trajectories[:, player, self.idx_of.a, self.timestep]
        ang_vel = self.trajectories[:, player, self.idx_of.a_vel, self.timestep]
        ang_acc = self.max_angle_acc * direction

        # Set new angular velocity
        new_ang_vel = max(-self.max_angular_velocity, min(self.max_angular_velocity, ang_vel + ang_acc))
        self.trajectories[:, player, self.idx_of.a_vel, self.timestep] = new_ang_vel

        # Set new angle
        new_ang = (new_ang_vel + angle) % (2 * np.pi)
        self.trajectories[:, player, self.idx_of.a, self.timestep] = new_ang

        # Set new velocity
        speed = np.linalg.norm(self.trajectories[:, player, self.idx_of.vel, self.timestep])
        new_vel = torch.Tensor([speed * np.cos(new_ang), speed * np.sin(new_ang)])
        self.trajectories[:, player, self.idx_of.vel, self.timestep] = new_vel

    def apply_action(self, player, action):
        """
        Applies action to the player. Happens to both players in teaching and only the student in testing.
        :param player:
        :return: done
        """
        # All actions discrete
        if action == self.actions.left:
            self.turn(player, 1)
        elif action == self.actions.right:
            self.turn(player, -1)
        elif action == self.actions.up:
            self.accelerate(player, 1)
        elif action == self.actions.down:
            self.accelerate(player, -1)
        elif action == self.actions.noop:
            pass
        else:  # not supported move
            raise ValueError('Not supported action')

        self.move(player)

    def step(self, actions):
        """
        Makes one step for player in [teacher, student].
        :param action: length 2 where action[0] is the teacher action and action[1] is the student action
        :return:
        """
        actions = actions.squeeze(0)
        done = False

        # print(self.trajectories[:,:,:,self.timestep])
        self.timestep += 1

        # move
        self.trajectories[:,:,:,self.timestep] = torch.clone(self.trajectories[:,:,:,self.timestep - 1])

        if self.is_teaching:
            # apply force or change state for both players
            self.apply_action(self.teacher, actions[self.teacher])

        # print('env:165')
        self.apply_action(self.student, actions[self.student])

        # detect done, verifier result if in testing mode
        aux_reward = None

        # determine reward
        F1, self.success, touched_new_things = self.check_collision()
        self.F1s[-1] = F1
        avg_F1_for_task = np.mean(self.F1s)

        reward = F1 + touched_new_things * 0.5 - (1 - int(self.success)) - 0.01 # exploration bonus, time penalty

        # determine whether to change state i.e. move to the next map
        if self.is_teaching:
            if self.timestep == self.max_teach_iters:
                self.state_change()
        else:
            if (self.timestep - self.max_teach_iters) % self.max_test_iters == 0:
                done = True
                self.state_change()

        # print('timestep: ', self.timestep)
        return self.feature(), reward, self.is_done_with_task, {
            'is_teaching': self.is_teaching,
            'points': self.points,
            'F1': avg_F1_for_task,
            'next_map': done,
            'trajectory': self.trajectories,
            'aux_reward': aux_reward}

    def feature(self):
        """
        Outputs the next observations for the player.
        :param player: [teacher, student]
        :return: (relative) map features, self velocity, self angle, other relative position, other velocity, other angle
        """
        # change observation space
        self.observation_space = spaces.Dict({
            'movement_features': spaces.Box(low=-self.max_velocity, high=2 * np.pi, shape=(1, 2, 5)),
            'map': spaces.MultiDiscrete(2 + torch.zeros((self.map.shape[0], self.map.shape[1] - 1)))  # change shape as we go
        })

        obs_dict = {
            'movement_features': self.trajectories[:,:,:self.idx_of.a + 1,self.timestep],
            'map': self.map[:, 1:] # block out first element (id),
        }

        return obs_dict

    def get_angle_between_vectors(self, obj_pos, agent_pos, a):
        vector_to_obj = (obj_pos - agent_pos).squeeze(0)
        vector_to_heading = torch.Tensor([np.cos(a), np.sin(a)])
        unit_vector_obj = vector_to_obj / torch.linalg.norm(vector_to_obj)
        dot_product = torch.dot(unit_vector_obj, vector_to_heading)
        angle = torch.arccos(dot_product)
        return angle

    def check_collision(self):
        """
        Check if the agent has collided with a shape. If so, adds to visited set and runs verifier.
        :return:
        """
        test_map = self.map
        agent_pos = self.trajectories[:, self.student, self.idx_of.pos, self.timestep]
        a = self.trajectories[:, self.student, self.idx_of.a, self.timestep]
        touched_new_things = 0

        for obj in test_map:
            obj_pos = obj[1:3]

            if torch.linalg.norm(obj_pos - agent_pos) < self.collision_tol:
                old_number_of_things_touched = len(self.visited)

                # report forwards/backwards
                # if the angle between the current heading and the vector (x,y) makes w the object =< 90, it's forwards
                angle = self.get_angle_between_vectors(obj_pos, agent_pos, a)

                # get id
                obj_id = obj[0]
                if angle <= np.pi / 2:
                    # register forwards
                    self.visited.add((int(obj_id.item()), 'forwards'))
                else:
                    self.visited.add((int(obj_id.item()), 'backwards'))
                self.visited.add((int(obj_id.item()), None))
                touched_new_things += (len(self.visited) - old_number_of_things_touched) / 2
                self.success = self.run_verifier()

        # small positive for each new thing touched and small negative for each time step
        return self.F1(), self.success, touched_new_things

    def F1(self):
        """
        Calculates F1 of visited set against solutions and nonsolutions.
        :return:
        """
        if self.success: return 1
        F1_ = 0

        # get the size of visited intersections w solutions
        size_intersections_with_solutions = [len(self.visited.intersection(soln)) for soln in self.solutions]

        # calculate TP, FN, FP
        TP = max(size_intersections_with_solutions)

        closest_solution_idx = np.argmax(size_intersections_with_solutions)
        FN = max(0, len(self.solutions[closest_solution_idx] - self.visited))
        FP = int(len(self.visited) / 2) - TP # divide by 2 as self.visited adds a directionless and directioned tuple for each visit

        if (TP + 0.5 * (FP + FN)) > 0:
            F1_ = TP / (TP + 0.5 * (FP + FN))

        return F1_

    def run_verifier(self):
        """
        Checks self.visited against solutions and nonsolutions.
        :return: F1 score, Success
        """
        if self.task_str == "Touch any object.":
            return True
        elif self.task_str == "Touch all objects.":
            return len(self.visited) / 2 == len(self.map)

        for nonsoln in self.nonsolutions:
            if self.visited.issuperset(nonsoln):
                return False

        for soln in self.solutions:
            if self.visited.issuperset(soln):
                return True

        return False

    def new_task(self, new_task, custom_task=None):
        self.task_no, self.task = new_task
        self.sample_maps(custom_task=custom_task)
        self.reset()

    def sample_maps(self, custom_task=None):
        """
        Sample a task, teaching map, and test maps
        :return:
        """
        self.teaching_map = torch.Tensor(random.choice(self.task['teaching_maps']))
        if self.num_test_rounds == 1:
            self.test_maps = [[torch.Tensor(test_map) for test_map in self.task['test_maps']][0]]
        else:
            self.test_maps = random.sample([torch.Tensor(test_map) for test_map in self.task['test_maps']], self.num_test_rounds)

        if custom_task is None:
            self.task_str = self.task['task_str']
        else:
            assert custom_task in ('Touch any object.', 'Touch all objects.')
            self.task_str = custom_task

    def sample_start_from_edge(self):
        # Returns x, y, angle coordinate
        edge = random.randint(1, 4)

        if edge == 1:
            # Bottom edge
            return random.random() * self.dim, 0, np.pi / 2
        elif edge == 2:
            # Top edge
            return random.random() * self.dim, self.dim, 3 * np.pi / 2
        elif edge == 3:
            # left edge
            return 0, random.random() * self.dim, 0
        else:
            # right edge
            return self.dim, random.random() * self.dim, np.pi

    def reset(self):
        if self.max_teach_iters > 0:
            self.is_teaching = True
        else:
            self.is_teaching = False
        self.success = False
        self.visited = set()  # tracks visited shapes so far
        self.timestep = 0
        self.is_done_with_task = False

        # set new map
        if self.is_teaching: # no teaching round
            self.map = self.teaching_map
        else:
            self.map = self.test_maps[0]

        self.test_no = 0

        # set new solutions
        self.solutions = [set([tuple(obj) for obj in soln]) for soln in self.task['solutions'][self.test_no]]
        self.nonsolutions = self.task['nonsolutions'][self.test_no]

        # Set map, fol, points, reset trajectories
        self.fol = torch.LongTensor(self.task['task_fol'])
        self.sentence_emb = torch.FloatTensor(self.task['task_features'])

        # Track points
        self.points = 0
        self.F1s = [-1]

        # Initialize trajectories and initial position
        x, y, angle = self.sample_start_from_edge()
        # start = torch.rand([1, 2])
        start = torch.Tensor([x, y])
        start_angle = torch.Tensor([angle, angle])
        # start = torch.zeros([1, 2])
        # start_angle = torch.zeros([1, 2])
        # start_angle = torch.rand([1, 2]) * 2 * np.pi

        # "batch size" x num agents x [x, y, vx, vy, ang, ang_v]
        self.trajectories = torch.zeros(1, 2, self.idx_of.a_vel + 1, self.max_teach_iters + self.num_test_rounds * (self.max_test_iters + 1)) # +1 because store first state also
        self.trajectories[:, :, self.idx_of.pos, self.timestep] = start
        self.trajectories[:, :, self.idx_of.a, self.timestep] = start_angle

        return {
            'movement_features': self.trajectories[:,:,:self.idx_of.a + 1, self.timestep],
            'map': self.map[:, 1:] # change shape as we go
        }

    def state_change(self):
        # Move to testing stage
        if not self.is_teaching:
            # print('MOVING ON TO NEXT TASK')
            # track points
            if self.success:
                # print('SUCCSES')
                self.points += 1
                # print('POINTS: ', self.points)

            # Move to next task
            if self.test_no < self.num_test_rounds - 1:
                # print('MOVE TO NEXT TEST MAP')
                # input()
                self.test_no += 1
                self.F1s.append(-1)

                # reset trajectory information and sample a position for student
                x, y, angle = self.sample_start_from_edge()
                start = torch.Tensor([x, y])
                start_angle = torch.Tensor([angle, angle])
                                
                # start = torch.zeros([1, 2])
                # start_angle = torch.zeros([1, 2])

                self.trajectories[:,:,:,self.timestep] = torch.zeros_like(self.trajectories[:,:,:,self.timestep])
                self.trajectories[:, self.student, self.idx_of.pos, self.timestep] = start
                self.trajectories[:, :, self.idx_of.a, self.timestep] = start_angle

            else:
                self.is_done_with_task = True
                return # end of task-- next task will be set

        self.is_teaching = False

        # set new map
        self.map = self.test_maps[self.test_no]

        # set new solutions
        self.solutions = [set([tuple(obj) for obj in soln]) for soln in self.task['solutions'][self.test_no]]
        self.nonsolutions = self.task['nonsolutions'][self.test_no]

    def get_fol(self):
        return self.fol

    def get_sentence_embed(self):
        return self.sentence_emb

    def render(self, mode='human'):
        # unimportant
        pass

    def close(self):
        pass