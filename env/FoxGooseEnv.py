import gymnasium as gym
import numpy as np
from env.GridRule import GridRule


class FoxGooseEnv(gym.Env):
    def __init__(self):
        super(FoxGooseEnv, self).__init__()
        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 0  1  2  3  4  5  6
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 7  8  9 10 11 12 13
            ['.', '.', '.', '.', '.', '.', '.'],  # 14 15 16 17 18 19 20
            ['G', '.', '.', 'F', '.', '.', 'G'],  # 21 22 23 24 25 26 27
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],  # 28 29 30 31 32 33 34
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],  # 35 36 37 38 39 40 41
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']   # 42 43 44 45 46 47 48
        ]
        self.fox_location = (3, 3)
        self.goose_location = [(3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 2), (5, 3),
                               (5, 4), (6, 2), (6, 3), (6, 4)]
        self._base_move = [
            (0, 1),  # →
            (1, 1),  # ↘
            (1, 0),  # ↓
            (1, -1),  # ↙
            (0, -1),  # ←
            (-1, -1),  # ↖
            (-1, 0),  # ↑
            (-1, 1)  # ↗
        ]
        self._fox_base_multi = [1, 2, 4, 6]
        self.fox_action_space = gym.spaces.Discrete(33)
        self._fox_action_to_move = {}
        for i in range(4):
            for j in range(8):
                self._fox_action_to_move[i * 8 + j] = (
                    self._base_move[j][0] * self._fox_base_multi[i], self._base_move[j][1] * self._fox_base_multi[i])
        self._fox_action_to_move[32] = (0, 0)
        self.goose_action_space = gym.spaces.Discrete(15 * 8)
        self._goose_action_to_move = {}
        for i in range(15):
            for j in range(8):
                self._goose_action_to_move[i * 8 + j] = {i: self._base_move[j]}
        self.grid_rule = GridRule()
        self.role = "fox"
        self.fox_jump = False
        self.fox_mask = self.grid_rule.get_fox_mask(
            self.state, self.fox_location, self.fox_jump)
        self.goose_mask = self.grid_rule.get_goose_mask(
            self.state, self.goose_location)

    def reset(self):
        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],
            [' ', ' ', '.', '.', '.', ' ', ' '],
            ['.', '.', '.', '.', '.', '.', '.'],
            ['G', '.', '.', 'F', '.', '.', 'G'],
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']
        ]
        self.role = "fox"
        self.fox_location = (3, 3)
        self.goose_location = [(3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 2), (5, 3),
                               (5, 4), (6, 2), (6, 3), (6, 4)]
        self.fox_jump = False
        self.fox_mask = self.grid_rule.get_fox_mask(
            self.state, self.fox_location, self.fox_jump)
        self.goose_mask = self.grid_rule.get_goose_mask(
            self.state, self.goose_location)

        return self.get_binary_state(self.state)

    def step(self, action):
        if self.role == "fox":
            move = self._fox_action_to_move[action]
            if move == (0, 0):
                self.role = "goose"
                self.fox_jump = False
            else:
                if move not in self._base_move:
                    # eat goose
                    self.fox_jump = True
                    self.state[self.fox_location[0]+int(move[0] /
                                                        2)][self.fox_location[1]+int(move[1]/2)] = '.'
                    for g in self.goose_location:
                        if g == (self.fox_location[0]+int(move[0]/2), self.fox_location[1]+int(move[1]/2)):
                            self.goose_location.remove(g)
                            break
                self.state[self.fox_location[0]][self.fox_location[1]] = '.'
                self.fox_location = (
                    self.fox_location[0] + move[0], self.fox_location[1] + move[1])
                self.state[self.fox_location[0]
                           ][self.fox_location[1]] = 'F'
                if not self.has_next_step() or not self.fox_jump:
                    self.role = "goose"
                    self.fox_jump = False
        elif self.role == "goose":
            goose_action = self._goose_action_to_move[action]
            goose_idx = list(goose_action.keys())[0]
            move = goose_action[goose_idx]
            self.state[self.goose_location[goose_idx][0]
                       ][self.goose_location[goose_idx][1]] = '.'
            self.goose_location[goose_idx] = (
                self.goose_location[goose_idx][0] + move[0], self.goose_location[goose_idx][1] + move[1])
            self.state[self.goose_location[goose_idx][0]
                       ][self.goose_location[goose_idx][1]] = 'G'

            self.role = "fox"
        else:
            raise ValueError("invalid role")

        # update action space mask
        self.fox_mask = self.grid_rule.get_fox_mask(
            self.state, self.fox_location, self.fox_jump)
        self.goose_mask = self.grid_rule.get_goose_mask(
            self.state, self.goose_location)

        done, winner = self.is_done()

        return self.get_binary_state(self.state), self.reward(), done, {"winner": winner}

    def render(self, mode='human'):
        print(
            "======== goose count " + str(len(self.goose_location)) + " ===========    role: ", self.role)
        for row in self.state:
            print(" ".join(row))
        print()

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def has_next_step(self):
        return np.count_nonzero(self.grid_rule.get_fox_mask(self.state, self.fox_location, self.fox_jump)) > 0

    def is_done(self):
        if self.role == "fox" and np.count_nonzero(self.fox_mask) == 0:
            return True, "goose"
        if self.role == "goose" and len(self.goose_location) < 4:
            return True, "fox"
        return False, None

    def reward(self):
        if self.role == "fox":
            return np.count_nonzero(self.fox_mask) * 0.1 + len(self.goose_location) * -0.02
        elif self.role == "goose":
            return np.count_nonzero(self.goose_mask) * -0.02 + len(self.goose_location) * 0.1

    def get_binary_state(self, state):
        binary_state = []
        for row in state:
            for cell in row:
                if cell == 'F':
                    binary_state.append(1)
                elif cell == 'G':
                    binary_state.append(2)
                elif cell == '.':
                    binary_state.append(0)
        return binary_state
