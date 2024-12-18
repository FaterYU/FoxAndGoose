import gymnasium as gym
import numpy as np
import copy
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
        now_role = self.role
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
        self.goose_location.sort()
        self.goose_mask = self.grid_rule.get_goose_mask(
            self.state, self.goose_location)

        done, winner = self.is_done()
        return self.get_binary_state(self.state), self.reward(now_role), done, {"winner": winner}

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

    def reward(self, role=None):
        if role == "fox":
            return self.fox_reward()
        elif role == "goose":
            return self.goose_reward()

    def fox_reward(self):
        # goose 个数
        reward_goose_count = np.log(
            15 - len(self.goose_location) + 1) * (15 - len(self.goose_location))

        # fox 当前可走的位置
        reward_fox_space = 0
        for i in range(len(self.fox_mask)):
            if self.fox_mask[i] == 1:
                reward_fox_space += 1

        # fox 正在吃goose
        reward_fox_eat = 1 if self.fox_jump else 0

        # fox 可能到达的位置
        # [[step, (x, y), is_eat], ...]
        fox_search = [[0, self.fox_location, False]]
        state_without_fox = copy.deepcopy(self.state)
        state_without_fox[self.fox_location[0]][self.fox_location[1]] = '.'
        search_idx = 0
        step = 0
        while search_idx < len(fox_search):
            step = fox_search[search_idx][0] + 1
            fox_mask = self.grid_rule.get_fox_mask(
                state_without_fox, fox_search[search_idx][1])
            fox_location = fox_search[search_idx][1]
            for i in range(len(fox_mask)):
                if fox_mask[i] == 1:
                    fox_action = self._fox_action_to_move[i]
                    location = (fox_location[0] + fox_action[0],
                                fox_location[1] + fox_action[1])

                    # check location not in fox_search
                    is_in = False
                    for item in fox_search:
                        if item[1] == location:
                            is_in = True
                            break
                    if is_in:
                        continue

                    if max(abs(fox_action[0]), abs(fox_action[1])) > 1:
                        fox_search.append([step, location, True])
                    elif state_without_fox[location[0]][location[1]] == '.':
                        fox_search.append([step, location, False])
                    else:
                        continue
            search_idx += 1

         # 可以到达的位置越多，reward越高
        reward_fox_space = len(fox_search)

        # 统计潜在机会
        reward_fox_opportunity = 0
        for item in fox_search[1:]:
            if item[2]:
                # 潜在机会步数越大，reward越低
                reward_fox_opportunity += 10 / item[0]
            else:
                reward_fox_opportunity -= 1 / item[0]

        # 加权
        reward = reward_goose_count * 2 + reward_fox_space + \
            reward_fox_eat * 20 + reward_fox_opportunity

        return reward

    def goose_reward(self):
        reward = 0
        # 有危险，直接返回0
        fox_mask = self.grid_rule.get_fox_mask(self.state, self.fox_location)
        for i in range(len(fox_mask)):
            if fox_mask[i] == 1:
                fox_action = self._fox_action_to_move[i]
                if max(abs(fox_action[0]), abs(fox_action[1])) > 1:
                    return 0

        # goose 个数
        reward_goose_count = len(self.goose_location) - 3

        # fox 可能到达的位置
        # [[step, (x, y), is_eat], ...]
        fox_search = [[0, self.fox_location, False]]
        state_without_fox = copy.deepcopy(self.state)
        state_without_fox[self.fox_location[0]][self.fox_location[1]] = '.'
        search_idx = 0
        step = 0
        while search_idx < len(fox_search):
            step = fox_search[search_idx][0] + 1
            fox_mask = self.grid_rule.get_fox_mask(
                state_without_fox, fox_search[search_idx][1])
            fox_location = fox_search[search_idx][1]
            for i in range(len(fox_mask)):
                if fox_mask[i] == 1:
                    fox_action = self._fox_action_to_move[i]
                    location = (fox_location[0] + fox_action[0],
                                fox_location[1] + fox_action[1])

                    # check location not in fox_search
                    is_in = False
                    for item in fox_search:
                        if item[1] == location:
                            is_in = True
                            break
                    if is_in:
                        continue

                    if max(abs(fox_action[0]), abs(fox_action[1])) > 1:
                        fox_search.append([step, location, True])
                    elif state_without_fox[location[0]][location[1]] == '.':
                        fox_search.append([step, location, False])
                    else:
                        continue
            search_idx += 1

        # 可以到达的位置越少，reward越高
        reward_fox_space = 18 / (len(fox_search))

        # 统计潜在风险
        reward_goose_risk = 0
        for item in fox_search[1:]:
            if item[2]:
                # 潜在风险步数越大，reward越低
                reward_goose_risk -= 15 / item[0]
            else:
                reward_goose_risk += item[0] / 5

        # fox可能到达的位置在goose以下，reward越低
        reward_fox_prob_location = 0
        for item in fox_search:
            fox_location = item[1]
            goose_row = [0] * 7
            for goose in self.goose_location:
                goose_row[goose[1]] = max(goose_row[goose[1]], goose[0])
            goose_col = [[-1, 7]] * 7
            for goose in self.goose_location:
                goose_col[goose[0]][0] = max(goose_col[goose[0]][0], goose[1])
                goose_col[goose[0]][1] = min(goose_col[goose[0]][1], goose[1])
            if fox_location[0] < goose_row[fox_location[1]]:
                reward_fox_prob_location -= 1
            if goose_col[fox_location[0]][0] == -1 and goose_col[fox_location[0]][1] == 7:
                continue
            elif goose_col[fox_location[0]][0] == goose_col[fox_location[0]][1]:
                if fox_location[1] == 0 or fox_location[1] == 6:
                    reward_fox_prob_location -= 1
            elif goose_col[fox_location[0]][0] < fox_location[1] or goose_col[fox_location[0]][1] > fox_location[1]:
                reward_fox_prob_location -= 1

        # 统计goose的连通块个数
        reward_goose_connect = 0
        goose_connect = []
        for goose in self.goose_location:
            is_in = False
            for item in goose_connect:
                if goose in item:
                    is_in = True
                    break
            if is_in:
                continue
            connect = [goose]
            goose_connect.append(connect)
            search = [goose]
            while len(search) > 0:
                location = search.pop()
                for move in self._base_move:
                    new_location = (
                        location[0] + move[0], location[1] + move[1])
                    if new_location in self.goose_location and new_location not in connect:
                        connect.append(new_location)
                        search.append(new_location)
        reward_goose_connect = - \
            len(goose_connect) if len(goose_connect) != 1 else 1

        # 加权
        reward = reward_goose_count + reward_fox_space + \
            reward_goose_risk*5 + reward_fox_prob_location + reward_goose_connect*10

        return reward

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
