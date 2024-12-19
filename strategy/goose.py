import numpy as np
from env.GridRule import GridRule
from env.FoxGooseEnv import FoxGooseEnv
from models.PPO import PPO


class Goose():

    def get_action(self, binary_state):
        state = self.binary_state2state(binary_state)
        fox_location = self.get_fox_location(state)
        goose_locations = self.get_goose_location(state)
        self.danger_step(state, fox_location, goose_locations)

    def get_fox_location(self, state):
        for i in range(7):
            for j in range(7):
                if state[i][j] == 'F':
                    return (i, j)

    def get_goose_location(self, state):
        locations = []
        for i in range(7):
            for j in range(7):
                if state[i][j] == 'G':
                    locations.append((i, j))
        return locations

    def binary_state2state(self, binary_state):
        state = []
        row = [3, 3, 7, 7, 7, 3, 3]
        trans = {
            0: '.',
            1: 'F',
            2: 'G',
        }
        j = 0
        for i in range(7):
            sub = []
            if row[i] == 3:
                sub.append(' ')
                sub.append(' ')
                for _ in range(3):
                    sub.append(trans[binary_state[j]])
                    j += 1
                sub.append(' ')
                sub.append(' ')
            else:
                for _ in range(7):
                    sub.append(trans[binary_state[j]])
                    j += 1
            state.append(sub)
        return state

    def danger_step(self, state, fox_location, goose_location):
        step = 0

        # 当前已有的危险
        fox_mask = self.grid_rule.get_fox_mask(state, fox_location)
        for i in range(len(fox_mask)):
            if fox_mask[i] == 1:
                fox_action = self.env._fox_action_to_move[i]
                if max(abs(fox_action[0]), abs(fox_action[1])) > 1:
                    step = 1
                    break

        # 未来的危险

        return False
