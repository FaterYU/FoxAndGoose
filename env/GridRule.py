import numpy as np


class GridRule():
    def __init__(self):
        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 0  1  2  3  4  5  6
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 7  8  9 10 11 12 13
            ['.', '.', '.', '.', '.', '.', '.'],  # 14 15 16 17 18 19 20
            ['G', '.', '.', 'F', '.', '.', 'G'],  # 21 22 23 24 25 26 27
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],  # 28 29 30 31 32 33 34
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],  # 35 36 37 38 39 40 41
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']   # 42 43 44 45 46 47 48
        ]

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

        self.fox_rule = {
            #            →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            2: self.mix([2, 3, 4, 0, 0, 0, 0, 0]),
            3: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            4: self.mix([0, 0, 4, 3, 2, 0, 0, 0]),
            9: self.mix([2, 0, 3, 0, 0, 0, 1, 0]),
            #             →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            10: self.mix([1, 2, 3, 2, 1, 1, 1, 1]),
            11: self.mix([0, 0, 3, 0, 2, 0, 1, 0]),
            14: self.mix([4, 3, 2, 0, 0, 0, 0, 0]),
            15: self.mix([3, 0, 2, 0, 1, 0, 0, 0]),
            16: self.mix([3, 2, 3, 2, 2, 0, 2, 2]),
            17: self.mix([2, 0, 3, 0, 2, 0, 2, 0]),
            18: self.mix([2, 2, 3, 2, 3, 2, 2, 0]),
            19: self.mix([1, 0, 2, 0, 3, 0, 0, 0]),
            20: self.mix([0, 0, 2, 3, 4, 0, 0, 0]),
            21: self.mix([4, 0, 1, 0, 0, 0, 1, 0]),
            22: self.mix([3, 2, 1, 1, 1, 1, 1, 2]),
            23: self.mix([3, 0, 2, 0, 2, 0, 2, 0]),
            24: self.mix([2, 1, 2, 1, 2, 1, 2, 1]),
            25: self.mix([2, 0, 2, 0, 3, 0, 2, 0]),
            26: self.mix([1, 1, 1, 2, 3, 2, 1, 1]),
            27: self.mix([0, 0, 1, 0, 4, 0, 1, 0]),
            28: self.mix([4, 0, 0, 0, 0, 0, 2, 3]),
            29: self.mix([3, 0, 0, 0, 1, 0, 2, 0]),
            30: self.mix([3, 2, 2, 0, 2, 2, 3, 2]),
            31: self.mix([2, 0, 2, 0, 2, 0, 3, 0]),
            32: self.mix([2, 0, 2, 2, 3, 2, 3, 2]),
            33: self.mix([1, 0, 0, 0, 3, 0, 2, 0]),
            34: self.mix([0, 0, 0, 0, 4, 3, 2, 0]),
            37: self.mix([2, 0, 1, 0, 0, 0, 3, 0]),
            38: self.mix([1, 1, 1, 1, 1, 2, 3, 2]),
            39: self.mix([0, 0, 1, 0, 2, 0, 3, 0]),
            44: self.mix([2, 0, 0, 0, 0, 0, 4, 3]),
            45: self.mix([1, 0, 0, 0, 1, 0, 4, 0]),
            46: self.mix([0, 0, 0, 0, 2, 3, 4, 0]),
        }

        self.goose_rule = {
            #            →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            2: self.mix([1, 1, 1, 0, 0, 0, 0, 0]),
            3: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            4: self.mix([0, 0, 1, 1, 1, 0, 0, 0]),
            9: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            #             →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            10: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            11: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            14: self.mix([1, 1, 1, 0, 0, 0, 0, 0]),
            15: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            16: self.mix([1, 1, 1, 1, 1, 0, 1, 1]),
            17: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            18: self.mix([1, 1, 1, 1, 1, 1, 1, 0]),
            19: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            20: self.mix([0, 0, 1, 1, 1, 0, 0, 0]),
            21: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            22: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            23: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            24: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            25: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            26: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            27: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            28: self.mix([1, 0, 0, 0, 0, 0, 1, 1]),
            29: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            30: self.mix([1, 1, 1, 0, 1, 1, 1, 1]),
            31: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            32: self.mix([1, 0, 1, 1, 1, 1, 1, 1]),
            33: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            34: self.mix([0, 0, 0, 0, 1, 1, 1, 0]),
            37: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            38: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            39: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            44: self.mix([1, 0, 0, 0, 0, 0, 1, 1]),
            45: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            46: self.mix([0, 0, 0, 0, 1, 1, 1, 0]),
        }

    def mix(self, multi_list):
        result = []
        for i in range(8):
            for j in range(multi_list[i]):
                result.append(
                    (self._base_move[i][0] * self._fox_base_multi[j], self._base_move[i][1] * self._fox_base_multi[j]))
        return result

    def get_fox_mask(self, state, location, jump=False):
        # TODO: bug to fix
        next_space = self.fox_rule[location[0]*7+location[1]]
        mask = [0] * 33
        for move in next_space:
            if state[location[0]+move[0]][location[1]+move[1]] == '.':
                multi = self._fox_base_multi.index(
                    max(abs(move[0]), abs(move[1])))
                base = self._base_move.index(
                    (move[0]//self._fox_base_multi[multi], move[1]//self._fox_base_multi[multi]))
                if multi > 0 and state[location[0]+move[0]//2][location[1]+move[1]//2] != 'G':
                    continue
                if not jump:
                    mask[multi*8+base] = 1
                else:
                    if multi == 0:
                        mask[32] = 1
                    else:
                        mask[multi*8+base] = 1

        return np.array(mask, dtype=np.int8)

    def get_goose_mask(self, state, location):
        mask = [0] * 15 * 8
        for i in range(len(location)):
            g = location[i]
            next_space = self.goose_rule[g[0]*7+g[1]]
            for move in next_space:
                if state[g[0]+move[0]][g[1]+move[1]] == '.':
                    mask[i*8+self._base_move.index(move)] = 1
        return np.array(mask, dtype=np.int8)
