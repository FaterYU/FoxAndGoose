# Fox and Geese manual play example
# JC4004 Computational Intelligence 2024-25

from models.PPO import PPO
from env.FoxGooseEnv import FoxGooseEnv
import torch


class Player:

    def __init__(self):
        self.weight_path = "weights"
        self.env = FoxGooseEnv()
        self._load_model()

    def _load_model(self):
        gamma = 0.9
        actor_lr = 1e-3
        critic_lr = 1e-2
        epsilon = 0.2
        eps = 0.2
        epochs = 500
        lmdba = 0.95
        n_state = 33
        n_hidden = 128
        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_fox_action = self.env.fox_action_space.n
        n_goose_action = self.env.goose_action_space.n

        self.fox_ppo = PPO(n_state, n_hidden, n_fox_action, epochs,
                           actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)
        self.goose_ppo = PPO(n_state, n_hidden, n_goose_action, epochs,
                             actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)

        self.fox_ppo.load(f"{self.weight_path}/fox")
        self.goose_ppo.load(f"{self.weight_path}/goose")

    # =================================================
    # Print the board

    def print_board(self, board):

        # Prints the current board
        print('')
        print('  0 1 2 3 4 5 6')
        for i in range(len(board)):
            txt = str(i) + ' '
            for j in range(len(board[i])):
                txt += board[i][j] + " "
            print(txt)
        print('')

    # =================================================
    # Play one move as a fox
    def play_fox(self, board):

        # First, print the current board
        self.print_board(board)

        # Get player input for new fox position
        print("Fox plays next!")

        fox_location = self._get_fox_location(board)
        cmd = [list(fox_location)]
        while True:
            state = self.env.get_binary_state(board)
            mask = self.env.grid_rule.get_fox_mask(
                board, cmd[-1], len(cmd) != 1)
            action = self.fox_ppo.get_action(
                state, self.env.fox_action_space, mask)
            for i in range(len(mask)):
                if mask[i] == 1:
                    m = self.env._fox_action_to_move[i]
                    if max(abs(m[0]), abs(m[1])) > 1:
                        action = i
            move = self.env._fox_action_to_move[action]
            if move == (0, 0):
                break
            cmd.append([cmd[-1][0] + move[0], cmd[-1][1] + move[1]])
            board[cmd[-1][0]][cmd[-1][1]] = 'F'
            board[cmd[-2][0]][cmd[-2][1]] = '.'
            if max(abs(move[0]), abs(move[1])) < 2:
                break
            else:
                board[(cmd[-1][0]+cmd[-2][0]) //
                      2][(cmd[-1][1]+cmd[-2][1])//2] = '.'

        return cmd

    # =================================================
    # Play one move as a goose

    def play_goose(self, board):

        # First, print the current board
        self.print_board(board)

        # Get goose start position
        print("Goose plays next!")

        # PPO
        state = self.env.get_binary_state(board)
        goose_location = self._get_goose_location(board)
        mask = self.env.grid_rule.get_goose_mask(board, goose_location)
        action = self.goose_ppo.get_action(
            state, self.env.goose_action_space, mask)
        move = self.env._goose_action_to_move[action]
        move_id = list(move.keys())[0]

        origin = goose_location[move_id]
        target = [origin[0] + move[move_id][0], origin[1] + move[move_id][1]]

        cmd = [origin, target]

        # Hold the tie
        fox_location = self._get_fox_location(board)
        if fox_location[1] != 0:
            if board[3][0] == 'G':
                cmd = [[3, 0], [2, 0]]
            elif board[2][0] == 'G':
                cmd = [[2, 0], [3, 0]]
        else:
            if board[3][6] == 'G':
                cmd = [[3, 6], [2, 6]]
            elif board[2][6] == 'G':
                cmd = [[2, 6], [3, 6]]

        return cmd

    def _get_fox_location(self, state):
        for i in range(7):
            for j in range(7):
                if state[i][j] == 'F':
                    return (i, j)

    def _get_goose_location(self, state):
        locations = []
        for i in range(7):
            for j in range(7):
                if state[i][j] == 'G':
                    locations.append((i, j))
        return locations


# ==== End of file
